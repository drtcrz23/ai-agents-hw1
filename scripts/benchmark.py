#!/usr/bin/env python3
"""Загрузить модель, прогреть, измерить каждый промпт три раза, сохранить лог."""
import argparse
import fcntl
import hashlib
import json
import platform
import subprocess
from pathlib import Path

from common import ROOT, MemorySampler, dump, public_text, stream_request, summarize, timestamp
from lifecycle import load_exact, memory_root_pid, unload_owned


def loaded():
    result = subprocess.run(['lms', 'ps', '--json'], capture_output=True, text=True, timeout=10)
    if result.returncode:
        raise RuntimeError(result.stderr or result.stdout)
    return json.loads(result.stdout)


def verify_loaded(current, model):
    if len(current) != 1 or current[0]['identifier'] != model['identifier']:
        raise RuntimeError('Ожидался единственный экземпляр ' + model['identifier'])
    item = current[0]
    if item.get('quantization', {}).get('name') != model['quant']:
        raise RuntimeError('Загружено другое квантование')
    if 'qwen3.5-9b' not in item.get('path', '').lower():
        raise RuntimeError('Загружена другая модель')


def environment():
    info = {'platform': platform.platform(), 'python': platform.python_version()}
    commands = {'chip': ['sysctl', '-n', 'machdep.cpu.brand_string'],
                'ram_bytes': ['sysctl', '-n', 'hw.memsize'],
                'power': ['pmset', '-g', 'batt'], 'thermal': ['pmset', '-g', 'therm'],
                'lms_version': ['lms', '--version'], 'runtimes': ['lms', 'runtime', 'ls']}
    for name, cmd in commands.items():
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        info[name] = (result.stdout or result.stderr).strip()
    info['script_hashes'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in (ROOT/'scripts').glob('*.py')}
    return info


def measure(text, model, cfg, pid):
    payload = {'model': model['identifier'], 'input': text, 'stream': True,
               'store': False, **cfg['generation']}
    memory = MemorySampler(pid, cfg['memory_sample_seconds'])
    row, events = stream_request(cfg['base_url'], payload, cfg['timeout_seconds'], memory)
    return row, events, {'samples': memory.samples, 'errors': memory.errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['q4', 'q8'], required=True)
    parser.add_argument('--mode', choices=['smoke', 'speed', 'context'], required=True)
    parser.add_argument('--lengths', type=int, nargs='+', help='Длины готовых промптов')
    parser.add_argument('--memory-pid', type=int, help='PID приложения; обычно определяется сам')
    parser.add_argument('--plan', action='store_true', help='Показать план без запуска модели')
    args = parser.parse_args()
    cfg = json.loads((ROOT/'experiment.json').read_text())
    model = cfg['models'][args.model]
    targets = args.lengths or ([128] if args.mode == 'smoke' else cfg[args.mode+'_lengths'])
    repeats = 1 if args.mode == 'smoke' else cfg['repeats']
    if targets != sorted(set(targets)) or repeats < (1 if args.mode == 'smoke' else 3):
        parser.error('Длины должны возрастать без повторов; основных повторов должно быть >=3')
    if args.mode != 'smoke':
        cfg[args.mode+'_lengths'] = targets
    prompts = {}
    for target in targets:
        window = cfg['speed_context'] if args.mode != 'context' else target + cfg['context_reserve']
        if target < 64 or target + cfg['generation']['max_output_tokens'] + 16 > window:
            parser.error('В окне не хватает места для промпта и ответа')
        path = ROOT/'prompts'/f'input_{target}.txt'
        if not path.exists():
            parser.error(f'Нет готового промпта: {path.name}')
        text = path.read_text()
        meta = json.loads(path.with_suffix('.json').read_text())
        if hashlib.sha256(text.encode()).hexdigest() != meta['sha256']:
            parser.error(f'Промпт изменён: {path.name}')
        prompts[target] = (text, meta)
        print(f'{args.model}: вход {target}, окно {window}, повторов {repeats}', flush=True)
    if args.plan:
        return

    # Блокировка не позволяет случайно запустить два эксперимента одновременно.
    lock = (ROOT/'.run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pid = args.memory_pid or memory_root_pid()
    import lmstudio
    import psutil
    process = psutil.Process(pid)
    if process.name().lower() not in ('bionic', 'lm studio', 'lm-studio', 'llmster'):
        parser.error('Для памяти нужен PID основного процесса LM Studio/Bionic')
    client = lmstudio.get_default_client()
    run_id = timestamp().replace(':', '-') + f'-{args.model}-{args.mode}'
    folder = ROOT/'logs'/run_id
    folder.mkdir()
    rows, status, reason = [], 'complete', 'All selected lengths tested; this is not a hardware limit'
    with (folder/'run.log').open('w', buffering=1) as log:
        def emit(kind, **data):
            row = {'time': timestamp(), 'kind': kind, **data}
            log.write(public_text(json.dumps(row, ensure_ascii=False)) + '\n')
            if kind in ('measurement', 'error', 'summary'):
                print(public_text(json.dumps(row, ensure_ascii=False)), flush=True)
            return row

        emit('start', run_id=run_id, model=args.model, mode=args.mode, config=cfg,
             management='automatic', memory_process={'pid': pid, 'name': process.name()})
        try:
            dump(folder/'environment.json', environment())
            for target in targets:
                window = cfg['speed_context'] if args.mode != 'context' else target + cfg['context_reserve']
                text, meta = prompts[target]
                for repeat in range(1, repeats + 1):
                    print(f'\nВход {target}, повтор {repeat}/{repeats}: загрузка -> прогрев -> замер', flush=True)
                    unload_owned(model, loaded, verify_loaded, client, emit)
                    emit('unloaded', rss_bytes=MemorySampler(pid).sample())
                    handle, current = load_exact(model, window, client, loaded, verify_loaded, emit)
                    dump(folder/f'{target}-{repeat}-load.json',
                         {'ps': current, 'load_config': handle.get_load_config().to_dict()})
                    for kind, prompt in [('warmup', 'Write a cheerful four-line poem about a red balloon.'),
                                         ('measurement', text)]:
                        row, events, memory = measure(prompt, model, cfg, pid)
                        if kind == 'measurement' and row['status'] == 'ok':
                            if (row['cold_load'] or abs(row['input_tokens'] - target) > 64
                                    or row['input_tokens'] + cfg['generation']['max_output_tokens'] > window
                                    or row['server_stats'].get('reasoning_output_tokens', 0) > 0):
                                row.update(status='error', error='Холодная загрузка, неверная длина входа или reasoning')
                        stem = f'{target}-{repeat}-{kind}'
                        dump(folder/(stem+'-stream.json'), events)
                        dump(folder/(stem+'-memory.json'), memory)
                        record = emit(kind, model=args.model, target_tokens=target, repeat=repeat,
                                      context_length=window, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(), **row)
                        if kind == 'measurement':
                            rows.append(record)
                        if row['status'] != 'ok':
                            raise RuntimeError(row['error'])
        except (Exception, KeyboardInterrupt) as exc:
            status, reason = 'stopped', f'{type(exc).__name__}: {exc}'
            emit('error', reason=reason)
        finally:
            try:
                unload_owned(model, loaded, verify_loaded, client, emit)
            except Exception as exc:
                status, reason = 'stopped', reason + f'; выгрузка: {exc}'
                emit('error', reason=str(exc))
            groups = summarize(rows, repeats)
            result = dict(run_id=run_id, model=args.model, mode=args.mode, status=status,
                          stop_reason=reason, repeats_required=repeats, groups=groups,
                          max_stable_target_tokens=max((g['target_tokens'] for g in groups if g['stable']), default=None))
            emit('summary', **result)
            dump(folder/'summary.json', result)
    lock.close()
    print(f'Логи: {folder.relative_to(ROOT)}')
    if status != 'complete':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
