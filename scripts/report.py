#!/usr/bin/env python3
"""Build a report from explicitly selected real benchmark run directories."""
import argparse
import json
import math
from pathlib import Path
from common import ROOT, dump, public_text


def merge_continuations(runs):
    """Combine disjoint steps only; never pool repeated attempts at one length."""
    merged = {}
    for folder, result, cfg in sorted(runs, key=lambda item: item[1]['run_id']):
        key = (result['model'], result['mode'])
        if key not in merged:
            merged[key] = (str(folder), dict(result, groups=list(result['groups'])), cfg)
            continue
        paths, prior, original_cfg = merged[key]
        if result['mode'] != 'context':
            raise ValueError(f'Multiple speed runs for {key}; select one explicitly')
        old_targets = {g['target_tokens'] for g in prior['groups']}
        new_targets = {g['target_tokens'] for g in result['groups']}
        overlap = old_targets & new_targets
        if overlap:
            raise ValueError(f'Overlapping context steps for {key}: {sorted(overlap)}; select runs explicitly')
        prior['groups'] = sorted(prior['groups'] + result['groups'], key=lambda g: g['target_tokens'])
        prior['stop_reason'] += ' | Continuation: ' + result['stop_reason']
        prior['status'] = result['status']
        merged[key] = (paths + '; ' + str(folder), prior, original_cfg)
    return list(merged.values())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('runs', nargs='*', type=Path, help='Run directories containing summary.json')
    args = ap.parse_args()
    if not args.runs:
        args.runs = [ROOT / p for p in json.loads((ROOT/'results/selected_runs.json').read_text())]
    runs, seen = [], set()
    for folder in args.runs:
        result = json.loads((folder/'summary.json').read_text())
        if result['mode'] in ('smoke', 'load-check'):
            raise SystemExit('Smoke tests are not experimental results')
        key = (result['model'], result['mode'])
        seen.add(key)
        start = json.loads((folder/'run.log').read_text().splitlines()[0])
        if start.get('kind') != 'start':
            raise SystemExit(f'Missing start record: {folder}')
        runs.append((folder, result, start['config']))
    # Refuse silently comparing different generation/measurement protocols.
    # Context sweeps can stop at different upper bounds; each step still uses
    # target+reserve and the same generation/timeout/repeat protocol.
    compare_keys = ('generation', 'repeats', 'speed_context', 'speed_lengths',
                    'context_reserve', 'cache_policy', 'timeout_seconds', 'memory_sample_seconds')
    first = runs[0][2]
    for folder, _, cfg in runs[1:]:
        if any(cfg[k] != first[k] for k in compare_keys):
            raise SystemExit(f'Protocol differs in {folder}; use separate reports')
    source_runs = list(runs)
    try:
        runs = merge_continuations(runs)
    except ValueError as exc:
        raise SystemExit(str(exc))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    table = ['# Результаты измерений', '',
             'Медианы рассчитаны отдельно для каждой конфигурации и длины входа. '
             'Неполные группы не используются для сравнений. Память - RSS дерева процессов, '
             'включая служебные расходы; это не точный объём Metal-аллокаций.', '',
             'Конфигурации: q4 = Qwen3.5-9B Q4_K_M; q8 = Qwen3.5-9B Q8_0. '
             'Движок, методика и ограничения сравнения описаны в [README](../README.md). '
             'Питание и фоновая память между сессиями различались; результаты отражают '
             'эти условия и не изолируют влияние квантования.', '',
             '| Модель | Тест | Вход, токены (медиана) | Успехи | TTFT, с | Токенов/с | Latency, с | RSS до, ГиБ | Пик RSS, ГиБ (медиана / максимум) |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(x, scale=1):
        return '-' if x is None else f'{x / scale:.3f}'
    for folder, result, _ in runs:
        for g in result['groups']:
            note = '' if g['stable'] else ' (неполная группа)'
            table.append('| ' + ' | '.join([
                result['model'], result['mode']+note,
                fmt(g['input_tokens_median']) if g['input_tokens_median'] is not None else f'цель {g["target_tokens"]}', f'{g["successful"]}/{result["repeats_required"]}',
                fmt(g['ttft_s_median']), fmt(g['tokens_per_second_median']), fmt(g['latency_s_median']),
                fmt(g['rss_before_bytes_median'], 2**30),
                fmt(g['rss_peak_bytes_median'], 2**30)+' / '+fmt(g['rss_peak_bytes_max'], 2**30)
            ]) + ' |')
    table += ['', '## Границы проверенного контекста', '']
    for folder, result, _ in runs:
        if result['mode'] != 'context':
            continue
        stable = [g for g in result['groups'] if g['stable']]
        if stable:
            g = max(stable, key=lambda x: x['target_tokens'])
            table.append(f'- {result["model"]}: целевая длина {g["target_tokens"]}, '
                         f'фактический вход {fmt(g["input_tokens_median"])} токенов; '
                         f'TTFT {fmt(g["ttft_s_median"])} с; '
                         f'пик RSS до {fmt(g["rss_peak_bytes_max"], 2**30)} ГиБ. '
                         f'Остановка: {result["stop_reason"]}.')
        else:
            table.append(f'- {result["model"]}: стабильных ступеней нет. {result["stop_reason"]}.')
    table += ['', 'Завершение всех выбранных ступеней даёт нижнюю границу рабочего контекста. '
              'Таймаут запроса означает практическое временное ограничение, а не доказательство '
              'нехватки памяти. Ожидание ручной загрузки не является пределом инференса.',
              '', '## Неуспешные запросы', '']
    for folder, result, _ in source_runs:
        for line in (folder/'run.log').read_text().splitlines():
            row = json.loads(line)
            if row.get('kind') == 'measurement' and row.get('status') != 'ok':
                table.append(f'- {result["model"]}, цель {row["target_tokens"]}, повтор {row["repeat"]}: '
                             f'{row.get("error")}; время до ошибки '
                             f'{fmt(row.get("elapsed_until_error_s"))} с; наблюдаемый пик RSS '
                             f'{fmt(row.get("rss_peak_bytes"), 2**30)} ГиБ. '
                             'Полный ответ не получен; успешные TTFT и throughput не подставляются.')
    table += ['', '## Исходные прогоны', '']
    table += [f'- `{folder}`: {r["status"]}; {r["stop_reason"]}.' for folder, r, _ in source_runs]
    expected = {('q4','speed'), ('q4','context'), ('q8','speed'), ('q8','context')}
    missing = expected-seen
    if missing:
        table += ['', '**Отчёт неполный:** отсутствуют '+', '.join(str(x) for x in sorted(missing))+'.']
    for mode, metric, label, filename, scale in [
        ('context','ttft_s_median','TTFT (s)','ttft_vs_context.png',1),
        ('context','rss_peak_bytes_median','Median peak process-tree RSS (GiB)','memory_vs_context.png',2**30),
        ('speed','tokens_per_second_median','Generation tokens/s (server)','throughput.png',1),
        ('speed','latency_s_median','Request latency (s)','latency.png',1),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        count = 0
        for _, r, _ in runs:
            points = [g for g in r['groups'] if g['stable']]
            if r['mode'] != mode or not points:
                continue
            ax.plot([g['input_tokens_median'] for g in points],
                    [g[metric]/scale for g in points], marker='o', label=r['model'])
            count += 1
        if count:
            ax.set_xscale('log', base=2)
            ax.set_xlabel('Actual input tokens (median)')
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(ROOT/'figures'/filename, dpi=160)
            table += ['', f'![{label}](../figures/{filename})']
        plt.close(fig)
    dump(ROOT/'results'/'selected_runs.json',
         [str(p.resolve().relative_to(ROOT)) if p.resolve().is_relative_to(ROOT)
          else str(p.resolve()) for p, _, _ in source_runs])
    path = ROOT/'results'/'report.md'
    path.write_text(public_text('\n'.join(table))+'\n')
    print(path.relative_to(ROOT))


if __name__ == '__main__':
    main()
