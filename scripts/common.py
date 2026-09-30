"""Shared parsing, deadline, memory and aggregation code. No inference on import."""
import contextlib
import datetime
import http.client
import json
import math
import signal
import statistics
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]


def public_text(text):
    """Сохранять пути проекта относительно папки, домашние пути - через ~."""
    return text.replace(str(ROOT) + '/', '').replace(str(Path.home()), '~')


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(public_text(json.dumps(value, ensure_ascii=False, indent=2)) + '\n')


def timestamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


@contextlib.contextmanager
def deadline(seconds):
    """Hard wall-clock deadline, including a server that keeps sending progress."""
    def expired(*_):
        raise TimeoutError(f'Request exceeded {seconds}s wall-clock limit')
    old = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def sse_events(lines):
    data = []
    event = None
    for raw in lines:
        line = raw.decode('utf-8').rstrip('\r\n')
        if not line:
            if data:
                obj = json.loads('\n'.join(data))
                if event and 'type' not in obj:
                    obj['type'] = event
                yield obj
            data, event = [], None
        elif line.startswith('data:'):
            data.append(line[5:].lstrip(' '))
        elif line.startswith('event:'):
            event = line[6:].strip()
    if data:
        raise ValueError('Truncated SSE event: missing final blank line')


class StreamMetrics:
    def __init__(self, expected_model):
        self.expected_model = expected_model
        self.first = None
        self.first_answer = None
        self.end = None
        self.result = None
        self.cold = False
        self.errors = []

    def accept(self, event, elapsed):
        kind = event.get('type')
        if kind == 'error':
            self.errors.append(event.get('error', event))
        if kind == 'model_load.start':
            self.cold = True
        if kind in ('chat.start', 'chat.end'):
            model = (event.get('result') or event).get('model_instance_id')
            if model != self.expected_model:
                raise ValueError(f'Wrong model instance: {model!r}, expected {self.expected_model!r}')
        if kind in ('message.delta', 'reasoning.delta') and event.get('content'):
            if self.first is None:
                self.first = elapsed
            if kind == 'message.delta' and self.first_answer is None:
                self.first_answer = elapsed
        if kind == 'chat.end':
            self.result = event['result']
            self.end = elapsed

    def finish(self):
        if self.errors:
            raise ValueError(f'Server stream error: {self.errors}')
        if self.result is None or self.first is None:
            raise ValueError('Incomplete stream or no generated text')
        stats = self.result['stats']
        for key in ('input_tokens', 'total_output_tokens', 'tokens_per_second'):
            value = stats.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'Invalid/missing server statistic {key}: {value!r}')
        return {
            'ttft_s': self.first, 'first_answer_s': self.first_answer,
            'latency_s': self.end, 'tokens_per_second': stats['tokens_per_second'],
            'input_tokens': stats['input_tokens'],
            'output_tokens': stats['total_output_tokens'],
            'cold_load': self.cold or stats.get('model_load_time_seconds', 0) > 0,
            'server_stats': stats,
        }


class MemorySampler:
    """RSS of an explicitly selected app/service process tree; not Metal allocation size."""
    def __init__(self, pid, interval=0.1):
        import psutil
        self.psutil = psutil
        self.root = psutil.Process(pid)
        self.interval = interval
        self.samples = []
        self.errors = []
        self.stop_event = threading.Event()
        self.thread = None

    def sample(self):
        try:
            processes = [self.root] + self.root.children(recursive=True)
            total, pids = 0, []
            for p in processes:
                try:
                    total += p.memory_info().rss
                    pids.append(p.pid)
                except self.psutil.NoSuchProcess:
                    continue
            if self.root.pid not in pids:
                raise RuntimeError('Memory root process exited')
            row = {'time': timestamp(), 'rss_bytes': total, 'pids': pids}
            self.samples.append(row)
            return total
        except Exception as exc:
            self.errors.append(str(exc))
            return None

    def _loop(self):
        while not self.stop_event.wait(self.interval):
            self.sample()

    def start(self):
        self.sample()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2)
        self.sample()

    def metrics(self):
        if self.errors or not self.samples:
            raise RuntimeError(f'Incomplete memory measurement: {self.errors}')
        return {'rss_before_bytes': self.samples[0]['rss_bytes'],
                'rss_peak_bytes': max(s['rss_bytes'] for s in self.samples),
                'rss_after_bytes': self.samples[-1]['rss_bytes']}


def stream_request(base_url, payload, timeout, sampler):
    url = urlparse(base_url)
    if url.scheme != 'http' or url.hostname not in ('127.0.0.1', 'localhost', '::1'):
        raise ValueError('Only local HTTP inference is supported')
    conn = http.client.HTTPConnection(url.hostname, url.port or 1234, timeout=timeout)
    metrics = StreamMetrics(payload['model'])
    events = []
    row = {'started_at': timestamp(), 'status': 'error'}
    sampler.start()
    started = time.perf_counter()
    try:
        with deadline(timeout):
            conn.request('POST', '/api/v1/chat', json.dumps(payload).encode(),
                         {'Content-Type': 'application/json', 'Accept': 'text/event-stream'})
            response = conn.getresponse()
            if response.status != 200:
                raise RuntimeError(f'HTTP {response.status}: {response.read(4096).decode(errors="replace")}')
            for event in sse_events(response):
                elapsed = time.perf_counter() - started
                events.append({'elapsed_s': elapsed, 'event': event})
                metrics.accept(event, elapsed)
                if event.get('type') == 'chat.end':
                    break
            row.update(metrics.finish())
            row['status'] = 'ok'
    except Exception as exc:
        row.update(error=f'{type(exc).__name__}: {exc}', elapsed_until_error_s=time.perf_counter()-started)
    finally:
        conn.close()
        sampler.stop()
    try:
        row.update(sampler.metrics())
    except Exception as exc:
        row.update(status='error', error=str(exc))
    return row, events


def summarize(rows, repeats):
    groups = {}
    for row in rows:
        if row.get('kind') == 'measurement':
            groups.setdefault(row['target_tokens'], []).append(row)
    result = []
    fields = ('input_tokens', 'ttft_s', 'tokens_per_second', 'latency_s',
              'rss_before_bytes', 'rss_peak_bytes', 'rss_after_bytes', 'output_tokens')
    for target, items in sorted(groups.items()):
        good = [r for r in items if r['status'] == 'ok' and not r.get('cold_load')]
        group = {'target_tokens': target, 'attempted': len(items), 'successful': len(good),
                 'stable': len(good) == repeats and len(items) == repeats}
        for key in fields:
            group[key + '_median'] = statistics.median(r[key] for r in good) if good else None
        group['rss_peak_bytes_max'] = max((r['rss_peak_bytes'] for r in good), default=None)
        result.append(group)
    return result
