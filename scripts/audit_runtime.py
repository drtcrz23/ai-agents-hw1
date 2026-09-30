#!/usr/bin/env python3
"""Match successful measurements to full-input prompt-evaluation runtime timings."""
import argparse
import datetime as dt
import json
import re
from pathlib import Path
from zoneinfo import ZoneInfo
from common import ROOT, dump


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--server-logs', type=Path, default=Path.home()/'.lmstudio/apps/bionic/server-logs')
    ap.add_argument('--timezone', default='Europe/Moscow', help='Timezone used by runtime log timestamps')
    args = ap.parse_args()
    entries = []
    for p in sorted(args.server_logs.rglob('*.log')):
        for line in p.read_text(errors='replace').splitlines():
            m = re.search(r'^\[([\d-]+ [\d:]+)\].*prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens', line)
            if m:
                stamp = dt.datetime.strptime(m[1], '%Y-%m-%d %H:%M:%S').replace(tzinfo=ZoneInfo(args.timezone))
                entries.append((stamp, int(m[3]), line))
    checks = []
    for folder in json.loads((ROOT/'results/selected_runs.json').read_text()):
        folder = ROOT / folder  # Absolute paths remain absolute; local paths are portable.
        for line in Path(folder, 'run.log').read_text().splitlines():
            r = json.loads(line)
            if r['kind'] != 'measurement' or r['status'] != 'ok':
                continue
            # Runtime timestamps have only one-second precision.
            start = dt.datetime.fromisoformat(r['started_at']) - dt.timedelta(seconds=1)
            end = dt.datetime.fromisoformat(r['time']) + dt.timedelta(seconds=1)
            hits = [v for t,n,v in entries if start <= t <= end and n == r['input_tokens']]
            checks.append({'run':Path(folder).name, 'target':r['target_tokens'], 'repeat':r['repeat'],
                           'input_tokens':r['input_tokens'], 'matching_runtime_lines':hits})
    dump(ROOT/'results/runtime-input-audit.json', checks)
    matched = sum(len(c['matching_runtime_lines']) == 1 for c in checks)
    print(f'Full-input runtime timings matched: {matched}/{len(checks)}')
    if matched != len(checks) or not checks:
        raise SystemExit('Some timings are missing or ambiguous; inspect runtime-input-audit.json')


if __name__ == '__main__':
    main()
