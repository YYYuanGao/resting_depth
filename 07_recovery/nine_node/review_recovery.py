#!/usr/bin/env python3
"""Summarize existing recovery CSVs and package small reports; never starts MATLAB."""
import argparse
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import statistics as st
import zipfile

PAIRS = ('EVC_PPC', 'EVC_cauPFC', 'PPC_cauPFC')
TRUTHS = ('observed', 'null', 'reversed')


def summarize(rows, snrs, reps, seed):
    expected = {(t, float(s), r) for t in TRUTHS for s in snrs for r in range(1, reps+1)}
    keys = [(x['truth'], float(x['SNR']), int(x['repetition'])) for x in rows]
    if len(set(keys)) != len(keys) or set(keys) != expected:
        raise ValueError('Missing/duplicate/unexpected simulation cases; retain the CSV and inspect status/logs')
    groups = []
    for t in TRUTHS:
        for s in snrs:
            block = [x for x in rows if x['truth'] == t and float(x['SNR']) == s]
            valid = [x for x in block if str(x['success']).lower() in ('1', 'true')]
            for x in block:
                if int(x['seed']) != seed + 10000*(snrs.index(s)+1) + int(x['repetition']):
                    raise ValueError('Unexpected seed')
            for pair in PAIRS:
                for metric in ('FF', 'FB', 'H'):
                    k = pair+'_'+metric
                    truth = [float(x['true_'+k]) for x in block]
                    if not all(math.isfinite(v) for v in truth) or max(truth)-min(truth)>1e-12:
                        raise ValueError('Inconsistent truth')
                    estimates = [float(x['estimated_'+k]) for x in valid]
                    if not all(math.isfinite(v) for v in estimates):
                        raise ValueError('Successful case has invalid estimates')
                    errors = [v-truth[0] for v in estimates]
                    sign = [int(v*truth[0]>0) for v in estimates] if abs(truth[0])>1e-10 else []
                    row = dict(truth=t, SNR=s, pair=pair, metric=metric, expected=reps,
                               successful=len(valid), failed=reps-len(valid), true_value=truth[0],
                               mean_estimate=st.mean(estimates) if estimates else '',
                               mean_bias=st.mean(errors) if errors else '',
                               rmse=math.sqrt(st.mean([e*e for e in errors])) if errors else '',
                               estimate_sd=st.stdev(estimates) if len(estimates)>1 else '',
                               estimate_min=min(estimates) if estimates else '',
                               estimate_max=max(estimates) if estimates else '',
                               sign_recovery_fraction=st.mean(sign) if sign else '',
                               summary_scope='all_cases_succeeded' if len(valid)==reps else 'INCOMPLETE_successful_cases_only_not_final',
                               range_definition='simulation variability, not participant confidence interval')
                    groups.append(row)
    return groups


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, type=Path)
    p.add_argument('--mode', choices=('pilot', 'formal'), default='pilot')
    p.add_argument('--pack', action='store_true')
    a = p.parse_args(); root = a.run.resolve(); phase = root/a.mode
    lock = json.loads((root/'run_lock.json').read_text()); c = lock['config']
    status = json.loads((phase/'status.json').read_text())
    if lock['token'] != status['token']:
        raise ValueError('Run token mismatch')
    with (phase/'recovery_cases.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    summary = summarize(rows, c['snrs'], c[a.mode+'_repeats'], c['master_seed'])
    with (phase/'recovery_summary.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    print('Summary:', phase/'recovery_summary.csv')
    print('Technical status:', status['status'], '(not a finding that recovery is accurate)')
    if a.pack:
        name = 'nine_recovery_'+a.mode+'_'+dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.zip'
        target = root/name
        files = [x for x in root.iterdir() if x.suffix in ('.json', '.csv')]
        files += [x for x in phase.rglob('*') if x.suffix in ('.json', '.csv', '.log')]
        inventory = []
        with zipfile.ZipFile(target, 'x', zipfile.ZIP_DEFLATED) as z:
            for f in sorted(set(files)):
                if f.is_symlink() or f.stat().st_size>10*1024*1024:
                    raise ValueError('Unexpected symlink/large report: '+str(f))
                raw = f.read_bytes(); rel = str(f.relative_to(root))
                z.writestr(rel, raw)
                inventory.append(dict(path=rel, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
            z.writestr('pack_manifest.json', json.dumps(inventory, indent=2))
        with zipfile.ZipFile(target) as z:
            if z.testzip() is not None:
                raise ValueError('ZIP integrity check failed')
        print('Upload:', target)

if __name__ == '__main__':
    main()
