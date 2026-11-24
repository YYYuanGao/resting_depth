#!/usr/bin/env python3
"""Per-pair FF/FB from the voxel-matched nine-node fits (Table S6 matched rows, Fig. S3).

Reads every successful nine-node formal fit listed in <run>/formal_fits.csv, extracts
FF and FB for the three regional pairs from A(target, source), checks the EVC-cauPFC
values against formal_fits.csv, averages the 100 repetitions within each participant
and writes matched_pair_pathways.csv (subject, pair, FF, FB). Read-only on the run.
Node order: EVC, PPC, cauPFC x deep, middle, superficial.
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

STAGE = {'EVC': 0, 'PPC': 3, 'cauPFC': 6}
PAIRS = [('EVC', 'PPC'), ('EVC', 'cauPFC'), ('PPC', 'cauPFC')]
DEEP, MID, SUP = 0, 1, 2


def ff_fb(A, lo, hi):
    l, h = STAGE[lo], STAGE[hi]
    ff = (A[h + DEEP, l + SUP] + A[h + MID, l + SUP]) / 2     # lower superficial -> higher deep/middle
    fb = (A[l + DEEP, h + DEEP] + A[l + SUP, h + DEEP]) / 2   # higher deep -> lower deep/superficial
    return ff, fb


def result_file(run, row):
    p = Path(str(row['attempt']))
    if p.is_file():
        return p
    if p.is_dir() and (p / 'result.json').exists():
        return p / 'result.json'
    c = sorted((run / 'formal' / row['subject'] / 'nine' / f"rep-{int(row['rep']):03d}").glob('attempt-*/result.json'))
    return c[-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    fits = pd.read_csv(a.run / 'formal_fits.csv')
    fits = fits[(fits.model == 'nine') & (fits.status == 'SUCCESS')]
    rows, maxdiff = [], 0.0
    for _, r in fits.iterrows():
        js = json.load(open(result_file(a.run, r)))
        A = np.asarray(js['A'], float)
        assert A.shape == (9, 9), A.shape
        for lo, hi in PAIRS:
            ff, fb = ff_fb(A, lo, hi)
            if (lo, hi) == ('EVC', 'cauPFC'):
                maxdiff = max(maxdiff, abs(ff - r.FF), abs(fb - r.FB))
            rows.append(dict(subject=r.subject, rep=int(r.rep), pair=f'{lo}–{hi}', FF=ff, FB=fb))
    long = pd.DataFrame(rows)
    n = long.groupby(['subject', 'pair']).size()
    print('fits:', len(fits), ' reps per subject x pair:', n.min(), '-', n.max(),
          ' max |EVC-cauPFC - formal_fits| =', f'{maxdiff:.2e}')
    assert maxdiff < 1e-9, 'EVC-cauPFC values do not match formal_fits.csv: check node order'
    m = long.groupby(['subject', 'pair'], as_index=False)[['FF', 'FB']].mean()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(a.out, index=False)
    ps = pd.read_csv(a.run / 'participant_summary.csv'); ps = ps[ps.model == 'nine'].set_index('subject')
    chk = m[m.pair == 'EVC–cauPFC'].set_index('subject')
    print('max |mean EVC-cauPFC FB - participant_summary| =', f"{(chk.FB - ps.FB).abs().max():.2e}")
    print('written:', a.out)


if __name__ == '__main__':
    main()
