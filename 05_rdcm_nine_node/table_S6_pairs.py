#!/usr/bin/env python3
"""Per-pair FF, FB and H in the nine-node model (Results 2.5, Supplementary Table S6, Fig. 4c-d).

Input: one row per participant and regional pair with signed FF and FB, either
  * evc_pooled_subject_stage_pair_pathways.csv (05_rdcm_nine_node, original estimates), or
  * matched_pair_pathways.csv (06_voxel_matching/collect_matched_pairs.py, voxel-matched).

For each pair, FF, FB and H = FB - FF are tested against zero (two-sided one-sample t,
t-based 95% CI, exact sign-flip over all 2^15 sign patterns); Holm correction across the
three pairs within each measure. Exploratory paired comparisons of H between pairs use
paired t and exact sign-flip tests with Holm correction across the three contrasts.
"""
import argparse
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

PAIRS = ['EVC–PPC', 'EVC–cauPFC', 'PPC–cauPFC']


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); out = np.empty_like(p); run = 0
    for k, i in enumerate(o):
        run = max(run, min(1, (len(p) - k) * p[i])); out[i] = run
    return out


def signflip(x):
    x = np.asarray(x, float); s = np.array(list(product([1, -1], repeat=len(x))))
    null = np.abs((s * x).mean(1)); return float(np.mean(null >= abs(x.mean()) - 1e-15))


def one_sample(x):
    t = stats.ttest_1samp(x, 0); ci = stats.t.interval(.95, len(x) - 1, loc=x.mean(), scale=stats.sem(x))
    return dict(mean=x.mean(), t=t.statistic, p_t=t.pvalue, ci_low=ci[0], ci_high=ci[1],
                p_signflip=signflip(x), n_positive=int((x > 0).sum()), n=len(x))


def load(path):
    d = pd.read_csv(path)
    if 'lower_stage' in d:
        d['pair'] = d.lower_stage + '–' + d.higher_stage
        d = d.rename(columns={'signed_FF': 'FF', 'signed_FB': 'FB'})
    d['H'] = d.FB - d.FF
    return d[['subject', 'pair', 'FF', 'FB', 'H']]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('pairs_csv', type=Path)
    ap.add_argument('--label', default='original')
    ap.add_argument('--out-dir', type=Path, default=Path('table_S6'))
    a = ap.parse_args(); a.out_dir.mkdir(parents=True, exist_ok=True)
    d = load(a.pairs_csv)
    w = {m: d.pivot(index='subject', columns='pair', values=m)[PAIRS] for m in ('FF', 'FB', 'H')}
    assert all(len(v) == 15 and not v.isna().any().any() for v in w.values()), 'expected 15 complete participants'
    rows = []
    for m in ('FF', 'FB', 'H'):
        r = [dict(sampling=a.label, pair=p, measure=m, **one_sample(w[m][p].to_numpy())) for p in PAIRS]
        for x, hp, hs in zip(r, holm([x['p_t'] for x in r]), holm([x['p_signflip'] for x in r])):
            x['p_t_holm_3pairs'] = hp; x['p_signflip_holm_3pairs'] = hs
        rows += r
    tab = pd.DataFrame(rows); tab.to_csv(a.out_dir / f'pair_tests_{a.label}.csv', index=False)
    comp = []
    for hi, lo in [('EVC–cauPFC', 'EVC–PPC'), ('PPC–cauPFC', 'EVC–PPC'), ('PPC–cauPFC', 'EVC–cauPFC')]:
        x = (w['H'][hi] - w['H'][lo]).to_numpy()
        comp.append(dict(sampling=a.label, contrast=f'{hi} minus {lo}', **one_sample(x)))
    for x, hp, hs in zip(comp, holm([c['p_t'] for c in comp]), holm([c['p_signflip'] for c in comp])):
        x['p_t_holm_3contrasts'] = hp; x['p_signflip_holm_3contrasts'] = hs
    cmp = pd.DataFrame(comp); cmp.to_csv(a.out_dir / f'pair_H_comparisons_{a.label}.csv', index=False)
    pd.set_option('display.width', 200)
    print(tab[['pair', 'measure', 'mean', 'p_t', 'p_t_holm_3pairs', 'ci_low', 'ci_high', 'n_positive']].round(5).to_string(index=False))
    print(cmp[['contrast', 'mean', 'p_t', 'p_t_holm_3contrasts', 'p_signflip', 'p_signflip_holm_3contrasts']].round(5).to_string(index=False))


if __name__ == '__main__':
    main()
