#!/usr/bin/env python3
"""Six-node voxel-matched behaviour associations (Supplementary Table S4, six-node 'matched' rows).

summarize_roi_size.py reports the nine-node matched behaviour tests only. This script applies
the same test (summarize_roi_size.behavior_test: Spearman rho, 100,000 label permutations,
20,000 paired-participant bootstrap samples, matched-minus-original change) to the six-node
participant means in <run>/participant_summary.csv, with Holm correction over FF and FB
within each outcome. Run from this folder: python six_matched_behaviour.py --config config.json --run <run dir>
"""
import argparse
from pathlib import Path
import numpy as np
from roi_size import csvread, csvwrite, expand, readj, SUBJECTS
from summarize_roi_size import behavior_test, holm


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='config.json')
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--seed', type=int, default=None, help='default: config statistics_seed')
    ap.add_argument('--out-dir', type=Path, default=None, help='default: <run>_six_behaviour next to the run')
    a = ap.parse_args()
    c = readj(a.config)
    beh = {r['subject']: r for r in csvread(expand(c, c['behavior_table']))}
    six = {r['subject']: r for r in csvread(a.run / 'participant_summary.csv') if r['model'] == 'six'}
    assert set(six) == set(SUBJECTS) and set(beh) >= set(SUBJECTS)
    rng = np.random.default_rng(c['statistics_seed'] if a.seed is None else a.seed)
    rows = []
    for outcome in ['Threshold', 'ACC_all']:
        for k in ['FB', 'FF']:
            x = np.array([float(six[s][k]) for s in SUBJECTS])
            old = np.array([float(six[s]['original_' + k]) for s in SUBJECTS])
            y = np.array([float(beh[s][outcome]) for s in SUBJECTS])
            rows.append(dict(outcome=outcome, metric=k, model='six',
                             **behavior_test(x, y, old, c['behavior_bootstrap'], c['behavior_permutations'], rng)))
    for outcome in ['Threshold', 'ACC_all']:
        g = [r for r in rows if r['outcome'] == outcome]
        for r, p in zip(g, holm([r['p_permutation'] for r in g])):
            r['p_holm_within_outcome_two_pathways'] = p
    out = a.out_dir or a.run.parent / (a.run.name + '_six_behaviour')
    out.mkdir(parents=True, exist_ok=True)
    csvwrite(out / 'six_behavior_statistics.csv', rows)
    for r in rows:
        print(f"{r['outcome']:9s} {r['metric']}  rho={r['rho']:+.4f}  p={r['p_permutation']:.5f}  "
              f"holm={r['p_holm_within_outcome_two_pathways']:.5f}  delta={r['delta_rho']:+.3f} "
              f"[{r['paired_delta_ci_low']:.2f}, {r['paired_delta_ci_high']:.2f}]")
    print('written:', out / 'six_behavior_statistics.csv')


if __name__ == '__main__':
    main()
