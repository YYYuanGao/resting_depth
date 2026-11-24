#!/usr/bin/env python3
"""Sensitivity (power) analysis for the qT1/MPC–rDCM correlations (Methods, n = 15).

For each population Pearson correlation r on a 0.01 grid, 20,000 bivariate-normal
datasets of n = 15 are simulated; power is the fraction whose two-sided Spearman
|rho| reaches the critical value of a Monte Carlo null built from 1,000,000 random
rank permutations. Linear interpolation gives the r needed for 80% power at
alpha = 0.05 and 0.0125 (first, most stringent step of the four-test Holm procedure).
"""
import argparse
import numpy as np
import pandas as pd


def spearman_rows(a, b):
    ra = a.argsort(1).argsort(1).astype(float); rb = b.argsort(1).argsort(1).astype(float)
    ra -= ra.mean(1, keepdims=True); rb -= rb.mean(1, keepdims=True)
    return (ra * rb).sum(1) / np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--n', type=int, default=15)
    ap.add_argument('--n-sim', type=int, default=20_000)
    ap.add_argument('--n-null', type=int, default=1_000_000)
    ap.add_argument('--alphas', type=float, nargs='+', default=[0.05, 0.0125])
    ap.add_argument('--seed', type=int, default=20261124)
    ap.add_argument('--out', default='power_qt1_rdcm.csv')
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    x = np.arange(a.n, dtype=float); x -= x.mean()
    null = []
    for _ in range(a.n_null // 100_000):
        P = np.argsort(rng.random((100_000, a.n)), 1).astype(float); P -= P.mean(1, keepdims=True)
        null.append(np.abs(P @ x) / (x @ x))
    null = np.sort(np.concatenate(null))
    # smallest |rho| whose two-sided Monte Carlo p = P(|rho0| >= |rho|) is <= alpha
    crit = {al: null[np.searchsorted(np.arange(1, null.size + 1) / null.size, 1 - al)] for al in a.alphas}
    grid = np.round(np.arange(0, 1.0, 0.01), 2)
    rows = []
    for r in grid:
        z = rng.standard_normal((a.n_sim, a.n, 2))
        u = z[..., 0]; v = r * z[..., 0] + np.sqrt(1 - r ** 2) * z[..., 1]
        rho = np.abs(spearman_rows(u, v))
        rows.append({'r': r, **{f'power_alpha_{al}': float((rho >= crit[al] - 1e-12).mean()) for al in a.alphas}})
    df = pd.DataFrame(rows); df.to_csv(a.out, index=False)
    for al in a.alphas:
        col = f'power_alpha_{al}'; print(f'alpha {al}: critical |rho| = {crit[al]:.4f}; '
                                        f'80% power at r = {np.interp(0.8, df[col], df.r):.3f}')


if __name__ == '__main__':
    main()
