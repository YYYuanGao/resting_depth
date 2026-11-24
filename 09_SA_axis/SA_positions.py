#!/usr/bin/env python3
"""S–A axis positions of the EVC, PPC and cauPFC composites (Results 2.4, Fig. 4a,
Supplementary Table S5).

Input: the public S–A axis ranks of Sydnor et al. (2021), Glasser HCP-MMP version
(`Sensorimotor_Association_Axis_AverageRanks.csv`, column `final.rank`,
https://github.com/PennLINC/S-A_ArchetypalAxis). No participant data are needed.

Outputs (in --out-dir):
  sa_parcel_ranks.csv       rank of every composite parcel
  sa_composite_positions.csv  mean, median, range per composite
  sa_group_tests.csv        exact two-sided Mann–Whitney tests between composites
  sa_separations.csv        Table S5: pairwise separations under alternative definitions,
                            with parcel-bootstrap CIs for the mean-rank definition
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

COMPOSITES = {
    "EVC": ["V1", "V2", "V3"],
    # IPS1 and AIP were removed before analysis (incomplete time series; Methods 4.5)
    "PPC": ["LIPv", "LIPd", "VIP", "MIP", "IP0", "IP1", "IP2",
            "7PC", "7Am", "7AL", "7PL", "7Pm", "7m"],
    "cauPFC": ["6d", "6v", "6mp", "FEF", "PEF", "55b", "SFL"],
}
PAIRS = [("EVC", "PPC"), ("EVC", "cauPFC"), ("PPC", "cauPFC")]
# alternative cauPFC definitions in Table S5
VARIANTS = {
    "Mean rank (main analysis)": ("mean", COMPOSITES["cauPFC"]),
    "Median rank": ("median", COMPOSITES["cauPFC"]),
    "cauPFC without 6d and 6mp": ("mean", [p for p in COMPOSITES["cauPFC"] if p not in ("6d", "6mp")]),
    "cauPFC without 55b": ("mean", [p for p in COMPOSITES["cauPFC"] if p != "55b"]),
    "cauPFC as FEF only": ("mean", ["FEF"]),
}


def load_ranks(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    df["region"] = df["region"].astype(str).str.replace(r"^[LR]_|_ROI$", "", regex=True)
    s = df.groupby("region")["final.rank"].first()
    missing = [p for ps in COMPOSITES.values() for p in ps if p not in s.index]
    if missing:
        raise KeyError(f"parcels not found in S–A file: {missing}")
    return s


def separations(ranks, agg, caupfc):
    f = np.mean if agg == "mean" else np.median
    pos = {"EVC": f(ranks[COMPOSITES["EVC"]]), "PPC": f(ranks[COMPOSITES["PPC"]]),
           "cauPFC": f(ranks[caupfc])}
    return {f"{a}–{b}": abs(pos[a] - pos[b]) for a, b in PAIRS}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sa-ranks", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("sa_positions"))
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=2026)  # reproduces the CIs reported in the manuscript
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    ranks = load_ranks(a.sa_ranks)

    rows = [{"composite": c, "parcel": p, "sa_rank": ranks[p]} for c, ps in COMPOSITES.items() for p in ps]
    pd.DataFrame(rows).to_csv(a.out_dir / "sa_parcel_ranks.csv", index=False)

    pos = pd.DataFrame([{"composite": c, "n_parcels": len(ps), "mean_rank": ranks[ps].mean(),
                         "median_rank": ranks[ps].median(), "min_rank": ranks[ps].min(),
                         "max_rank": ranks[ps].max()} for c, ps in COMPOSITES.items()])
    pos.to_csv(a.out_dir / "sa_composite_positions.csv", index=False)

    tests = []
    for x, y in combinations(COMPOSITES, 2):
        r = stats.mannwhitneyu(ranks[COMPOSITES[x]], ranks[COMPOSITES[y]],
                               alternative="two-sided", method="exact")
        tests.append({"a": x, "b": y, "U": r.statistic, "p_exact": r.pvalue})
    pd.DataFrame(tests).to_csv(a.out_dir / "sa_group_tests.csv", index=False)

    # parcel bootstrap: resample parcels with replacement within each composite
    rng = np.random.default_rng(a.seed)
    vals = {c: ranks[ps].to_numpy(float) for c, ps in COMPOSITES.items()}
    boot = {c: rng.choice(v, size=(a.n_boot, v.size), replace=True).mean(1) for c, v in vals.items()}
    bsep = {f"{x}–{y}": np.abs(boot[x] - boot[y]) for x, y in PAIRS}
    bmat = np.column_stack(list(bsep.values()))
    closest = (bmat.argmin(1) == 2).mean()

    out = []
    for name, (agg, cau) in VARIANTS.items():
        sep = separations(ranks, agg, cau)
        row = {"definition": name, **{k: round(v, 3) for k, v in sep.items()},
               "smallest": min(sep, key=sep.get)}
        if name.startswith("Mean rank"):
            for k in sep:
                lo, hi = np.percentile(bsep[k], [2.5, 97.5])
                row[f"{k}_ci95"] = f"[{lo:.1f}, {hi:.1f}]"
            row["boot_PPC–cauPFC_smallest"] = round(closest, 3)
        out.append(row)
    pd.DataFrame(out).to_csv(a.out_dir / "sa_separations.csv", index=False)

    print(pos.round(1).to_string(index=False))
    print(pd.DataFrame(tests).to_string(index=False))
    print(pd.DataFrame(out).to_string(index=False))


if __name__ == "__main__":
    main()
