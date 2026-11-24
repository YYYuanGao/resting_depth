#!/usr/bin/env python3
"""Node-matched pooled-EVC rDCM pathway associations with behavior.

This is a sensitivity analysis for the already estimated pooled V1/V2/V3
multistage rDCM. It does not re-estimate rDCM. It selects the pooled
EVC--cauPFC stage pair, tests signed feedback and feedforward separately
against Threshold and ACC_all, and controls Holm FWER across the two pathway
tests within each behavioral outcome.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import os
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "/path/to/project")  # see config/paths.sh


DEFAULT_BEHAVIOR = Path(PROJECT_ROOT + "/wmpfc_cau_350.csv")
DEFAULT_OUT = Path(
    PROJECT_ROOT + "/derivatives_0331/S6_validation/"
    "pial5_only_update/pooled_evc_rdcm_behavior"
)
METRICS = ("signed_FB", "signed_FF")
OUTCOMES = ("Threshold", "ACC_all")
COLORS = {"signed_FB": "#D55E00", "signed_FF": "#0072B2"}
LABELS = {"signed_FB": "Feedback  cauPFC → pooled EVC", "signed_FF": "Feedforward  pooled EVC → cauPFC"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "pathways_csv",
        type=Path,
        help="evc_pooled_subject_stage_pair_pathways.csv from the pooled multistage rDCM",
    )
    p.add_argument("--behavior-csv", type=Path, default=DEFAULT_BEHAVIOR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--expected-n", type=int, default=15)
    p.add_argument("--n-permutations", type=int, default=100_000)
    p.add_argument("--n-bootstrap", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=20260724)
    return p.parse_args()


def normalize_subject(value: object) -> str:
    m = re.search(r"(\d+)", str(value))
    if not m:
        raise ValueError(f"Cannot normalize participant identifier: {value!r}")
    return f"sub-{int(m.group(1)):02d}"


def stable_seed(base: int, *parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, (base, *parts))).encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def holm_adjust(values: np.ndarray) -> np.ndarray:
    p = np.asarray(values, float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted_ranked = np.maximum.accumulate((len(p) - np.arange(len(p))) * ranked)
    out = np.empty_like(p)
    out[order] = np.clip(adjusted_ranked, 0, 1)
    return out


def rho(x: np.ndarray, y: np.ndarray) -> float:
    return float(stats.spearmanr(x, y).statistic)


def bootstrap_ci(x, y, n_bootstrap, rng):
    n = len(x)
    draws = np.full(n_bootstrap, np.nan)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        if np.ptp(x[idx]) and np.ptp(y[idx]):
            draws[i] = rho(x[idx], y[idx])
    valid = draws[np.isfinite(draws)]
    if len(valid) < 0.9 * n_bootstrap:
        raise RuntimeError(f"Only {len(valid)}/{n_bootstrap} valid bootstrap draws")
    return (*np.quantile(valid, [0.025, 0.975]), len(valid))


def permutation_p(x, y, n_permutations, rng, observed):
    exceed = 0
    for _ in range(n_permutations):
        perm = rng.permutation(y)
        exceed += abs(rho(x, perm)) >= abs(observed) - 1e-15
    return (exceed + 1) / (n_permutations + 1)


def find_behavior_subject_column(df: pd.DataFrame) -> str:
    preferred = ("subject", "participant", "sub", "subj", "ID", "id")
    for name in preferred:
        if name in df.columns:
            return name
    raise KeyError(f"No participant column found. Columns: {list(df.columns)}")


def prepare(pathways_csv: Path, behavior_csv: Path, expected_n: int) -> pd.DataFrame:
    paths = pd.read_csv(pathways_csv)
    required = {"subject", "lower_stage", "higher_stage", *METRICS}
    missing = required - set(paths)
    if missing:
        raise KeyError(f"Pathway CSV missing columns: {sorted(missing)}")
    paths["subject"] = paths["subject"].map(normalize_subject)
    pair = paths[(paths["lower_stage"] == "EVC") & (paths["higher_stage"] == "cauPFC")].copy()
    if pair["subject"].duplicated().any():
        raise RuntimeError("Expected exactly one pooled EVC--cauPFC row per participant")

    beh = pd.read_csv(behavior_csv)
    subject_col = find_behavior_subject_column(beh)
    beh["subject"] = beh[subject_col].map(normalize_subject)
    for outcome in OUTCOMES:
        if outcome not in beh:
            raise KeyError(f"Behavior CSV missing {outcome!r}")
        beh[outcome] = pd.to_numeric(beh[outcome], errors="coerce")
    beh = beh[["subject", *OUTCOMES]].drop_duplicates("subject")

    merged = pair[["subject", *METRICS]].merge(beh, on="subject", how="inner", validate="one_to_one")
    merged = merged.dropna()
    if len(merged) != expected_n:
        raise RuntimeError(f"Expected {expected_n} complete participants, found {len(merged)}")
    return merged.sort_values("subject").reset_index(drop=True)


def run_tests(data: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for outcome in OUTCOMES:
        for metric in METRICS:
            x = data[metric].to_numpy(float)
            y = data[outcome].to_numpy(float)
            observed = rho(x, y)
            ci_rng = np.random.default_rng(stable_seed(args.seed, outcome, metric, "ci"))
            perm_rng = np.random.default_rng(stable_seed(args.seed, outcome, metric, "perm"))
            low, high, n_valid = bootstrap_ci(x, y, args.n_bootstrap, ci_rng)
            p_perm = permutation_p(x, y, args.n_permutations, perm_rng, observed)
            loo = [rho(np.delete(x, i), np.delete(y, i)) for i in range(len(x))]
            rows.append(
                {
                    "analysis_role": "node_matched_pooled_evc_sensitivity",
                    "outcome": outcome,
                    "metric": metric,
                    "n_subjects": len(x),
                    "spearman_rho": observed,
                    "bootstrap_ci95_low": low,
                    "bootstrap_ci95_high": high,
                    "permutation_p_two_sided": p_perm,
                    "loo_rho_min": min(loo),
                    "loo_rho_max": max(loo),
                    "loo_same_sign_fraction": np.mean(np.sign(loo) == np.sign(observed)),
                    "n_bootstrap": args.n_bootstrap,
                    "n_valid_bootstrap": n_valid,
                    "n_permutations": args.n_permutations,
                }
            )
    out = pd.DataFrame(rows)
    out["permutation_p_holm_within_2_pathway_outcome"] = np.nan
    for outcome in OUTCOMES:
        idx = out.index[out["outcome"].eq(outcome)]
        out.loc[idx, "permutation_p_holm_within_2_pathway_outcome"] = holm_adjust(
            out.loc[idx, "permutation_p_two_sided"].to_numpy()
        )
    return out


def style():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_forest(results: pd.DataFrame, out_dir: Path):
    style()
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.75), sharey=True)
    y = np.array([1, 0])
    titles = {"Threshold": "a  Threshold", "ACC_all": "b  Accuracy"}
    for ax, outcome in zip(axes, OUTCOMES):
        sub = results[results["outcome"].eq(outcome)].set_index("metric").loc[list(METRICS)]
        ax.axvline(0, color="#8B8F94", lw=0.9, ls=(0, (4, 3)), zorder=0)
        for yi, metric in zip(y, METRICS):
            row = sub.loc[metric]
            significant = row["permutation_p_holm_within_2_pathway_outcome"] < 0.05
            ax.errorbar(
                row["spearman_rho"],
                yi,
                xerr=[
                    [row["spearman_rho"] - row["bootstrap_ci95_low"]],
                    [row["bootstrap_ci95_high"] - row["spearman_rho"]],
                ],
                fmt="s" if metric == "signed_FB" else "o",
                ms=6,
                mfc=COLORS[metric] if significant else "white",
                mec=COLORS[metric],
                mew=1.4,
                ecolor=COLORS[metric],
                elinewidth=1.5,
                capsize=3,
            )
            ax.text(
                1.03,
                yi,
                f"ρ={row['spearman_rho']:+.2f}\\n"
                f"P_Holm={row['permutation_p_holm_within_2_pathway_outcome']:.3f}",
                transform=ax.get_yaxis_transform(),
                va="center",
                ha="left",
                fontsize=7.4,
                color="#34373B",
            )
        ax.set_title(titles[outcome], loc="left", weight="bold", fontsize=11)
        ax.set_xlim(-1, 1)
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_xlabel("Spearman ρ (bootstrap 95% CI)")
        ax.set_ylim(-0.55, 1.55)
        ax.grid(axis="x", color="#E2E5E9", lw=0.7)
    axes[0].set_yticks(y, [LABELS[m] for m in METRICS])
    fig.suptitle(
        "Pooled V1–V3 EVC pathways and behavior",
        x=0.08,
        ha="left",
        weight="bold",
        fontsize=13,
    )
    fig.text(
        0.08,
        0.01,
        "Node-matched sensitivity; two-sided behavior-label permutation; Holm FWER within the two-pathway family for each outcome.",
        fontsize=7.2,
        color="#6F747A",
    )
    fig.subplots_adjust(left=0.31, right=0.87, bottom=0.25, top=0.76, wspace=0.30)
    for ext in ("pdf", "png"):
        kwargs = {"bbox_inches": "tight"}
        if ext == "png":
            kwargs["dpi"] = 600
        fig.savefig(out_dir / f"pooled_evc_behavior_forest.{ext}", **kwargs)
    plt.close(fig)


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = prepare(args.pathways_csv, args.behavior_csv, args.expected_n)
    results = run_tests(data, args)
    data.to_csv(args.out_dir / "pooled_evc_behavior_subjects.csv", index=False)
    results.to_csv(args.out_dir / "pooled_evc_behavior_statistics.csv", index=False)
    plot_forest(results, args.out_dir)
    print(results.to_string(index=False))
    print(f"\nSaved: {args.out_dir}")


if __name__ == "__main__":
    main()
