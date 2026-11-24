#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit matched pial5 rDCM sensitivity and current-pipeline replication.

The script keeps two paired comparisons conceptually separate:

1. pial_depth_effect:
   pial5_current - baseline_current (only pial exclusion differs)
2. pipeline_replication:
   baseline_current - old_primary (pipeline-version difference; not pial)

Non-significance of a paired difference is not labelled equivalence.  The
outputs therefore include confidence intervals, standardized paired effects,
subject agreement, ICC(A,1), sign agreement, and whole-A-matrix agreement.
"""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import os
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "/path/to/project")  # see config/paths.sh


DEFAULT_BASE = Path(
    PROJECT_ROOT + "/derivatives_0331/S6_validation/"
    "pial_depth_exclusion/matched_current_rdcm/group_audit"
)
METRICS = [
    "signed_FF", "signed_FB", "signed_H",
    "absolute_FF", "absolute_FB", "absolute_H",
]
NETWORKS = ["cauPFC_EVC", "cauPFC_V1"]
CONDITIONS = ["baseline_current", "pial5_current", "old_primary"]
COMPARISONS = {
    "pial_depth_effect": ("baseline_current", "pial5_current"),
    "pipeline_replication": ("old_primary", "baseline_current"),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--subject-csv", type=Path,
        default=DEFAULT_BASE / "matched_pial5_subject_metrics.csv")
    p.add_argument(
        "--edge-csv", type=Path,
        default=DEFAULT_BASE / "matched_pial5_A_edges_long.csv")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_BASE)
    return p.parse_args()


def holm_adjust(values):
    values = np.asarray(values, dtype=float)
    result = np.full_like(values, np.nan)
    valid = np.isfinite(values)
    p = values[valid]
    if not len(p):
        return result
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.maximum.accumulate((len(p) - np.arange(len(p))) * ranked)
    adjusted = np.minimum(adjusted, 1.0)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    result[valid] = restored
    return result


def exact_signflip_p(x):
    x = np.asarray(x, dtype=float)
    if len(x) > 20:
        return np.nan, 0
    signs = np.asarray(list(product((-1.0, 1.0), repeat=len(x))))
    permuted = np.abs((signs * x).mean(axis=1))
    observed = abs(x.mean())
    return float(np.mean(permuted >= observed - 1e-15)), len(permuted)


def safe_wilcoxon(x):
    x = np.asarray(x, dtype=float)
    if np.allclose(x, 0.0, atol=1e-15, rtol=0):
        return 0.0, 1.0
    result = stats.wilcoxon(x, alternative="two-sided", method="auto")
    return float(result.statistic), float(result.pvalue)


def safe_pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(stats.pearsonr(x, y).statistic)


def icc_a1(x, y):
    """Two-way random, single-measure, absolute-agreement ICC(A,1)."""
    z = np.column_stack([x, y]).astype(float)
    n, k = z.shape
    grand = z.mean()
    row_means = z.mean(axis=1)
    col_means = z.mean(axis=0)
    ss_rows = k * np.sum((row_means - grand) ** 2)
    ss_cols = n * np.sum((col_means - grand) ** 2)
    residual = z - row_means[:, None] - col_means[None, :] + grand
    ss_error = np.sum(residual ** 2)
    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))
    denominator = (
        ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
    )
    return float((ms_rows - ms_error) / denominator) if denominator != 0 else np.nan


def validate_subject_table(d):
    required = {"subject", "network", "condition", *METRICS}
    missing = sorted(required - set(d.columns))
    if missing:
        raise ValueError(f"Subject CSV missing columns: {missing}")
    if d[list(required)].isna().any().any():
        raise ValueError("Required subject-level fields contain missing values.")
    if d.duplicated(["subject", "network", "condition"]).any():
        dup = d[d.duplicated(
            ["subject", "network", "condition"], keep=False)]
        raise ValueError(f"Duplicate subject/network/condition rows:\n{dup}")

    rows = []
    for network in NETWORKS:
        for condition in CONDITIONS:
            g = d[(d.network == network) & (d.condition == condition)]
            rows.append({
                "network": network,
                "condition": condition,
                "n_rows": len(g),
                "n_subjects": g.subject.nunique(),
                "expected_n": 15,
                "complete": len(g) == 15 and g.subject.nunique() == 15,
            })
    audit = pd.DataFrame(rows)
    if not audit.complete.all():
        raise ValueError(f"Incomplete condition cells:\n{audit}")
    return audit


def condition_statistics(d):
    rows = []
    for (condition, network), g in d.groupby(
            ["condition", "network"], sort=True):
        for metric in METRICS:
            x = g[metric].to_numpy(float)
            n = len(x)
            mean = x.mean()
            sd = x.std(ddof=1)
            sem = stats.sem(x)
            ci = stats.t.interval(.95, n - 1, loc=mean, scale=sem)
            tt = stats.ttest_1samp(x, 0.0)
            wil_stat, wil_p = safe_wilcoxon(x)
            sf_p, patterns = exact_signflip_p(x)
            rows.append({
                "condition": condition, "network": network,
                "metric": metric, "n_subjects": n,
                "mean": mean, "sd": sd, "sem": sem,
                "ci95_low_t": float(ci[0]), "ci95_high_t": float(ci[1]),
                "t_stat": tt.statistic, "p_t_two_sided": tt.pvalue,
                "cohen_dz_vs_zero": mean / sd if sd > 0 else np.nan,
                "n_positive": int(np.sum(x > 0)),
                "n_negative": int(np.sum(x < 0)),
                "wilcoxon_stat": wil_stat,
                "wilcoxon_p_two_sided": wil_p,
                "signflip_p_two_sided": sf_p,
                "signflip_n_patterns": patterns,
            })
    out = pd.DataFrame(rows)
    out["p_t_holm_across_networks_for_condition_metric"] = np.nan
    out["p_signflip_holm_across_networks_for_condition_metric"] = np.nan
    for (_, _), idx in out.groupby(["condition", "metric"]).groups.items():
        idx = list(idx)
        out.loc[idx, "p_t_holm_across_networks_for_condition_metric"] = \
            holm_adjust(out.loc[idx, "p_t_two_sided"])
        out.loc[idx, "p_signflip_holm_across_networks_for_condition_metric"] = \
            holm_adjust(out.loc[idx, "signflip_p_two_sided"])
    return out


def paired_statistics(d):
    comparison_rows = []
    delta_rows = []
    for comparison, (reference, test) in COMPARISONS.items():
        for network in NETWORKS:
            left = d[(d.network == network) & (d.condition == reference)]
            right = d[(d.network == network) & (d.condition == test)]
            merged = left.merge(
                right, on=["subject", "network"], suffixes=("_reference", "_test"),
                validate="one_to_one")
            if len(merged) != 15:
                raise ValueError(
                    f"{comparison}/{network}: expected 15 pairs, got {len(merged)}")

            for metric in METRICS:
                x = merged[f"{metric}_reference"].to_numpy(float)
                y = merged[f"{metric}_test"].to_numpy(float)
                delta = y - x
                n = len(delta)
                mean_delta = delta.mean()
                sd_delta = delta.std(ddof=1)
                sem = stats.sem(delta)
                ci = stats.t.interval(.95, n - 1, loc=mean_delta, scale=sem)
                tt = stats.ttest_rel(y, x)
                wil_stat, wil_p = safe_wilcoxon(delta)
                sf_p, patterns = exact_signflip_p(delta)
                shapiro = stats.shapiro(delta)
                comparison_rows.append({
                    "comparison": comparison,
                    "reference_condition": reference,
                    "test_condition": test,
                    "network": network, "metric": metric,
                    "n_subjects": n,
                    "reference_mean": x.mean(),
                    "test_mean": y.mean(),
                    "delta_test_minus_reference_mean": mean_delta,
                    "delta_sd": sd_delta, "delta_sem": sem,
                    "delta_ci95_low_t": float(ci[0]),
                    "delta_ci95_high_t": float(ci[1]),
                    "paired_t": tt.statistic,
                    "paired_p_t_two_sided": tt.pvalue,
                    "paired_cohen_dz": (
                        mean_delta / sd_delta if sd_delta > 0 else np.nan),
                    "delta_wilcoxon_stat": wil_stat,
                    "delta_wilcoxon_p_two_sided": wil_p,
                    "delta_signflip_p_two_sided": sf_p,
                    "delta_signflip_n_patterns": patterns,
                    "delta_shapiro_W": shapiro.statistic,
                    "delta_shapiro_p": shapiro.pvalue,
                    "pearson_r_subjects": safe_pearson(x, y),
                    "icc_A1_absolute_agreement": icc_a1(x, y),
                    "rmse": np.sqrt(np.mean(delta ** 2)),
                    "mean_absolute_difference": np.mean(np.abs(delta)),
                    "max_absolute_difference": np.max(np.abs(delta)),
                    "same_sign_n": int(np.sum(np.sign(x) == np.sign(y))),
                    "same_sign_rate": float(np.mean(np.sign(x) == np.sign(y))),
                })
                for subject, xv, yv, dv in zip(merged.subject, x, y, delta):
                    delta_rows.append({
                        "comparison": comparison,
                        "reference_condition": reference,
                        "test_condition": test,
                        "subject": subject, "network": network,
                        "metric": metric, "reference_value": xv,
                        "test_value": yv, "delta_test_minus_reference": dv,
                    })

    out = pd.DataFrame(comparison_rows)
    out["paired_p_t_holm_across_networks_for_metric"] = np.nan
    out["delta_signflip_p_holm_across_networks_for_metric"] = np.nan
    for (_, _), idx in out.groupby(["comparison", "metric"]).groups.items():
        idx = list(idx)
        out.loc[idx, "paired_p_t_holm_across_networks_for_metric"] = \
            holm_adjust(out.loc[idx, "paired_p_t_two_sided"])
        out.loc[idx, "delta_signflip_p_holm_across_networks_for_metric"] = \
            holm_adjust(out.loc[idx, "delta_signflip_p_two_sided"])
    return out, pd.DataFrame(delta_rows)


def matrix_agreement(edges):
    required = {
        "subject", "network", "condition", "target_index", "source_index", "A"
    }
    missing = sorted(required - set(edges.columns))
    if missing:
        raise ValueError(f"Edge CSV missing columns: {missing}")
    if edges.duplicated([
            "subject", "network", "condition", "target_index", "source_index"
    ]).any():
        raise ValueError("Duplicate A-edge keys detected.")

    rows = []
    for comparison, (reference, test) in COMPARISONS.items():
        for network in NETWORKS:
            for subject in sorted(edges.subject.unique()):
                ref = edges[(edges.condition == reference) &
                            (edges.network == network) &
                            (edges.subject == subject)]
                tst = edges[(edges.condition == test) &
                            (edges.network == network) &
                            (edges.subject == subject)]
                merged = ref.merge(
                    tst, on=["subject", "network", "target_index", "source_index"],
                    suffixes=("_reference", "_test"), validate="one_to_one")
                merged = merged[merged.target_index != merged.source_index]
                if len(merged) != 30:
                    raise ValueError(
                        f"{comparison}/{network}/{subject}: expected 30 off-diagonal "
                        f"edges, got {len(merged)}")
                x = merged.A_reference.to_numpy(float)
                y = merged.A_test.to_numpy(float)
                delta = y - x
                rows.append({
                    "comparison": comparison,
                    "reference_condition": reference,
                    "test_condition": test,
                    "subject": subject, "network": network,
                    "n_offdiagonal_edges": len(x),
                    "pearson_r_offdiagonal": safe_pearson(x, y),
                    "rmse_offdiagonal": float(np.sqrt(np.mean(delta ** 2))),
                    "mean_abs_diff_offdiagonal": float(np.mean(np.abs(delta))),
                    "max_abs_diff_offdiagonal": float(np.max(np.abs(delta))),
                    "edge_sign_agreement_rate": float(
                        np.mean(np.sign(x) == np.sign(y))),
                })
    subjects = pd.DataFrame(rows)
    summary = subjects.groupby(["comparison", "network"], as_index=False).agg(
        n_subjects=("subject", "nunique"),
        pearson_r_offdiagonal_mean=("pearson_r_offdiagonal", "mean"),
        pearson_r_offdiagonal_min=("pearson_r_offdiagonal", "min"),
        rmse_offdiagonal_mean=("rmse_offdiagonal", "mean"),
        mean_abs_diff_offdiagonal_mean=("mean_abs_diff_offdiagonal", "mean"),
        max_abs_diff_offdiagonal_max=("max_abs_diff_offdiagonal", "max"),
        edge_sign_agreement_rate_mean=("edge_sign_agreement_rate", "mean"),
    )
    return subjects, summary


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(args.subject_csv)
    edges = pd.read_csv(args.edge_csv)
    integrity = validate_subject_table(d)
    conditions = condition_statistics(d)
    paired, deltas = paired_statistics(d)
    matrix_subjects, matrix_summary = matrix_agreement(edges)

    files = {
        "integrity": args.out_dir / "matched_pial5_integrity_audit.csv",
        "conditions": args.out_dir / "matched_pial5_condition_stats.csv",
        "paired": args.out_dir / "matched_pial5_paired_comparisons.csv",
        "deltas": args.out_dir / "matched_pial5_subject_deltas_long.csv",
        "matrix_subjects": args.out_dir / "matched_pial5_matrix_agreement_subjects.csv",
        "matrix_summary": args.out_dir / "matched_pial5_matrix_agreement_summary.csv",
    }
    integrity.to_csv(files["integrity"], index=False)
    conditions.to_csv(files["conditions"], index=False)
    paired.to_csv(files["paired"], index=False)
    deltas.to_csv(files["deltas"], index=False)
    matrix_subjects.to_csv(files["matrix_subjects"], index=False)
    matrix_summary.to_csv(files["matrix_summary"], index=False)

    focus = paired[paired.metric.isin(["signed_H", "absolute_H"])]
    print("=" * 110)
    print("Matched pial5 rDCM audit")
    print("Integrity: PASS; 15 subjects in every network/condition cell")
    print("\nHierarchy contrast comparisons (test - reference):")
    print(focus.to_string(index=False))
    print("\nWhole-A off-diagonal agreement:")
    print(matrix_summary.to_string(index=False))
    print("\nImportant: p > .05 for a paired difference is not proof of equivalence.")
    for label, file in files.items():
        print(f"{label:16s}: {file}")
    print("=" * 110)


if __name__ == "__main__":
    main()
