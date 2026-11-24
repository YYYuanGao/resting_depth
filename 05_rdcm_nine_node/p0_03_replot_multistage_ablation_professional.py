#!/usr/bin/env python3
"""Replot existing P0-3 ablation results without rerunning bootstrap tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from p0_03_multistage_pair_ablation import VARIANTS, plot_forest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group-csv", type=Path, required=True)
    parser.add_argument("--changes-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def validate(group: pd.DataFrame, changes: pd.DataFrame) -> None:
    required_group = {
        "variant", "mean_within_subject_spearman_rho", "bootstrap_ci95_low",
        "bootstrap_ci95_high", "exact_signflip_p_two_sided",
        "exact_signflip_p_holm_across_3_ablations", "n_negative", "n_subjects",
    }
    required_changes = {
        "variant", "mean_delta_rho_ablation_minus_full",
        "bootstrap_delta_ci95_low", "bootstrap_delta_ci95_high",
        "paired_p_holm_across_3_ablation_changes", "same_sign_fraction_as_full",
        "n_subjects",
    }
    missing_group = required_group - set(group.columns)
    missing_changes = required_changes - set(changes.columns)
    if missing_group or missing_changes:
        raise KeyError(
            f"Missing group columns={sorted(missing_group)}; "
            f"missing change columns={sorted(missing_changes)}"
        )

    expected_group = {variant for variant, _ in VARIANTS}
    expected_changes = {variant for variant, _ in VARIANTS[1:]}
    if set(group["variant"]) != expected_group:
        raise RuntimeError(
            f"Expected group variants={sorted(expected_group)}; "
            f"found={sorted(group['variant'].unique())}"
        )
    if set(changes["variant"]) != expected_changes:
        raise RuntimeError(
            f"Expected change variants={sorted(expected_changes)}; "
            f"found={sorted(changes['variant'].unique())}"
        )
    if group["variant"].duplicated().any() or changes["variant"].duplicated().any():
        raise RuntimeError("Each variant must occur exactly once in each input CSV")

    group_base = group[
        [
            "mean_within_subject_spearman_rho", "bootstrap_ci95_low",
            "bootstrap_ci95_high", "exact_signflip_p_two_sided", "n_negative",
            "n_subjects",
        ]
    ].to_numpy(float)
    ablation_holm = group.loc[
        ~group["variant"].eq("full_all_pairs"),
        "exact_signflip_p_holm_across_3_ablations",
    ].to_numpy(float)
    change_values = changes[
        [
            "mean_delta_rho_ablation_minus_full", "bootstrap_delta_ci95_low",
            "bootstrap_delta_ci95_high", "paired_p_holm_across_3_ablation_changes",
            "same_sign_fraction_as_full", "n_subjects",
        ]
    ].to_numpy(float)
    if not (
        np.isfinite(group_base).all()
        and np.isfinite(ablation_holm).all()
        and np.isfinite(change_values).all()
    ):
        raise RuntimeError("Plot inputs contain non-finite values")


def main() -> None:
    args = parse_args()
    group = pd.read_csv(args.group_csv)
    changes = pd.read_csv(args.changes_csv)
    validate(group, changes)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_forest(group, changes, args.out_dir)
    print(f"Saved professional two-panel P0-3 figure to {args.out_dir}")


if __name__ == "__main__":
    main()
