#!/usr/bin/env python3
"""Paired diagnostic of positive-versus-reversed rDCM sign recovery.

This analysis asks whether recovery of the empirical positive FB-FF truth is
different from recovery of the sign-reversed control. Truth conditions are
paired by SNR, repetition and shared random seed. It is a diagnostic of
directional estimator performance, not a test of the empirical biological
hypothesis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from scipy import stats


TRUTHS = ("observed", "null", "reversed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("long_csv", type=Path, help="rdcm_recovery_long.csv")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expected-reps", type=int, default=100)
    parser.add_argument("--n-bootstrap", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument(
        "--prefix", default="paired_directional_recovery_asymmetry_n100"
    )
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def stable_seed(base: int, *parts: object) -> int:
    payload = "|".join(map(str, (base, *parts))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (
        2**32 - 1
    )


def parse_boolean(series: pd.Series, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="raise")
        if not values.isin([0, 1]).all():
            raise ValueError(f"{name} contains values outside 0/1")
        return values.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    if not normalized.isin(mapping).all():
        bad = sorted(normalized.loc[~normalized.isin(mapping)].unique())
        raise ValueError(f"{name} contains invalid Boolean values: {bad}")
    return normalized.map(mapping).astype(bool)


def load_paired_data(path: Path, expected_reps: int) -> pd.DataFrame:
    data = pd.read_csv(path, keep_default_na=False)
    required = {
        "case_id",
        "truth_condition",
        "repetition",
        "seed",
        "SNR",
        "generation_success",
        "fit_success",
        "true_H",
        "estimated_H",
    }
    missing = required - set(data.columns)
    if missing:
        raise KeyError(f"Input lacks required columns: {sorted(missing)}")
    if data.empty:
        raise RuntimeError("Input CSV is empty")
    if data["case_id"].astype(str).duplicated().any():
        raise RuntimeError("Input contains duplicate case_id values")

    data = data.copy()
    for column in ("generation_success", "fit_success"):
        data[column] = parse_boolean(data[column], column)
    for column in ("repetition", "seed", "SNR", "true_H", "estimated_H"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if not data["generation_success"].all() or not data["fit_success"].all():
        raise RuntimeError("Input contains failed generation or rDCM fits")
    if not np.isfinite(
        data[["repetition", "seed", "SNR", "true_H", "estimated_H"]].to_numpy()
    ).all():
        raise RuntimeError("Input contains non-finite required values")
    if set(data["truth_condition"].astype(str)) != set(TRUTHS):
        raise RuntimeError("Truth conditions must be observed, null and reversed")

    grid = data.groupby(["truth_condition", "SNR"], observed=True).agg(
        n_rows=("case_id", "size"),
        n_repetitions=("repetition", "nunique"),
        n_seeds=("seed", "nunique"),
    )
    if not grid["n_rows"].eq(expected_reps).all():
        raise RuntimeError(
            f"Expected {expected_reps} rows in every truth-by-SNR cell:\n{grid}"
        )
    if not grid["n_repetitions"].eq(expected_reps).all() or not grid[
        "n_seeds"
    ].eq(expected_reps).all():
        raise RuntimeError(f"Incomplete or duplicated repetition grid:\n{grid}")

    pairing = data.groupby(["SNR", "repetition"], observed=True).agg(
        n_truths=("truth_condition", "nunique"),
        n_seeds=("seed", "nunique"),
    )
    if not pairing["n_truths"].eq(3).all() or not pairing["n_seeds"].eq(1).all():
        raise RuntimeError("Truth conditions are not paired by a shared seed")

    true_h = data.groupby("truth_condition", observed=True)["true_H"].agg(
        ["first", "nunique"]
    )
    if not true_h["nunique"].eq(1).all():
        raise RuntimeError("A truth condition contains inconsistent true_H values")
    observed_h = float(true_h.loc["observed", "first"])
    null_h = float(true_h.loc["null", "first"])
    reversed_h = float(true_h.loc["reversed", "first"])
    if observed_h <= 0 or abs(null_h) > 1e-10 or not np.isclose(
        reversed_h, -observed_h, atol=1e-10, rtol=0
    ):
        raise RuntimeError("Known observed/null/reversed truths are malformed")

    wide = data.pivot(
        index=["SNR", "repetition", "seed"],
        columns="truth_condition",
        values="estimated_H",
    ).reset_index()
    if wide.groupby("SNR", observed=True).size().ne(expected_reps).any():
        raise RuntimeError("Paired wide table does not contain the expected grid")
    return wide.sort_values(["SNR", "repetition"]).reset_index(drop=True)


def paired_bootstrap_ci(
    differences: np.ndarray, n_bootstrap: int, rng: np.random.Generator
) -> tuple[float, float]:
    indices = rng.integers(
        0, len(differences), size=(n_bootstrap, len(differences))
    )
    boot = differences[indices].mean(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975])
    return float(low), float(high)


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    ranked = p_values[order]
    factors = len(ranked) - np.arange(len(ranked))
    adjusted_ranked = np.minimum(1.0, np.maximum.accumulate(ranked * factors))
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    return adjusted


def analyze(
    wide: pd.DataFrame, n_bootstrap: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paired_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for snr, block in wide.groupby("SNR", sort=True, observed=True):
        block = block.copy()
        block["observed_sign_recovered"] = block["observed"] > 0
        block["reversed_sign_recovered"] = block["reversed"] < 0
        block["paired_recovery_difference"] = (
            block["observed_sign_recovered"].astype(int)
            - block["reversed_sign_recovered"].astype(int)
        )
        block["both_recovered"] = (
            block["observed_sign_recovered"]
            & block["reversed_sign_recovered"]
        )
        block["observed_only"] = (
            block["observed_sign_recovered"]
            & ~block["reversed_sign_recovered"]
        )
        block["reversed_only"] = (
            ~block["observed_sign_recovered"]
            & block["reversed_sign_recovered"]
        )
        block["neither_recovered"] = (
            ~block["observed_sign_recovered"]
            & ~block["reversed_sign_recovered"]
        )
        paired_rows.append(block)

        differences = block["paired_recovery_difference"].to_numpy(float)
        rng = np.random.default_rng(stable_seed(seed, "paired_difference", snr))
        ci_low, ci_high = paired_bootstrap_ci(differences, n_bootstrap, rng)
        observed_only = int(block["observed_only"].sum())
        reversed_only = int(block["reversed_only"].sum())
        discordant = observed_only + reversed_only
        exact_p = (
            1.0
            if discordant == 0
            else float(
                stats.binomtest(
                    observed_only, discordant, p=0.5, alternative="two-sided"
                ).pvalue
            )
        )
        summary_rows.append(
            {
                "SNR": float(snr),
                "n_paired_repetitions": len(block),
                "observed_successes": int(block["observed_sign_recovered"].sum()),
                "observed_recovery_rate": float(
                    block["observed_sign_recovered"].mean()
                ),
                "reversed_successes": int(block["reversed_sign_recovered"].sum()),
                "reversed_recovery_rate": float(
                    block["reversed_sign_recovered"].mean()
                ),
                "paired_rate_difference_observed_minus_reversed": float(
                    differences.mean()
                ),
                "paired_rate_difference_bootstrap_ci95_low": ci_low,
                "paired_rate_difference_bootstrap_ci95_high": ci_high,
                "both_recovered": int(block["both_recovered"].sum()),
                "observed_only": observed_only,
                "reversed_only": reversed_only,
                "neither_recovered": int(block["neither_recovered"].sum()),
                "discordant_pairs": discordant,
                "exact_mcnemar_p_unadjusted": exact_p,
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("SNR").reset_index(drop=True)
    summary["exact_mcnemar_p_holm_across_3_snr"] = holm_adjust(
        summary["exact_mcnemar_p_unadjusted"].to_numpy(float)
    )
    summary["diagnostic_conclusion"] = np.where(
        summary["exact_mcnemar_p_holm_across_3_snr"] < 0.05,
        "directional recovery rates differ after Holm correction",
        "no evidence of a directional recovery-rate difference after Holm correction",
    )
    paired = pd.concat(paired_rows, ignore_index=True).sort_values(
        ["SNR", "repetition"]
    )
    return paired, summary


def format_p(value: float) -> str:
    if value < 0.001:
        return "<.001"
    return f"{value:.3f}".lstrip("0")


def set_plot_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7.5,
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def plot_paired_asymmetry(
    summary: pd.DataFrame, out_dir: Path, prefix: str, dpi: int
) -> None:
    """Forest plot of the paired difference in correct-sign recovery."""
    set_plot_style()
    styles = {
        0.5: {"face": "#EEF1F3", "edge": "#66727A", "marker": "o"},
        1.0: {"face": "#F1C66D", "edge": "#96670D", "marker": "^"},
        3.0: {"face": "#76B7D5", "edge": "#246B8C", "marker": "s"},
    }
    summary = summary.sort_values("SNR").reset_index(drop=True)
    y_positions = np.arange(len(summary) - 1, -1, -1, dtype=float)

    low_all = summary[
        "paired_rate_difference_bootstrap_ci95_low"
    ].to_numpy(float)
    high_all = summary[
        "paired_rate_difference_bootstrap_ci95_high"
    ].to_numpy(float)
    span = max(float(high_all.max() - low_all.min()), 0.20)
    lower = min(float(low_all.min()), 0.0) - 0.12 * span
    upper = max(float(high_all.max()), 0.0) + 0.12 * span

    figure = plt.figure(figsize=(7.05, 3.05), constrained_layout=False)
    ax = figure.add_axes([0.16, 0.30, 0.48, 0.43])
    ax.axvline(0, color="#7A8389", linewidth=0.85, linestyle=(0, (4, 3)))
    ax.axvspan(lower, 0, color="#F4F5F6", zorder=-2)
    ax.grid(axis="x", color="#E7EAEC", linewidth=0.55, zorder=-1)

    for y, row in zip(y_positions, summary.itertuples(index=False)):
        snr = float(row.SNR)
        style = styles.get(
            snr, {"face": "white", "edge": "#246B8C", "marker": "D"}
        )
        value = float(row.paired_rate_difference_observed_minus_reversed)
        low = float(row.paired_rate_difference_bootstrap_ci95_low)
        high = float(row.paired_rate_difference_bootstrap_ci95_high)
        ax.errorbar(
            value,
            y,
            xerr=np.array([[value - low], [high - value]]),
            fmt=style["marker"],
            ms=6.0,
            color=style["edge"],
            markerfacecolor=style["face"],
            markeredgecolor=style["edge"],
            markeredgewidth=0.75,
            capsize=2.3,
            elinewidth=1.0,
            zorder=3,
        )
        p_holm = float(row.exact_mcnemar_p_holm_across_3_snr)
        annotation = (
            f"Δ = {value:+.0%}  "
            f"95% CI [{low:+.0%}, {high:+.0%}]  "
            f"$P_{{Holm}}$ = {format_p(p_holm)}"
        )
        ax.text(
            1.04,
            y,
            annotation,
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=6.7,
            color="#364047",
            clip_on=False,
        )

    ax.set_xlim(lower, upper)
    ax.set_ylim(-0.65, len(summary) - 0.35)
    ax.set_yticks(y_positions, [f"SNR {value:g}" for value in summary["SNR"]])
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel(
        "Difference in correct-sign recovery\nObserved truth − reversed control"
    )

    figure.text(
        0.16,
        0.90,
        "Paired asymmetry in directional recovery",
        ha="left",
        va="center",
        fontsize=10,
        weight="bold",
        color="#202428",
    )
    figure.text(
        0.16,
        0.82,
        "Paired by shared noise realization; intervals are paired-bootstrap 95% CIs.",
        ha="left",
        va="center",
        fontsize=7,
        color="#66727A",
    )
    figure.text(
        0.16,
        0.045,
        "Positive values indicate better recovery of the empirical positive truth. "
        "P values: exact McNemar tests, Holm-adjusted across three SNRs.",
        ha="left",
        va="center",
        fontsize=6.3,
        color="#66727A",
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        kwargs = {"dpi": dpi} if suffix == "png" else {}
        figure.savefig(
            out_dir / f"{prefix}.{suffix}",
            facecolor="white",
            bbox_inches="tight",
            **kwargs,
        )
    plt.close(figure)

    caption = (
        "Fig. X | Paired asymmetry in directional recovery. Points show the "
        "difference between correct-sign recovery of the observed positive "
        "FB−FF truth and the sign-reversed control within each SNR. Error bars "
        "show paired-repetition bootstrap 95% confidence intervals. P values "
        "are from exact McNemar tests of discordant paired outcomes and are "
        "Holm-adjusted across the three SNRs. Positive values indicate better "
        "recovery of the empirical positive direction. This is an estimator "
        "diagnostic and not a biological hypothesis test."
    )
    (out_dir / f"{prefix}_caption.txt").write_text(
        caption + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    if args.expected_reps < 2:
        raise ValueError("--expected-reps must be at least 2")
    if args.n_bootstrap < 10_000:
        raise ValueError("Use at least 10,000 bootstrap resamples")
    if args.dpi < 300:
        raise ValueError("Use at least 300 dpi for the PNG export")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    wide = load_paired_data(args.long_csv, args.expected_reps)
    paired, summary = analyze(wide, args.n_bootstrap, args.seed)
    paired.to_csv(
        args.out_dir / "paired_directional_recovery_by_repetition.csv", index=False
    )
    summary.to_csv(
        args.out_dir / "paired_directional_recovery_asymmetry_summary.csv",
        index=False,
    )
    plot_paired_asymmetry(summary, args.out_dir, args.prefix, args.dpi)

    manifest = {
        "analysis": "paired positive-versus-reversed sign-recovery diagnostic",
        "input_csv": str(args.long_csv.resolve()),
        "pairing": "SNR + repetition + shared seed",
        "observed_success": "estimated_H(observed) > 0",
        "reversed_success": "estimated_H(reversed) < 0",
        "effect": "observed recovery rate minus reversed recovery rate",
        "effect_interval": "paired-repetition bootstrap 95% CI",
        "test": "exact McNemar test implemented as an exact binomial test on discordant pairs",
        "multiplicity": "Holm adjustment across the three SNR-specific tests",
        "scope": "algorithmic directional-recovery asymmetry; not biological inference",
        "expected_repetitions_per_truth_snr": args.expected_reps,
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "figure": {
            "prefix": args.prefix,
            "formats": ["png", "pdf", "svg"],
            "dpi": args.dpi,
        },
    }
    (args.out_dir / "paired_directional_recovery_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print("Input and pairing validation: PASS")
    print(summary.to_string(index=False))
    print(f"\nSaved paired directional diagnostic to: {args.out_dir}")


if __name__ == "__main__":
    main()
