#!/usr/bin/env python3
"""Validate, summarize and plot final-pial5 rDCM FF/FB recovery.

The input must be ``rdcm_recovery_long.csv`` produced by
``p1_01_run_final_pial5_rdcm_recovery.m``.  Inference is deliberately
descriptive: this simulation quantifies numerical recovery under a specified
generative/noise model and does not test whether the empirical biological
effect is true.
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
from scipy import stats


TRUTH_ORDER = ("observed", "null", "reversed")
TRUTH_LABELS = {
    "observed": "Observed FB advantage",
    "null": "Null (FB − FF = 0)",
    "reversed": "Reversed asymmetry",
}
COLORS = {0.5: "#B35806", 1.0: "#5E3C99", 3.0: "#1B9E77"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("long_csv", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expected-reps", type=int)
    parser.add_argument("--expected-n-timepoints", type=int, default=218)
    parser.add_argument("--expected-tr", type=float, default=2.2)
    parser.add_argument("--expected-te", type=float, default=0.026)
    parser.add_argument("--n-bootstrap", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260819)
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
        raise ValueError(f"{name} contains invalid Boolean strings: {bad}")
    return normalized.map(mapping).astype(bool)


def load_and_validate(args: argparse.Namespace) -> pd.DataFrame:
    # Keep the literal truth label "null"; pandas otherwise parses it as NA.
    data = pd.read_csv(args.long_csv, keep_default_na=False)
    required = {
        "case_id",
        "run_mode",
        "truth_condition",
        "generator",
        "fit_method",
        "repetition",
        "seed",
        "SNR",
        "n_timepoints",
        "n_nodes",
        "TR",
        "TE",
        "matrix_convention",
        "generation_success",
        "generation_runtime_sec",
        "generation_error",
        "fit_success",
        "fit_runtime_sec",
        "fit_error",
        "true_FF",
        "true_FB",
        "true_H",
        "estimated_FF",
        "estimated_FB",
        "estimated_H",
        "FF_bias",
        "FB_bias",
        "H_bias",
        "offdiag_rmse",
        "offdiag_pearson_r",
        "offdiag_sign_accuracy",
        "key_edge_sign_accuracy",
    }
    missing = required - set(data.columns)
    if missing:
        raise KeyError(f"Recovery CSV lacks columns: {sorted(missing)}")
    data = data.copy()
    if data.empty:
        raise RuntimeError("Recovery CSV is empty")
    if data["case_id"].astype(str).duplicated().any():
        duplicated = data.loc[
            data["case_id"].astype(str).duplicated(False), "case_id"
        ].tolist()
        raise RuntimeError(f"Duplicate recovery case IDs: {duplicated[:10]}")

    for column in ("generation_success", "fit_success"):
        data[column] = parse_boolean(data[column], column)
    numeric = (
        "repetition",
        "seed",
        "SNR",
        "n_timepoints",
        "n_nodes",
        "TR",
        "TE",
        "generation_runtime_sec",
        "fit_runtime_sec",
        "true_FF",
        "true_FB",
        "true_H",
        "estimated_FF",
        "estimated_FB",
        "estimated_H",
        "FF_bias",
        "FB_bias",
        "H_bias",
        "offdiag_rmse",
        "offdiag_pearson_r",
        "offdiag_sign_accuracy",
        "key_edge_sign_accuracy",
    )
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    if set(data["truth_condition"].astype(str)) != set(TRUTH_ORDER):
        raise RuntimeError(
            "Truth conditions must be exactly observed/null/reversed; found "
            f"{sorted(data['truth_condition'].astype(str).unique())}"
        )
    if set(data["generator"].astype(str)) != {"spm_biophysical"}:
        raise RuntimeError("Expected only the SPM biophysical generator")
    if set(data["fit_method"].astype(str)) != {"rDCM"}:
        raise RuntimeError("Expected only rDCM fits")
    if set(data["matrix_convention"].astype(str)) != {"A(target,source)"}:
        raise RuntimeError("A matrix convention is not uniquely A(target,source)")
    if data["run_mode"].nunique() != 1:
        raise RuntimeError("Do not mix different recovery run modes in one CSV")
    if not data["n_timepoints"].eq(args.expected_n_timepoints).all():
        raise RuntimeError("Unexpected number of timepoints")
    if not data["n_nodes"].eq(6).all():
        raise RuntimeError("Recovery is not uniformly six-node")
    if not np.allclose(data["TR"], args.expected_tr, atol=1e-12, rtol=0):
        raise RuntimeError("Unexpected TR")
    if not np.allclose(data["TE"], args.expected_te, atol=1e-12, rtol=0):
        raise RuntimeError("Unexpected TE")

    grid = data.groupby(["truth_condition", "SNR"], observed=True).agg(
        n_rows=("case_id", "size"),
        n_reps=("repetition", "nunique"),
        n_seeds=("seed", "nunique"),
    )
    if not (grid["n_rows"] == grid["n_reps"]).all() or not (
        grid["n_reps"] == grid["n_seeds"]
    ).all():
        raise RuntimeError(f"Duplicate/missing repetitions within cells:\n{grid}")
    inferred_reps = int(grid["n_reps"].iloc[0])
    if not grid["n_reps"].eq(inferred_reps).all():
        raise RuntimeError(f"Unequal repetitions across cells:\n{grid}")
    if args.expected_reps is not None and inferred_reps != args.expected_reps:
        raise RuntimeError(
            f"Expected {args.expected_reps} repetitions/cell; found {inferred_reps}"
        )

    # The same innovation/noise seed must pair all three truths within SNR/rep.
    pairing = data.groupby(["SNR", "repetition"], observed=True).agg(
        n_truths=("truth_condition", "nunique"),
        n_seeds=("seed", "nunique"),
    )
    if not pairing["n_truths"].eq(3).all() or not pairing["n_seeds"].eq(1).all():
        raise RuntimeError("Common-random-number truth pairing failed")

    truth = data.groupby("truth_condition", observed=True).agg(
        true_FF=("true_FF", "first"),
        true_FB=("true_FB", "first"),
        true_H=("true_H", "first"),
        n_unique_FF=("true_FF", "nunique"),
        n_unique_FB=("true_FB", "nunique"),
        n_unique_H=("true_H", "nunique"),
    )
    if not truth[["n_unique_FF", "n_unique_FB", "n_unique_H"]].eq(1).all().all():
        raise RuntimeError("A truth condition contains inconsistent FF/FB/H values")
    observed_h = float(truth.loc["observed", "true_H"])
    null_h = float(truth.loc["null", "true_H"])
    reversed_h = float(truth.loc["reversed", "true_H"])
    if observed_h <= 0:
        raise RuntimeError("Observed final-pial5 truth is not an FB advantage")
    if abs(null_h) > 1e-10:
        raise RuntimeError(f"Null truth H is not zero: {null_h}")
    if not np.isclose(reversed_h, -observed_h, atol=1e-10, rtol=0):
        raise RuntimeError("Reversed H is not the exact negative of observed H")
    reconstructed_h = data["true_FB"] - data["true_FF"]
    if not np.allclose(reconstructed_h, data["true_H"], atol=1e-12, rtol=0):
        raise RuntimeError("true_H != true_FB - true_FF")

    failures = data.loc[~data["generation_success"] | ~data["fit_success"]]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not failures.empty:
        failures.to_csv(args.out_dir / "rdcm_recovery_failures.csv", index=False)
        counts = failures.groupby(
            ["truth_condition", "SNR", "generation_success", "fit_success"],
            observed=True,
        ).size()
        raise RuntimeError(
            "Recovery contains failed generation/fits; failure CSV was saved. "
            f"Counts:\n{counts}"
        )
    runtimes = data[["generation_runtime_sec", "fit_runtime_sec"]].to_numpy(float)
    if not np.isfinite(runtimes).all() or (runtimes < 0).any():
        raise RuntimeError("Successful recovery rows contain invalid runtimes")
    needed = (
        "estimated_FF",
        "estimated_FB",
        "estimated_H",
        "FF_bias",
        "FB_bias",
        "H_bias",
        "offdiag_rmse",
        "offdiag_pearson_r",
        "offdiag_sign_accuracy",
        "key_edge_sign_accuracy",
    )
    if not np.isfinite(data[list(needed)].to_numpy(float)).all():
        raise RuntimeError("Successful recovery rows contain non-finite metrics")
    if not np.allclose(
        data["estimated_FB"] - data["estimated_FF"],
        data["estimated_H"],
        atol=1e-12,
        rtol=0,
    ):
        raise RuntimeError("estimated_H != estimated_FB - estimated_FF")
    for metric in ("FF", "FB", "H"):
        expected_bias = data[f"estimated_{metric}"] - data[f"true_{metric}"]
        if not np.allclose(expected_bias, data[f"{metric}_bias"], atol=1e-12, rtol=0):
            raise RuntimeError(f"{metric}_bias is internally inconsistent")

    return data.sort_values(
        ["SNR", "repetition", "truth_condition"]
    ).reset_index(drop=True)


def bootstrap_mean_ci(
    values: np.ndarray, n_bootstrap: int, rng: np.random.Generator
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    indices = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    z = stats.norm.ppf(0.975)
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    margin = z * np.sqrt(
        proportion * (1 - proportion) / total + z**2 / (4 * total**2)
    ) / denominator
    return float(center - margin), float(center + margin)


def summarize_cells(data: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (condition, snr), block in data.groupby(
        ["truth_condition", "SNR"], sort=True, observed=True
    ):
        estimated = block["estimated_H"].to_numpy(float)
        bias = block["H_bias"].to_numpy(float)
        rng_est = np.random.default_rng(
            stable_seed(args.seed, "estimated_H", condition, snr)
        )
        rng_bias = np.random.default_rng(
            stable_seed(args.seed, "H_bias", condition, snr)
        )
        est_low, est_high = bootstrap_mean_ci(
            estimated, args.n_bootstrap, rng_est
        )
        bias_low, bias_high = bootstrap_mean_ci(
            bias, args.n_bootstrap, rng_bias
        )
        true_h = float(block["true_H"].iloc[0])
        if abs(true_h) > 1e-10:
            successes = int((np.sign(estimated) == np.sign(true_h)).sum())
            sign_rate = successes / len(block)
            sign_low, sign_high = wilson_interval(successes, len(block))
        else:
            successes = np.nan
            sign_rate = np.nan
            sign_low = np.nan
            sign_high = np.nan
        rows.append(
            {
                "truth_condition": condition,
                "SNR": float(snr),
                "n_repetitions": len(block),
                "true_FF": float(block["true_FF"].iloc[0]),
                "true_FB": float(block["true_FB"].iloc[0]),
                "true_H": true_h,
                "estimated_H_mean": float(estimated.mean()),
                "estimated_H_sd": (
                    float(estimated.std(ddof=1)) if len(estimated) > 1 else np.nan
                ),
                "estimated_H_median": float(np.median(estimated)),
                "estimated_H_mean_bootstrap_ci95_low": est_low,
                "estimated_H_mean_bootstrap_ci95_high": est_high,
                "H_bias_mean": float(bias.mean()),
                "H_bias_mean_bootstrap_ci95_low": bias_low,
                "H_bias_mean_bootstrap_ci95_high": bias_high,
                "H_rmse": float(np.sqrt(np.mean(bias**2))),
                "H_mae": float(np.mean(np.abs(bias))),
                "H_sign_recovered_n": successes,
                "H_sign_recovery_rate": sign_rate,
                "H_sign_recovery_wilson_ci95_low": sign_low,
                "H_sign_recovery_wilson_ci95_high": sign_high,
                "estimated_H_positive_rate": float(np.mean(estimated > 0)),
                "offdiag_rmse_mean": float(block["offdiag_rmse"].mean()),
                "offdiag_pearson_r_mean": float(
                    block["offdiag_pearson_r"].mean()
                ),
                "offdiag_sign_accuracy_mean": float(
                    block["offdiag_sign_accuracy"].mean()
                ),
                "key_edge_sign_accuracy_mean": float(
                    block["key_edge_sign_accuracy"].mean()
                ),
                "generation_runtime_sec_mean": float(
                    block["generation_runtime_sec"].mean()
                ),
                "fit_runtime_sec_mean": float(block["fit_runtime_sec"].mean()),
            }
        )
    output = pd.DataFrame(rows)
    order = {name: index for index, name in enumerate(TRUTH_ORDER)}
    output["_truth_order"] = output["truth_condition"].map(order)
    return output.sort_values(["SNR", "_truth_order"]).drop(
        columns="_truth_order"
    ).reset_index(drop=True)


def calibration_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for snr, block in data.groupby("SNR", sort=True, observed=True):
        x = block["true_H"].to_numpy(float)
        y = block["estimated_H"].to_numpy(float)
        regression = stats.linregress(x, y)
        rows.append(
            {
                "SNR": float(snr),
                "n_cases": len(block),
                "calibration_intercept": float(regression.intercept),
                "calibration_slope": float(regression.slope),
                "pearson_r": float(regression.rvalue),
                "pearson_r_squared": float(regression.rvalue**2),
                "spearman_rho": float(stats.spearmanr(x, y).statistic),
                "H_rmse_all_truths": float(np.sqrt(np.mean((y - x) ** 2))),
                "H_mae_all_truths": float(np.mean(np.abs(y - x))),
            }
        )
    return pd.DataFrame(rows)


def paired_truth_separation(data: pd.DataFrame) -> pd.DataFrame:
    wide = data.pivot(
        index=["SNR", "repetition", "seed"],
        columns="truth_condition",
        values="estimated_H",
    ).reset_index()
    if set(TRUTH_ORDER) - set(wide.columns):
        raise RuntimeError("Cannot form paired truth-condition table")
    wide["observed_minus_null"] = wide["observed"] - wide["null"]
    wide["null_minus_reversed"] = wide["null"] - wide["reversed"]
    wide["correct_full_order"] = (
        (wide["observed"] > wide["null"])
        & (wide["null"] > wide["reversed"])
    )
    rows: list[dict[str, object]] = []
    for snr, block in wide.groupby("SNR", sort=True, observed=True):
        successes = int(block["correct_full_order"].sum())
        low, high = wilson_interval(successes, len(block))
        rows.append(
            {
                "SNR": float(snr),
                "n_repetitions": len(block),
                "observed_minus_null_mean": float(
                    block["observed_minus_null"].mean()
                ),
                "null_minus_reversed_mean": float(
                    block["null_minus_reversed"].mean()
                ),
                "correct_observed_gt_null_gt_reversed_n": successes,
                "correct_full_order_rate": successes / len(block),
                "correct_full_order_wilson_ci95_low": low,
                "correct_full_order_wilson_ci95_high": high,
            }
        )
    return pd.DataFrame(rows)


def color_for_snr(snr: float, index: int) -> str:
    for key, value in COLORS.items():
        if np.isclose(snr, key):
            return value
    fallback = ("#0072B2", "#D55E00", "#CC79A7", "#009E73")
    return fallback[index % len(fallback)]


def plot_recovery(
    data: pd.DataFrame, summary: pd.DataFrame, out_dir: Path, seed: int
) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    snrs = sorted(data["SNR"].unique())
    colors = {float(snr): color_for_snr(float(snr), i) for i, snr in enumerate(snrs)}
    figure, (left, right) = plt.subplots(1, 2, figsize=(10.2, 4.5))
    rng = np.random.default_rng(stable_seed(seed, "plot_jitter"))
    centers = np.arange(len(TRUTH_ORDER), dtype=float)
    offsets = np.linspace(-0.18, 0.18, len(snrs)) if len(snrs) > 1 else np.array([0.0])

    for truth_index, truth in enumerate(TRUTH_ORDER):
        true_h = float(
            data.loc[data["truth_condition"].eq(truth), "true_H"].iloc[0]
        )
        left.scatter(
            truth_index,
            true_h,
            marker="D",
            s=42,
            facecolor="#27292B",
            edgecolor="white",
            linewidth=0.7,
            zorder=5,
        )
        left.plot(
            [truth_index - 0.34, truth_index + 0.34],
            [true_h, true_h],
            color="#27292B",
            lw=1.0,
            ls=(0, (3, 2)),
            zorder=1,
        )
        for snr_index, snr in enumerate(snrs):
            block = data.loc[
                data["truth_condition"].eq(truth) & data["SNR"].eq(snr)
            ]
            x = truth_index + offsets[snr_index]
            jitter = rng.normal(0, 0.018, size=len(block))
            left.scatter(
                np.full(len(block), x) + jitter,
                block["estimated_H"],
                s=20,
                facecolor="white",
                edgecolor=colors[float(snr)],
                linewidth=0.85,
                alpha=0.65,
                zorder=2,
            )
            row = summary.loc[
                summary["truth_condition"].eq(truth) & summary["SNR"].eq(snr)
            ].iloc[0]
            mean = float(row["estimated_H_mean"])
            low = float(row["estimated_H_mean_bootstrap_ci95_low"])
            high = float(row["estimated_H_mean_bootstrap_ci95_high"])
            left.errorbar(
                x,
                mean,
                yerr=np.array([[mean - low], [high - mean]]),
                fmt="o",
                ms=5.2,
                color=colors[float(snr)],
                mec="#25282A",
                mew=0.5,
                capsize=3,
                lw=1.5,
                zorder=4,
                label=f"SNR {snr:g}" if truth_index == 0 else None,
            )

    left.axhline(0, color="#959A9E", lw=0.8, ls=(0, (4, 3)))
    left.set_xticks(centers, [TRUTH_LABELS[name] for name in TRUTH_ORDER])
    left.set_ylabel("Estimated rDCM asymmetry, H = FB − FF (Hz)")
    left.set_title("a  Condition-wise recovery", loc="left", weight="bold")
    left.grid(axis="y", color="#E4E7EA", lw=0.7)
    left.legend(frameon=False, loc="best", title="Noise condition")

    for snr_index, snr in enumerate(snrs):
        block = data.loc[data["SNR"].eq(snr)]
        x = block["true_H"].to_numpy(float)
        x_jitter = x + rng.normal(0, max(np.ptp(x), 0.1) * 0.012, size=len(x))
        right.scatter(
            x_jitter,
            block["estimated_H"],
            s=23,
            facecolor="white",
            edgecolor=colors[float(snr)],
            linewidth=0.9,
            alpha=0.70,
            label=f"SNR {snr:g}",
        )
        regression = stats.linregress(x, block["estimated_H"].to_numpy(float))
        x_line = np.array([x.min(), x.max()])
        right.plot(
            x_line,
            regression.intercept + regression.slope * x_line,
            color=colors[float(snr)],
            lw=1.6,
        )
    limits = np.array(
        [
            data["true_H"].min(),
            data["true_H"].max(),
            data["estimated_H"].min(),
            data["estimated_H"].max(),
        ]
    )
    padding = max(np.ptp(limits) * 0.08, 0.02)
    lower, upper = float(limits.min() - padding), float(limits.max() + padding)
    right.plot([lower, upper], [lower, upper], color="#8C9195", lw=1, ls="--")
    right.axhline(0, color="#D0D3D5", lw=0.7)
    right.axvline(0, color="#D0D3D5", lw=0.7)
    right.set_xlim(lower, upper)
    right.set_ylim(lower, upper)
    right.set_xlabel("Known true asymmetry, H (Hz)")
    right.set_ylabel("Estimated asymmetry, H (Hz)")
    right.set_title("b  Calibration across known truths", loc="left", weight="bold")
    right.legend(frameon=False, loc="best")

    figure.suptitle(
        "Recovery of the final pial5 EVC–cauPFC rDCM asymmetry",
        x=0.06,
        y=1.01,
        ha="left",
        fontsize=14,
        weight="bold",
    )
    figure.text(
        0.06,
        -0.025,
        "Open points: individual simulations; coloured points/error bars: mean and bootstrap 95% CI; "
        "black diamonds/dashed segments: known truth.",
        color="#666B70",
        fontsize=8,
    )
    figure.tight_layout(rect=(0.0, 0.05, 1.0, 0.98), w_pad=3.2)
    for suffix in ("png", "pdf", "svg"):
        kwargs = {"dpi": 600} if suffix == "png" else {}
        figure.savefig(
            out_dir / f"final_pial5_rdcm_recovery.{suffix}",
            bbox_inches="tight",
            **kwargs,
        )
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.n_bootstrap < 10_000:
        raise ValueError("Use at least 10,000 bootstrap resamples")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = load_and_validate(args)
    summary = summarize_cells(data, args)
    calibration = calibration_summary(data)
    separation = paired_truth_separation(data)

    data.to_csv(args.out_dir / "rdcm_recovery_validated_cases.csv", index=False)
    summary.to_csv(args.out_dir / "rdcm_recovery_cell_summary.csv", index=False)
    calibration.to_csv(
        args.out_dir / "rdcm_recovery_calibration_summary.csv", index=False
    )
    separation.to_csv(
        args.out_dir / "rdcm_recovery_paired_truth_separation.csv", index=False
    )
    plot_recovery(data, summary, args.out_dir, args.seed)

    manifest = {
        "analysis": "Final pial5 EVC-cauPFC rDCM FF/FB recovery summary",
        "input_csv": str(args.long_csv.resolve()),
        "run_mode": str(data["run_mode"].iloc[0]),
        "n_cases": len(data),
        "n_repetitions_per_cell": int(
            data.groupby(["truth_condition", "SNR"]).size().iloc[0]
        ),
        "truth_conditions": list(TRUTH_ORDER),
        "SNR": sorted(map(float, data["SNR"].unique())),
        "generator": "SPM one-state biophysical fMRI model",
        "fit": "rDCM using the final empirical pial5 settings",
        "matrix_convention": "A(target,source)",
        "FF": "mean(A(4,3),A(5,3))",
        "FB": "mean(A(1,4),A(3,4))",
        "H": "FB-FF",
        "intervals": "Monte Carlo-repetition bootstrap 95% CIs",
        "sign_recovery_interval": "Wilson 95% interval",
        "inference": "descriptive recovery; no null-hypothesis significance claim",
        "interpretation": (
            "Numerical identifiability under the specified simulation and noise "
            "model; not proof that the empirical biological effect is true."
        ),
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
    }
    (args.out_dir / "rdcm_recovery_analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(f"\nSaved validated recovery results to {args.out_dir}")


if __name__ == "__main__":
    main()
