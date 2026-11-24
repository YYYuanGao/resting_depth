#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Publication figure for the rDCM recovery simulation.

Standalone plotting program: it does not import the simulation or analysis
code. It validates the complete paired grid, writes the exact summaries used
in the figure, and exports PNG, PDF and SVG.

Layout
------
Both panels use one grammar: one row per condition, estimate plus interval
inside the axes, every number in aligned columns outside the axes.

    a  Recovery rate      rows grouped by success definition, one row per SNR
    b  Calibration slope  one row per SNR

The previous version wrote the percentages at x = 1.025 and the slope values
at x = 0.34, i.e. inside the data area and on top of the bars; the x-limit was
stretched to 1.12 purely to make room for that text. Nothing is written over
data here. Panel rows also had different meanings between a and b (metric vs
SNR), which is now consistent.

Visual system matches the other publication scripts: Arial 7 pt, hairline
spines trimmed to the tick range, no gridlines, exact output dimensions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
from scipy import stats

TRUTH_ORDER = ("reversed", "null", "observed")

INK = "#1A1A1A"
MUTED = "#6B7075"
HAIRLINE = "#C9CED4"
# SNR is an ordinal design factor with no sign, so the default ramp is
# neutral grey getting darker. Red and blue stay reserved for signed
# quantities elsewhere in the paper (e.g. the A-matrix panels), which avoids a
# reader carrying "blue = negative coupling" across figures.
SNR_RAMPS = {
    "purple": ("#CFC2E2", "#A78BC9", "#7B5EA7", "#583E7E", "#3B2857"),
    "neutral": ("#C4CAD0", "#8B959E", "#525D68", "#2B3640", "#1A2129"),
    "blue": ("#9CC3D9", "#5C93B4", "#2C5D7C", "#173E5B", "#0E2A3E"),
}
# Tones for the three success definitions in the curve layout: dark to light
# within one hue, so the lines stay separable in greyscale.
METRIC_RAMPS = {
    "purple": ("#3F2A5C", "#7B5EA7", "#BCAAD8"),
    "neutral": ("#22303A", "#7C8790", "#B9C0C6"),
    "blue": ("#1F4A63", "#4E86A8", "#A9C6D8"),
}

METRIC_ORDER = ("Full order", "Observed sign", "Reversed sign")
METRIC_KEYS = {
    "order": "Full order",
    "observed": "Observed sign",
    "reversed": "Reversed sign",
}
METRIC_LABELS = {
    "Full order": "Paired ordering  obs. > null > rev.",
    "Observed sign": "Observed sign  H(obs) > 0",
    "Reversed sign": "Reversed sign  H(rev) < 0",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("long_csv", type=Path, help="rdcm_recovery_long.csv")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="final_pial5_rdcm_recovery")
    parser.add_argument("--n-bootstrap", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--expected-reps", type=int,
        help="Require this many repetitions in every truth-by-SNR cell",
    )
    parser.add_argument(
        "--direction-only", action="store_true",
        help="Export only the directional-recovery panel (drop calibration)",
    )
    parser.add_argument(
        "--metrics", nargs="+", default=["observed", "reversed", "order"],
        choices=["observed", "reversed", "order"],
        help=(
            "Which success definitions to draw, in order. Dropping 'reversed' "
            "removes the specificity control that separates true directional "
            "recovery from a positive bias, so it is kept by default."
        ),
    )
    parser.add_argument(
        "--layout", choices=("curves", "rows"), default="curves",
        help=(
            "curves: recovery and calibration as compact lines over SNR "
            "(default). rows: one row per SNR with numeric columns."
        ),
    )
    parser.add_argument(
        "--rate-ymin", choices=("auto", "zero"), default="auto",
        help=(
            "auto: start the y axis just below the lowest interval so the "
            "informative range fills the panel. zero: always start at 0."
        ),
    )
    parser.add_argument(
        "--annotate", action="store_true",
        help="Print the exact value beside every point in the curve layout.",
    )
    parser.add_argument(
        "--palette", choices=tuple(SNR_RAMPS), default="purple",
        help=(
            "Colour family for both layouts (default purple). Every colour is "
            "one hue at different lightness, so nothing is confused with the "
            "signed red/blue used for coupling elsewhere."
        ),
    )
    parser.add_argument(
        "--snr-palette", choices=tuple(SNR_RAMPS), default=None,
        help="Deprecated alias for --palette.",
    )
    parser.add_argument(
        "--no-titles", action="store_true",
        help="Drop panel titles and footnotes (text goes in the caption).",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Validation and summaries (unchanged behaviour)
# --------------------------------------------------------------------------- #
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


def load_and_validate(
    path: Path, expected_reps: int | None = None
) -> tuple[pd.DataFrame, int]:
    # Preserve the literal truth label "null"; pandas otherwise treats it as NA.
    data = pd.read_csv(path, keep_default_na=False)
    required = {
        "case_id", "run_mode", "truth_condition", "generator", "fit_method",
        "repetition", "seed", "SNR", "matrix_convention", "generation_success",
        "fit_success", "true_FF", "true_FB", "true_H", "estimated_FF",
        "estimated_FB", "estimated_H",
    }
    missing = required - set(data.columns)
    if missing:
        raise KeyError(f"Input lacks required columns: {sorted(missing)}")
    if data.empty:
        raise RuntimeError("Input CSV is empty")
    if data["case_id"].astype(str).duplicated().any():
        duplicates = data.loc[
            data["case_id"].astype(str).duplicated(False), "case_id"
        ].tolist()
        raise RuntimeError(f"Duplicate case_id values: {duplicates[:10]}")

    data = data.copy()
    for column in ("generation_success", "fit_success"):
        data[column] = parse_boolean(data[column], column)
    numeric = (
        "repetition", "seed", "SNR", "true_FF", "true_FB", "true_H",
        "estimated_FF", "estimated_FB", "estimated_H",
    )
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    if set(data["truth_condition"].astype(str)) != set(TRUTH_ORDER):
        raise RuntimeError("Truth conditions must be observed, null and reversed")
    if data["run_mode"].nunique() != 1:
        raise RuntimeError("Do not mix run modes in one figure")
    if set(data["generator"].astype(str)) != {"spm_biophysical"}:
        raise RuntimeError("Expected only the SPM biophysical generator")
    if set(data["fit_method"].astype(str)) != {"rDCM"}:
        raise RuntimeError("Expected only rDCM fits")
    if set(data["matrix_convention"].astype(str)) != {"A(target,source)"}:
        raise RuntimeError("Matrix convention must be A(target,source)")
    if not data["generation_success"].all() or not data["fit_success"].all():
        failures = data.loc[
            ~data["generation_success"] | ~data["fit_success"], "case_id"
        ].tolist()
        raise RuntimeError(f"Input contains failed simulations/fits: {failures[:10]}")
    if not np.isfinite(data[list(numeric)].to_numpy(float)).all():
        raise RuntimeError("Successful rows contain non-finite numeric values")
    if not np.allclose(
        data["true_FB"] - data["true_FF"], data["true_H"], atol=1e-12, rtol=0
    ):
        raise RuntimeError("true_H is inconsistent with true_FB - true_FF")
    if not np.allclose(
        data["estimated_FB"] - data["estimated_FF"], data["estimated_H"],
        atol=1e-12, rtol=0,
    ):
        raise RuntimeError(
            "estimated_H is inconsistent with estimated_FB - estimated_FF"
        )

    truth = data.groupby("truth_condition", observed=True)["true_H"].agg(
        ["first", "nunique"]
    )
    if not truth["nunique"].eq(1).all():
        raise RuntimeError("A truth condition contains inconsistent true_H values")
    observed_h = float(truth.loc["observed", "first"])
    null_h = float(truth.loc["null", "first"])
    reversed_h = float(truth.loc["reversed", "first"])
    if observed_h <= 0 or abs(null_h) > 1e-10 or not np.isclose(
        reversed_h, -observed_h, atol=1e-10, rtol=0
    ):
        raise RuntimeError("Observed/null/reversed true_H values are not symmetric")

    grid = data.groupby(["truth_condition", "SNR"], observed=True).agg(
        n_rows=("case_id", "size"),
        n_repetitions=("repetition", "nunique"),
        n_seeds=("seed", "nunique"),
    )
    if not (grid["n_rows"] == grid["n_repetitions"]).all() or not (
        grid["n_repetitions"] == grid["n_seeds"]
    ).all():
        raise RuntimeError(f"Duplicate or missing rows within grid cells:\n{grid}")
    n_repetitions = int(grid["n_repetitions"].iloc[0])
    if not grid["n_repetitions"].eq(n_repetitions).all():
        raise RuntimeError(f"Unequal repetitions across grid cells:\n{grid}")
    if expected_reps is not None and n_repetitions != expected_reps:
        raise RuntimeError(
            f"Expected {expected_reps} repetitions/cell; found {n_repetitions}"
        )

    pairing = data.groupby(["SNR", "repetition"], observed=True).agg(
        n_truths=("truth_condition", "nunique"), n_seeds=("seed", "nunique")
    )
    if not pairing["n_truths"].eq(3).all() or not pairing["n_seeds"].eq(1).all():
        raise RuntimeError("Truth conditions are not paired by common seed")
    return data.sort_values(["SNR", "repetition", "truth_condition"]), n_repetitions


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


def summarize_cells(
    data: pd.DataFrame, n_bootstrap: int, seed: int
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (snr, truth), block in data.groupby(
        ["SNR", "truth_condition"], sort=True, observed=True
    ):
        values = block["estimated_H"].to_numpy(float)
        rng = np.random.default_rng(stable_seed(seed, "cell", snr, truth))
        low, high = bootstrap_mean_ci(values, n_bootstrap, rng)
        true_h = float(block["true_H"].iloc[0])
        if truth == "observed":
            successes = int((values > 0).sum())
        elif truth == "reversed":
            successes = int((values < 0).sum())
        else:
            successes = -1
        if successes >= 0:
            sign_rate = successes / len(values)
            sign_low, sign_high = wilson_interval(successes, len(values))
        else:
            sign_rate = sign_low = sign_high = float("nan")
        rows.append(
            {
                "SNR": float(snr),
                "truth_condition": truth,
                "n_repetitions": len(values),
                "true_H": true_h,
                "estimated_H_mean": float(values.mean()),
                "estimated_H_sd": float(values.std(ddof=1)),
                "estimated_H_median": float(np.median(values)),
                "estimated_H_bootstrap_ci95_low": low,
                "estimated_H_bootstrap_ci95_high": high,
                "H_bias_mean": float(np.mean(values - true_h)),
                "H_rmse": float(np.sqrt(np.mean((values - true_h) ** 2))),
                "sign_recovered_n": successes if successes >= 0 else np.nan,
                "sign_recovery_rate": sign_rate,
                "sign_recovery_wilson_ci95_low": sign_low,
                "sign_recovery_wilson_ci95_high": sign_high,
            }
        )
    return pd.DataFrame(rows)


def paired_diagnostics(
    data: pd.DataFrame, n_bootstrap: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    wide = data.pivot(
        index=["SNR", "repetition", "seed"],
        columns="truth_condition",
        values="estimated_H",
    ).reset_index()
    true_h = float(
        data.loc[data["truth_condition"].eq("observed"), "true_H"].iloc[0]
    )
    wide["calibration_slope"] = (wide["observed"] - wide["reversed"]) / (2 * true_h)
    wide["calibration_intercept"] = wide[
        ["observed", "null", "reversed"]
    ].mean(axis=1)
    wide["correct_full_order"] = (
        (wide["observed"] > wide["null"]) & (wide["null"] > wide["reversed"])
    )

    calibration_rows: list[dict[str, object]] = []
    direction_rows: list[dict[str, object]] = []
    for snr, block in wide.groupby("SNR", sort=True, observed=True):
        slopes = block["calibration_slope"].to_numpy(float)
        intercepts = block["calibration_intercept"].to_numpy(float)
        slope_rng = np.random.default_rng(stable_seed(seed, "slope", snr))
        intercept_rng = np.random.default_rng(stable_seed(seed, "intercept", snr))
        slope_low, slope_high = bootstrap_mean_ci(slopes, n_bootstrap, slope_rng)
        int_low, int_high = bootstrap_mean_ci(intercepts, n_bootstrap, intercept_rng)
        calibration_rows.append(
            {
                "SNR": float(snr),
                "n_paired_repetitions": len(block),
                "calibration_slope_mean": float(slopes.mean()),
                "calibration_slope_sd": float(slopes.std(ddof=1)),
                "calibration_slope_bootstrap_ci95_low": slope_low,
                "calibration_slope_bootstrap_ci95_high": slope_high,
                "calibration_intercept_mean": float(intercepts.mean()),
                "calibration_intercept_bootstrap_ci95_low": int_low,
                "calibration_intercept_bootstrap_ci95_high": int_high,
            }
        )
        successes = int(block["correct_full_order"].sum())
        low, high = wilson_interval(successes, len(block))
        direction_rows.append(
            {
                "SNR": float(snr),
                "metric": "Full order",
                "successes": successes,
                "n_repetitions": len(block),
                "rate": successes / len(block),
                "ci95_low": low,
                "ci95_high": high,
            }
        )
    return pd.DataFrame(calibration_rows), pd.DataFrame(direction_rows)


def direction_summary(cells: pd.DataFrame, full_order: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for truth, label in (("observed", "Observed sign"), ("reversed", "Reversed sign")):
        block = cells.loc[cells["truth_condition"].eq(truth)]
        for _, row in block.iterrows():
            rows.append(
                {
                    "SNR": float(row["SNR"]),
                    "metric": label,
                    "successes": int(row["sign_recovered_n"]),
                    "n_repetitions": int(row["n_repetitions"]),
                    "rate": float(row["sign_recovery_rate"]),
                    "ci95_low": float(row["sign_recovery_wilson_ci95_low"]),
                    "ci95_high": float(row["sign_recovery_wilson_ci95_high"]),
                }
            )
    return pd.concat([pd.DataFrame(rows), full_order], ignore_index=True)


def format_full_order_result(direction: pd.DataFrame, n_repetitions: int) -> str:
    block = direction.loc[direction["metric"].eq("Full order")].sort_values("SNR")
    if block.empty:
        raise RuntimeError("Directional summary lacks the paired full-order metric")
    if block["n_repetitions"].astype(int).ne(n_repetitions).any():
        raise RuntimeError("Full-order denominators disagree with the validated grid")
    if block["successes"].astype(int).eq(n_repetitions).all():
        return f"{n_repetitions}/{n_repetitions} at each SNR"
    return "; ".join(
        f"SNR {float(row.SNR):g}: {int(row.successes)}/{n_repetitions}"
        for row in block.itertuples(index=False)
    )


# --------------------------------------------------------------------------- #
# Style
# --------------------------------------------------------------------------- #
def set_plot_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.titlesize": 7.2,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.9,
            "legend.fontsize": 6.4,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.6,
            "ytick.major.size": 2.6,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.edgecolor": INK,
            "axes.grid": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def report_font() -> None:
    from matplotlib.font_manager import FontProperties, findfont

    resolved = Path(
        findfont(FontProperties(family=list(mpl.rcParams["font.sans-serif"])))
    ).name
    if "arial" not in resolved.lower():
        print(
            f"[warning] Arial not found; matplotlib resolved to {resolved}. "
            "Install Arial (or msttcorefonts) for the intended typography."
        )


def snr_colors(snrs: list[float], palette: str = "neutral") -> dict[float, str]:
    ramp = SNR_RAMPS[palette]
    if len(snrs) <= len(ramp):
        chosen = ramp[: len(snrs)]
    else:
        cmap = mpl.colormaps["Greys" if palette == "neutral" else "Blues"]
        chosen = [
            mpl.colors.to_hex(cmap(value))
            for value in np.linspace(0.30, 0.85, len(snrs))
        ]
    return {float(snr): color for snr, color in zip(snrs, chosen)}


def strip_spines(ax: plt.Axes, keep=("bottom",)) -> None:
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)
    ax.grid(False)


def trim_x_spine(ax: plt.Axes) -> None:
    low, high = ax.get_xlim()
    ticks = [t for t in ax.get_xticks() if low <= t <= high]
    if ticks:
        ax.spines["bottom"].set_bounds(min(ticks), max(ticks))


def darken(color: str, factor: float = 0.72) -> tuple[float, float, float]:
    return tuple(component * factor for component in mpl.colors.to_rgb(color))


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
LAYOUT = {
    "fig_w": 7.10,
    "left": 0.300, "width": 0.300,
    "col_value": 0.625, "col_count": 0.995,
    "row_in": 0.190, "header_in": 0.220,
}


def _draw_rows(
    ax: plt.Axes,
    fig: plt.Figure,
    entries: list[dict[str, object]],
    colors: dict[float, str],
    value_fmt,
    count_fmt,
) -> None:
    """Shared row renderer: point + interval inside, numbers in outside columns."""
    n_slots = len(entries)
    axes_transform = mpl.transforms.blended_transform_factory(
        ax.transAxes, ax.transData
    )
    figure_transform = mpl.transforms.blended_transform_factory(
        fig.transFigure, ax.transData
    )
    for index, entry in enumerate(entries):
        y = n_slots - 1 - index
        if entry["kind"] == "header":
            ax.text(-0.015, y, str(entry["label"]), transform=axes_transform,
                    ha="right", va="center", fontsize=7.0, fontweight="bold",
                    color=INK)
            continue
        color = colors[float(entry["snr"])]
        accent = darken(color)
        value = float(entry["value"])
        low = float(entry["low"])
        high = float(entry["high"])
        ax.plot([low, high], [y, y], color=accent, linewidth=1.4,
                solid_capstyle="butt", zorder=3)
        ax.plot(value, y, marker="o", markersize=4.6, markerfacecolor=color,
                markeredgecolor=accent, markeredgewidth=1.0, zorder=4)
        ax.text(-0.015, y, str(entry["label"]), transform=axes_transform,
                ha="right", va="center", fontsize=6.9, color=INK)
        ax.text(LAYOUT["col_value"], y, value_fmt(value, low, high),
                transform=figure_transform, ha="left", va="center",
                fontsize=6.6, color=INK)
        if entry.get("count") is not None:
            ax.text(LAYOUT["col_count"], y, count_fmt(entry["count"]),
                    transform=figure_transform, ha="right", va="center",
                    fontsize=6.6, color=MUTED)


def plot_recovery(
    direction: pd.DataFrame,
    calibration: pd.DataFrame,
    snrs: list[float],
    n_repetitions: int,
    args: argparse.Namespace,
) -> None:
    set_plot_style()
    colors = snr_colors(snrs, args.snr_palette or args.palette)
    metrics = [METRIC_KEYS[key] for key in args.metrics]

    rate_entries: list[dict[str, object]] = []
    for metric in metrics:
        rate_entries.append({"kind": "header", "label": METRIC_LABELS[metric]})
        for snr in snrs:
            row = direction.loc[
                direction["metric"].eq(metric) & direction["SNR"].eq(snr)
            ].iloc[0]
            rate_entries.append({
                "kind": "row", "snr": snr, "label": f"SNR {snr:g}",
                "value": float(row["rate"]), "low": float(row["ci95_low"]),
                "high": float(row["ci95_high"]),
                "count": (int(row["successes"]), int(row["n_repetitions"])),
            })

    slope_entries: list[dict[str, object]] = []
    if not args.direction_only:
        calibration = calibration.sort_values("SNR")
        for _, row in calibration.iterrows():
            snr = float(row["SNR"])
            slope_entries.append({
                "kind": "row", "snr": snr, "label": f"SNR {snr:g}",
                "value": float(row["calibration_slope_mean"]),
                "low": float(row["calibration_slope_bootstrap_ci95_low"]),
                "high": float(row["calibration_slope_bootstrap_ci95_high"]),
                "count": None,
            })

    rate_h = LAYOUT["row_in"] * sum(e["kind"] == "row" for e in rate_entries) + \
        LAYOUT["header_in"] * sum(e["kind"] == "header" for e in rate_entries)
    slope_h = LAYOUT["row_in"] * len(slope_entries)
    top_pad = 0.42 if args.no_titles else 0.58
    gap = 0.92
    bottom_pad = 0.30 if args.no_titles else 0.32
    trailing = 0.42 if args.no_titles else 0.44
    fig_h = top_pad + rate_h + (
        gap + slope_h + trailing if slope_entries else trailing
    ) + bottom_pad
    fig_w = LAYOUT["fig_w"]
    fig = plt.figure(figsize=(fig_w, fig_h))

    def frac(inches: float) -> float:
        return inches / fig_h

    # ---- panel a: recovery rates ----
    rate_bottom = fig_h - top_pad - rate_h
    ax_rate = fig.add_axes(
        [LAYOUT["left"], frac(rate_bottom), LAYOUT["width"], frac(rate_h)]
    )
    ax_rate.axvline(1.0, color=HAIRLINE, linewidth=0.7, linestyle=(0, (3, 3)),
                    zorder=0)
    _draw_rows(
        ax_rate, fig, rate_entries, colors,
        value_fmt=lambda v, lo, hi: f"{v:.0%}  [{lo:.0%}, {hi:.0%}]",
        count_fmt=lambda c: f"{c[0]}/{c[1]}",
    )
    n_slots = len(rate_entries)
    ax_rate.set_xlim(0.0, 1.02)
    ax_rate.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_rate.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax_rate.set_ylim(-0.6, n_slots - 0.4)
    ax_rate.set_yticks([])
    ax_rate.set_xlabel("Recovery rate (Wilson 95% CI)")
    strip_spines(ax_rate)
    trim_x_spine(ax_rate)

    header_transform = mpl.transforms.blended_transform_factory(
        fig.transFigure, ax_rate.transData
    )
    ax_rate.text(LAYOUT["col_value"], n_slots - 0.35, "rate [95% CI]",
                 transform=header_transform, ha="left", va="center",
                 fontsize=6.4, color=MUTED)
    ax_rate.text(LAYOUT["col_count"], n_slots - 0.35, "successes",
                 transform=header_transform, ha="right", va="center",
                 fontsize=6.4, color=MUTED)

    letter_y = frac(rate_bottom + rate_h + 0.20)
    fig.text(0.020, letter_y, "a", ha="left", va="bottom", fontsize=9,
             fontweight="bold", color=INK)
    if not args.no_titles:
        fig.text(0.048, letter_y + 0.001, "Directional recovery", ha="left",
                 va="bottom", fontsize=7.2, color=INK)
        fig.text(LAYOUT["col_count"], letter_y + 0.001,
                 f"n = {n_repetitions} paired repetitions per cell",
                 ha="right", va="bottom", fontsize=6.5, color=MUTED)

    # ---- panel b: calibration slope ----
    if slope_entries:
        slope_bottom = rate_bottom - gap - slope_h
        ax_slope = fig.add_axes(
            [LAYOUT["left"], frac(slope_bottom), LAYOUT["width"], frac(slope_h)]
        )
        ax_slope.axvline(1.0, color=HAIRLINE, linewidth=0.7,
                         linestyle=(0, (3, 3)), zorder=0)
        _draw_rows(
            ax_slope, fig, slope_entries, colors,
            value_fmt=lambda v, lo, hi: f"{v:.2f}  [{lo:.2f}, {hi:.2f}]",
            count_fmt=lambda c: "",
        )
        highest = max(
            [float(e["high"]) for e in slope_entries] + [1.0]
        )
        ax_slope.set_xlim(0.0, max(1.02, highest * 1.05))
        ax_slope.xaxis.set_major_locator(
            mpl.ticker.MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10])
        )
        ax_slope.set_ylim(-0.6, len(slope_entries) - 0.4)
        ax_slope.set_yticks([])
        ax_slope.set_xlabel("Calibration slope, estimated / true")
        strip_spines(ax_slope)
        trim_x_spine(ax_slope)

        slope_header = mpl.transforms.blended_transform_factory(
            fig.transFigure, ax_slope.transData
        )
        ax_slope.text(LAYOUT["col_value"], len(slope_entries) - 0.35,
                      "slope [95% CI]", transform=slope_header, ha="left",
                      va="center", fontsize=6.4, color=MUTED)

        letter_y = frac(slope_bottom + slope_h + 0.20)
        fig.text(0.020, letter_y, "b", ha="left", va="bottom", fontsize=9,
                 fontweight="bold", color=INK)
        if not args.no_titles:
            fig.text(0.048, letter_y + 0.001, "Magnitude calibration",
                     ha="left", va="bottom", fontsize=7.2, color=INK)
            fig.text(LAYOUT["col_count"], letter_y + 0.001,
                     "1.00 = unbiased", ha="right", va="bottom",
                     fontsize=6.5, color=MUTED)

    if not args.no_titles:
        fig.text(
            0.020, frac(0.185),
            "Dashed lines mark perfect recovery. Marker shade encodes SNR, which "
            "is also labelled on every row.",
            ha="left", va="bottom", fontsize=6.0, color=MUTED,
        )
        fig.text(
            0.020, frac(0.065),
            "Intervals are Wilson 95% CIs for rates and paired-bootstrap 95% CIs "
            "for the slope. These are recovery diagnostics, not null-hypothesis "
            "tests.",
            ha="left", va="bottom", fontsize=6.0, color=MUTED,
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        kwargs = {"dpi": args.dpi} if suffix == "png" else {}
        fig.savefig(
            args.out_dir / f"{args.prefix}.{suffix}", facecolor="white", **kwargs
        )
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Curve layout: recovery and calibration as a function of SNR
# --------------------------------------------------------------------------- #
METRIC_SHORT = {
    "Observed sign": "H(obs) > 0",
    "Reversed sign": "H(rev) < 0",
    "Full order": "obs > null > rev",
}


def plot_curves(
    direction: pd.DataFrame,
    calibration: pd.DataFrame,
    snrs: list[float],
    metrics: list[str],
    n_repetitions: int,
    args: argparse.Namespace,
) -> None:
    """Compact two-panel figure: both panels share SNR as the x axis."""
    set_plot_style()
    palette = args.snr_palette or args.palette
    tones = dict(zip(
        ("Observed sign", "Reversed sign", "Full order"), METRIC_RAMPS[palette]
    ))
    x = np.arange(len(snrs), dtype=float)
    show_calibration = not args.direction_only

    fig_w = 5.60 if show_calibration else 3.30
    fig_h = 2.55 if not args.no_titles else 2.20
    fig = plt.figure(figsize=(fig_w, fig_h))

    def frac_x(inches: float) -> float:
        return inches / fig_w

    def frac_y(inches: float) -> float:
        return inches / fig_h

    bottom = frac_y(0.52 if not args.no_titles else 0.45)
    height = frac_y(1.42)
    panel_w = frac_x(1.62)
    left_a = frac_x(0.55)
    left_b = frac_x(3.55)

    # ---- panel a: recovery rate vs SNR ----
    ax = fig.add_axes([left_a, bottom, panel_w, height])
    ax.axhline(1.0, color=HAIRLINE, linewidth=0.7, linestyle=(0, (3, 3)), zorder=0)
    for metric in metrics:
        tone = tones[metric]
        block = direction.loc[direction["metric"].eq(metric)].sort_values("SNR")
        values = block["rate"].to_numpy(float)
        low = block["ci95_low"].to_numpy(float)
        high = block["ci95_high"].to_numpy(float)
        ax.plot(x, values, color=tone, linewidth=1.1, zorder=2)
        ax.errorbar(
            x, values, yerr=np.vstack([values - low, high - values]),
            fmt="o", markersize=4.0, markerfacecolor=tone, markeredgecolor="white",
            markeredgewidth=0.5, ecolor=tone, elinewidth=0.9, capsize=0, zorder=3,
        )
        if args.annotate:
            for xi, value in zip(x, values):
                ax.text(xi, value + 0.035, f"{value:.0%}", ha="center",
                        va="bottom", fontsize=5.8, color=tone)

    ax.set_xlim(-0.25, len(snrs) - 1 + 0.25)
    ax.set_xticks(x, [f"{snr:g}" for snr in snrs])
    # The informative range of a recovery rate is near the ceiling, so the axis
    # starts just below the lowest confidence bound instead of wasting two
    # thirds of the panel on empty space. Percentage ticks keep this explicit.
    lowest = float(
        direction.loc[direction["metric"].isin(metrics), "ci95_low"].min()
    )
    rate_floor = 0.0 if args.rate_ymin == "zero" else min(
        0.9, max(0.0, np.floor((lowest - 0.05) * 10) / 10)
    )
    ax.set_ylim(rate_floor, 1.0 + 0.06 * (1.0 - rate_floor))
    ax.yaxis.set_major_locator(
        mpl.ticker.MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10])
    )
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))

    handles = [
        Line2D([], [], color=tones[metric], linewidth=1.1, marker="o",
               markersize=3.6, markerfacecolor=tones[metric],
               markeredgecolor="white", markeredgewidth=0.5,
               label=METRIC_SHORT[metric])
        for metric in metrics
    ]
    ax.legend(
        handles=handles, loc="lower right", frameon=False, fontsize=6.2,
        handlelength=1.3, handletextpad=0.45, borderpad=0.0, labelspacing=0.3,
    )
    ax.set_xlabel("SNR")
    ax.set_ylabel("Recovery rate")
    strip_spines(ax, keep=("left", "bottom"))
    trim_x_spine(ax)
    low_lim, high_lim = ax.get_ylim()
    yticks = [t for t in ax.get_yticks() if low_lim <= t <= high_lim]
    if yticks:
        ax.spines["left"].set_bounds(min(yticks), max(yticks))

    fig.text(frac_x(0.055), 1 - frac_y(0.20), "a", ha="left", va="top",
             fontsize=9, fontweight="bold", color=INK)
    if not args.no_titles:
        fig.text(frac_x(0.22), 1 - frac_y(0.205), "Directional recovery",
                 ha="left", va="top", fontsize=7.2, color=INK)

    # ---- panel b: calibration slope vs SNR ----
    if show_calibration:
        ax_b = fig.add_axes([left_b, bottom, panel_w, height])
        calibration = calibration.sort_values("SNR")
        values = calibration["calibration_slope_mean"].to_numpy(float)
        low = calibration["calibration_slope_bootstrap_ci95_low"].to_numpy(float)
        high = calibration["calibration_slope_bootstrap_ci95_high"].to_numpy(float)
        tone = METRIC_RAMPS[palette][0]
        ax_b.axhline(1.0, color=HAIRLINE, linewidth=0.7, linestyle=(0, (3, 3)),
                     zorder=0)
        # the mid tone keeps the band visibly in the same hue as the line
        ax_b.fill_between(x, low, high, color=METRIC_RAMPS[palette][1],
                          alpha=0.22, linewidth=0, zorder=1)
        ax_b.plot(x, values, color=tone, linewidth=1.1, zorder=2)
        ax_b.plot(x, values, marker="o", linestyle="none", markersize=4.0,
                  markerfacecolor=tone, markeredgecolor="white",
                  markeredgewidth=0.5, zorder=3)
        if args.annotate:
            for xi, value in zip(x, values):
                ax_b.text(xi, value + 0.035, f"{value:.2f}", ha="center",
                          va="bottom", fontsize=5.8, color=tone)
        ax_b.text(x[-1] + 0.12, 1.0, "unbiased", ha="left", va="center",
                  fontsize=6.3, color=MUTED, clip_on=False)

        ax_b.set_xlim(-0.25, len(snrs) - 1 + 0.25)
        ax_b.set_xticks(x, [f"{snr:g}" for snr in snrs])
        slope_floor = 0.0 if args.rate_ymin == "zero" else max(
            0.0, np.floor((float(low.min()) - 0.05) * 10) / 10
        )
        ax_b.set_ylim(slope_floor, max(1.0, float(high.max())) + 0.06)
        ax_b.yaxis.set_major_locator(
            mpl.ticker.MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10])
        )
        ax_b.set_xlabel("SNR")
        ax_b.set_ylabel("Calibration slope")
        strip_spines(ax_b, keep=("left", "bottom"))
        trim_x_spine(ax_b)
        low_lim, high_lim = ax_b.get_ylim()
        yticks = [t for t in ax_b.get_yticks() if low_lim <= t <= high_lim]
        if yticks:
            ax_b.spines["left"].set_bounds(min(yticks), max(yticks))

        fig.text(frac_x(3.05), 1 - frac_y(0.20), "b", ha="left", va="top",
                 fontsize=9, fontweight="bold", color=INK)
        if not args.no_titles:
            fig.text(frac_x(3.22), 1 - frac_y(0.205), "Magnitude calibration",
                     ha="left", va="top", fontsize=7.2, color=INK)

    if not args.no_titles:
        note = (
            f"n = {n_repetitions} per cell. Bars: Wilson 95% CI"
            + ("; shading: bootstrap 95% CI." if show_calibration else ".")
        )
        fig.text(
            frac_x(0.055), frac_y(0.055), note,
            ha="left", va="bottom", fontsize=5.9, color=MUTED,
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        kwargs = {"dpi": args.dpi} if suffix == "png" else {}
        fig.savefig(
            args.out_dir / f"{args.prefix}.{suffix}", facecolor="white", **kwargs
        )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.n_bootstrap < 10_000:
        raise ValueError("Use at least 10,000 bootstrap resamples")
    if args.dpi < 300:
        raise ValueError("Use at least 300 dpi for the PNG export")
    report_font()

    data, n_repetitions = load_and_validate(args.long_csv, args.expected_reps)
    cells = summarize_cells(data, args.n_bootstrap, args.seed)
    calibration, full_order = paired_diagnostics(data, args.n_bootstrap, args.seed)
    direction = direction_summary(cells, full_order)
    snrs = sorted(map(float, data["SNR"].unique()))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cells.to_csv(args.out_dir / "recovery_plot_cell_summary.csv", index=False)
    calibration.to_csv(
        args.out_dir / "recovery_plot_calibration_summary.csv", index=False
    )
    direction.to_csv(
        args.out_dir / "recovery_plot_direction_summary.csv", index=False
    )

    metrics = [METRIC_KEYS[key] for key in args.metrics]
    if args.layout == "curves":
        plot_curves(
            direction, calibration, snrs, metrics, n_repetitions, args
        )
    else:
        plot_recovery(direction, calibration, snrs, n_repetitions, args)

    order_result = format_full_order_result(direction, n_repetitions)
    if args.direction_only:
        caption = (
            "Fig. X | Directional recovery of the final pial5 EVC–cauPFC rDCM "
            "FF–FB asymmetry. Recovery rates quantify the positive sign of the "
            "observed asymmetry and the negative sign of the reversed control. "
            "Points show proportions across repetitions and error bars show "
            "Wilson 95% confidence intervals. The estimated observed > null > "
            f"reversed ordering was preserved as follows: {order_result}. "
            "H = FB − FF. SNR is the node-wise signal standard deviation divided "
            f"by the observation-noise standard deviation (n = {n_repetitions} "
            "paired repetitions per truth condition and SNR)."
        )
    else:
        caption = (
            "Fig. X | Directional and magnitude recovery of the final pial5 "
            "EVC–cauPFC rDCM asymmetry. a, Recovery rates for the paired ordering "
            "(estimated observed > null > reversed), the positive sign of the "
            "observed asymmetry and the negative sign of the reversed control. "
            "Points show proportions across paired repetitions and error bars show "
            "Wilson 95% confidence intervals. b, Paired calibration slope, defined "
            "as the recovered observed-to-reversed contrast divided by the known "
            "true contrast; a slope of 1 denotes unbiased magnitude recovery. Error "
            "bars show paired-bootstrap 95% confidence intervals. H = FB − FF. "
            "SNR is the node-wise signal standard deviation divided by the "
            f"observation-noise standard deviation (n = {n_repetitions} paired "
            "repetitions per truth condition and SNR). Truth conditions within a "
            "repetition shared the same simulated innovations and observation-noise "
            "realization."
        )
    (args.out_dir / f"{args.prefix}_caption.txt").write_text(
        caption + "\n", encoding="utf-8"
    )

    manifest = {
        "program": "standalone final-pial5 rDCM recovery plotter",
        "input_csv": str(args.long_csv.resolve()),
        "run_mode": str(data["run_mode"].iloc[0]),
        "n_cases": len(data),
        "n_repetitions_per_truth_snr": n_repetitions,
        "truth_conditions": list(TRUTH_ORDER),
        "SNR": snrs,
        "matrix_convention": "A(target,source)",
        "H": "FB-FF",
        "cell_intervals": "paired-repetition bootstrap 95% confidence intervals",
        "rate_intervals": "Wilson 95% confidence intervals",
        "calibration_slope": (
            "mean paired slope (observed-reversed)/(2*true observed H)"
        ),
        "inference_note": "Recovery diagnostics; no null-hypothesis p values",
        "figure_scope": (
            "direction_only" if args.direction_only else "direction_and_magnitude"
        ),
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "outputs": [f"{args.prefix}.{suffix}" for suffix in ("png", "pdf", "svg")]
        + [f"{args.prefix}_caption.txt"],
    }
    (args.out_dir / "recovery_plot_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print("Input validation: PASS")
    print(f"Cases: {len(data)} | repetitions/cell: {n_repetitions}")
    print("\nCalibration:")
    print(calibration.to_string(index=False))
    print("\nDirectional recovery:")
    print(direction.to_string(index=False))
    print(f"\nSaved figure and summaries to: {args.out_dir}")


if __name__ == "__main__":
    main()