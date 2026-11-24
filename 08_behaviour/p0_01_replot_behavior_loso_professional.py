#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figures for the P0-1 pathway-behaviour analysis and its LOSO robustness check.

Plotting only. Every rho, confidence interval, leave-one-subject-out value and
adjusted P is read from the frozen CSVs; nothing is recomputed except the
Theil-Sen guide and its bootstrap band in the optional scatter figure, which
are display aids.

This script is deliberately standalone: it does not import from the analysis
module, so the figures can be re-cut without touching or rerunning the
statistics. The outcome/pathway grid is read from the statistics CSV itself.

Figure 1  ``fig_behavior_loso_robustness``
    One row per outcome x pathway. The 15 leave-one-participant-out rho values
    are plotted as individual points beside the full-sample estimate, which is
    the only way to see whether an association depends on a single participant.
    Numbers sit in aligned columns outside the data area.

Figure 2  ``fig_behavior_primary_scatter`` (only with --merged-csv)
    Participant-level scatter for each cell, with a Theil-Sen fit and a
    participant-bootstrap 95% pointwise band.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats

# Direction colours shared with the other rDCM figures in the paper.
FB_COLOR = "#A63D40"
FF_COLOR = "#2C5D7C"
NEUTRAL = "#5A6068"
INK = "#1A1A1A"
MUTED = "#6B7075"
HAIRLINE = "#C9CED4"

OUTCOME_LABELS = {
    "Threshold": "Behavioural threshold",
    "ACC_all": "Accuracy",
    "ACC": "Accuracy",
}
METRIC_LABELS = {
    "signed_FB": "Feedback: cauPFC → EVC",
    "signed_FF": "Feedforward: EVC → cauPFC",
}
METRIC_LABELS_SHORT = {
    "signed_FB": "Feedback\ncauPFC → EVC",
    "signed_FF": "Feedforward\nEVC → cauPFC",
}

REQUIRED_RESULTS = {
    "outcome", "metric", "spearman_rho", "bootstrap_ci95_low",
    "bootstrap_ci95_high", "loso_same_sign_fraction",
    "permutation_p_holm_within_2_pathway_outcome",
}
REQUIRED_LOSO = {"outcome", "metric", "omitted_subject", "loso_rho"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics-csv", type=Path, required=True)
    parser.add_argument("--loso-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--subjects-csv", "--merged-csv", dest="subjects_csv", type=Path,
        default=None,
        help=(
            "pooled_evc_behavior_subjects.csv; enables the correlation figure."
        ),
    )
    parser.add_argument(
        "--expected-n", type=int, default=15,
        help="Expected number of leave-one-out omissions per cell.",
    )
    parser.add_argument("--trend-bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument(
        "--no-titles", action="store_true",
        help="Drop the headline and footnotes (text goes in the caption).",
    )
    return parser.parse_args()


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.titlesize": 7.2,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 7,
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
    """Warn if Arial is unavailable, because the fallback changes metrics."""
    from matplotlib.font_manager import findfont, FontProperties

    # a bare "sans-serif" string is parsed as a fontconfig pattern, so pass a list
    resolved = Path(findfont(FontProperties(family=list(mpl.rcParams["font.sans-serif"])))).name
    if "arial" not in resolved.lower():
        print(
            f"[warning] Arial not found; matplotlib resolved to {resolved}. "
            "Install Arial (or msttcorefonts) for the intended typography."
        )


def strip_spines(ax: plt.Axes, keep=("bottom",)) -> None:
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)
    ax.grid(False)


def trim_spine(ax: plt.Axes, axis: str = "x") -> None:
    if axis == "x" and ax.spines["bottom"].get_visible():
        low, high = ax.get_xlim()
        ticks = [t for t in ax.get_xticks() if low <= t <= high]
        if ticks:
            ax.spines["bottom"].set_bounds(min(ticks), max(ticks))
    if axis == "y" and ax.spines["left"].get_visible():
        low, high = ax.get_ylim()
        ticks = [t for t in ax.get_yticks() if low <= t <= high]
        if ticks:
            ax.spines["left"].set_bounds(min(ticks), max(ticks))


def format_p(value: float) -> str:
    return "< .001" if value < 0.001 else f"{value:.3f}".replace("0.", ".")


def metric_color(metric: str) -> str:
    lowered = str(metric).lower()
    if "fb" in lowered or "feedback" in lowered:
        return FB_COLOR
    if "ff" in lowered or "feedforward" in lowered:
        return FF_COLOR
    return NEUTRAL


def label_for(mapping: dict[str, str], key: str) -> str:
    return mapping.get(str(key), str(key).replace("signed_", "").replace("_", " "))


def darken(color: str, factor: float = 0.68) -> tuple[float, float, float]:
    return tuple(component * factor for component in mpl.colors.to_rgb(color))


def save(fig: plt.Figure, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix, kwargs in ((".png", {"dpi": 600}), (".pdf", {}), (".svg", {})):
        path = stem.with_suffix(suffix)
        fig.savefig(path, facecolor="white", **kwargs)
        paths.append(path)
    plt.close(fig)
    return paths


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate(
    results: pd.DataFrame, loso: pd.DataFrame, expected_n: int
) -> tuple[list[str], list[str]]:
    missing_results = REQUIRED_RESULTS - set(results.columns)
    missing_loso = REQUIRED_LOSO - set(loso.columns)
    if missing_results or missing_loso:
        raise KeyError(
            f"Missing statistics columns={sorted(missing_results)}; "
            f"missing LOSO columns={sorted(missing_loso)}"
        )

    outcomes = list(dict.fromkeys(results["outcome"].astype(str)))
    metrics = list(dict.fromkeys(results["metric"].astype(str)))
    expected_cells = {(o, m) for o in outcomes for m in metrics}
    result_cells = set(
        zip(results["outcome"].astype(str), results["metric"].astype(str))
    )
    loso_cells = set(zip(loso["outcome"].astype(str), loso["metric"].astype(str)))
    if result_cells != expected_cells or loso_cells != expected_cells:
        raise RuntimeError(
            f"Outcome/pathway grid mismatch. expected={sorted(expected_cells)}; "
            f"statistics={sorted(result_cells)}; LOSO={sorted(loso_cells)}"
        )
    if results.duplicated(["outcome", "metric"]).any():
        raise RuntimeError("Duplicate outcome/metric rows in the statistics CSV")

    counts = loso.groupby(["outcome", "metric"])["omitted_subject"].nunique()
    if not counts.eq(expected_n).all():
        raise RuntimeError(
            f"Expected {expected_n} unique LOSO omissions per cell:\n{counts}"
        )

    numeric = results[
        [
            "spearman_rho", "bootstrap_ci95_low", "bootstrap_ci95_high",
            "loso_same_sign_fraction",
            "permutation_p_holm_within_2_pathway_outcome",
        ]
    ].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise RuntimeError("Statistics CSV contains non-finite values")
    if not np.isfinite(loso["loso_rho"].to_numpy(float)).all():
        raise RuntimeError("LOSO CSV contains non-finite values")
    if (results["bootstrap_ci95_low"] > results["spearman_rho"]).any():
        raise RuntimeError("A lower confidence bound exceeds its estimate")
    if (results["bootstrap_ci95_high"] < results["spearman_rho"]).any():
        raise RuntimeError("An upper confidence bound is below its estimate")
    return outcomes, metrics


# --------------------------------------------------------------------------- #
# Figure 1: LOSO robustness forest
# --------------------------------------------------------------------------- #
def plot_loso_robustness(
    results: pd.DataFrame,
    loso: pd.DataFrame,
    outcomes: list[str],
    metrics: list[str],
    out_dir: Path,
    expected_n: int,
    seed: int,
    no_titles: bool,
) -> list[Path]:
    configure_style()
    indexed = results.set_index(
        [results["outcome"].astype(str), results["metric"].astype(str)]
    )

    rows: list[dict[str, object]] = []
    for outcome in outcomes:
        rows.append({"kind": "header", "label": label_for(OUTCOME_LABELS, outcome)})
        for metric in metrics:
            rows.append({"kind": "row", "outcome": outcome, "metric": metric})
    n_slots = len(rows)

    fig_w = 7.10
    fig_h = 0.245 * n_slots + (1.78 if not no_titles else 1.05)
    left, width = 0.215, 0.250
    bottom = (0.82 if not no_titles else 0.45) / fig_h
    height = 1 - bottom - (0.58 if not no_titles else 0.28) / fig_h
    col_estimate, col_range, col_sign, col_p = 0.487, 0.775, 0.890, 0.995

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([left, bottom, width, height])
    axes_transform = mpl.transforms.blended_transform_factory(
        ax.transAxes, ax.transData
    )
    figure_transform = mpl.transforms.blended_transform_factory(
        fig.transFigure, ax.transData
    )
    rng = np.random.default_rng(seed)

    for index, entry in enumerate(rows):
        y = n_slots - 1 - index
        if entry["kind"] == "header":
            ax.text(-0.015, y, str(entry["label"]), transform=axes_transform,
                    ha="right", va="center", fontsize=7.0, fontweight="bold",
                    color=INK)
            continue

        outcome = str(entry["outcome"])
        metric = str(entry["metric"])
        row = indexed.loc[(outcome, metric)]
        color = metric_color(metric)
        accent = darken(color)
        rho = float(row["spearman_rho"])
        low = float(row["bootstrap_ci95_low"])
        high = float(row["bootstrap_ci95_high"])
        p_adj = float(row["permutation_p_holm_within_2_pathway_outcome"])
        significant = p_adj < 0.05

        values = loso.loc[
            (loso["outcome"].astype(str) == outcome)
            & (loso["metric"].astype(str) == metric),
            "loso_rho",
        ].to_numpy(float)
        same_sign = int(round(float(row["loso_same_sign_fraction"]) * values.size))

        # the leave-one-out cloud sits above the summary, never on top of it
        ax.scatter(
            values, y + 0.20 + rng.uniform(-0.055, 0.055, values.size),
            s=8, facecolor=color, edgecolor="white", linewidth=0.3,
            alpha=0.75, zorder=3,
        )
        ax.plot([low, high], [y - 0.14] * 2, color=accent, linewidth=1.5,
                solid_capstyle="butt", zorder=4)
        ax.plot(
            rho, y - 0.14, marker="o", markersize=4.8,
            markerfacecolor=accent if significant else "white",
            markeredgecolor=accent, markeredgewidth=1.15, zorder=5,
        )
        ax.text(-0.015, y, label_for(METRIC_LABELS, metric),
                transform=axes_transform, ha="right", va="center", fontsize=6.9,
                color=INK if significant else MUTED,
                fontweight="bold" if significant else "normal")

        for column, text, align in (
            (col_estimate, f"{rho:+.2f} [{low:+.2f}, {high:+.2f}]", "left"),
            (col_range, f"{values.min():+.2f} to {values.max():+.2f}", "right"),
            (col_sign, f"{same_sign}/{values.size}", "right"),
            (col_p, format_p(p_adj), "right"),
        ):
            ax.text(
                column, y, text, transform=figure_transform, ha=align,
                va="center", fontsize=6.6,
                color=INK if significant else MUTED,
                fontweight="bold" if significant else "normal",
            )

    ax.axvline(0, color=HAIRLINE, linewidth=0.7, linestyle=(0, (3, 3)), zorder=0)
    ax.set_xlim(-1.0, 1.0)
    ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_ylim(-0.75, n_slots - 0.25)
    ax.set_yticks([])
    ax.set_xlabel("Spearman ρ")
    strip_spines(ax)
    trim_spine(ax, "x")

    header_y = n_slots - 0.35
    for column, text, align in (
        (col_estimate, "ρ [95% CI]", "left"),
        (col_range, "LOSO range", "right"),
        (col_sign, "same sign", "right"),
        (col_p, "P adj.", "right"),
    ):
        ax.text(column, header_y, text, transform=figure_transform, ha=align,
                va="center", fontsize=6.4, color=MUTED)

    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=3.2,
               markerfacecolor=NEUTRAL, markeredgecolor="white",
               markeredgewidth=0.3, label="leave-one-participant-out ρ"),
        Line2D([], [], color=NEUTRAL, linewidth=1.5, marker="o", markersize=4.8,
               markerfacecolor="white", markeredgecolor=NEUTRAL,
               markeredgewidth=1.15, label="full sample ρ ± bootstrap 95% CI"),
        Line2D([], [], marker="o", linestyle="none", markersize=4.8,
               markerfacecolor=NEUTRAL, markeredgecolor=NEUTRAL,
               label="filled: Holm-adjusted P < 0.05"),
    ]
    fig.legend(
        handles=handles, frameon=False, ncols=3, loc="lower left",
        bbox_to_anchor=(0.020, 0.300 / fig_h), handlelength=1.2,
        handletextpad=0.45, columnspacing=1.5, borderpad=0.0,
    )

    if not no_titles:
        adjusted = np.array([
            float(indexed.loc[(str(e["outcome"]), str(e["metric"])),
                              "permutation_p_holm_within_2_pathway_outcome"])
            for e in rows if e["kind"] == "row"
        ])
        labels = [
            f"{label_for(OUTCOME_LABELS, e['outcome'])} / "
            f"{label_for(METRIC_LABELS, e['metric']).split(':')[0]}"
            for e in rows if e["kind"] == "row"
        ]
        if (adjusted < 0.05).any():
            hits = [
                name for name, p in zip(labels, adjusted) if p < 0.05
            ]
            headline = "Survives correction and every leave-one-out fit: " + \
                ", ".join(hits)
        else:
            headline = (
                "No association survives correction "
                f"(all adjusted P ≥ {adjusted.min():.2f})"
            )
        fig.text(0.020, 1 - 0.215 / fig_h, headline, ha="left", va="top",
                 fontsize=7.4, color=INK)
        fig.text(
            col_p, 1 - 0.215 / fig_h, f"n = {expected_n} participants",
            ha="right", va="top", fontsize=6.5, color=MUTED,
            transform=fig.transFigure,
        )
        fig.text(
            0.020, 0.155 / fig_h,
            "Each cloud is the 15 recomputations of ρ with one participant "
            "removed; 'same sign' counts how many keep the full-sample sign.",
            ha="left", va="bottom", fontsize=6.0, color=MUTED,
        )
        fig.text(
            0.020, 0.045 / fig_h,
            "Holm correction is applied across the two pathways within each "
            "outcome. Significance is decided by that adjusted permutation P, "
            "not by the bootstrap CI.",
            ha="left", va="bottom", fontsize=6.0, color=MUTED,
        )
    return save(fig, out_dir / "fig_behavior_loso_robustness")


# --------------------------------------------------------------------------- #
# Figure 2: participant-level correlations
# --------------------------------------------------------------------------- #
def theil_sen_band(
    x: np.ndarray, y: np.ndarray, n_draws: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Theil-Sen fit with a participant-bootstrap pointwise 95% band.

    Display only: this band is not the confidence interval of Spearman rho and
    it is not the inferential test.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    grid = np.linspace(x.min(), x.max(), 200)
    slope, intercept, _, _ = stats.theilslopes(y, x, alpha=0.95)
    fitted = intercept + slope * grid

    rng = np.random.default_rng(seed)
    predictions = np.full((n_draws, grid.size), np.nan)
    for draw in range(n_draws):
        index = rng.integers(0, x.size, size=x.size)
        if np.ptp(x[index]) == 0:
            continue
        try:
            draw_slope, draw_intercept, _, _ = stats.theilslopes(
                y[index], x[index], alpha=0.95
            )
        except ValueError:
            continue
        predictions[draw] = draw_intercept + draw_slope * grid
    valid = np.isfinite(predictions).all(axis=1)
    if valid.sum() < max(100, int(0.9 * n_draws)):
        raise RuntimeError(f"Only {valid.sum()}/{n_draws} valid bootstrap fits")
    low, high = np.percentile(predictions[valid], [2.5, 97.5], axis=0)
    return grid, fitted, low, high


def plot_scatter(
    results: pd.DataFrame,
    merged: pd.DataFrame,
    outcomes: list[str],
    metrics: list[str],
    out_dir: Path,
    n_draws: int,
    seed: int,
    no_titles: bool,
) -> list[Path]:
    configure_style()
    indexed = results.set_index(
        [results["outcome"].astype(str), results["metric"].astype(str)]
    )
    cells = [(o, m) for o in outcomes for m in metrics]
    missing = [
        name for name in set(outcomes) | set(metrics)
        if name not in merged.columns
    ]
    if missing:
        raise KeyError(
            f"--merged-csv has no column for {missing}. Available columns: "
            f"{list(merged.columns)}"
        )

    n_cols = len(metrics)
    n_rows = len(outcomes)
    fig_w = 7.10
    fig_h = 1.62 * n_rows + (0.62 if not no_titles else 0.40)
    fig = plt.figure(figsize=(fig_w, fig_h))
    left_margin, right_margin, gap = 0.085, 0.020, 0.115
    panel_w = (1 - left_margin - right_margin - gap * (n_cols - 1)) / n_cols
    panel_h = 1.10 / fig_h
    bottom_margin = (0.68 if not no_titles else 0.42) / fig_h
    row_gap = 0.50 / fig_h

    outputs_letters = "abcdefgh"
    for index, (outcome, metric) in enumerate(cells):
        column = index % n_cols
        row_index = index // n_cols
        panel_left = left_margin + column * (panel_w + gap)
        panel_bottom = (
            bottom_margin + (n_rows - 1 - row_index) * (panel_h + row_gap)
        )
        ax = fig.add_axes([panel_left, panel_bottom, panel_w, panel_h])

        color = metric_color(metric)
        row_for_ring = indexed.loc[(outcome, metric)]
        x = merged[outcome].to_numpy(float)
        y = merged[metric].to_numpy(float)
        grid, fitted, band_low, band_high = theil_sen_band(
            x, y, n_draws, seed + 13 * index
        )
        ax.fill_between(grid, band_low, band_high, color=color, alpha=0.13,
                        linewidth=0, zorder=1)
        ax.plot(grid, fitted, color=color, linewidth=1.0,
                linestyle=(0, (4, 2.5)), zorder=2)
        ax.axhline(0, color=HAIRLINE, linewidth=0.6, linestyle=(0, (3, 3)),
                   zorder=0)
        ax.scatter(x, y, s=18, facecolor=color, edgecolor="white",
                   linewidth=0.6, alpha=0.9, zorder=3)

        # The participant whose removal moves rho the most is already stored by
        # the analysis; ringing it here is the only way a reader can connect the
        # LOSO spread in figure 1 to an actual data point.
        influential = str(row_for_ring.get("loso_most_influential_subject", ""))
        if influential and "subject" in merged.columns:
            match = merged["subject"].astype(str) == influential
            if match.any():
                ax.scatter(
                    x[match.to_numpy()], y[match.to_numpy()], s=62,
                    facecolor="none", edgecolor=INK, linewidth=0.9, zorder=4,
                )

        y_low = min(float(y.min()), float(band_low.min()))
        y_high = max(float(y.max()), float(band_high.max()))
        x_pad = 0.06 * (x.max() - x.min())
        y_pad = 0.08 * (y_high - y_low)
        ax.set_xlim(x.min() - x_pad, x.max() + x_pad)
        ax.set_ylim(y_low - y_pad, y_high + y_pad)
        ax.set_xlabel(label_for(OUTCOME_LABELS, outcome))
        ax.set_ylabel(label_for(METRIC_LABELS, metric).split(": ")[0] + " (Hz)")
        strip_spines(ax, keep=("left", "bottom"))
        trim_spine(ax, "x")
        trim_spine(ax, "y")

        row = row_for_ring
        rho = float(row["spearman_rho"])
        low = float(row["bootstrap_ci95_low"])
        high = float(row["bootstrap_ci95_high"])
        p_adj = float(row["permutation_p_holm_within_2_pathway_outcome"])
        significant = p_adj < 0.05
        fig.text(panel_left - 0.062, panel_bottom + panel_h + 0.050,
                 outputs_letters[index], ha="left", va="bottom", fontsize=9,
                 fontweight="bold", color=INK)
        fig.text(
            panel_left, panel_bottom + panel_h + 0.018,
            f"ρ = {rho:+.2f}   95% CI [{low:+.2f}, {high:+.2f}]   "
            f"P adj. {format_p(p_adj)}",
            ha="left", va="bottom", fontsize=6.5,
            color=INK if significant else MUTED,
            fontweight="bold" if significant else "normal",
        )

    if not no_titles:
        fig.text(
            0.020, 0.195 / fig_h,
            "Dashed lines are Theil–Sen robust fits; shading is a "
            "participant-bootstrap 95% pointwise band (display only, not the CI of ρ).",
            ha="left", va="bottom", fontsize=6.1, color=MUTED,
        )
        fig.text(
            0.020, 0.070 / fig_h,
            "Ringed point: the participant whose removal shifts ρ the most "
            f"(see the LOSO clouds in the robustness figure). n = {len(merged)} "
            "participants.",
            ha="left", va="bottom", fontsize=6.1, color=MUTED,
        )
    return save(fig, out_dir / "fig_behavior_primary_scatter")




# --------------------------------------------------------------------------- #
# Figure: combined forest + correlation overview (single main figure)
# --------------------------------------------------------------------------- #
OVERVIEW = {
    "fig_w": 7.10,
    "fig_h": 6.90,
    "forest_bottom": 5.10, "forest_h": 0.95,
    "row1_bottom": 2.85, "row2_bottom": 0.72, "scatter_h": 1.30,
    "a_left": 0.215, "b_left": 0.620, "forest_w": 0.165,
    "a_text": 0.395, "b_text": 0.800,
    "col_left": 0.085, "col_right": 0.590, "scatter_w": 0.360,
}


def plot_overview(
    results: pd.DataFrame,
    merged: pd.DataFrame,
    outcomes: list[str],
    metrics: list[str],
    out_dir: Path,
    n_draws: int,
    seed: int,
    no_titles: bool,
) -> list[Path]:
    """One clean main figure: two forests on top, four correlations below."""
    configure_style()
    layout = OVERVIEW
    fig_w, fig_h = layout["fig_w"], layout["fig_h"]
    fig = plt.figure(figsize=(fig_w, fig_h))
    indexed = results.set_index(
        [results["outcome"].astype(str), results["metric"].astype(str)]
    )

    def frac_y(inches: float) -> float:
        return inches / fig_h

    # ---------------- forest row ----------------
    forest_bottom = frac_y(layout["forest_bottom"])
    forest_h = frac_y(layout["forest_h"])
    y_positions = np.arange(len(metrics))[::-1]

    for panel_index, outcome in enumerate(outcomes):
        left = layout["a_left"] if panel_index == 0 else layout["b_left"]
        text_x = layout["a_text"] if panel_index == 0 else layout["b_text"]
        ax = fig.add_axes([left, forest_bottom, layout["forest_w"], forest_h])
        text_transform = mpl.transforms.blended_transform_factory(
            fig.transFigure, ax.transData
        )

        for y, metric in zip(y_positions, metrics):
            row = indexed.loc[(outcome, metric)]
            color = metric_color(metric)
            rho = float(row["spearman_rho"])
            low = float(row["bootstrap_ci95_low"])
            high = float(row["bootstrap_ci95_high"])
            p_adj = float(row["permutation_p_holm_within_2_pathway_outcome"])
            significant = p_adj < 0.05

            ax.plot([low, high], [y, y], color=color, linewidth=1.4,
                    solid_capstyle="butt", zorder=3)
            ax.plot(
                rho, y, marker="o", markersize=5.4,
                markerfacecolor=color if significant else "white",
                markeredgecolor=color, markeredgewidth=1.2, zorder=4,
            )
            ax.text(
                text_x, y,
                f"ρ = {rho:+.2f}\nP (Holm) {format_p(p_adj)}",
                transform=text_transform, ha="left", va="center",
                fontsize=6.6, linespacing=1.35,
                color=INK if significant else MUTED,
                fontweight="bold" if significant else "normal",
            )

        ax.axvline(0, color=HAIRLINE, linewidth=0.7, linestyle=(0, (3, 3)),
                   zorder=0)
        ax.set_xlim(-1.0, 1.0)
        ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        ax.set_ylim(-0.6, len(metrics) - 0.4)
        ax.set_yticks(y_positions)
        ax.set_xlabel("Spearman ρ (bootstrap 95% CI)")
        ax.tick_params(axis="y", length=0, pad=5)
        strip_spines(ax)
        trim_spine(ax, "x")
        if panel_index == 0:
            ax.set_yticklabels(
                [label_for(METRIC_LABELS_SHORT, m) for m in metrics],
                fontsize=6.9, linespacing=1.3,
            )
        else:
            ax.set_yticklabels([])

        letter_y = forest_bottom + forest_h + frac_y(0.30)
        fig.text(left - 0.062 if panel_index else 0.020, letter_y,
                 "ab"[panel_index], ha="left", va="bottom", fontsize=9,
                 fontweight="bold", color=INK)
        fig.text(left, letter_y + 0.001, label_for(OUTCOME_LABELS, outcome),
                 ha="left", va="bottom", fontsize=7.2, color=INK)

    # ---------------- scatter rows ----------------
    letters = "cdef"
    cells = [(o, m) for o in outcomes for m in metrics]
    for index, (outcome, metric) in enumerate(cells):
        column = index % 2
        row_index = index // 2
        left = layout["col_left"] if column == 0 else layout["col_right"]
        bottom = frac_y(
            layout["row1_bottom"] if row_index == 0 else layout["row2_bottom"]
        )
        height = frac_y(layout["scatter_h"])
        ax = fig.add_axes([left, bottom, layout["scatter_w"], height])

        row = indexed.loc[(outcome, metric)]
        color = metric_color(metric)
        x = merged[outcome].to_numpy(float)
        y = merged[metric].to_numpy(float)
        grid, fitted, band_low, band_high = theil_sen_band(
            x, y, n_draws, seed + 13 * index
        )
        ax.fill_between(grid, band_low, band_high, color=color, alpha=0.14,
                        linewidth=0, zorder=1)
        ax.plot(grid, fitted, color=color, linewidth=1.1, zorder=2)
        ax.axhline(0, color=HAIRLINE, linewidth=0.6, linestyle=(0, (3, 3)),
                   zorder=0)
        ax.scatter(x, y, s=20, facecolor=color, edgecolor="white",
                   linewidth=0.6, alpha=0.92, zorder=3)

        y_low = min(float(y.min()), float(band_low.min()))
        y_high = max(float(y.max()), float(band_high.max()))
        x_pad = 0.06 * (x.max() - x.min())
        y_pad = 0.10 * (y_high - y_low)
        ax.set_xlim(x.min() - x_pad, x.max() + x_pad)
        ax.set_ylim(y_low - y_pad, y_high + y_pad)
        ax.set_xlabel(label_for(OUTCOME_LABELS, outcome))
        if column == 0:
            ax.set_ylabel("Signed mean connection (Hz)")
        strip_spines(ax, keep=("left", "bottom"))
        trim_spine(ax, "x")
        trim_spine(ax, "y")

        rho = float(row["spearman_rho"])
        p_adj = float(row["permutation_p_holm_within_2_pathway_outcome"])
        significant = p_adj < 0.05
        letter_y = bottom + height + frac_y(0.30)
        fig.text(left - 0.062, letter_y, letters[index], ha="left",
                 va="bottom", fontsize=9, fontweight="bold", color=INK)
        fig.text(left, letter_y + 0.001,
                 label_for(METRIC_LABELS, metric).split(":")[0],
                 ha="left", va="bottom", fontsize=7.2, color=INK)
        fig.text(
            left + layout["scatter_w"], letter_y + 0.001,
            f"ρ = {rho:+.2f}   P (Holm) {format_p(p_adj)}",
            ha="right", va="bottom", fontsize=6.6,
            color=INK if significant else MUTED,
            fontweight="bold" if significant else "normal",
        )

    if not no_titles:
        fig.text(
            0.020, frac_y(0.185),
            "Points are participants; lines are Theil–Sen robust fits with a "
            "participant-bootstrap 95% pointwise band.",
            ha="left", va="bottom", fontsize=6.1, color=MUTED,
        )
        fig.text(
            0.020, frac_y(0.065),
            "Filled markers in a and b mark Holm-adjusted P < 0.05, corrected "
            f"across the two pathways within each outcome. n = {len(merged)} "
            "participants.",
            ha="left", va="bottom", fontsize=6.1, color=MUTED,
        )
    return save(fig, out_dir / "fig_behavior_overview")


def main() -> None:
    args = parse_args()
    report_font()
    results = pd.read_csv(args.statistics_csv)
    loso = pd.read_csv(args.loso_csv)
    outcomes, metrics = validate(results, loso, args.expected_n)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    outputs = plot_loso_robustness(
        results, loso, outcomes, metrics, args.out_dir,
        args.expected_n, args.seed, args.no_titles,
    )
    if args.subjects_csv is not None:
        merged = pd.read_csv(args.subjects_csv)
        outputs = plot_overview(
            results, merged, outcomes, metrics, args.out_dir,
            args.trend_bootstrap_draws, args.seed, args.no_titles,
        ) + outputs

    print(f"Saved figures to {args.out_dir}")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()