#!/usr/bin/env python3
"""
Focused pathway-level association between final outer-pial-5% rDCM estimates
and behavior.

The rDCM long table must follow the SPM/DCM convention A(target, source).
For every participant and network, this script reconstructs the four
literature-defined laminar edges used by the hierarchy analysis:

    FF = mean(visual superficial -> cauPFC deep,
              visual superficial -> cauPFC middle)

    FB = mean(cauPFC deep -> visual deep,
              cauPFC deep -> visual superficial)

Primary inferential family (Threshold) and secondary family (ACC_all):

    cauPFC_EVC signed_FF, cauPFC_EVC signed_FB,
    cauPFC_V1  signed_FF, cauPFC_V1  signed_FB.

For each test the script reports:
    - Spearman rho
    - paired-participant percentile-bootstrap 95% CI
    - two-sided behavior-label permutation p-value
    - Holm-adjusted permutation p within the four-test outcome family
    - leave-one-participant-out rho range and sign stability

It also writes publication-oriented scatter and forest plots. Lines in the
scatter plots are Theil-Sen visual guides; inference is based on Spearman rho
and label permutation, not on the fitted line.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import os
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "/path/to/project")  # see config/paths.sh


DERIV = Path(PROJECT_ROOT + "/derivatives_0331")
DEFAULT_RDCM_LONG = (
    DERIV
    / "S6_validation"
    / "pial5_only_update"
    / "rdcm_long"
    / "pial5_current"
    / "merged_layer_rdcm_connections_long.csv"
)
DEFAULT_BEHAVIOR = Path(PROJECT_ROOT + "/wmpfc_cau_350.csv")
DEFAULT_OUT = (
    DERIV
    / "S6_validation"
    / "pial5_only_update"
    / "rdcm_behavior_pathway"
)

NETWORK_ORDER = ("cauPFC_EVC", "cauPFC_V1")
NETWORK_LABELS = {
    "cauPFC_EVC": "EVC–cauPFC",
    "cauPFC_V1": "V1–cauPFC",
}
NETWORK_VISUAL = {
    "cauPFC_EVC": "EVC",
    "cauPFC_V1": "V1",
}
METRIC_ORDER = ("signed_FB", "signed_FF")
METRIC_LABELS = {
    "signed_FB": "Feedback (FB)",
    "signed_FF": "Feedforward (FF)",
}
OUTCOME_ORDER = ("Threshold", "ACC_all")
OUTCOME_LABELS = {
    "Threshold": "Threshold",
    "ACC_all": "Accuracy",
}

# Colorblind-safe Okabe-Ito-inspired colors.
BLUE = "#0072B2"
VERMILLION = "#D55E00"
DARK = "#202124"
GRID = "#D9DDE3"
PURPLE = "#6F4E9C"

REQUIRED_RDCM_COLUMNS = {
    "subject",
    "network",
    "source_group",
    "target_group",
    "source_layer",
    "target_layer",
    "A",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rdcm-long", type=Path, default=DEFAULT_RDCM_LONG)
    parser.add_argument("--behavior-csv", type=Path, default=DEFAULT_BEHAVIOR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--expected-n", type=int, default=15)
    parser.add_argument("--n-permutations", type=int, default=100_000)
    parser.add_argument("--n-bootstrap", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260724)
    return parser.parse_args()


def normalize_subject(value: object) -> str:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        raise ValueError(f"Invalid participant identifier: {value!r}")
    match = re.search(r"(\d+)", text)
    if match is None:
        raise ValueError(f"Could not parse participant identifier: {value!r}")
    return f"sub-{int(match.group(1)):02d}"


def stable_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join([str(base_seed), *(str(x) for x in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Holm family-wise adjusted p-values in the original row order."""
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all():
        raise ValueError("Holm adjustment requires finite one-dimensional p-values.")
    order = np.argsort(p)
    ranked = p[order]
    adjusted_ranked = np.maximum.accumulate(
        (len(ranked) - np.arange(len(ranked))) * ranked
    )
    adjusted_ranked = np.clip(adjusted_ranked, 0.0, 1.0)
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    return adjusted


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y) or len(x) < 3:
        return np.nan
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    xr = stats.rankdata(x, method="average")
    yr = stats.rankdata(y, method="average")
    return float(np.corrcoef(xr, yr)[0, 1])


def permutation_p_spearman(
    x: np.ndarray,
    y: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
    chunk_size: int = 20_000,
) -> tuple[float, int]:
    """
    Monte Carlo two-sided label-permutation p-value with a +1 correction.

    Correlation of average ranks is computed directly, so tied ACC values are
    handled in the same way as scipy.stats.spearmanr.
    """
    x_rank = stats.rankdata(np.asarray(x, dtype=float), method="average")
    y_rank = stats.rankdata(np.asarray(y, dtype=float), method="average")
    x_rank = x_rank - x_rank.mean()
    y_rank = y_rank - y_rank.mean()
    x_norm = np.linalg.norm(x_rank)
    y_norm = np.linalg.norm(y_rank)
    if x_norm == 0 or y_norm == 0:
        return np.nan, 0
    x_rank = x_rank / x_norm
    y_rank = y_rank / y_norm
    observed = float(np.dot(x_rank, y_rank))

    exceedances = 0
    completed = 0
    n = len(x_rank)
    while completed < n_permutations:
        m = min(chunk_size, n_permutations - completed)
        # Sorting independent continuous random values gives uniformly random
        # permutations for each row.
        indices = np.argsort(rng.random((m, n)), axis=1)
        permuted_rho = y_rank[indices] @ x_rank
        exceedances += int(
            np.count_nonzero(np.abs(permuted_rho) >= abs(observed) - 1e-15)
        )
        completed += m
    p_value = (exceedances + 1.0) / (n_permutations + 1.0)
    return float(p_value), exceedances


def bootstrap_spearman_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float, int]:
    """Paired-participant percentile-bootstrap 95% CI for Spearman rho."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    estimates = np.empty(n_bootstrap, dtype=float)
    estimates.fill(np.nan)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        estimates[i] = spearman_rho(x[idx], y[idx])
    valid = estimates[np.isfinite(estimates)]
    if len(valid) < max(100, int(0.90 * n_bootstrap)):
        raise RuntimeError(
            f"Only {len(valid)}/{n_bootstrap} valid bootstrap estimates."
        )
    low, high = np.quantile(valid, [0.025, 0.975])
    return float(low), float(high), int(len(valid))


def loo_summary(x: np.ndarray, y: np.ndarray, observed: float) -> dict[str, float]:
    values = []
    for omitted in range(len(x)):
        keep = np.arange(len(x)) != omitted
        values.append(spearman_rho(x[keep], y[keep]))
    loo = np.asarray(values, dtype=float)
    valid = loo[np.isfinite(loo)]
    if not len(valid):
        return {
            "loo_rho_min": np.nan,
            "loo_rho_max": np.nan,
            "loo_same_sign_fraction": np.nan,
        }
    if observed > 0:
        same = valid > 0
    elif observed < 0:
        same = valid < 0
    else:
        same = valid == 0
    return {
        "loo_rho_min": float(valid.min()),
        "loo_rho_max": float(valid.max()),
        "loo_same_sign_fraction": float(np.mean(same)),
    }


def one_edge(group: pd.DataFrame, mask: pd.Series, label: str) -> tuple[float, str]:
    selected = group.loc[mask]
    if len(selected) != 1:
        subject = group["subject"].iloc[0]
        network = group["network"].iloc[0]
        raise RuntimeError(
            f"{subject}/{network}: expected one {label} edge, found {len(selected)}."
        )
    row = selected.iloc[0]
    edge_label = (
        f"{row['source_group']}_{row['source_layer']}->"
        f"{row['target_group']}_{row['target_layer']}"
    )
    return float(row["A"]), edge_label


def load_and_reconstruct_rdcm(
    path: Path,
    expected_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(path)
    long_df = pd.read_csv(path)
    missing = REQUIRED_RDCM_COLUMNS - set(long_df.columns)
    if missing:
        raise RuntimeError(f"{path} missing columns: {sorted(missing)}")

    long_df = long_df.copy()
    long_df["subject"] = long_df["subject"].map(normalize_subject)
    for column in [
        "network",
        "source_group",
        "target_group",
        "source_layer",
        "target_layer",
    ]:
        long_df[column] = long_df[column].astype(str).str.strip()
    long_df["A"] = pd.to_numeric(long_df["A"], errors="raise")
    if not np.isfinite(long_df["A"].to_numpy(float)).all():
        raise RuntimeError("rDCM long table contains non-finite A values.")

    rows = []
    audit_rows = []
    for network in NETWORK_ORDER:
        part_network = long_df.loc[long_df["network"] == network]
        if part_network.empty:
            raise RuntimeError(f"Missing required network: {network}")
        if part_network["subject"].nunique() != expected_n:
            raise RuntimeError(
                f"{network}: expected {expected_n} participants, found "
                f"{part_network['subject'].nunique()}."
            )
        visual = NETWORK_VISUAL[network]
        for subject, group in part_network.groupby("subject", sort=True):
            ff_base = (
                (group["source_group"] == visual)
                & (group["target_group"] == "cau_PFC")
                & (group["source_layer"] == "sup")
            )
            fb_base = (
                (group["source_group"] == "cau_PFC")
                & (group["target_group"] == visual)
                & (group["source_layer"] == "deep")
            )

            ff_deep, ff_deep_label = one_edge(
                group,
                ff_base & (group["target_layer"] == "deep"),
                "FF visual-sup -> cauPFC-deep",
            )
            ff_mid, ff_mid_label = one_edge(
                group,
                ff_base & (group["target_layer"] == "mid"),
                "FF visual-sup -> cauPFC-mid",
            )
            fb_deep, fb_deep_label = one_edge(
                group,
                fb_base & (group["target_layer"] == "deep"),
                "FB cauPFC-deep -> visual-deep",
            )
            fb_sup, fb_sup_label = one_edge(
                group,
                fb_base & (group["target_layer"] == "sup"),
                "FB cauPFC-deep -> visual-sup",
            )

            signed_ff = float(np.mean([ff_deep, ff_mid]))
            signed_fb = float(np.mean([fb_deep, fb_sup]))
            rows.append(
                {
                    "subject": subject,
                    "network": network,
                    "visual_group": visual,
                    "signed_FF": signed_ff,
                    "signed_FB": signed_fb,
                    "signed_H": signed_fb - signed_ff,
                    "A_FF_visual_sup_to_cau_deep": ff_deep,
                    "A_FF_visual_sup_to_cau_mid": ff_mid,
                    "A_FB_cau_deep_to_visual_deep": fb_deep,
                    "A_FB_cau_deep_to_visual_sup": fb_sup,
                }
            )
            audit_rows.append(
                {
                    "subject": subject,
                    "network": network,
                    "FF_edge_1": ff_deep_label,
                    "FF_edge_2": ff_mid_label,
                    "FB_edge_1": fb_deep_label,
                    "FB_edge_2": fb_sup_label,
                    "matrix_convention": (
                        "A(target,source): directed influence source->target"
                    ),
                    "status": "PASS",
                }
            )

    metrics = pd.DataFrame(rows)
    if metrics.duplicated(["subject", "network"]).any():
        raise RuntimeError("Duplicate subject/network rows after reconstruction.")
    return metrics, pd.DataFrame(audit_rows)


def find_subject_column(df: pd.DataFrame) -> str:
    lower_map = {str(column).strip().lower(): column for column in df.columns}
    for candidate in [
        "subject",
        "participant_id",
        "participant",
        "subject_id",
        "sub",
        "id",
    ]:
        if candidate in lower_map:
            return str(lower_map[candidate])

    first = str(df.columns[0])
    values = df[first].dropna().astype(str)
    if len(values) and values.str.contains(r"sub-?\d+|^\d+$", case=False, regex=True).all():
        return first
    raise RuntimeError(
        "Could not identify the participant column in the behavior CSV. "
        f"Columns are: {list(df.columns)}"
    )


def load_behavior(path: Path, expected_n: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    raw = pd.read_csv(path)
    raw.columns = raw.columns.astype(str).str.strip()
    subject_column = find_subject_column(raw)
    missing = set(OUTCOME_ORDER) - set(raw.columns)
    if missing:
        raise RuntimeError(f"{path} missing behavior columns: {sorted(missing)}")

    behavior = raw[[subject_column, *OUTCOME_ORDER]].copy()
    behavior = behavior.rename(columns={subject_column: "subject"})
    behavior["subject"] = behavior["subject"].map(normalize_subject)
    for outcome in OUTCOME_ORDER:
        behavior[outcome] = pd.to_numeric(behavior[outcome], errors="coerce")
    if behavior.duplicated("subject").any():
        duplicated = behavior.loc[
            behavior.duplicated("subject", keep=False), "subject"
        ].tolist()
        raise RuntimeError(f"Duplicate behavior rows: {duplicated}")
    if len(behavior) != expected_n:
        raise RuntimeError(
            f"Expected {expected_n} behavior rows, found {len(behavior)}."
        )
    if behavior[list(OUTCOME_ORDER)].isna().any().any():
        bad = behavior.loc[
            behavior[list(OUTCOME_ORDER)].isna().any(axis=1),
            ["subject", *OUTCOME_ORDER],
        ]
        raise RuntimeError(
            "Missing/non-numeric behavior values:\n" + bad.to_string(index=False)
        )
    return behavior.sort_values("subject").reset_index(drop=True)


def merge_inputs(
    metrics: pd.DataFrame,
    behavior: pd.DataFrame,
    expected_n: int,
) -> pd.DataFrame:
    rdcm_subjects = set(metrics["subject"])
    behavior_subjects = set(behavior["subject"])
    if rdcm_subjects != behavior_subjects:
        raise RuntimeError(
            "Participant mismatch.\n"
            f"Only in rDCM: {sorted(rdcm_subjects - behavior_subjects)}\n"
            f"Only in behavior: {sorted(behavior_subjects - rdcm_subjects)}"
        )
    merged = metrics.merge(
        behavior,
        on="subject",
        how="inner",
        validate="many_to_one",
    )
    counts = merged.groupby("network")["subject"].nunique()
    if not counts.eq(expected_n).all():
        raise RuntimeError(f"Unexpected merged participant counts:\n{counts}")
    return merged.sort_values(["network", "subject"]).reset_index(drop=True)


def run_statistics(
    merged: pd.DataFrame,
    n_permutations: int,
    n_bootstrap: int,
    base_seed: int,
) -> pd.DataFrame:
    rows = []
    for outcome in OUTCOME_ORDER:
        for network in NETWORK_ORDER:
            part = merged.loc[merged["network"] == network].sort_values("subject")
            y = part[outcome].to_numpy(float)
            for metric in METRIC_ORDER:
                x = part[metric].to_numpy(float)
                rho, p_asymptotic = stats.spearmanr(x, y)
                rho = float(rho)
                p_asymptotic = float(p_asymptotic)

                perm_rng = np.random.default_rng(
                    stable_seed(base_seed, "permutation", outcome, network, metric)
                )
                p_perm, n_exceed = permutation_p_spearman(
                    x,
                    y,
                    n_permutations=n_permutations,
                    rng=perm_rng,
                )
                boot_rng = np.random.default_rng(
                    stable_seed(base_seed, "bootstrap", outcome, network, metric)
                )
                ci_low, ci_high, n_valid_boot = bootstrap_spearman_ci(
                    x,
                    y,
                    n_bootstrap=n_bootstrap,
                    rng=boot_rng,
                )
                loo = loo_summary(x, y, rho)
                rows.append(
                    {
                        "family": f"{outcome}_four_pathway_tests",
                        "outcome": outcome,
                        "network": network,
                        "network_label": NETWORK_LABELS[network],
                        "metric": metric,
                        "metric_label": METRIC_LABELS[metric],
                        "n_subjects": len(part),
                        "spearman_rho": rho,
                        "bootstrap_ci95_low": ci_low,
                        "bootstrap_ci95_high": ci_high,
                        "p_spearman_asymptotic": p_asymptotic,
                        "permutation_p_two_sided": p_perm,
                        "permutation_exceedances": n_exceed,
                        "n_permutations": n_permutations,
                        "n_valid_bootstrap": n_valid_boot,
                        **loo,
                        "is_focused_primary": (
                            outcome == "Threshold"
                            and network == "cauPFC_EVC"
                            and metric == "signed_FB"
                        ),
                    }
                )

    result = pd.DataFrame(rows)
    result["permutation_p_holm_within_outcome_family"] = np.nan
    for outcome in OUTCOME_ORDER:
        mask = result["outcome"] == outcome
        if int(mask.sum()) != 4:
            raise RuntimeError(f"Expected four tests for {outcome}, got {mask.sum()}.")
        result.loc[
            mask, "permutation_p_holm_within_outcome_family"
        ] = holm_adjust(result.loc[mask, "permutation_p_two_sided"].to_numpy())
    result["_outcome_order"] = result["outcome"].map(
        {value: index for index, value in enumerate(OUTCOME_ORDER)}
    )
    result["_network_order"] = result["network"].map(
        {value: index for index, value in enumerate(NETWORK_ORDER)}
    )
    result["_metric_order"] = result["metric"].map(
        {value: index for index, value in enumerate(METRIC_ORDER)}
    )
    return (
        result.sort_values(
            ["_outcome_order", "_network_order", "_metric_order"]
        )
        .drop(
            columns=["_outcome_order", "_network_order", "_metric_order"]
        )
        .reset_index(drop=True)
    )


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
        }
    )


def theil_sen_line(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    slope, intercept, _, _ = stats.theilslopes(y, x, alpha=0.95)
    line_x = np.linspace(float(np.min(x)), float(np.max(x)), 100)
    return line_x, intercept + slope * line_x


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_scatter_panels(
    merged: pd.DataFrame,
    result: pd.DataFrame,
    out_dir: Path,
) -> None:
    configure_style()
    colors = {"signed_FF": BLUE, "signed_FB": VERMILLION}
    markers = {"signed_FF": "o", "signed_FB": "s"}

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.15, 5.8),
        constrained_layout=True,
    )
    panel_letters = iter("abcd")
    for row_index, outcome in enumerate(OUTCOME_ORDER):
        for column_index, network in enumerate(NETWORK_ORDER):
            ax = axes[row_index, column_index]
            part = merged.loc[merged["network"] == network].sort_values("subject")
            behavior = part[outcome].to_numpy(float)
            for metric in ("signed_FF", "signed_FB"):
                connection = part[metric].to_numpy(float)
                ax.scatter(
                    behavior,
                    connection,
                    s=27,
                    marker=markers[metric],
                    facecolors="white",
                    edgecolors=colors[metric],
                    linewidths=1.2,
                    label=METRIC_LABELS[metric],
                    zorder=3,
                )
                line_x, line_y = theil_sen_line(behavior, connection)
                ax.plot(line_x, line_y, color=colors[metric], linewidth=1.2, zorder=2)

            stats_part = result.loc[
                (result["outcome"] == outcome)
                & (result["network"] == network)
            ].set_index("metric")
            annotation_lines = []
            for metric in ("signed_FB", "signed_FF"):
                row = stats_part.loc[metric]
                annotation_lines.append(
                    f"{metric[-2:]}: ρ={row['spearman_rho']:+.2f} "
                    f"[{row['bootstrap_ci95_low']:+.2f}, "
                    f"{row['bootstrap_ci95_high']:+.2f}], "
                    f"Holm P={row['permutation_p_holm_within_outcome_family']:.3f}"
                )
            ax.text(
                0.02,
                0.98,
                "\n".join(annotation_lines),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=6.8,
                color=DARK,
            )
            ax.axhline(0, color="#999999", linewidth=0.7, linestyle=(0, (3, 2)))
            ax.grid(True, color=GRID, linewidth=0.5, zorder=0)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_title(NETWORK_LABELS[network], loc="left", fontweight="bold")
            x_label = OUTCOME_LABELS[outcome]
            if outcome == "Threshold":
                x_label += " (lower = better, if task-defined)"
            ax.set_xlabel(x_label)
            ax.set_ylabel("Signed mean connection (Hz)")
            ax.text(
                -0.16,
                1.04,
                next(panel_letters),
                transform=ax.transAxes,
                fontsize=10,
                fontweight="bold",
            )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper right",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.98, 1.055),
    )
    fig.suptitle(
        "Resting-state laminar pathway strength and behavior",
        x=0.02,
        y=1.045,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.02,
        -0.015,
        (
            "N=15; final outer-pial-5% rDCM inputs. Points are participants; "
            "lines are Theil–Sen visual guides. Brackets are paired-participant "
            "bootstrap 95% CIs for Spearman ρ; Holm P values use two-sided "
            "behavior-label permutation within each four-test outcome family."
        ),
        ha="left",
        va="top",
        fontsize=6.8,
        color="#666666",
    )
    save_figure(fig, out_dir / "figure_rdcm_pathway_behavior_scatter")


def plot_forest(
    result: pd.DataFrame,
    out_dir: Path,
) -> None:
    configure_style()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.15, 3.65),
        sharex=True,
        constrained_layout=True,
    )
    ordered_pairs = [
        ("cauPFC_EVC", "signed_FB"),
        ("cauPFC_EVC", "signed_FF"),
        ("cauPFC_V1", "signed_FB"),
        ("cauPFC_V1", "signed_FF"),
    ]

    for panel, outcome in enumerate(OUTCOME_ORDER):
        ax = axes[panel]
        part = result.loc[result["outcome"] == outcome].set_index(
            ["network", "metric"]
        )
        y_positions = np.arange(len(ordered_pairs))[::-1]
        for y_position, pair in zip(y_positions, ordered_pairs):
            row = part.loc[pair]
            metric = pair[1]
            color = VERMILLION if metric == "signed_FB" else BLUE
            significant = (
                row["permutation_p_holm_within_outcome_family"] < 0.05
            )
            ax.errorbar(
                row["spearman_rho"],
                y_position,
                xerr=np.asarray(
                    [
                        [
                            row["spearman_rho"]
                            - row["bootstrap_ci95_low"]
                        ],
                        [
                            row["bootstrap_ci95_high"]
                            - row["spearman_rho"]
                        ],
                    ]
                ),
                fmt="s" if metric == "signed_FB" else "o",
                markersize=5,
                markerfacecolor=color if significant else "white",
                markeredgecolor=color,
                markeredgewidth=1.2,
                ecolor=color,
                elinewidth=1.2,
                capsize=2.5,
            )
            ax.text(
                1.03,
                y_position,
                (
                    f"ρ={row['spearman_rho']:+.2f}; "
                    f"P={row['permutation_p_holm_within_outcome_family']:.3f}"
                ),
                transform=ax.get_yaxis_transform(),
                ha="left",
                va="center",
                fontsize=6.6,
            )

        labels = [
            f"{NETWORK_LABELS[network]}  {METRIC_LABELS[metric]}"
            for network, metric in ordered_pairs
        ]
        ax.set_yticks(y_positions, labels=labels)
        ax.axvline(0, color="#777777", linewidth=0.8, linestyle=(0, (3, 2)))
        ax.set_xlim(-1.0, 1.0)
        ax.set_xlabel("Spearman ρ [bootstrap 95% CI]")
        ax.set_title(
            OUTCOME_LABELS[outcome],
            loc="left",
            fontweight="bold",
        )
        ax.grid(axis="x", color=GRID, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(
            -0.14,
            1.04,
            "a" if panel == 0 else "b",
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
        )

    fig.suptitle(
        "Pathway-level associations with behavior",
        x=0.02,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.02,
        -0.02,
        (
            "Open markers denote Holm-adjusted P≥0.05; filled markers denote "
            "Holm-adjusted P<0.05 within the corresponding four-test outcome "
            "family. Confidence intervals are marginal paired-participant "
            "bootstrap intervals."
        ),
        ha="left",
        va="top",
        fontsize=6.8,
        color="#666666",
    )
    save_figure(fig, out_dir / "figure_rdcm_pathway_behavior_forest")


def write_audit(
    args: argparse.Namespace,
    metrics: pd.DataFrame,
    behavior: pd.DataFrame,
    merged: pd.DataFrame,
    out_dir: Path,
) -> None:
    audit = {
        "rdcm_long": str(args.rdcm_long.resolve()),
        "behavior_csv": str(args.behavior_csv.resolve()),
        "output_directory": str(out_dir.resolve()),
        "matrix_convention": "A(target,source)",
        "FF_definition": (
            "mean(visual_sup->cauPFC_deep, visual_sup->cauPFC_mid)"
        ),
        "FB_definition": (
            "mean(cauPFC_deep->visual_deep, cauPFC_deep->visual_sup)"
        ),
        "expected_n": args.expected_n,
        "n_rdcm_subjects": int(metrics["subject"].nunique()),
        "n_behavior_subjects": int(behavior["subject"].nunique()),
        "n_merged_subjects_per_network": {
            key: int(value)
            for key, value in merged.groupby("network")["subject"].nunique().items()
        },
        "n_permutations": args.n_permutations,
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "multiplicity": (
            "Holm correction separately within the four Threshold tests and "
            "the four ACC_all tests"
        ),
        "status": "PASS",
    }
    with (out_dir / "analysis_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    if args.n_permutations < 999:
        raise ValueError("--n-permutations should be at least 999.")
    if args.n_bootstrap < 1_000:
        raise ValueError("--n-bootstrap should be at least 1000.")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    metrics, edge_audit = load_and_reconstruct_rdcm(
        args.rdcm_long,
        expected_n=args.expected_n,
    )
    behavior = load_behavior(args.behavior_csv, expected_n=args.expected_n)
    merged = merge_inputs(metrics, behavior, expected_n=args.expected_n)
    result = run_statistics(
        merged,
        n_permutations=args.n_permutations,
        n_bootstrap=args.n_bootstrap,
        base_seed=args.seed,
    )

    metrics.to_csv(args.out_dir / "rdcm_pathway_metrics_per_subject.csv", index=False)
    edge_audit.to_csv(args.out_dir / "rdcm_pathway_edge_audit.csv", index=False)
    behavior.to_csv(args.out_dir / "behavior_input_audit.csv", index=False)
    merged.to_csv(args.out_dir / "rdcm_pathway_behavior_merged.csv", index=False)
    result.to_csv(args.out_dir / "rdcm_pathway_behavior_statistics.csv", index=False)

    plot_scatter_panels(merged, result, args.out_dir)
    plot_forest(result, args.out_dir)
    write_audit(args, metrics, behavior, merged, args.out_dir)

    display_columns = [
        "outcome",
        "network",
        "metric",
        "n_subjects",
        "spearman_rho",
        "bootstrap_ci95_low",
        "bootstrap_ci95_high",
        "permutation_p_two_sided",
        "permutation_p_holm_within_outcome_family",
        "loo_rho_min",
        "loo_rho_max",
        "loo_same_sign_fraction",
    ]
    print("\n" + "=" * 100)
    print("Focused pathway-level rDCM–behavior analysis")
    print("=" * 100)
    print(result[display_columns].to_string(index=False))
    print("\nSaved output directory:")
    print(args.out_dir)
    print("\nPrimary focused row:")
    print(
        result.loc[result["is_focused_primary"], display_columns].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
