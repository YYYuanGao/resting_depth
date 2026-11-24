#!/usr/bin/env python3
"""P0-3: leave-one-stage-pair-out sensitivity for the G1 rDCM bridge.

The input is ``evc_pooled_subject_directed_pathways.csv`` from the final
pooled-EVC G1 alignment pipeline.  For each participant the script recomputes
Spearman correlation between directed G1 displacement and signed pathway
strength for:

1. all three stage pairs (six directed pathways), and
2. three variants that each omit one unordered stage pair (four pathways).

The critical reviewer-facing variant omits EVC--cauPFC.  Because each ablated
participant correlation contains only four observations, all ablations are
explicitly labelled low-resolution sensitivity analyses, not replacements for
the prespecified six-pathway primary test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from itertools import product
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


STAGE_ORDER = ("EVC", "PPC", "cauPFC")
PAIR_ORDER = ("EVC--PPC", "EVC--cauPFC", "PPC--cauPFC")
VARIANTS = (
    ("full_all_pairs", None),
    ("drop_EVC--PPC", "EVC--PPC"),
    ("drop_EVC--cauPFC", "EVC--cauPFC"),
    ("drop_PPC--cauPFC", "PPC--cauPFC"),
)
DISPLAY = {
    "full_all_pairs": "All three stage pairs (primary reference)",
    "drop_EVC--PPC": "Omit EVC–PPC",
    "drop_EVC--cauPFC": "Omit EVC–cauPFC (targeted)",
    "drop_PPC--cauPFC": "Omit PPC–cauPFC",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directed_pathways_csv", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expected-n", type=int, default=15)
    parser.add_argument("--n-bootstrap", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument(
        "--reference-primary-csv",
        type=Path,
        default=None,
        help="Optional evc_pooled_primary_group_test.csv for exact full-model audit",
    )
    return parser.parse_args()


def normalize_subject(value: object) -> str:
    match = re.search(r"(\d+)", str(value))
    if not match:
        raise ValueError(f"Cannot normalize participant: {value!r}")
    return f"sub-{int(match.group(1)):02d}"


def stable_seed(base: int, *parts: object) -> int:
    value = "|".join(map(str, (base, *parts))).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little") % (2**32 - 1)


def holm_adjust(values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(values, float)
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = np.maximum.accumulate((len(ranked) - np.arange(len(ranked))) * ranked)
    output = np.empty_like(adjusted)
    output[order] = np.clip(adjusted, 0, 1)
    return output


def exact_signflip_p(values: np.ndarray) -> float:
    values = np.asarray(values, float)
    if not np.isfinite(values).all() or len(values) == 0:
        raise ValueError("Sign flip requires finite non-empty values")
    observed = abs(float(values.mean()))
    signs = np.asarray(list(product((-1.0, 1.0), repeat=len(values))), float)
    null = np.abs((signs * values).mean(axis=1))
    return float(np.mean(null >= observed - 1e-15))


def bootstrap_mean_ci(
    values: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    indices = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    draws = values[indices].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 4 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(stats.spearmanr(x, y).statistic)


def pair_key(source: str, target: str) -> str:
    unknown = {source, target} - set(STAGE_ORDER)
    if unknown:
        raise ValueError(f"Unknown stages: {sorted(unknown)}")
    ordered = sorted((source, target), key=STAGE_ORDER.index)
    return "--".join(ordered)


def load_and_validate(path: Path, expected_n: int) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = {
        "subject", "pathway", "source_stage", "target_stage",
        "directed_G1_displacement", "signed_pathway_strength",
    }
    missing = required - set(data.columns)
    if missing:
        raise KeyError(f"Directed-pathway CSV missing columns: {sorted(missing)}")
    data = data.copy()
    data["subject"] = data["subject"].map(normalize_subject)
    data["pathway"] = data["pathway"].astype(str)
    if set(data["pathway"]) != {"FF", "FB"}:
        raise RuntimeError(f"Expected FF and FB pathways; found {sorted(data['pathway'].unique())}")
    data["pair_key"] = [
        pair_key(source, target)
        for source, target in zip(data["source_stage"], data["target_stage"])
    ]
    for column in ("directed_G1_displacement", "signed_pathway_strength"):
        data[column] = pd.to_numeric(data[column], errors="raise")
    if not np.isfinite(
        data[["directed_G1_displacement", "signed_pathway_strength"]].to_numpy(float)
    ).all():
        raise RuntimeError("Directed-pathway input contains non-finite values")
    if data["subject"].nunique() != expected_n:
        raise RuntimeError(f"Expected {expected_n} participants, found {data['subject'].nunique()}")
    counts = data.groupby("subject").size()
    if not counts.eq(6).all():
        raise RuntimeError(f"Expected six directed pathways per participant:\n{counts}")
    pair_counts = data.groupby(["subject", "pair_key"]).size()
    if set(data["pair_key"]) != set(PAIR_ORDER) or not pair_counts.eq(2).all():
        raise RuntimeError("Expected exactly FF and FB for each of the three unordered stage pairs")
    if data.duplicated(["subject", "pair_key", "pathway"]).any():
        raise RuntimeError("Duplicate subject/pair/pathway rows")
    return data.sort_values(["subject", "pair_key", "pathway"]).reset_index(drop=True)


def participant_statistics(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for subject, participant in data.groupby("subject", sort=True):
        for variant, omitted_pair in VARIANTS:
            selected = participant if omitted_pair is None else participant.loc[
                ~participant["pair_key"].eq(omitted_pair)
            ]
            rho = safe_spearman(
                selected["directed_G1_displacement"].to_numpy(float),
                selected["signed_pathway_strength"].to_numpy(float),
            )
            if not np.isfinite(rho):
                raise RuntimeError(f"Undefined correlation: {subject} {variant}")
            rows.append(
                {
                    "subject": subject,
                    "variant": variant,
                    "omitted_pair": omitted_pair or "none",
                    "n_directed_pathways": len(selected),
                    "within_subject_spearman_rho": rho,
                    "analysis_role": (
                        "primary_reference" if omitted_pair is None
                        else "low_resolution_stage_pair_ablation"
                    ),
                }
            )
    return pd.DataFrame(rows)


def group_statistics(participant: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variant, _ in VARIANTS:
        values = participant.loc[
            participant["variant"].eq(variant), "within_subject_spearman_rho"
        ].to_numpy(float)
        rng = np.random.default_rng(stable_seed(args.seed, variant, "bootstrap"))
        low, high = bootstrap_mean_ci(values, args.n_bootstrap, rng)
        rows.append(
            {
                "variant": variant,
                "display_label": DISPLAY[variant],
                "analysis_role": (
                    "primary_reference" if variant == "full_all_pairs"
                    else "low_resolution_stage_pair_ablation"
                ),
                "n_subjects": len(values),
                "n_pathways_per_subject": int(
                    participant.loc[participant["variant"].eq(variant), "n_directed_pathways"].iloc[0]
                ),
                "mean_within_subject_spearman_rho": float(values.mean()),
                "sd_within_subject_spearman_rho": float(values.std(ddof=1)),
                "median_within_subject_spearman_rho": float(np.median(values)),
                "bootstrap_ci95_low": low,
                "bootstrap_ci95_high": high,
                "exact_signflip_p_two_sided": exact_signflip_p(values),
                "n_negative": int((values < 0).sum()),
                "n_positive": int((values > 0).sum()),
                "n_zero": int((values == 0).sum()),
                "n_bootstrap": args.n_bootstrap,
            }
        )
    output = pd.DataFrame(rows)
    output["exact_signflip_p_holm_across_3_ablations"] = np.nan
    ablation = output["analysis_role"].eq("low_resolution_stage_pair_ablation")
    output.loc[ablation, "exact_signflip_p_holm_across_3_ablations"] = holm_adjust(
        output.loc[ablation, "exact_signflip_p_two_sided"].to_numpy(float)
    )
    return output


def compare_ablations_to_full(
    participant: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    wide = participant.pivot(
        index="subject", columns="variant", values="within_subject_spearman_rho"
    )
    full = wide["full_all_pairs"].to_numpy(float)
    rows: list[dict[str, object]] = []
    for variant, omitted_pair in VARIANTS[1:]:
        ablated = wide[variant].to_numpy(float)
        delta = ablated - full
        rng = np.random.default_rng(stable_seed(args.seed, variant, "delta"))
        low, high = bootstrap_mean_ci(delta, args.n_bootstrap, rng)
        rows.append(
            {
                "variant": variant,
                "omitted_pair": omitted_pair,
                "n_subjects": len(delta),
                "mean_delta_rho_ablation_minus_full": float(delta.mean()),
                "bootstrap_delta_ci95_low": low,
                "bootstrap_delta_ci95_high": high,
                "paired_exact_signflip_p_two_sided": exact_signflip_p(delta),
                "same_sign_fraction_as_full": float(
                    np.mean(np.sign(ablated) == np.sign(full))
                ),
                "spearman_across_subjects_vs_full": safe_spearman(ablated, full),
            }
        )
    output = pd.DataFrame(rows)
    output["paired_p_holm_across_3_ablation_changes"] = holm_adjust(
        output["paired_exact_signflip_p_two_sided"].to_numpy(float)
    )
    return output


def audit_reference(group: pd.DataFrame, path: Path | None) -> dict[str, object]:
    if path is None:
        return {"reference_primary_csv": None, "full_model_match_checked": False}
    reference = pd.read_csv(path)
    column = "mean_within_subject_spearman_rho"
    if column not in reference.columns or len(reference) != 1:
        raise RuntimeError(f"Reference primary CSV must have one row and {column!r}")
    expected = float(reference.iloc[0][column])
    observed = float(
        group.loc[group["variant"].eq("full_all_pairs"), column].iloc[0]
    )
    if not np.isclose(expected, observed, atol=1e-12, rtol=0):
        raise RuntimeError(
            f"Full-model mismatch versus reference: expected={expected}, observed={observed}"
        )
    return {
        "reference_primary_csv": str(path.resolve()),
        "full_model_match_checked": True,
        "reference_mean_rho": expected,
        "recomputed_mean_rho": observed,
    }


def format_p(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    if value < 0.001:
        return "< 0.001"
    return f"= {value:.3f}"


def variant_style(variant: str) -> dict[str, object]:
    if variant == "full_all_pairs":
        return {
            "color": "#6F55A6", "marker": "D", "mfc": "#6F55A6",
            "ms": 7.2, "mew": 0.9,
        }
    if variant == "drop_EVC--cauPFC":
        return {
            "color": "#D55E00", "marker": "s", "mfc": "#D55E00",
            "ms": 7.0, "mew": 0.9,
        }
    return {
        "color": "#496A79", "marker": "o", "mfc": "white",
        "ms": 7.0, "mew": 1.4,
    }


def plot_forest(group: pd.DataFrame, changes: pd.DataFrame, out_dir: Path) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    ordered = group.set_index("variant").loc[[variant for variant, _ in VARIANTS]]
    change_variants = [variant for variant, _ in VARIANTS[1:]]
    ordered_changes = changes.set_index("variant").loc[change_variants]

    figure = plt.figure(figsize=(12.8, 5.0))
    grid = figure.add_gridspec(
        1, 4, width_ratios=[1.0, 0.57, 1.0, 0.57], wspace=0.18,
    )
    absolute_axis = figure.add_subplot(grid[0, 0])
    absolute_info = figure.add_subplot(grid[0, 1])
    delta_axis = figure.add_subplot(grid[0, 2])
    delta_info = figure.add_subplot(grid[0, 3])

    absolute_y = np.arange(len(ordered))[::-1]
    for position, (variant, row) in zip(absolute_y, ordered.iterrows()):
        mean = float(row["mean_within_subject_spearman_rho"])
        low = float(row["bootstrap_ci95_low"])
        high = float(row["bootstrap_ci95_high"])
        style = variant_style(variant)
        absolute_axis.errorbar(
            mean, position, xerr=np.asarray([[mean - low], [high - mean]]),
            fmt=str(style["marker"]), ms=float(style["ms"]),
            color=str(style["color"]), mfc=str(style["mfc"]),
            mec="#252A30" if variant in {"full_all_pairs", "drop_EVC--cauPFC"}
            else str(style["color"]),
            mew=float(style["mew"]), capsize=4, lw=1.8,
        )
        if variant == "full_all_pairs":
            p_value = float(row["exact_signflip_p_two_sided"])
            p_line = f"Exact sign-flip $P$ {format_p(p_value)}"
        else:
            p_value = float(row["exact_signflip_p_holm_across_3_ablations"])
            p_line = f"Sign-flip $P_{{\\mathrm{{Holm}}}}$ {format_p(p_value)}"
        absolute_info.text(
            0.02, position,
            f"ρ = {mean:+.2f}\n"
            f"95% CI [{low:+.2f}, {high:+.2f}]\n"
            f"{p_line}\n"
            f"negative {int(row['n_negative'])}/{int(row['n_subjects'])}",
            ha="left", va="center", fontsize=7.8, color="#34383D",
            linespacing=1.24,
        )

    absolute_axis.axvline(0, color="#858A90", lw=1.0, ls=(0, (4, 3)))
    absolute_axis.set_xlim(-1.05, 1.05)
    absolute_axis.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    absolute_axis.set_ylim(-0.5, 3.5)
    absolute_axis.set_yticks(
        absolute_y,
        [
            "Full model (primary)\n3 pairs · 6 pathways",
            "Omit EVC–PPC\n2 pairs · 4 pathways",
            "Omit EVC–cauPFC (targeted)\n2 pairs · 4 pathways",
            "Omit PPC–cauPFC\n2 pairs · 4 pathways",
        ],
    )
    absolute_axis.set_xlabel("Mean within-participant Spearman correlation (ρ)")
    absolute_axis.set_title("a   Absolute alignment", loc="left", weight="bold", fontsize=11)
    absolute_axis.grid(axis="x", color="#E4E7EA", lw=0.7)
    absolute_axis.tick_params(axis="y", length=0)
    absolute_axis.spines["left"].set_visible(False)
    absolute_info.set_xlim(0, 1)
    absolute_info.set_ylim(-0.5, 3.5)
    absolute_info.axis("off")

    delta_y = np.arange(len(ordered_changes))[::-1]
    largest_interval = 0.0
    for position, (variant, row) in zip(delta_y, ordered_changes.iterrows()):
        mean = float(row["mean_delta_rho_ablation_minus_full"])
        low = float(row["bootstrap_delta_ci95_low"])
        high = float(row["bootstrap_delta_ci95_high"])
        largest_interval = max(largest_interval, abs(low), abs(high))
        style = variant_style(variant)
        delta_axis.errorbar(
            mean, position, xerr=np.asarray([[mean - low], [high - mean]]),
            fmt=str(style["marker"]), ms=float(style["ms"]),
            color=str(style["color"]), mfc=str(style["mfc"]),
            mec="#252A30" if variant == "drop_EVC--cauPFC" else str(style["color"]),
            mew=float(style["mew"]), capsize=4, lw=1.8,
        )
        p_value = float(row["paired_p_holm_across_3_ablation_changes"])
        same_sign_n = int(round(float(row["same_sign_fraction_as_full"]) * int(row["n_subjects"])))
        delta_info.text(
            0.02, position,
            f"Δρ = {mean:+.2f}\n"
            f"95% CI [{low:+.2f}, {high:+.2f}]\n"
            f"Paired $P_{{\\mathrm{{Holm}}}}$ {format_p(p_value)}\n"
            f"same direction {same_sign_n}/{int(row['n_subjects'])}",
            ha="left", va="center", fontsize=7.8, color="#34383D",
            linespacing=1.24,
        )

    delta_limit = max(0.5, np.ceil(largest_interval * 1.20 * 4) / 4)
    delta_limit = min(2.05, float(delta_limit))
    delta_axis.axvline(0, color="#858A90", lw=1.0, ls=(0, (4, 3)))
    delta_axis.set_xlim(-delta_limit, delta_limit)
    delta_axis.xaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
    delta_axis.set_ylim(-0.5, 2.5)
    delta_axis.set_yticks(
        delta_y,
        ["EVC–PPC", "EVC–cauPFC", "PPC–cauPFC"],
    )
    delta_axis.set_xlabel("Change in correlation: Δρ = ablated − full model")
    delta_axis.set_title("b   Paired change from full model", loc="left", weight="bold", fontsize=11)
    delta_axis.grid(axis="x", color="#E4E7EA", lw=0.7)
    delta_axis.tick_params(axis="y", length=0)
    delta_axis.spines["left"].set_visible(False)
    delta_axis.text(
        0.01, 0.985, "← more negative", transform=delta_axis.transAxes,
        ha="left", va="top", fontsize=7.3, color="#6B7076",
    )
    delta_axis.text(
        0.99, 0.985, "attenuation toward zero →", transform=delta_axis.transAxes,
        ha="right", va="top", fontsize=7.3, color="#6B7076",
    )
    delta_info.set_xlim(0, 1)
    delta_info.set_ylim(-0.5, 2.5)
    delta_info.axis("off")

    figure.suptitle(
        "Stage-pair ablation of G1-directed rDCM alignment",
        x=0.055, y=0.975, ha="left", weight="bold", fontsize=13.0,
    )
    figure.text(
        0.055, 0.915,
        "Participant-level Spearman correlations; points are group means and intervals are participant-bootstrap 95% CIs (n=15). Orange squares mark the reviewer-targeted EVC–cauPFC omission.",
        fontsize=8.4, color="#666A70",
    )
    figure.text(
        0.055, 0.035,
        "Full model: 6 directed pathways. Each ablation: 4 pathways (low-resolution sensitivity analysis). ",
        fontsize=7.3, color="#666A70",
    )
    figure.text(
        0.055, 0.012,
        "Panel a: full-model P is unadjusted; ablation P values are Holm-adjusted across 3 variants. Panel b: paired P values are Holm-adjusted across 3 changes.",
        fontsize=7.3, color="#666A70",
    )
    figure.subplots_adjust(left=0.255, right=0.99, bottom=0.20, top=0.82)
    for suffix in ("png", "pdf", "svg"):
        kwargs = {"dpi": 600} if suffix == "png" else {}
        figure.savefig(out_dir / f"multistage_pair_ablation_forest.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.n_bootstrap < 10_000:
        raise ValueError("Use at least 10,000 bootstrap resamples")
    data = load_and_validate(args.directed_pathways_csv, args.expected_n)
    participant = participant_statistics(data)
    group = group_statistics(participant, args)
    changes = compare_ablations_to_full(participant, args)
    reference_audit = audit_reference(group, args.reference_primary_csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.out_dir / "directed_pathways_validated.csv", index=False)
    participant.to_csv(args.out_dir / "multistage_pair_ablation_subject_results.csv", index=False)
    group.to_csv(args.out_dir / "multistage_pair_ablation_group_results.csv", index=False)
    changes.to_csv(args.out_dir / "multistage_pair_ablation_vs_full_changes.csv", index=False)
    plot_forest(group, changes, args.out_dir)
    manifest = {
        "analysis": "P0-3 leave-one-stage-pair-out G1 bridge sensitivity",
        "input_csv": str(args.directed_pathways_csv.resolve()),
        "n_subjects": args.expected_n,
        "full_model": "six directed pathways: three stage pairs x FF/FB",
        "ablations": [variant for variant, _ in VARIANTS[1:]],
        "critical_ablation": "drop_EVC--cauPFC",
        "inference": "participant bootstrap CI and exact participant sign flip",
        "multiplicity": "Holm across three ablations; primary reference unchanged",
        "caveat": "each ablation uses only four pathways per participant",
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        **reference_audit,
    }
    (args.out_dir / "multistage_pair_ablation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(group.to_string(index=False))
    print(f"\nSaved stage-pair ablation outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
