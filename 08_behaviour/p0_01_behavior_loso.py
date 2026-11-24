#!/usr/bin/env python3
"""P0-1: pooled-EVC rDCM pathways, behavior, and LOSO influence.

The analysis unit is the participant.  For the pooled EVC--cauPFC stage pair,
the script tests signed feedback (FB) and feedforward (FF) against Threshold
and ACC_all using Spearman correlation, participant bootstrap confidence
intervals, behavior-label permutation, and Holm correction across FB/FF
within each behavioral outcome.

The reviewer-facing addition is a complete leave-one-subject-out (LOSO) table:
one row for every omitted participant, outcome, and pathway.  LOSO is an
influence/sensitivity analysis, not predictive cross-validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import stats


METRICS = ("signed_FB", "signed_FF")
OUTCOMES = ("Threshold", "ACC_all")
COLORS = {"signed_FB": "#D55E00", "signed_FF": "#0072B2"}
LABELS = {
    "signed_FB": "Feedback  cauPFC-deep → EVC-superficial/deep",
    "signed_FF": "Feedforward  EVC-superficial → cauPFC-middle/deep",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "pathways_csv",
        type=Path,
        help="evc_pooled_subject_stage_pair_pathways.csv from final pooled-EVC rDCM",
    )
    parser.add_argument("--behavior-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expected-n", type=int, default=15)
    parser.add_argument("--n-permutations", type=int, default=100_000)
    parser.add_argument("--n-bootstrap", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260818)
    return parser.parse_args()


def normalize_subject(value: object) -> str:
    match = re.search(r"(\d+)", str(value))
    if not match:
        raise ValueError(f"Cannot normalize participant identifier: {value!r}")
    return f"sub-{int(match.group(1)):02d}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(base: int, *parts: object) -> int:
    payload = "|".join(map(str, (base, *parts))).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32 - 1)


def holm_adjust(values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(values, dtype=float)
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = np.maximum.accumulate((len(ranked) - np.arange(len(ranked))) * ranked)
    output = np.empty_like(adjusted)
    output[order] = np.clip(adjusted, 0.0, 1.0)
    return output


def safe_rho(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y) or len(x) < 3:
        raise ValueError("Spearman correlation requires matched vectors with n >= 3")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Spearman inputs contain non-finite values")
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(stats.spearmanr(x, y).statistic)


def bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float, int]:
    draws = np.full(n_bootstrap, np.nan, dtype=float)
    n = len(x)
    for index in range(n_bootstrap):
        selected = rng.integers(0, n, n)
        draws[index] = safe_rho(x[selected], y[selected])
    valid = draws[np.isfinite(draws)]
    if len(valid) < 0.9 * n_bootstrap:
        raise RuntimeError(f"Only {len(valid)}/{n_bootstrap} valid bootstrap draws")
    low, high = np.quantile(valid, [0.025, 0.975])
    return float(low), float(high), int(len(valid))


def permutation_p(
    x: np.ndarray,
    y: np.ndarray,
    observed: float,
    n_permutations: int,
    rng: np.random.Generator,
) -> float:
    exceed = 0
    for _ in range(n_permutations):
        permuted = safe_rho(x, rng.permutation(y))
        exceed += abs(permuted) >= abs(observed) - 1e-15
    return float((exceed + 1) / (n_permutations + 1))


def find_subject_column(frame: pd.DataFrame) -> str:
    # ``Unnamed: 0`` is accepted last because the current behavior CSV stores
    # the real identifiers (sub-01 ... sub-14, sub-16) in a saved index column.
    # Subsequent normalization, uniqueness, expected-n, and one-to-one merge
    # checks prevent an ordinary 0..N row index from being used silently.
    for name in (
        "subject", "participant", "sub", "subj", "ID", "id", "Unnamed: 0"
    ):
        if name in frame.columns:
            return name
    raise KeyError(f"No participant column found. Columns: {list(frame.columns)}")


def prepare(
    pathways_csv: Path,
    behavior_csv: Path,
    expected_n: int,
) -> pd.DataFrame:
    pathways = pd.read_csv(pathways_csv)
    required = {"subject", "lower_stage", "higher_stage", *METRICS}
    missing = required - set(pathways.columns)
    if missing:
        raise KeyError(f"Pathway CSV missing columns: {sorted(missing)}")
    pathways = pathways.copy()
    pathways["subject"] = pathways["subject"].map(normalize_subject)
    selected = pathways.loc[
        pathways["lower_stage"].eq("EVC")
        & pathways["higher_stage"].eq("cauPFC"),
        ["subject", *METRICS],
    ].copy()
    if selected["subject"].duplicated(keep=False).any():
        duplicates = selected.loc[
            selected["subject"].duplicated(keep=False), "subject"
        ].tolist()
        raise RuntimeError(f"Expected one EVC--cauPFC row per participant: {duplicates}")

    behavior = pd.read_csv(behavior_csv)
    subject_column = find_subject_column(behavior)
    behavior = behavior.copy()
    behavior["subject"] = behavior[subject_column].map(normalize_subject)
    for outcome in OUTCOMES:
        if outcome not in behavior.columns:
            raise KeyError(f"Behavior CSV missing {outcome!r}")
        behavior[outcome] = pd.to_numeric(behavior[outcome], errors="coerce")
    behavior = behavior[["subject", *OUTCOMES]]
    if behavior["subject"].duplicated(keep=False).any():
        duplicates = sorted(
            behavior.loc[behavior["subject"].duplicated(keep=False), "subject"].unique()
        )
        raise RuntimeError(
            "Behavior table has duplicate participant rows; resolve rather than "
            f"silently dropping them: {duplicates}"
        )

    merged = selected.merge(behavior, on="subject", how="inner", validate="one_to_one")
    missing_rows = merged[[*METRICS, *OUTCOMES]].isna().any(axis=1)
    if missing_rows.any():
        bad = merged.loc[missing_rows, "subject"].tolist()
        raise RuntimeError(f"Participants have incomplete pathway/behavior data: {bad}")
    if len(merged) != expected_n:
        raise RuntimeError(f"Expected {expected_n} complete participants, found {len(merged)}")
    if selected["subject"].nunique() != expected_n:
        raise RuntimeError(
            "Pathway input does not contain the expected final participant set "
            f"(found {selected['subject'].nunique()})"
        )
    return merged.sort_values("subject").reset_index(drop=True)


def run_tests(
    data: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_rows: list[dict[str, object]] = []
    loso_rows: list[dict[str, object]] = []
    for outcome in OUTCOMES:
        for metric in METRICS:
            x = data[metric].to_numpy(float)
            y = data[outcome].to_numpy(float)
            observed = safe_rho(x, y)
            ci_rng = np.random.default_rng(stable_seed(args.seed, outcome, metric, "ci"))
            perm_rng = np.random.default_rng(stable_seed(args.seed, outcome, metric, "perm"))
            low, high, n_valid = bootstrap_ci(x, y, args.n_bootstrap, ci_rng)
            p_perm = permutation_p(x, y, observed, args.n_permutations, perm_rng)

            for omitted_index, omitted_subject in enumerate(data["subject"]):
                retained = np.arange(len(data)) != omitted_index
                loso_rho = safe_rho(x[retained], y[retained])
                loso_rows.append(
                    {
                        "outcome": outcome,
                        "metric": metric,
                        "omitted_subject": omitted_subject,
                        "n_retained": int(retained.sum()),
                        "full_sample_rho": observed,
                        "loso_rho": loso_rho,
                        "delta_rho_loso_minus_full": loso_rho - observed,
                        "same_sign_as_full": bool(np.sign(loso_rho) == np.sign(observed)),
                    }
                )

            current_loso = [
                row for row in loso_rows
                if row["outcome"] == outcome and row["metric"] == metric
            ]
            loso_values = np.asarray([row["loso_rho"] for row in current_loso], float)
            changes = np.abs(loso_values - observed)
            most_influential = current_loso[int(np.argmax(changes))]["omitted_subject"]
            result_rows.append(
                {
                    "analysis_role": "final_pooled_evc_behavior_with_loso",
                    "outcome": outcome,
                    "metric": metric,
                    "n_subjects": len(x),
                    "spearman_rho": observed,
                    "bootstrap_ci95_low": low,
                    "bootstrap_ci95_high": high,
                    "permutation_p_two_sided": p_perm,
                    "loso_rho_min": float(np.min(loso_values)),
                    "loso_rho_max": float(np.max(loso_values)),
                    "loso_same_sign_fraction": float(
                        np.mean(np.sign(loso_values) == np.sign(observed))
                    ),
                    "loso_max_abs_delta_rho": float(np.max(changes)),
                    "loso_most_influential_subject": most_influential,
                    "n_bootstrap": args.n_bootstrap,
                    "n_valid_bootstrap": n_valid,
                    "n_permutations": args.n_permutations,
                }
            )

    results = pd.DataFrame(result_rows)
    results["permutation_p_holm_within_2_pathway_outcome"] = np.nan
    for outcome in OUTCOMES:
        indices = results.index[results["outcome"].eq(outcome)]
        results.loc[indices, "permutation_p_holm_within_2_pathway_outcome"] = (
            holm_adjust(results.loc[indices, "permutation_p_two_sided"].to_numpy(float))
        )
    return results, pd.DataFrame(loso_rows)


def set_style() -> None:
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


def format_p(value: float) -> str:
    """Format an exact P value without confusing it with Spearman rho."""
    if not np.isfinite(value):
        return "NA"
    if value < 0.001:
        return "< 0.001"
    return f"= {value:.3f}"


def plot_loso(results: pd.DataFrame, loso: pd.DataFrame, out_dir: Path) -> None:
    set_style()
    figure = plt.figure(figsize=(11.4, 4.35))
    grid = figure.add_gridspec(
        1,
        4,
        width_ratios=[1.0, 0.52, 1.0, 0.52],
        wspace=0.16,
    )
    axes = [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 2])]
    axes[1].sharey(axes[0])
    info_axes = [figure.add_subplot(grid[0, 1]), figure.add_subplot(grid[0, 3])]
    y_base = {"signed_FB": 1.0, "signed_FF": 0.0}
    titles = {
        "Threshold": "a   Perceptual threshold  (primary)",
        "ACC_all": "b   Accuracy  (secondary)",
    }
    for axis, info_axis, outcome in zip(axes, info_axes, OUTCOMES):
        axis.axvline(0, color="#6F747A", lw=1.0, ls=(0, (4, 3)), zorder=0)
        for metric in METRICS:
            row = results.loc[
                results["outcome"].eq(outcome) & results["metric"].eq(metric)
            ].iloc[0]
            subset = loso.loc[
                loso["outcome"].eq(outcome) & loso["metric"].eq(metric)
            ].sort_values("loso_rho")
            y = y_base[metric]
            loso_low = float(subset["loso_rho"].min())
            loso_high = float(subset["loso_rho"].max())
            full = float(row["spearman_rho"])
            ci_low = float(row["bootstrap_ci95_low"])
            ci_high = float(row["bootstrap_ci95_high"])
            p_holm = float(row["permutation_p_holm_within_2_pathway_outcome"])

            # Pale, thick interval: complete LOSO range.  This is deliberately
            # visually secondary to the full-sample bootstrap interval.
            axis.plot(
                [loso_low, loso_high], [y, y], color=COLORS[metric],
                lw=11, alpha=0.16, solid_capstyle="round", zorder=1,
            )

            # Small rug marks preserve all 15 leave-one-subject-out estimates
            # without turning them into a second scatterplot.
            jitter = np.linspace(-0.075, 0.075, len(subset))
            axis.scatter(
                subset["loso_rho"],
                y + jitter,
                s=28,
                marker="|",
                color=COLORS[metric],
                linewidths=1.1,
                alpha=0.60,
                zorder=2,
            )

            # Full n=15 estimate and participant-bootstrap 95% CI.
            axis.errorbar(
                full,
                y,
                xerr=np.asarray([[full - ci_low], [ci_high - full]]),
                fmt="D" if metric == "signed_FB" else "o",
                ms=7.3,
                color=COLORS[metric],
                mfc=COLORS[metric],
                mec="#252A30",
                mew=0.8,
                ecolor=COLORS[metric],
                elinewidth=2.0,
                capsize=4,
                capthick=1.5,
                zorder=4,
            )

            same_sign_n = int(round(float(row["loso_same_sign_fraction"]) * len(subset)))
            info_axis.text(
                0.02,
                y,
                f"ρ = {full:+.2f}\n"
                f"95% CI [{ci_low:+.2f}, {ci_high:+.2f}]\n"
                f"Permutation $P_{{\\mathrm{{Holm}}}}$ {format_p(p_holm)}\n"
                f"LOSO [{loso_low:+.2f}, {loso_high:+.2f}]\n"
                f"same direction {same_sign_n}/{len(subset)}",
                ha="left",
                va="center",
                fontsize=8.0,
                color="#34383D",
                linespacing=1.25,
            )
        axis.set_title(titles[outcome], loc="left", weight="bold", fontsize=11)
        axis.set_xlim(-1.02, 1.02)
        axis.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        axis.set_xlabel("Spearman ρ")
        axis.set_ylim(-0.45, 1.45)
        axis.grid(axis="x", color="#E2E5E9", lw=0.7, zorder=0)
        axis.tick_params(axis="y", length=0)
        axis.spines["left"].set_visible(False)
        info_axis.set_xlim(0.0, 1.0)
        info_axis.set_ylim(-0.45, 1.45)
        info_axis.axis("off")
    axes[0].set_yticks(
        [1, 0],
        [
            "Feedback (FB)\ncauPFC-deep → EVC-superficial/deep",
            "Feedforward (FF)\nEVC-superficial → cauPFC-middle/deep",
        ],
    )
    axes[1].tick_params(axis="y", labelleft=False)
    figure.suptitle(
        "Participant influence on pooled-EVC pathway–behavior associations",
        x=0.075, y=0.975, ha="left", weight="bold", fontsize=13.0,
    )
    figure.text(
        0.075,
        0.905,
        "Full-sample Spearman correlations with participant-bootstrap 95% CIs and Holm-adjusted two-sided permutation P values; pale bands show the complete LOSO range.",
        fontsize=8.4,
        color="#666A70",
    )
    legend_handles = [
        Line2D(
            [0], [0], color="#34383D", marker="D", markersize=6,
            markerfacecolor="#34383D", lw=1.8, solid_capstyle="projecting",
            label="Full n=15 estimate (bootstrap 95% CI)",
        ),
        Line2D(
            [0], [0], color="#7A8086", marker="|", markersize=9,
            lw=8, alpha=0.24, label="15 leave-one-subject-out estimates and range (n=14)",
        ),
    ]
    figure.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(0.075, 0.075),
        frameon=False,
        ncol=2,
        fontsize=7.7,
        handlelength=2.5,
        columnspacing=1.8,
    )
    figure.text(
        0.075,
        0.025,
        "Holm correction is applied across FB and FF within each outcome. LOSO is descriptive; inference uses the complete n=15 sample.",
        fontsize=7.3,
        color="#74797F",
    )
    figure.subplots_adjust(left=0.255, right=0.985, bottom=0.25, top=0.79)
    for suffix in ("png", "pdf", "svg"):
        kwargs = {"dpi": 600} if suffix == "png" else {}
        figure.savefig(out_dir / f"pooled_evc_behavior_loso.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.n_bootstrap < 10_000:
        raise ValueError("Use at least 10,000 bootstrap resamples for final output")
    if args.n_permutations < 10_000:
        raise ValueError("Use at least 10,000 permutations for final output")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = prepare(args.pathways_csv, args.behavior_csv, args.expected_n)
    results, loso = run_tests(data, args)

    data.to_csv(args.out_dir / "pooled_evc_behavior_subjects.csv", index=False)
    results.to_csv(args.out_dir / "pooled_evc_behavior_statistics.csv", index=False)
    loso.to_csv(args.out_dir / "pooled_evc_behavior_loso_by_subject.csv", index=False)
    plot_loso(results, loso, args.out_dir)

    manifest = {
        "analysis": "P0-1 pooled-EVC behavior LOSO",
        "pathways_csv": str(args.pathways_csv.resolve()),
        "pathways_sha256": sha256_file(args.pathways_csv),
        "behavior_csv": str(args.behavior_csv.resolve()),
        "behavior_sha256": sha256_file(args.behavior_csv),
        "n_subjects": len(data),
        "subjects": data["subject"].tolist(),
        "metrics": list(METRICS),
        "outcomes": list(OUTCOMES),
        "multiplicity": "Holm across signed_FB and signed_FF within each outcome",
        "loso_interpretation": "participant influence analysis; not predictive CV",
        "n_bootstrap": args.n_bootstrap,
        "n_permutations": args.n_permutations,
        "seed": args.seed,
    }
    (args.out_dir / "pooled_evc_behavior_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(results.to_string(index=False))
    print(f"\nSaved final LOSO outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
