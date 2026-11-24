#!/usr/bin/env python3
"""G1-directed rDCM bridge for EVC(V1/V2/V3), PPC and cauPFC.

This is a node-definition sensitivity analysis aligned with the endpoint
rDCM.  It uses newly inverted nine-node rDCMs; it never averages parameters
from the earlier four-stage model.

Definitions
-----------
Stages:
    EVC = equal ROI mean of V1, V2 and V3
    PPC = the retained frozen PPC members from the exporter
    cauPFC = the seven frozen cauPFC members

Laminar pathways:
    FF = lower-stage superficial -> higher-stage deep/middle
    FB = higher-stage deep -> lower-stage deep/superficial

Primary statistic:
    Within each participant, Spearman correlation between directed G1
    displacement and signed pathway strength across six directed pathways
    (three stage pairs x FF/FB). Group inference uses an exact two-sided
    sign-flip test across the 15 participant correlations.

Secondary decomposition:
    Within each participant, Spearman correlations of signed FF, signed FB,
    |FF| and |FB| with |delta G1| across the three unordered stage pairs.
    These tests have only three observations per participant and are therefore
    explicitly labelled low-resolution sensitivity analyses.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from itertools import combinations
from pathlib import Path

import matplotlib

import os
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "/path/to/project")  # see config/paths.sh

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


DERIV = Path(PROJECT_ROOT + "/derivatives_0331")
BASE = DERIV / "S6_validation" / "pial5_multistage_bridge_evc_pooled"
GRADIENT_BASE = (
    DERIV
    / "S6_validation"
    / "pial5_only_update"
    / "functional_gradient_bianca_reference"
)
DEFAULT_EDGES = BASE / "group" / "pial5_evc_pooled_multistage_A_edges_long.csv"
DEFAULT_EXPORT_AUDIT = (
    BASE / "timeseries" / "evc_pooled_multistage_export_audit.csv"
)
DEFAULT_MEMBERSHIP = (
    BASE / "timeseries" / "evc_pooled_multistage_membership_qc.csv"
)
DEFAULT_GRADIENT = GRADIENT_BASE / "reference_gradient_scores.csv"
DEFAULT_OUT = BASE / "final_G1_evc_pooled_directed_alignment"
PROTECTED_OUTPUT_ROOTS = (
    DERIV / "S6_validation" / "pial5_multistage_bridge",
    DERIV
    / "S6_validation"
    / "pial_depth_exclusion"
    / "matched_current_rdcm",
    GRADIENT_BASE,
)

STAGES = ("EVC", "PPC", "cauPFC")
DISPLAY_STAGE = {
    "EVC": "Visual cortex (V1–V3)",
    "PPC": "PPC",
    "cauPFC": "cauPFC",
}
LAYERS = ("deep", "mid", "sup")
SECONDARY_METRICS = ("signed_FF", "signed_FB", "absolute_FF", "absolute_FB")
ROI_NAME_MAP = {"7Al": "7AL", "8AD": "8Ad"}
INK = "#252A30"
MUTED = "#6B7280"
GRID = "#E5E7EB"
FF_BLUE = "#147FAE"
FB_ORANGE = "#E66101"
PURPLE = "#7158A5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edges-csv", type=Path, default=DEFAULT_EDGES)
    parser.add_argument(
        "--export-audit-csv", type=Path, default=DEFAULT_EXPORT_AUDIT
    )
    parser.add_argument("--membership-csv", type=Path, default=DEFAULT_MEMBERSHIP)
    parser.add_argument(
        "--reference-gradient-csv", type=Path, default=DEFAULT_GRADIENT
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--expected-subjects", type=int, default=15)
    parser.add_argument("--n-boot", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260724)
    return parser.parse_args()


def normalize_subject(value: object) -> str:
    text = str(value).replace("sub-", "").strip()
    return f"sub-{int(text):02d}"


def canonical_roi(value: object) -> str:
    text = str(value).strip()
    return ROI_NAME_MAP.get(text, text)


def paths_overlap(first: Path, second: Path) -> bool:
    first_resolved = first.resolve(strict=False)
    second_resolved = second.resolve(strict=False)
    return (
        first_resolved == second_resolved
        or first_resolved in second_resolved.parents
        or second_resolved in first_resolved.parents
    )


def validate_output_isolation(out_dir: Path) -> None:
    for protected in PROTECTED_OUTPUT_ROOTS:
        if paths_overlap(out_dir, protected):
            raise RuntimeError(
                "Refusing to write into or around a protected prior-analysis "
                f"path: {out_dir} versus {protected}"
            )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def boolean_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    normalized = series.astype(str).str.strip().str.lower()
    allowed = {"true", "false", "1", "0", "yes", "no"}
    unexpected = set(normalized.unique()) - allowed
    if unexpected:
        raise RuntimeError(f"Cannot parse boolean values: {sorted(unexpected)}")
    return normalized.isin({"true", "1", "yes"})


def load_edges(path: Path, expected_subjects: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    data = pd.read_csv(path)
    required = {
        "subject",
        "source_stage",
        "target_stage",
        "source_layer",
        "target_layer",
        "A",
        "source_index",
        "target_index",
    }
    missing = required - set(data.columns)
    if missing:
        raise RuntimeError(f"{path} missing columns: {sorted(missing)}")

    data = data.copy()
    data["subject"] = data["subject"].map(normalize_subject)
    for column in (
        "source_stage",
        "target_stage",
        "source_layer",
        "target_layer",
    ):
        data[column] = data[column].astype(str)
    data["A"] = pd.to_numeric(data["A"], errors="raise")
    if not np.isfinite(data["A"].to_numpy(float)).all():
        raise RuntimeError("A table contains non-finite values")
    if set(data["source_stage"]) != set(STAGES):
        raise RuntimeError("Source stages do not match EVC/PPC/cauPFC")
    if set(data["target_stage"]) != set(STAGES):
        raise RuntimeError("Target stages do not match EVC/PPC/cauPFC")
    if set(data["source_layer"]) != set(LAYERS):
        raise RuntimeError("Source layers are not exactly deep/mid/sup")
    if set(data["target_layer"]) != set(LAYERS):
        raise RuntimeError("Target layers are not exactly deep/mid/sup")

    subjects = sorted(data["subject"].unique())
    if len(subjects) != expected_subjects:
        raise RuntimeError(
            f"Expected {expected_subjects} participants, found {len(subjects)}"
        )
    counts = data.groupby("subject").size()
    if not counts.eq(81).all():
        raise RuntimeError(
            "Expected one complete 9 x 9 A matrix per participant:\n"
            + counts.to_string()
        )
    if data.duplicated(["subject", "target_index", "source_index"]).any():
        raise RuntimeError("Duplicated A(target,source) rows")
    return data


def validate_lineage(
    path: Path, edge_subjects: list[str], expected_subjects: int
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    data = pd.read_csv(path)
    required = {
        "subject",
        "n_nodes",
        "source_ts_base",
        "has_nonfinite",
        "min_node_sd",
        "EVC_members",
    }
    missing = required - set(data.columns)
    if missing:
        raise RuntimeError(f"{path} missing columns: {sorted(missing)}")
    data = data.copy()
    data["subject"] = data["subject"].map(normalize_subject)
    if len(data) != expected_subjects or data["subject"].nunique() != expected_subjects:
        raise RuntimeError("Expected one export-audit row per participant")
    if sorted(data["subject"].tolist()) != sorted(edge_subjects):
        raise RuntimeError("Participant mismatch between export and A edge tables")
    if not pd.to_numeric(data["n_nodes"], errors="raise").eq(9).all():
        raise RuntimeError("Export audit does not report nine nodes for everyone")
    if boolean_series(data["has_nonfinite"]).any():
        raise RuntimeError("Export audit reports non-finite node signals")
    if not (pd.to_numeric(data["min_node_sd"], errors="raise") > 0).all():
        raise RuntimeError("Export audit reports a zero-variance node")
    if not data["source_ts_base"].astype(str).str.contains(
        "Step3_TimeSeries_tSNRclean_pial5pct", regex=False
    ).all():
        raise RuntimeError("Not all inputs trace to the outer-pial-5% time series")
    if not data["EVC_members"].astype(str).eq("V1|V2|V3").all():
        raise RuntimeError("EVC membership is not exactly V1|V2|V3")
    return data


def load_g1_stage_geometry(
    membership_path: Path, gradient_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not membership_path.exists() or not gradient_path.exists():
        raise FileNotFoundError(f"Missing {membership_path} or {gradient_path}")
    membership = pd.read_csv(membership_path)
    required_membership = {"stage", "candidate_roi", "retained"}
    missing = required_membership - set(membership.columns)
    if missing:
        raise RuntimeError(f"Membership table missing columns: {sorted(missing)}")
    membership = membership.loc[boolean_series(membership["retained"])].copy()
    membership["stage"] = membership["stage"].astype(str)
    membership["candidate_roi"] = membership["candidate_roi"].map(canonical_roi)
    if set(membership["stage"]) != set(STAGES):
        raise RuntimeError("Membership table does not contain all three stages")
    evc_members = tuple(
        membership.loc[membership["stage"] == "EVC", "candidate_roi"].tolist()
    )
    if evc_members != ("V1", "V2", "V3"):
        raise RuntimeError(
            f"EVC membership/order is {evc_members}; expected ('V1','V2','V3')"
        )

    gradient = pd.read_csv(gradient_path)
    score_column = next(
        (
            candidate
            for candidate in ("G1_oriented", "reference_G1_oriented_score")
            if candidate in gradient.columns
        ),
        None,
    )
    if "roi" not in gradient.columns or score_column is None:
        raise RuntimeError(
            "Reference gradient table must contain roi and G1_oriented"
        )
    gradient = gradient[["roi", score_column]].copy()
    gradient.columns = ["roi", "G1_oriented"]
    gradient["roi"] = gradient["roi"].map(canonical_roi)
    gradient["G1_oriented"] = pd.to_numeric(
        gradient["G1_oriented"], errors="raise"
    )
    if gradient["roi"].duplicated().any():
        raise RuntimeError("Reference gradient has duplicate canonical ROI labels")

    joined = membership.merge(
        gradient,
        left_on="candidate_roi",
        right_on="roi",
        how="left",
        validate="one_to_one",
    )
    if joined["G1_oriented"].isna().any():
        bad = joined.loc[joined["G1_oriented"].isna(), "candidate_roi"].tolist()
        raise RuntimeError(f"Retained stage ROIs missing G1 scores: {bad}")

    stage = (
        joined.groupby("stage", sort=False)["G1_oriented"]
        .agg(["mean", "std", "count"])
        .reindex(STAGES)
        .reset_index()
        .rename(
            columns={
                "mean": "G1_oriented",
                "std": "within_stage_roi_sd",
                "count": "n_rois",
            }
        )
    )
    stage["display_stage"] = stage["stage"].map(DISPLAY_STAGE)
    stage["stage_order"] = np.arange(1, len(STAGES) + 1)
    scores = stage.set_index("stage")["G1_oriented"]
    if not scores["cauPFC"] > scores["EVC"]:
        raise RuntimeError(
            "Frozen orientation check failed: cauPFC must exceed pooled EVC. "
            "The bridge will not flip G1."
        )
    return stage, joined


def one_edge(
    data: pd.DataFrame,
    source_stage: str,
    target_stage: str,
    source_layer: str,
    target_layer: str,
) -> float:
    selected = data.loc[
        (data["source_stage"] == source_stage)
        & (data["target_stage"] == target_stage)
        & (data["source_layer"] == source_layer)
        & (data["target_layer"] == target_layer),
        "A",
    ]
    if len(selected) != 1:
        raise RuntimeError(
            f"Expected one edge {source_stage}_{source_layer} -> "
            f"{target_stage}_{target_layer}; found {len(selected)}"
        )
    return float(selected.iloc[0])


def pathway_values(
    data: pd.DataFrame, lower: str, higher: str
) -> dict[str, float]:
    ff_edges = np.asarray(
        [
            one_edge(data, lower, higher, "sup", "deep"),
            one_edge(data, lower, higher, "sup", "mid"),
        ],
        dtype=float,
    )
    fb_edges = np.asarray(
        [
            one_edge(data, higher, lower, "deep", "deep"),
            one_edge(data, higher, lower, "deep", "sup"),
        ],
        dtype=float,
    )
    return {
        "signed_FF": float(ff_edges.mean()),
        "signed_FB": float(fb_edges.mean()),
        "absolute_FF": float(np.abs(ff_edges).mean()),
        "absolute_FB": float(np.abs(fb_edges).mean()),
        "FF_sup_to_deep": float(ff_edges[0]),
        "FF_sup_to_mid": float(ff_edges[1]),
        "FB_deep_to_deep": float(fb_edges[0]),
        "FB_deep_to_sup": float(fb_edges[1]),
    }


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.ptp(x) <= 0 or np.ptp(y) <= 0:
        return np.nan
    return float(spearmanr(x, y).statistic)


def exact_signflip_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or not len(values):
        raise RuntimeError("Exact sign flipping requires finite statistics")
    observed = abs(float(values.mean()))
    signs = np.asarray(
        list(itertools.product((-1.0, 1.0), repeat=len(values))), dtype=float
    )
    null = np.abs((signs * values[None, :]).mean(axis=1))
    return float(np.mean(null >= observed - 1e-15))


def bootstrap_mean_ci(
    values: np.ndarray, rng: np.random.Generator, n_boot: int
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or not len(values):
        raise RuntimeError("Bootstrap requires finite statistics")
    indices = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    sorted_adjusted = np.maximum.accumulate(
        (len(p_values) - np.arange(len(p_values))) * p_values[order]
    )
    output = np.empty_like(p_values)
    output[order] = np.minimum(sorted_adjusted, 1.0)
    return output


def build_pair_table(edges: pd.DataFrame, g1: pd.DataFrame) -> pd.DataFrame:
    score = g1.set_index("stage")["G1_oriented"].to_dict()
    stage_index = {stage: index for index, stage in enumerate(STAGES)}
    rows: list[dict[str, object]] = []
    for subject, participant in edges.groupby("subject", sort=True):
        for lower, higher in combinations(STAGES, 2):
            delta = float(score[higher] - score[lower])
            rows.append(
                {
                    "subject": subject,
                    "lower_stage": lower,
                    "higher_stage": higher,
                    "lower_display": DISPLAY_STAGE[lower],
                    "higher_display": DISPLAY_STAGE[higher],
                    "stage_pair": (
                        f"{DISPLAY_STAGE[lower]}–{DISPLAY_STAGE[higher]}"
                    ),
                    "ordinal_gap": stage_index[higher] - stage_index[lower],
                    "G1_lower": score[lower],
                    "G1_higher": score[higher],
                    "G1_higher_minus_lower": delta,
                    "G1_absolute_separation": abs(delta),
                    **pathway_values(participant, lower, higher),
                }
            )
    output = pd.DataFrame(rows)
    if not output.groupby("subject").size().eq(3).all():
        raise RuntimeError("Expected three unordered stage pairs per participant")
    return output


def build_directed_rows(pair_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in pair_table.itertuples(index=False):
        common = {
            "subject": row.subject,
            "stage_pair": row.stage_pair,
            "ordinal_gap": row.ordinal_gap,
            "G1_absolute_separation": row.G1_absolute_separation,
        }
        rows.append(
            {
                **common,
                "pathway": "FF",
                "source_stage": row.lower_stage,
                "target_stage": row.higher_stage,
                "source_display": row.lower_display,
                "target_display": row.higher_display,
                "directed_G1_displacement": row.G1_higher_minus_lower,
                "signed_pathway_strength": row.signed_FF,
                "absolute_pathway_strength": row.absolute_FF,
            }
        )
        rows.append(
            {
                **common,
                "pathway": "FB",
                "source_stage": row.higher_stage,
                "target_stage": row.lower_stage,
                "source_display": row.higher_display,
                "target_display": row.lower_display,
                "directed_G1_displacement": -row.G1_higher_minus_lower,
                "signed_pathway_strength": row.signed_FB,
                "absolute_pathway_strength": row.absolute_FB,
            }
        )
    output = pd.DataFrame(rows)
    if not output.groupby("subject").size().eq(6).all():
        raise RuntimeError("Expected six directed pathways per participant")
    return output


def participant_primary(directed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for subject, participant in directed.groupby("subject", sort=True):
        rows.append(
            {
                "subject": subject,
                "primary_metric": "signed_directed_pathway_alignment",
                "spearman_rho_across_6_directed_pathways": safe_spearman(
                    participant["directed_G1_displacement"].to_numpy(float),
                    participant["signed_pathway_strength"].to_numpy(float),
                ),
                "n_directed_pathways": len(participant),
            }
        )
    output = pd.DataFrame(rows)
    if output["spearman_rho_across_6_directed_pathways"].isna().any():
        bad = output.loc[
            output["spearman_rho_across_6_directed_pathways"].isna(), "subject"
        ].tolist()
        raise RuntimeError(f"Undefined primary participant correlations: {bad}")
    return output


def participant_secondary(pair_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for subject, participant in pair_table.groupby("subject", sort=True):
        separation = participant["G1_absolute_separation"].to_numpy(float)
        for metric in SECONDARY_METRICS:
            rows.append(
                {
                    "subject": subject,
                    "metric": metric,
                    "spearman_rho_across_3_stage_pairs": safe_spearman(
                        separation, participant[metric].to_numpy(float)
                    ),
                    "n_stage_pairs": len(participant),
                }
            )
    output = pd.DataFrame(rows)
    if output["spearman_rho_across_3_stage_pairs"].isna().any():
        bad = output.loc[
            output["spearman_rho_across_3_stage_pairs"].isna(),
            ["subject", "metric"],
        ]
        raise RuntimeError(
            "Undefined low-resolution secondary correlations:\n"
            + bad.to_string(index=False)
        )
    return output


def summarize_primary(
    participant: pd.DataFrame, rng: np.random.Generator, n_boot: int
) -> pd.DataFrame:
    values = participant["spearman_rho_across_6_directed_pathways"].to_numpy(
        float
    )
    low, high = bootstrap_mean_ci(values, rng, n_boot)
    return pd.DataFrame(
        [
            {
                "analysis_role": "primary_node_definition_sensitivity",
                "metric": "signed_directed_pathway_alignment",
                "n_subjects": len(values),
                "n_pathways_per_subject": 6,
                "mean_within_subject_spearman_rho": float(values.mean()),
                "sd_within_subject_spearman_rho": float(values.std(ddof=1)),
                "bootstrap_ci95_low": low,
                "bootstrap_ci95_high": high,
                "exact_signflip_p_two_sided": exact_signflip_p(values),
                "n_signflip_patterns": 2 ** len(values),
                "n_bootstrap": n_boot,
                "multiplicity_adjustment": "none: one specified primary test",
            }
        ]
    )


def summarize_secondary(
    participant: pd.DataFrame, rng: np.random.Generator, n_boot: int
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric in SECONDARY_METRICS:
        values = participant.loc[
            participant["metric"] == metric,
            "spearman_rho_across_3_stage_pairs",
        ].to_numpy(float)
        low, high = bootstrap_mean_ci(values, rng, n_boot)
        rows.append(
            {
                "analysis_role": "secondary_low_resolution_decomposition",
                "family": "signed" if metric.startswith("signed") else "absolute",
                "metric": metric,
                "n_subjects": len(values),
                "n_stage_pairs_per_subject": 3,
                "mean_within_subject_spearman_rho": float(values.mean()),
                "sd_within_subject_spearman_rho": float(values.std(ddof=1)),
                "bootstrap_ci95_low": low,
                "bootstrap_ci95_high": high,
                "exact_signflip_p_two_sided": exact_signflip_p(values),
                "n_signflip_patterns": 2 ** len(values),
                "n_bootstrap": n_boot,
            }
        )
    output = pd.DataFrame(rows)
    output["exact_signflip_p_holm_within_2_test_family"] = np.nan
    for _, indices in output.groupby("family").groups.items():
        output.loc[
            indices, "exact_signflip_p_holm_within_2_test_family"
        ] = holm_adjust(
            output.loc[indices, "exact_signflip_p_two_sided"].to_numpy(float)
        )
    return output


def plot_forest(
    primary: pd.DataFrame, secondary: pd.DataFrame, output_stem: Path
) -> None:
    rows: list[dict[str, object]] = [
        {
            "label": "Primary: signed directed alignment",
            "color": PURPLE,
            "marker": "D",
            "significant": (
                primary.iloc[0]["exact_signflip_p_two_sided"] < 0.05
            ),
            **primary.iloc[0].to_dict(),
        }
    ]
    colors = {"FF": FF_BLUE, "FB": FB_ORANGE}
    labels = {
        "signed_FF": "Signed FF vs |ΔG1|",
        "signed_FB": "Signed FB vs |ΔG1|",
        "absolute_FF": "|FF| vs |ΔG1|",
        "absolute_FB": "|FB| vs |ΔG1|",
    }
    for result in secondary.itertuples(index=False):
        direction = result.metric.split("_")[-1]
        rows.append(
            {
                "label": labels[result.metric],
                "color": colors[direction],
                "marker": "o" if direction == "FF" else "s",
                "significant": (
                    result.exact_signflip_p_holm_within_2_test_family < 0.05
                ),
                **result._asdict(),
            }
        )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.edgecolor": INK,
            "axes.linewidth": 0.8,
        }
    )
    figure, axis = plt.subplots(figsize=(7.25, 4.15))
    positions = np.arange(len(rows))[::-1]
    for position, row in zip(positions, rows):
        mean = float(row["mean_within_subject_spearman_rho"])
        low = float(row["bootstrap_ci95_low"])
        high = float(row["bootstrap_ci95_high"])
        axis.errorbar(
            mean,
            position,
            xerr=np.asarray([[mean - low], [high - mean]]),
            fmt=str(row["marker"]),
            ms=6.5,
            lw=1.5,
            capsize=3,
            color=str(row["color"]),
            mec=str(row["color"]),
            mew=1.3,
            mfc=str(row["color"]) if row["significant"] else "white",
            zorder=3,
        )
    axis.axvline(0, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)
    axis.set_xlim(-1.05, 1.05)
    axis.set_yticks(positions, [str(row["label"]) for row in rows])
    axis.set_xlabel("Mean within-participant Spearman ρ (bootstrap 95% CI)")
    figure.suptitle(
        "EVC-pooled rDCM pathways along the principal gradient",
        x=0.055,
        y=0.975,
        ha="left",
        fontsize=12.3,
        fontweight="bold",
    )
    figure.text(
        0.055,
        0.91,
        "EVC is the equal-ROI composite of V1, V2 and V3; n = 15",
        color=MUTED,
        fontsize=8.4,
    )
    axis.grid(axis="x", color=GRID, lw=0.7, zorder=0)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    figure.text(
        0.055,
        0.012,
        "Primary: exact two-sided participant sign flip. Secondary FF/FB "
        "decompositions use only three stage pairs per participant and are "
        "low-resolution sensitivity analyses; Holm correction is applied "
        "within signed and absolute families.",
        fontsize=7.25,
        color=MUTED,
    )
    figure.tight_layout(rect=(0.03, 0.105, 1, 0.87))
    for suffix in (".png", ".pdf", ".svg"):
        kwargs = {"dpi": 600} if suffix == ".png" else {}
        figure.savefig(
            output_stem.with_suffix(suffix),
            facecolor="white",
            bbox_inches="tight",
            **kwargs,
        )
    plt.close(figure)


def write_manifest(
    args: argparse.Namespace,
    edges: pd.DataFrame,
    lineage: pd.DataFrame,
    stage_geometry: pd.DataFrame,
    joined_geometry: pd.DataFrame,
    output_path: Path,
) -> None:
    output_path.write_text(
        json.dumps(
            {
                "analysis_name": "G1_EVC_pooled_directed_pathway_alignment",
                "analysis_role": (
                    "node-definition sensitivity aligned with endpoint EVC"
                ),
                "inputs": {
                    "edges_csv": str(args.edges_csv.resolve()),
                    "edges_sha256": sha256_file(args.edges_csv),
                    "export_audit_csv": str(args.export_audit_csv.resolve()),
                    "export_audit_sha256": sha256_file(args.export_audit_csv),
                    "membership_csv": str(args.membership_csv.resolve()),
                    "membership_sha256": sha256_file(args.membership_csv),
                    "reference_gradient_csv": str(
                        args.reference_gradient_csv.resolve()
                    ),
                    "reference_gradient_sha256": sha256_file(
                        args.reference_gradient_csv
                    ),
                },
                "sample": {
                    "n_subjects": int(edges["subject"].nunique()),
                    "subjects": sorted(edges["subject"].unique().tolist()),
                    "source_ts_bases": sorted(
                        lineage["source_ts_base"].astype(str).unique().tolist()
                    ),
                    "outer_pial_exclusion": "5% final inputs",
                },
                "definitions": {
                    "stages": list(STAGES),
                    "display_stages": [
                        DISPLAY_STAGE[stage] for stage in STAGES
                    ],
                    "EVC": "equal ROI mean of V1, V2 and V3 within each layer",
                    "stage_members": {
                        stage: joined_geometry.loc[
                            joined_geometry["stage"] == stage, "candidate_roi"
                        ].tolist()
                        for stage in STAGES
                    },
                    "G1_stage_scores": {
                        row.stage: float(row.G1_oriented)
                        for row in stage_geometry.itertuples()
                    },
                    "FF": (
                        "lower-stage superficial -> higher-stage deep/middle; "
                        "equal mean of two A estimates"
                    ),
                    "FB": (
                        "higher-stage deep -> lower-stage deep/superficial; "
                        "equal mean of two A estimates"
                    ),
                    "primary_statistic": (
                        "participant Spearman rho across six directed pathways"
                    ),
                    "primary_inference": (
                        "two-sided exact sign flip across participant rhos"
                    ),
                    "secondary_statistic": (
                        "participant Spearman rho across three stage pairs; "
                        "low-resolution sensitivity"
                    ),
                    "H_metric": "not used",
                },
                "random_seed": args.seed,
                "n_bootstrap": args.n_boot,
                "code_file": str(Path(__file__).resolve()),
                "code_sha256": sha256_file(Path(__file__).resolve()),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    validate_output_isolation(args.out_dir)
    if args.expected_subjects != 15:
        raise ValueError("This analysis expects exactly 15 participants")
    if args.n_boot < 10_000:
        raise ValueError("Use at least 10,000 bootstrap resamples")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    edges = load_edges(args.edges_csv, args.expected_subjects)
    lineage = validate_lineage(
        args.export_audit_csv,
        sorted(edges["subject"].unique().tolist()),
        args.expected_subjects,
    )
    stage_geometry, joined_geometry = load_g1_stage_geometry(
        args.membership_csv, args.reference_gradient_csv
    )
    pair_table = build_pair_table(edges, stage_geometry)
    directed = build_directed_rows(pair_table)
    subject_primary = participant_primary(directed)
    subject_secondary = participant_secondary(pair_table)

    rng = np.random.default_rng(args.seed)
    primary_group = summarize_primary(subject_primary, rng, args.n_boot)
    secondary_group = summarize_secondary(subject_secondary, rng, args.n_boot)

    stage_geometry.to_csv(
        args.out_dir / "evc_pooled_G1_stage_geometry.csv", index=False
    )
    joined_geometry.to_csv(
        args.out_dir / "evc_pooled_G1_roi_membership.csv", index=False
    )
    pair_table.to_csv(
        args.out_dir / "evc_pooled_subject_stage_pair_pathways.csv", index=False
    )
    directed.to_csv(
        args.out_dir / "evc_pooled_subject_directed_pathways.csv", index=False
    )
    subject_primary.to_csv(
        args.out_dir / "evc_pooled_subject_primary_alignment.csv", index=False
    )
    primary_group.to_csv(
        args.out_dir / "evc_pooled_primary_group_test.csv", index=False
    )
    subject_secondary.to_csv(
        args.out_dir / "evc_pooled_subject_secondary_associations.csv",
        index=False,
    )
    secondary_group.to_csv(
        args.out_dir / "evc_pooled_secondary_group_tests.csv", index=False
    )
    write_manifest(
        args,
        edges,
        lineage,
        stage_geometry,
        joined_geometry,
        args.out_dir / "evc_pooled_alignment_manifest.json",
    )
    plot_forest(
        primary_group,
        secondary_group,
        args.out_dir / "evc_pooled_G1_directed_alignment_forest",
    )

    primary = primary_group.iloc[0]
    print("=" * 90)
    print("EVC(V1/V2/V3)-POOLED G1-DIRECTED rDCM PATHWAY ALIGNMENT")
    print("=" * 90)
    print(stage_geometry[["display_stage", "n_rois", "G1_oriented"]].to_string(index=False))
    print(
        "\nPrimary mean within-participant Spearman rho = "
        f"{primary['mean_within_subject_spearman_rho']:.6f}"
    )
    print(
        "Participant-bootstrap 95% CI = "
        f"[{primary['bootstrap_ci95_low']:.6f}, "
        f"{primary['bootstrap_ci95_high']:.6f}]"
    )
    print(
        "Two-sided exact sign-flip p = "
        f"{primary['exact_signflip_p_two_sided']:.8f}"
    )
    print("\nSecondary low-resolution decomposition:")
    print(
        secondary_group[
            [
                "metric",
                "mean_within_subject_spearman_rho",
                "bootstrap_ci95_low",
                "bootstrap_ci95_high",
                "exact_signflip_p_two_sided",
                "exact_signflip_p_holm_within_2_test_family",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved: {args.out_dir}")
    print("=" * 90)


if __name__ == "__main__":
    main()
