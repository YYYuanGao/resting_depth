#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functional-gradient analysis for the final outer-pial 5% data.

The primary implementation follows Serio et al. (Nature Communications, 2024):
1. Fit 10 diffusion-map gradients using the normalized-angle kernel and
   row-wise 90% sparsity (retain the strongest 10%).
2. Build the reference from the FC matrix averaged across all participants.
3. Procrustes-align every participant's gradients to that mean reference.

An optional leave-one-subject-out (LOSO) reference is retained as a sensitivity
analysis. LOSO is not the alignment used by Serio et al.

Additional safeguards
---------------------
1. No post-hoc choice of the component that best matches the pooled-EVC-to-PFC order.
2. Split-half reliability is inferred across participants, not across repeated
   random splits treated as independent observations.
3. The Serio/Bianca-style primary organization is the aligned principal
   gradient G1; a pooled-EVC (V1/V2/V3)-to-cauPFC axis in G1-G3 is a
   secondary bridge measure.

Input
-----
For each participant:
  <ts-base>/<tag>/sub-XX/ts_matrix.npy   [ROI, layer, time]
  <ts-base>/<tag>/sub-XX/roi_names.npy

The default input is the final outer-pial 5% branch. The script intentionally
has no fallback to the non-pial branch.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import matplotlib

import os
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "/path/to/project")  # see config/paths.sh
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

try:
    from brainspace.gradient import GradientMaps
except ImportError:
    GradientMaps = None


ROOT = Path(PROJECT_ROOT + "/derivatives_0331")
DEFAULT_TS_BASE = (
    ROOT / "S6_validation" / "pial_depth_exclusion"
    / "Step3_TimeSeries_tSNRclean_pial5pct"
)
DEFAULT_OUT = (
    ROOT / "S6_validation" / "pial5_only_update"
    / "functional_gradient_bianca_loso_pooled_evc"
)
DEFAULT_STAGE_DEFINITION = (
    ROOT / "S6_validation" / "pial5_only_update"
    / "functional_gradient_loso_stage_definition_pooled_evc.json"
)
DEFAULT_SUBJECTS = [
    "01", "02", "03", "04", "05", "06", "07", "08",
    "09", "10", "11", "12", "13", "14", "16",
]

STAGE_ORDER = ("EVC", "PPC", "cauPFC")
FROZEN_STAGE_CANDIDATES = {
    "EVC": ("V1", "V2", "V3"),
    "PPC": (
    "IPS1", "LIPv", "LIPd", "VIP", "AIP", "MIP", "IP0", "IP1", "IP2",
    "7PC", "7Am", "7AL", "7PL", "7Pm", "7m",
    ),
    "cauPFC": ("6d", "6v", "6mp", "FEF", "PEF", "55b", "SFL"),
}
STRICT_STAGE_MEMBERS = {"EVC", "cauPFC"}

# Only spelling/case aliases. In particular, 9-46v and p9-46v are distinct
# HCP-MMP parcels and must never be collapsed.
NAME_MAP = {"7Al": "7AL", "8AD": "8Ad"}

# This ordering is retained from the original all-ROI visualization. It is used
# only for display after G1 has been estimated and fixed. It never selects,
# rotates, reorders, or otherwise changes a gradient component.
DISPLAY_SEQUENCE = (
    "V1", "V2", "V3", "V3A", "V3B", "V4", "V8", "V6", "V6A", "V7",
    "MT", "MST", "LO1", "LO2", "LO3", "V4t", "FST", "V3CD",
    "FFC", "VMV1", "VMV2", "PCV",
    "TPOJ1", "TPOJ2", "TPOJ3", "STV", "PHA2", "PHA3",
    "IPS1", "LIPv", "LIPd", "VIP", "AIP", "MIP", "IP0", "IP1", "IP2",
    "7PC", "7Am", "7AL", "7PL", "7Pm", "7m",
    "PGp", "PGi", "PGs", "PF", "PFm", "PFop", "PFcm", "PFt",
    "OP1", "OP2-3", "OP4",
    "POS1", "POS2", "RSC", "DVT",
    "23c", "23d", "v23ab", "d23ab", "31pd", "31a", "31pv",
    "5L", "5m", "5mv",
    "2", "1", "3b", "3a", "4",
    "6a", "6ma", "6d", "6v", "6mp", "6r", "i6-8", "SCEF",
    "PEF", "FEF", "55b",
    "SFL", "8Ad", "8C", "p9-46v",
)

SEQUENCE_CMAP = LinearSegmentedColormap.from_list(
    "visual_to_prefrontal",
    ["#3B8FC2", "#61BBA6", "#E8D477", "#F1A35C", "#E76F63", "#9B6BB5"],
)
INK = "#252A30"
MUTED = "#6B7280"
LIGHT_GRID = "#E5E7EB"
FF_BLUE = "#147FAE"
FB_RED = "#E65C5C"
STAGE_LABELS = {
    "EVC": "EVC (V1/V2/V3)",
    "PPC": "PPC",
    "cauPFC": "cauPFC",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ts-base", type=Path, default=DEFAULT_TS_BASE)
    p.add_argument("--tag", default="ld1000_3layers_equivol")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--stage-definition",
        type=Path,
        default=DEFAULT_STAGE_DEFINITION,
        help=(
            "JSON whose retained_members field freezes the exact pooled-EVC "
            "(V1/V2/V3), PPC and cauPFC ROIs for this sensitivity analysis."
        ),
    )
    p.add_argument(
        "--allow-frozen-stage-fallback",
        action="store_true",
        help=(
            "Permit the frozen candidate list when --stage-definition is absent. "
            "Off by default because exact PPC membership should match rDCM."
        ),
    )
    p.add_argument("--subjects", nargs="+", default=DEFAULT_SUBJECTS)
    p.add_argument("--n-components", type=int, default=10)
    p.add_argument("--analysis-components", type=int, default=3)
    p.add_argument("--sparsity", type=float, default=0.90)
    p.add_argument(
        "--reference-mode",
        choices=("full_sample", "loso"),
        default="loso",
        help=(
            "full_sample reproduces the reference strategy of Serio et al.; "
            "loso is an optional sensitivity analysis"
        ),
    )
    p.add_argument("--max-nan-frac", type=float, default=0.20)
    p.add_argument("--n-bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20260723)
    return p.parse_args()


def norm_roi(x: object) -> str:
    s = str(x).strip()
    return NAME_MAP.get(s, s)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_and_average_layers(
    ts: np.ndarray,
    max_nan_frac: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Validate every ROI-layer series before averaging across the three layers.

    A ROI enters the functional-gradient analysis only when all three layer
    series pass the same finite-data and non-constant checks. This prevents
    np.nanmean from silently turning a one- or two-layer ROI into an apparently
    valid whole-cortex time series.
    """
    arr = np.asarray(ts, float)
    if arr.ndim != 3 or arr.shape[1] != 3:
        raise ValueError(f"Expected ROI x 3 layers x time, got {arr.shape}")
    cleaned = arr.copy()
    layer_valid = np.zeros(arr.shape[:2], dtype=bool)
    grid = np.arange(arr.shape[2])
    for ri in range(arr.shape[0]):
        for li in range(arr.shape[1]):
            row = cleaned[ri, li]
            finite = np.isfinite(row)
            nan_frac = 1.0 - finite.mean()
            if nan_frac > max_nan_frac or finite.sum() < 3:
                continue
            if not finite.all():
                row[~finite] = np.interp(grid[~finite], grid[finite], row[finite])
            if np.std(row) < 1e-8:
                continue
            layer_valid[ri, li] = True
    roi_valid = layer_valid.all(axis=1)
    averaged = np.full((arr.shape[0], arr.shape[2]), np.nan, dtype=float)
    averaged[roi_valid] = cleaned[roi_valid].mean(axis=1)
    return averaged, roi_valid, layer_valid


def load_subject(
    args: argparse.Namespace,
    sub: str,
) -> tuple[np.ndarray, list[str], np.ndarray, dict[str, object]]:
    d = args.ts_base / args.tag / f"sub-{sub}"
    ts_file = d / "ts_matrix.npy"
    names_file = d / "roi_names.npy"
    if not ts_file.exists() or not names_file.exists():
        raise FileNotFoundError(f"Missing final pial5 input for sub-{sub}: {d}")
    ts = np.load(ts_file)
    if ts.ndim != 3:
        raise ValueError(f"{ts_file}: expected ROI x layer x time, got {ts.shape}")
    names = [norm_roi(x) for x in np.load(names_file, allow_pickle=True)]
    if len(names) != ts.shape[0] or len(set(names)) != len(names):
        raise ValueError(f"{d}: ROI names are missing, duplicated, or misordered")
    averaged, roi_valid, layer_valid = clean_and_average_layers(ts, args.max_nan_frac)
    audit = {
        "subject": f"sub-{sub}",
        "ts_file": str(ts_file.resolve()),
        "roi_names_file": str(names_file.resolve()),
        "ts_sha256": sha256_file(ts_file),
        "roi_names_sha256": sha256_file(names_file),
        "ts_shape": "x".join(str(x) for x in ts.shape),
        "ts_dtype": str(ts.dtype),
        "n_rois_total": int(ts.shape[0]),
        "n_layers_expected": 3,
        "n_valid_roi_layers": int(layer_valid.sum()),
        "n_valid_rois_all_three_layers": int(roi_valid.sum()),
        "n_invalid_rois": int((~roi_valid).sum()),
        "pial5_path_token_present": "Step3_TimeSeries_tSNRclean_pial5pct" in str(ts_file),
    }
    return averaged, names, roi_valid, audit


def resolve_stages(
    args: argparse.Namespace,
) -> tuple[dict[str, tuple[str, ...]], str, dict[str, object]]:
    """Resolve pooled-EVC/PPC/cauPFC membership for this analysis."""
    stage_file = args.stage_definition
    if stage_file.exists():
        raw = json.loads(stage_file.read_text(encoding="utf-8"))
        retained = raw.get("retained_members")
        if not isinstance(retained, dict):
            raise ValueError(f"{stage_file}: missing dict field retained_members")
        missing = [s for s in STAGE_ORDER if s not in retained]
        if missing:
            raise ValueError(f"{stage_file}: missing stages {missing}")
        stages = {
            stage: tuple(norm_roi(x) for x in retained[stage])
            for stage in STAGE_ORDER
        }
        expected_source = (args.ts_base / args.tag).resolve()
        saved_source = raw.get("source_ts_base")
        if saved_source is None:
            raise ValueError(f"{stage_file}: missing source_ts_base")
        if Path(saved_source).resolve() != expected_source:
            raise ValueError(
                "Stage-definition input does not match gradient input:\n"
                f"  JSON source: {Path(saved_source).resolve()}\n"
                f"  gradient:    {expected_source}"
            )
        source = "multistage_stage_definition_json"
        provenance = raw
    else:
        if not args.allow_frozen_stage_fallback:
            raise FileNotFoundError(
                f"Missing exact stage definition: {stage_file}\n"
                "Run export_pial5_multistage_timeseries.py first, or use "
                "--allow-frozen-stage-fallback only as an explicitly documented "
                "sensitivity analysis."
            )
        stages = {
            stage: tuple(norm_roi(x) for x in FROZEN_STAGE_CANDIDATES[stage])
            for stage in STAGE_ORDER
        }
        source = "frozen_candidate_fallback"
        provenance = {
            "source_ts_base": str((args.ts_base / args.tag).resolve()),
            "retained_members": {k: list(v) for k, v in stages.items()},
        }

    for stage in STAGE_ORDER:
        members = stages[stage]
        if not members or len(set(members)) != len(members):
            raise ValueError(f"Stage {stage} has empty or duplicate membership: {members}")
        candidates = set(FROZEN_STAGE_CANDIDATES[stage])
        if stage in STRICT_STAGE_MEMBERS and set(members) != candidates:
            raise ValueError(
                f"Strict stage {stage} must equal {sorted(candidates)}, got {sorted(members)}"
            )
        if stage == "PPC":
            if len(members) < 5 or not set(members).issubset(candidates):
                raise ValueError(
                    "PPC must contain at least five members drawn from the frozen "
                    f"candidate set; got {members}"
                )

    membership = [roi for stage in STAGE_ORDER for roi in stages[stage]]
    duplicates = sorted({roi for roi in membership if membership.count(roi) > 1})
    if duplicates:
        raise ValueError(f"Stages are not mutually exclusive: {duplicates}")
    return stages, source, provenance


def fc_z(ts: np.ndarray) -> np.ndarray:
    if ts.ndim != 2 or ts.shape[0] < 3 or ts.shape[1] < 10:
        raise ValueError(f"Insufficient ROI x time array for FC: {ts.shape}")
    if not np.isfinite(ts).all():
        raise ValueError("FC input still contains non-finite values")
    r = np.corrcoef(ts)
    if not np.isfinite(r).all():
        raise ValueError("Pearson FC contains non-finite values after input QC")
    r = np.clip(r, -1 + 1e-7, 1 - 1e-7)
    z = np.arctanh(r)
    np.fill_diagonal(z, 0.0)
    return z


def embed_reference(
    fc: np.ndarray,
    n_components: int,
    sparsity: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the group reference exactly with the Serio et al. BrainSpace setup."""
    if GradientMaps is None:
        raise ImportError("BrainSpace is required: pip install brainspace")
    gm = GradientMaps(
        n_components=n_components,
        kernel="normalized_angle",
        approach="dm",
        random_state=0,
    )
    gm.fit(fc, sparsity=sparsity, alpha=0.5, diffusion_time=0)
    g = np.asarray(gm.gradients_, float)[:, :n_components]
    lambdas = np.asarray(gm.lambdas_, float).reshape(-1)[:n_components]
    if (
        g.shape != (fc.shape[0], n_components)
        or lambdas.shape != (n_components,)
        or not np.isfinite(g).all()
        or not np.isfinite(lambdas).all()
    ):
        raise RuntimeError("Reference gradient embedding returned invalid values")
    return g, lambdas


def embed_aligned(
    fc: np.ndarray,
    reference: np.ndarray,
    n_components: int,
    sparsity: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit one participant and Procrustes-align it to the supplied reference."""
    if GradientMaps is None:
        raise ImportError("BrainSpace is required: pip install brainspace")
    gm = GradientMaps(
        n_components=n_components,
        kernel="normalized_angle",
        approach="dm",
        random_state=0,
        alignment="procrustes",
    )
    gm.fit(
        fc,
        sparsity=sparsity,
        reference=reference,
        alpha=0.5,
        diffusion_time=0,
    )
    native = np.asarray(gm.gradients_, float)[:, :n_components]
    aligned = np.asarray(gm.aligned_, float)[:, :n_components]
    if (
        native.shape != (fc.shape[0], n_components)
        or aligned.shape != (fc.shape[0], n_components)
        or not np.isfinite(native).all()
        or not np.isfinite(aligned).all()
    ):
        raise RuntimeError("Participant gradient embedding/alignment returned invalid values")
    return native, aligned


def stage_axis(
    reference: np.ndarray,
    roi_names: list[str],
    stages: dict[str, tuple[str, ...]],
) -> tuple[np.ndarray, pd.DataFrame]:
    names = np.asarray(roi_names)
    centroids = []
    rows = []
    for stage_name in STAGE_ORDER:
        members = set(stages[stage_name])
        idx = np.array([n in members for n in names])
        if idx.sum() == 0:
            raise RuntimeError(f"No common ROI is available for stage {stage_name}")
        c = np.nanmean(reference[idx], axis=0)
        centroids.append(c)
        rows.append({"stage": stage_name, "n_rois": int(idx.sum())})
    cmat = np.vstack(centroids)
    _, _, vt = np.linalg.svd(cmat - cmat.mean(axis=0, keepdims=True), full_matrices=False)
    axis = vt[0]
    if float((cmat[-1] - cmat[0]) @ axis) < 0:
        axis *= -1
    scores = cmat @ axis
    for row, score in zip(rows, scores):
        row["reference_axis_score"] = float(score)
    return axis, pd.DataFrame(rows)


def oriented_g1_stage_scores(
    reference: np.ndarray,
    subject_gradient: np.ndarray,
    roi_names: list[str],
    stages: dict[str, tuple[str, ...]],
) -> tuple[float, list[float], list[float]]:
    """
    Orient G1 once using the reference pooled-EVC and cauPFC anchors.

    Diffusion-gradient sign is arbitrary. This operation never changes which
    component is G1; it only makes increasing values denote EVC-to-cauPFC for
    readable group and participant summaries.
    """
    ref_scores = []
    subject_scores = []
    names = np.asarray(roi_names)
    for stage in STAGE_ORDER:
        idx = np.array([r in set(stages[stage]) for r in names])
        if not idx.any():
            raise RuntimeError(f"No ROI is available for G1 stage {stage}")
        ref_scores.append(float(np.mean(reference[idx, 0])))
        subject_scores.append(float(np.mean(subject_gradient[idx, 0])))
    sign = 1.0 if ref_scores[-1] - ref_scores[0] >= 0 else -1.0
    return sign, [sign * x for x in ref_scores], [sign * x for x in subject_scores]


def exact_signflip_p(values: np.ndarray) -> float:
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    obs = abs(float(np.mean(x)))
    null = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(x)):
        null.append(abs(float(np.mean(x * np.asarray(signs)))))
    return float(np.mean(np.asarray(null) >= obs - 1e-15))


def bootstrap_mean_ci(values: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(x, size=(n_boot, len(x)), replace=True), axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]).astype(float))


def summarize(values: list[float], metric: str, args: argparse.Namespace) -> dict[str, object]:
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    lo, hi = bootstrap_mean_ci(x, args.n_bootstrap, args.seed)
    return {
        "metric": metric,
        "n_subjects": int(len(x)),
        "mean": float(np.mean(x)) if len(x) else np.nan,
        "sd": float(np.std(x, ddof=1)) if len(x) > 1 else np.nan,
        "bootstrap_ci95_low": lo,
        "bootstrap_ci95_high": hi,
        "exact_signflip_p_two_sided": exact_signflip_p(x),
    }


def set_publication_style() -> None:
    """Compact, restrained defaults suitable for journal figure assembly."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.0,
        "axes.titlesize": 10.0,
        "axes.labelsize": 8.5,
        "axes.linewidth": 0.8,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def clean_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, color=LIGHT_GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def save_publication_figure(fig: plt.Figure, base: Path) -> None:
    """Save review and editable publication formats from one source figure."""
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def add_title(
    fig: plt.Figure,
    title: str,
    subtitle: str,
    x: float = 0.02,
) -> None:
    fig.text(x, 0.975, title, ha="left", va="top", fontsize=12, weight="bold", color=INK)
    fig.text(x, 0.925, subtitle, ha="left", va="top", fontsize=7.5, color=MUTED)


def sequence_display_table(reference_scores: pd.DataFrame) -> pd.DataFrame:
    rank = {norm_roi(roi): i for i, roi in enumerate(DISPLAY_SEQUENCE)}
    out = reference_scores.copy()
    out["display_rank_full"] = out["roi"].map(rank)
    out = out[out["display_rank_full"].notna()].copy()
    out["display_rank_full"] = out["display_rank_full"].astype(int)
    out = out.sort_values("display_rank_full").reset_index(drop=True)
    if out.empty:
        raise RuntimeError("No common-valid ROI belongs to DISPLAY_SEQUENCE")
    out["display_position"] = np.arange(len(out))
    denom = max(len(DISPLAY_SEQUENCE) - 1, 1)
    out["display_rank_normalized"] = out["display_rank_full"] / denom
    return out


def plot_reference_spectrum(
    lambdas: np.ndarray,
    figure_dir: Path,
) -> None:
    values = np.asarray(lambdas, float)
    denom = float(np.sum(values))
    fractions = values / denom if denom > 0 else np.full_like(values, np.nan)
    x = np.arange(1, len(values) + 1)
    fig, ax = plt.subplots(figsize=(4.6, 3.25))
    fig.subplots_adjust(left=0.15, right=0.97, bottom=0.18, top=0.76)
    colors = [INK] + ["#BCC3CA"] * (len(values) - 1)
    ax.bar(x, fractions, width=0.72, color=colors, edgecolor="none", zorder=2)
    ax.plot(x, fractions, color="#8B939C", linewidth=0.8, zorder=1)
    clean_axis(ax)
    ax.set_xlabel("Diffusion-map component")
    ax.set_ylabel("Relative eigenvalue contribution")
    ax.set_xticks(x)
    add_title(
        fig,
        "Reference diffusion-map spectrum",
        "G1 is fixed a priori as the primary macroscale gradient.",
        x=0.05,
    )
    save_publication_figure(fig, figure_dir / "gradient_reference_spectrum")


def plot_reference_gradient_space(
    reference_scores: pd.DataFrame,
    figure_dir: Path,
) -> pd.DataFrame:
    seq = sequence_display_table(reference_scores)
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.65), sharex=True)
    fig.subplots_adjust(left=0.09, right=0.88, bottom=0.20, top=0.73, wspace=0.32)
    norm = Normalize(vmin=0, vmax=max(len(DISPLAY_SEQUENCE) - 1, 1))

    labels_to_show = {"V1", "V2", "V3", "LIPv", "7PC", "FEF", "55b", "SFL"}
    for panel, (ax, y_col) in enumerate(zip(axes, ("G2", "G3"))):
        ax.scatter(
            reference_scores["G1_oriented"],
            reference_scores[y_col],
            s=15,
            facecolor="#D7DCE0",
            edgecolor="white",
            linewidth=0.4,
            alpha=0.72,
            zorder=1,
        )
        ax.scatter(
            seq["G1_oriented"],
            seq[y_col],
            c=seq["display_rank_full"],
            cmap=SEQUENCE_CMAP,
            norm=norm,
            s=31,
            edgecolor="white",
            linewidth=0.65,
            zorder=2,
        )
        for _, row in seq[seq["roi"].isin(labels_to_show)].iterrows():
            ax.annotate(
                row["roi"],
                (row["G1_oriented"], row[y_col]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=6.2,
                color=INK,
            )
        clean_axis(ax, grid_axis="both")
        ax.axhline(0, color="#AAB0B6", linewidth=0.65, zorder=0)
        ax.axvline(0, color="#AAB0B6", linewidth=0.65, zorder=0)
        ax.set_xlabel("Oriented principal gradient G1")
        ax.set_ylabel(y_col)
        ax.text(
            -0.16, 1.04, chr(ord("a") + panel),
            transform=ax.transAxes, fontsize=10, weight="bold", color=INK,
        )

    cax = fig.add_axes([0.91, 0.20, 0.018, 0.53])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=SEQUENCE_CMAP), cax=cax)
    cb.set_label("Predefined display rank\nvisual → prefrontal", fontsize=7.5)
    cb.set_ticks([0, len(DISPLAY_SEQUENCE) - 1])
    cb.set_ticklabels(["1", str(len(DISPLAY_SEQUENCE))])
    add_title(
        fig,
        "Functional-gradient space",
        (
            "Grey: all common-valid ROIs; colour: the prespecified display sequence. "
            "Colour does not enter component selection."
        ),
    )
    save_publication_figure(fig, figure_dir / "gradient_reference_space")
    return seq


def plot_reference_sequence_bar(
    seq: pd.DataFrame,
    figure_dir: Path,
) -> None:
    x = seq["display_position"].to_numpy()
    colors = SEQUENCE_CMAP(seq["display_rank_normalized"].to_numpy())
    rho = float(spearmanr(seq["display_rank_full"], seq["G1_oriented"]).statistic)
    fig, ax = plt.subplots(figsize=(12.0, 3.65))
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.31, top=0.73)
    ax.bar(
        x,
        seq["G1_oriented"],
        width=0.78,
        color=colors,
        edgecolor="white",
        linewidth=0.35,
        zorder=2,
    )
    clean_axis(ax)
    ax.axhline(0, color=INK, linewidth=0.9, zorder=3)
    ax.set_xlim(-0.8, len(seq) - 0.2)
    ax.set_ylabel("Reference G1 score")
    ax.set_xticks(x)
    ax.set_xticklabels(seq["roi"], rotation=68, ha="right", fontsize=5.6)
    ax.text(
        0.99,
        0.96,
        f"Spearman ρ = {rho:.2f} (descriptive)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.4,
        color=INK,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": LIGHT_GRID},
    )
    add_title(
        fig,
        "Principal gradient along the prespecified visual-to-prefrontal sequence",
        "The ordering and colour are descriptive; G1 was fixed before this display was constructed.",
    )
    save_publication_figure(fig, figure_dir / "gradient_G1_predefined_sequence_bar")


def plot_stage_profile(
    stage_df: pd.DataFrame,
    args: argparse.Namespace,
    figure_dir: Path,
) -> pd.DataFrame:
    metric = "subject_G1_oriented_score"
    pivot = stage_df.pivot(index="subject", columns="stage", values=metric)
    pivot = pivot.loc[:, list(STAGE_ORDER)]
    colors = SEQUENCE_CMAP(np.linspace(0.04, 0.96, len(STAGE_ORDER)))
    summary_rows = []
    for i, stage in enumerate(STAGE_ORDER):
        values = pivot[stage].to_numpy(float)
        lo, hi = bootstrap_mean_ci(values, args.n_bootstrap, args.seed + 101 + i)
        summary_rows.append({
            "stage": stage,
            "display_label": STAGE_LABELS[stage],
            "n_subjects": int(np.isfinite(values).sum()),
            "mean_G1_oriented": float(np.nanmean(values)),
            "bootstrap_ci95_low": lo,
            "bootstrap_ci95_high": hi,
        })
    summary = pd.DataFrame(summary_rows)

    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    fig.subplots_adjust(left=0.15, right=0.97, bottom=0.19, top=0.73)
    xpos = np.arange(len(STAGE_ORDER))
    for _, row in pivot.iterrows():
        ax.plot(
            xpos,
            row.to_numpy(float),
            color="#C7CDD2",
            linewidth=0.65,
            alpha=0.55,
            zorder=1,
        )
        ax.scatter(
            xpos,
            row.to_numpy(float),
            s=12,
            facecolor="white",
            edgecolor="#9AA2AA",
            linewidth=0.55,
            alpha=0.8,
            zorder=2,
        )
    means = summary["mean_G1_oriented"].to_numpy()
    yerr = np.vstack([
        means - summary["bootstrap_ci95_low"].to_numpy(),
        summary["bootstrap_ci95_high"].to_numpy() - means,
    ])
    ax.plot(xpos, means, color=INK, linewidth=1.15, zorder=3)
    for i, (mean, color) in enumerate(zip(means, colors)):
        ax.errorbar(
            i,
            mean,
            yerr=yerr[:, i].reshape(2, 1),
            fmt="o",
            markersize=6,
            markerfacecolor=color,
            markeredgecolor=INK,
            markeredgewidth=0.7,
            ecolor=INK,
            elinewidth=1.0,
            capsize=2.5,
            zorder=4,
        )
    clean_axis(ax)
    ax.axhline(0, color="#AAB0B6", linewidth=0.65)
    ax.set_xticks(xpos)
    ax.set_xticklabels([STAGE_LABELS[s] for s in STAGE_ORDER])
    ax.set_ylabel("Oriented G1 stage score")
    add_title(
        fig,
        "Participant-level positions along the principal gradient",
        "Thin lines: participants; large markers: means; error bars: participant-bootstrap 95% CIs.",
        x=0.05,
    )
    save_publication_figure(
        fig,
        figure_dir / f"{args.reference_mode}_G1_stage_profile",
    )
    return summary


def plot_split_half_reliability(
    rel_df: pd.DataFrame,
    args: argparse.Namespace,
    figure_dir: Path,
) -> pd.DataFrame:
    order = [f"G{i}" for i in range(1, args.analysis_components + 1)]
    order.append(f"G1_G{args.analysis_components}_pairwise_distance")
    available = [x for x in order if x in set(rel_df["metric"])]
    labels = {
        **{f"G{i}": f"G{i}" for i in range(1, args.analysis_components + 1)},
        f"G1_G{args.analysis_components}_pairwise_distance": (
            f"G1–G{args.analysis_components}\nROI distance"
        ),
    }
    rng = np.random.default_rng(args.seed + 701)
    rows = []
    fig, ax = plt.subplots(figsize=(5.6, 3.75))
    fig.subplots_adjust(left=0.14, right=0.97, bottom=0.20, top=0.73)
    colors = ["#3B8FC2", "#E8C95F", "#9B6BB5", INK]
    for i, metric in enumerate(available):
        values = rel_df.loc[rel_df["metric"] == metric, "spearman_rho"].to_numpy(float)
        lo, hi = bootstrap_mean_ci(values, args.n_bootstrap, args.seed + 801 + i)
        mean = float(np.mean(values))
        rows.append({
            "metric": metric,
            "n_subjects": len(values),
            "mean_spearman_rho": mean,
            "bootstrap_ci95_low": lo,
            "bootstrap_ci95_high": hi,
        })
        jitter = rng.uniform(-0.09, 0.09, size=len(values))
        ax.scatter(
            np.full(len(values), i) + jitter,
            values,
            s=16,
            facecolor="white",
            edgecolor="#8E969E",
            linewidth=0.6,
            alpha=0.86,
            zorder=2,
        )
        ax.errorbar(
            i,
            mean,
            yerr=np.array([[mean - lo], [hi - mean]]),
            fmt="o",
            markersize=6,
            markerfacecolor=colors[i % len(colors)],
            markeredgecolor=INK,
            markeredgewidth=0.7,
            ecolor=INK,
            elinewidth=1.0,
            capsize=2.5,
            zorder=3,
        )
    clean_axis(ax)
    ax.axhline(0, color="#AAB0B6", linewidth=0.65)
    ax.axhline(0.70, color=MUTED, linestyle=(0, (3, 2)), linewidth=0.8)
    ax.text(
        0.995,
        0.70,
        "r = 0.70",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=6.5,
        color=MUTED,
    )
    ax.set_xticks(np.arange(len(available)))
    ax.set_xticklabels([labels[x] for x in available])
    ax.set_ylabel("Split-half Spearman ρ")
    add_title(
        fig,
        "Split-half reliability of aligned gradients",
        "Small points: participants; large markers: means; error bars: participant-bootstrap 95% CIs.",
        x=0.05,
    )
    save_publication_figure(
        fig,
        figure_dir / f"{args.reference_mode}_split_half_reliability",
    )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    set_publication_style()
    if args.analysis_components < 1 or args.analysis_components > args.n_components:
        raise ValueError("--analysis-components must be between 1 and --n-components")
    if args.analysis_components < 3:
        raise ValueError(
            "--analysis-components must be at least 3 because the prespecified "
            "secondary axis and publication figures use aligned G1-G3"
        )
    resolved_ts_base = args.ts_base.resolve()
    if "Step3_TimeSeries_tSNRclean_pial5pct" not in str(resolved_ts_base):
        raise ValueError(
            "Refusing a non-final input branch. --ts-base must contain "
            f"'Step3_TimeSeries_tSNRclean_pial5pct'; got {resolved_ts_base}"
        )
    stages, stage_definition_source, stage_provenance = resolve_stages(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "references").mkdir(exist_ok=True)
    (args.out_dir / "subject_gradients").mkdir(exist_ok=True)
    figure_dir = args.out_dir / "figures"
    figure_dir.mkdir(exist_ok=True)

    subjects = [str(s).replace("sub-", "") for s in args.subjects]
    loaded = {}
    canonical_names = None
    validity = {}
    input_audit_rows = []
    for sub in subjects:
        ts, names, valid, input_audit = load_subject(args, sub)
        if canonical_names is None:
            canonical_names = names
        elif names != canonical_names:
            raise ValueError(f"ROI order differs for sub-{sub}; explicit reconciliation is required")
        loaded[sub] = ts
        validity[sub] = valid
        input_audit_rows.append(input_audit)

    assert canonical_names is not None
    common = np.logical_and.reduce([validity[s] for s in subjects])
    roi_names = [r for r, keep in zip(canonical_names, common) if keep]
    if len(roi_names) < 20:
        raise RuntimeError(f"Only {len(roi_names)} ROIs are valid in all participants")
    ts_common = {s: loaded[s][common] for s in subjects}
    fcs = np.stack([fc_z(ts_common[s]) for s in subjects])

    np.save(args.out_dir / "roi_names.npy", np.asarray(roi_names, object))
    pd.DataFrame({"roi": roi_names}).to_csv(args.out_dir / "common_rois.csv", index=False)
    pd.DataFrame(input_audit_rows).to_csv(
        args.out_dir / "input_lineage_audit.csv", index=False
    )

    stage_audit_rows = []
    roi_name_set = set(canonical_names)
    common_name_set = set(roi_names)
    for stage in STAGE_ORDER:
        members = stages[stage]
        input_present = [r for r in members if r in roi_name_set]
        common_present = [r for r in members if r in common_name_set]
        missing_input = [r for r in members if r not in roi_name_set]
        missing_common = [r for r in members if r not in common_name_set]
        if missing_input or missing_common:
            raise RuntimeError(
                f"Stage {stage} is incomplete after pial5/common-valid QC. "
                f"Missing from input={missing_input}; missing from common set={missing_common}"
            )
        stage_audit_rows.append({
            "stage": stage,
            "definition_source": stage_definition_source,
            "n_defined_members": len(members),
            "defined_members": "|".join(members),
            "n_present_in_input": len(input_present),
            "n_present_common_valid": len(common_present),
            "missing_from_input": "",
            "missing_from_common_valid": "",
            "status": "PASS",
        })
    pd.DataFrame(stage_audit_rows).to_csv(
        args.out_dir / "stage_roi_audit.csv", index=False
    )
    (args.out_dir / "stage_definition_used.json").write_text(
        json.dumps({
            "definition_source": stage_definition_source,
            "definition_file": str(args.stage_definition.resolve()),
            "retained_members": {k: list(v) for k, v in stages.items()},
            "source_ts_base": str((args.ts_base / args.tag).resolve()),
            "source_provenance": stage_provenance,
        }, indent=2),
        encoding="utf-8",
    )

    full_reference = None
    if args.reference_mode == "full_sample":
        full_reference, full_lambdas = embed_reference(
            np.mean(fcs, axis=0), args.n_components, args.sparsity
        )
        np.save(
            args.out_dir / "references" / "full_sample_mean_reference.npy",
            full_reference.astype(np.float32),
        )
        full_axis, full_stage_ref = stage_axis(
            full_reference[:, : args.analysis_components], roi_names, stages
        )
        full_g1_sign, full_ref_g1_stage, _ = oriented_g1_stage_scores(
            full_reference, full_reference, roi_names, stages
        )
        np.save(
            args.out_dir / "references" / "full_sample_stage_axis.npy",
            full_axis.astype(np.float32),
        )
        reference_rows = []
        for ri, roi in enumerate(roi_names):
            row = {
                "roi": roi,
                "G1_oriented": float(full_reference[ri, 0] * full_g1_sign),
                "gradient_axis_score": float(
                    full_reference[ri, : args.analysis_components] @ full_axis
                ),
            }
            for c in range(args.analysis_components):
                row[f"G{c + 1}"] = float(full_reference[ri, c])
            reference_rows.append(row)
        reference_scores_df = pd.DataFrame(reference_rows)
        reference_scores_df.to_csv(
            args.out_dir / "reference_gradient_scores.csv", index=False
        )
        full_stage_ref.to_csv(
            args.out_dir / "reference_stage_axis_scores.csv", index=False
        )
        pd.DataFrame({
            "stage": STAGE_ORDER,
            "reference_G1_oriented_score": full_ref_g1_stage,
        }).to_csv(
            args.out_dir / "reference_G1_stage_scores.csv", index=False
        )

        # Transparent G1-G3 diagnostic only. The principal component remains G1
        # regardless of which component has the largest absolute correspondence
        # with the frozen pooled-EVC-to-cauPFC stage order.
        component_rows = []
        for c in range(args.analysis_components):
            stage_means = np.asarray([
                np.mean(full_reference[
                    np.isin(roi_names, list(stages[stage])), c
                ])
                for stage in STAGE_ORDER
            ], dtype=float)
            rho = float(
                spearmanr(np.arange(len(STAGE_ORDER)), stage_means).statistic
            )
            endpoint = float(stage_means[-1] - stage_means[0])
            sign = 1.0 if endpoint >= 0 else -1.0
            component_rows.append({
                "component": f"G{c + 1}",
                "component_index": c + 1,
                "is_prespecified_primary": c == 0,
                "eigenvalue": float(full_lambdas[c]),
                "eigenvalue_fraction_within_returned_components": float(
                    full_lambdas[c] / np.sum(full_lambdas)
                ),
                "raw_stage_order_spearman": rho,
                "absolute_stage_order_spearman": abs(rho),
                "raw_cauPFC_minus_V1": endpoint,
                "orientation_sign_for_cauPFC_above_V1": sign,
                "oriented_stage_order_spearman": rho * sign,
                **{
                    f"raw_{stage}_mean": float(stage_means[i])
                    for i, stage in enumerate(STAGE_ORDER)
                },
            })
        best_i = int(np.argmax([
            row["absolute_stage_order_spearman"] for row in component_rows
        ]))
        for i, row in enumerate(component_rows):
            row["exploratory_best_stage_match"] = i == best_i
            row["selection_rule"] = (
                "diagnostic_only; never substitutes for prespecified primary G1"
            )
        pd.DataFrame(component_rows).to_csv(
            args.out_dir / "reference_component_diagnostics.csv", index=False
        )
        pd.DataFrame({
            "component": [f"G{i}" for i in range(1, len(full_lambdas) + 1)],
            "eigenvalue": full_lambdas,
            "relative_eigenvalue_contribution": (
                full_lambdas / np.sum(full_lambdas)
            ),
            "is_prespecified_primary": (
                np.arange(len(full_lambdas)) == 0
            ),
        }).to_csv(
            args.out_dir / "reference_eigenvalues.csv", index=False
        )
        plot_reference_spectrum(full_lambdas, figure_dir)
        sequence_df = plot_reference_gradient_space(
            reference_scores_df, figure_dir
        )
        sequence_df.to_csv(
            args.out_dir / "figure_G1_predefined_sequence_data.csv", index=False
        )
        plot_reference_sequence_bar(sequence_df, figure_dir)

    grad_rows = []
    stage_rows = []
    embedding_rows = []
    g1_stage_rhos = []
    g1_endpoint_seps = []
    axis_stage_rhos = []
    axis_endpoint_seps = []
    reliability_rows = []

    for si, sub in enumerate(subjects):
        if args.reference_mode == "full_sample":
            assert full_reference is not None
            ref = full_reference
            n_reference_subjects = len(subjects)
        else:
            train = np.arange(len(subjects)) != si
            ref, _ = embed_reference(
                np.mean(fcs[train], axis=0), args.n_components, args.sparsity
            )
            n_reference_subjects = int(train.sum())
            np.save(
                args.out_dir / "references" / f"sub-{sub}_loso_reference.npy",
                ref.astype(np.float32),
            )

        native, aligned = embed_aligned(
            fcs[si], ref, args.n_components, args.sparsity
        )
        ref_analysis = ref[:, : args.analysis_components]
        aligned_analysis = aligned[:, : args.analysis_components]
        axis, stage_ref = stage_axis(ref_analysis, roi_names, stages)
        g1_sign, reference_g1_stage_scores, subject_g1_stage_scores = (
            oriented_g1_stage_scores(ref, aligned, roi_names, stages)
        )

        np.save(args.out_dir / "subject_gradients" / f"sub-{sub}.npy", aligned.astype(np.float32))
        np.save(
            args.out_dir / "subject_gradients" / f"sub-{sub}_native.npy",
            native.astype(np.float32),
        )
        embedding_rows.append({
            "subject": f"sub-{sub}",
            "reference_mode": args.reference_mode,
            "n_reference_subjects": n_reference_subjects,
            "reference_sparsity_used": args.sparsity,
            "subject_sparsity_used": args.sparsity,
            "n_common_rois": len(roi_names),
        })

        for ri, roi in enumerate(roi_names):
            row = {"subject": f"sub-{sub}", "roi": roi}
            for c in range(args.analysis_components):
                row[f"G{c + 1}"] = float(aligned[ri, c])
                row[f"reference_G{c + 1}"] = float(ref[ri, c])
            row["G1_oriented"] = float(aligned[ri, 0] * g1_sign)
            row["reference_G1_oriented"] = float(ref[ri, 0] * g1_sign)
            row["gradient_axis_score"] = float(aligned_analysis[ri] @ axis)
            grad_rows.append(row)

        subject_stage_scores = []
        for stage_order, (stage_name, (_, ref_row)) in enumerate(
            zip(STAGE_ORDER, stage_ref.iterrows())
        ):
            members = set(stages[stage_name])
            idx = np.array([r in members for r in roi_names])
            sub_score = float(np.mean(aligned_analysis[idx] @ axis))
            subject_stage_scores.append(sub_score)
            stage_rows.append({
                "subject": f"sub-{sub}", "stage": stage_name, "stage_order": stage_order,
                "n_rois": int(idx.sum()), "subject_axis_score": sub_score,
                "reference_axis_score": float(ref_row["reference_axis_score"]),
                "subject_G1_oriented_score": subject_g1_stage_scores[stage_order],
                "reference_G1_oriented_score": reference_g1_stage_scores[stage_order],
                "G1_reference_sign": g1_sign,
            })
        g1_stage_rhos.append(float(
            spearmanr(
                np.arange(len(STAGE_ORDER)), subject_g1_stage_scores
            ).statistic
        ))
        g1_endpoint_seps.append(float(
            subject_g1_stage_scores[-1] - subject_g1_stage_scores[0]
        ))
        axis_stage_rhos.append(float(
            spearmanr(np.arange(len(STAGE_ORDER)), subject_stage_scores).statistic
        ))
        axis_endpoint_seps.append(float(
            subject_stage_scores[-1] - subject_stage_scores[0]
        ))

        # One prespecified split per participant: first versus second temporal half.
        ts = ts_common[sub]
        midpoint = ts.shape[1] // 2
        _, h1 = embed_aligned(
            fc_z(ts[:, :midpoint]), ref, args.n_components, args.sparsity
        )
        _, h2 = embed_aligned(
            fc_z(ts[:, midpoint:]), ref, args.n_components, args.sparsity
        )
        h1_analysis = h1[:, : args.analysis_components]
        h2_analysis = h2[:, : args.analysis_components]
        for c in range(args.analysis_components):
            reliability_rows.append({
                "subject": f"sub-{sub}", "metric": f"G{c + 1}",
                "spearman_rho": float(spearmanr(h1[:, c], h2[:, c]).statistic),
                "half1_sparsity_used": args.sparsity,
                "half2_sparsity_used": args.sparsity,
            })
        reliability_rows.append({
            "subject": f"sub-{sub}",
            "metric": f"G1_G{args.analysis_components}_pairwise_distance",
            "spearman_rho": float(
                spearmanr(pdist(h1_analysis), pdist(h2_analysis)).statistic
            ),
            "half1_sparsity_used": args.sparsity,
            "half2_sparsity_used": args.sparsity,
        })

    grad_df = pd.DataFrame(grad_rows)
    stage_df = pd.DataFrame(stage_rows)
    rel_df = pd.DataFrame(reliability_rows)
    grad_df.to_csv(args.out_dir / "subject_functional_gradients.csv", index=False)
    stage_df.to_csv(args.out_dir / "subject_stage_axis_scores.csv", index=False)
    rel_df.to_csv(args.out_dir / "split_half_reliability_by_subject.csv", index=False)
    pd.DataFrame(embedding_rows).to_csv(args.out_dir / "embedding_audit.csv", index=False)
    stage_plot_stats = plot_stage_profile(stage_df, args, figure_dir)
    stage_plot_stats.to_csv(
        args.out_dir / "figure_G1_stage_profile_statistics.csv", index=False
    )
    reliability_plot_stats = plot_split_half_reliability(rel_df, args, figure_dir)
    reliability_plot_stats.to_csv(
        args.out_dir / "figure_split_half_reliability_statistics.csv", index=False
    )

    summaries = [
        summarize(
            g1_stage_rhos,
            f"{args.reference_mode}_primary_G1_stage_order_spearman",
            args,
        ),
        summarize(
            g1_endpoint_seps,
            f"{args.reference_mode}_primary_G1_cauPFC_minus_pooled_EVC",
            args,
        ),
        summarize(
            axis_stage_rhos,
            f"{args.reference_mode}_secondary_G1_G{args.analysis_components}_axis_stage_order_spearman",
            args,
        ),
        summarize(
            axis_endpoint_seps,
            f"{args.reference_mode}_secondary_G1_G{args.analysis_components}_axis_cauPFC_minus_pooled_EVC",
            args,
        ),
    ]
    for metric, sdf in rel_df.groupby("metric"):
        summaries.append(summarize(sdf["spearman_rho"].tolist(), f"split_half_{metric}", args))
    pd.DataFrame(summaries).to_csv(args.out_dir / "gradient_primary_statistics.csv", index=False)

    metadata = {
        "analysis": (
            "Serio-style full-sample mean reference"
            if args.reference_mode == "full_sample"
            else "LOSO reference sensitivity analysis"
        ),
        "input_semantics": "outer-pial 5% final ROI x layer x time arrays",
        "input_ts_base": str(resolved_ts_base),
        "input_tag": args.tag,
        "input_rule": (
            "all three layer time series must pass finite-data and non-constant "
            "QC; valid layers are then averaged equally"
        ),
        "subjects": [f"sub-{s}" for s in subjects],
        "n_subjects": len(subjects),
        "n_common_rois": len(roi_names),
        "n_components": args.n_components,
        "analysis_components": args.analysis_components,
        "kernel": "normalized_angle",
        "approach": "diffusion_map",
        "diffusion_alpha": 0.5,
        "diffusion_time": 0,
        "sparsity": args.sparsity,
        "reference_mode": args.reference_mode,
        "reference_definition": (
            "diffusion-map embedding of the Fisher-z FC matrix averaged across "
            "all 15 participants"
            if args.reference_mode == "full_sample"
            else "participant-specific mean FC of the other 14 participants"
        ),
        "alignment": "BrainSpace Procrustes alignment to the stated reference",
        "stage_definition_source": stage_definition_source,
        "stage_definition_file": str(args.stage_definition.resolve()),
        "stage_members": {k: list(v) for k, v in stages.items()},
        "primary_space": (
            "aligned principal gradient G1; the reference sign is oriented once "
            "so pooled EVC is lower than cauPFC"
        ),
        "secondary_space": (
            f"pooled-EVC-to-cauPFC axis in the aligned G1-G"
            f"{args.analysis_components} subspace"
        ),
        "hemisphere_handling": (
            "bilateral ROI time series supplied by the existing pipeline; "
            "hemispheres were not available as separate nodes"
        ),
        "reliability_unit": "participant",
        "split_definition": "first temporal half versus second temporal half",
    }
    (args.out_dir / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    figure_manifest = [
        {
            "figure": "gradient_reference_spectrum",
            "scope": "full_sample_only",
            "role": "diagnostic",
            "inference_unit": "none",
            "uncertainty": "not_applicable",
            "formats": "png|pdf|svg",
        },
        {
            "figure": "gradient_reference_space",
            "scope": "full_sample_only",
            "role": "descriptive_anatomical_display",
            "inference_unit": "none",
            "uncertainty": "not_applicable",
            "formats": "png|pdf|svg",
        },
        {
            "figure": "gradient_G1_predefined_sequence_bar",
            "scope": "full_sample_only",
            "role": "descriptive_ordered_display",
            "inference_unit": "none",
            "uncertainty": "not_applicable",
            "formats": "png|pdf|svg",
        },
        {
            "figure": f"{args.reference_mode}_G1_stage_profile",
            "scope": args.reference_mode,
            "role": "primary_result_support",
            "inference_unit": "participant",
            "uncertainty": "participant_bootstrap_95CI",
            "formats": "png|pdf|svg",
        },
        {
            "figure": f"{args.reference_mode}_split_half_reliability",
            "scope": args.reference_mode,
            "role": "quality_control",
            "inference_unit": "participant",
            "uncertainty": "participant_bootstrap_95CI",
            "formats": "png|pdf|svg",
        },
    ]
    if args.reference_mode != "full_sample":
        figure_manifest = [
            row for row in figure_manifest if row["scope"] != "full_sample_only"
        ]
    pd.DataFrame(figure_manifest).to_csv(
        args.out_dir / "figure_manifest.csv", index=False
    )
    print(f"Saved corrected gradient outputs: {args.out_dir}")


if __name__ == "__main__":
    main()
