#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MPC–functional-gradient alignment with a bilateral spatial null.

Why this script exists
----------------------
The previous descriptive Procrustes correlation used the held-out participant's
functional map to rotate that participant's MPC components.  That optimization
can inflate the subsequent cross-modal correlation.  In strict LOSO mode, all
rotations used for participant s are learned from the other N-1 participants:

  1. The functional gradients are the optional LOSO sensitivity outputs of
     ``gradient_bianca_reference.py --reference-mode loso``.
  2. Training-participant MPC maps are brought into a common MPC frame.
  3. A training-only MPC-to-functional rotation is estimated.
  4. The held-out MPC map is transformed with those training-only rotations and
     compared with the held-out functional map.

The script also permits a ``full_sample`` mode anchored to the original
Bianca/Serio-style full-sample mean functional reference.  This preserves the
primary functional-gradient reference, but participant s contributes to that
shared reference.  It must therefore be presented as a descriptive/secondary
cross-modal alignment rather than fully held-out validation.

Group summaries use two-sided exact sign flipping and participant bootstrap
confidence intervals.  A paired bilateral spin null is also calculated.
Because the available MPC and functional ROI values are bilateral averages,
each bilateral value is duplicated to homologous left/right parcels for the
spin.  This tests the spatial smoothness explanation without claiming
hemisphere-specific effects.

The script never falls back to the old, post-hoc functional-gradient output.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import nibabel as nib
import matplotlib

import os
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "/path/to/project")  # see config/paths.sh
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nibabel.freesurfer import read_annot
from scipy.linalg import orthogonal_procrustes
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation
from scipy.stats import spearmanr, theilslopes


ROOT = Path(PROJECT_ROOT + "/derivatives_0331")
DEFAULT_GRADIENT_LOSO = (
    ROOT / "S6_validation" / "pial5_only_update"
    / "functional_gradient_bianca_loso_pooled_evc"
)
DEFAULT_GRADIENT_FULL = (
    ROOT / "S6_validation" / "pial5_only_update"
    / "functional_gradient_bianca_reference"
)
DEFAULT_MPC_BASE = Path(PROJECT_ROOT + "/derivatives/micapipe_v0.2.0")
DEFAULT_LUT = (
    Path(PROJECT_ROOT + "/derivatives/HCPMMP_parcellation")
    / "sub-01" / "LUT_HCPMMP1.txt"
)
DEFAULT_OUT_LOSO = (
    ROOT / "S6_validation" / "pial5_only_update"
    / "mpc_functional_alignment_loso_pooled_evc"
)
DEFAULT_OUT_FULL = (
    ROOT / "S6_validation" / "pial5_only_update"
    / "mpc_functional_alignment_full_sample_reference"
)
DEFAULT_ANNOT_DIR = (
    ROOT / "Step4_MPC_G1"
    / "mpc_func_gradient_alignment_spatial_null_scanRangeExpanded_strict"
)
DEFAULT_SURF_DIR = Path(PROJECT_ROOT + "/templates/micapipe_surfaces")

# Only spelling/case aliases. HCP-MMP parcels 9-46v, a9-46v and p9-46v are
# anatomically distinct and must never be collapsed into one label.
NAME_MAP = {"7Al": "7AL", "8AD": "8Ad"}
INK = "#202124"
MUTED = "#687078"
GRID = "#DDE1E5"
PURPLE = "#6A51A3"
BLUE = "#2A6F97"
RED = "#D95F59"


def set_publication_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.8,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def clean_axis(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.65, zorder=0)
    ax.tick_params(width=0.8, length=3.5, color=INK)


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    plt.close(fig)


def format_p(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def reference_subtitle(mode: str, n_subjects: int) -> str:
    if mode == "loso":
        return (
            f"Training-only LOSO sensitivity analysis; n={n_subjects}; "
            "bars are participant-bootstrap 95% CIs"
        )
    return (
        f"Original full-sample functional reference; descriptive/secondary; "
        f"n={n_subjects}; bars are participant-bootstrap 95% CIs"
    )


def plot_participant_summary(
    subject_df: pd.DataFrame,
    stats: pd.DataFrame,
    mode: str,
    figure_dir: Path,
) -> None:
    components = stats["component"].tolist()
    x = np.arange(len(components), dtype=float)
    fig = plt.figure(figsize=(9.0, 4.25))
    grid = fig.add_gridspec(
        1, 2, width_ratios=(1.75, 1.25),
        left=0.09, right=0.98, bottom=0.18, top=0.76, wspace=0.18,
    )
    ax = fig.add_subplot(grid[0, 0])
    table_ax = fig.add_subplot(grid[0, 1])

    for i, component in enumerate(components):
        values = (
            subject_df.loc[
                subject_df["component"].eq(component), "spearman_rho"
            ]
            .to_numpy(float)
        )
        jitter = np.linspace(-0.095, 0.095, len(values))
        ax.scatter(
            np.full(len(values), x[i]) + jitter,
            values,
            s=28,
            facecolor="white",
            edgecolor=MUTED,
            linewidth=0.85,
            alpha=0.85,
            zorder=2,
        )
        row = stats.iloc[i]
        mean = float(row["mean_subject_spearman_rho"])
        lo = float(row["participant_bootstrap_ci95_low"])
        hi = float(row["participant_bootstrap_ci95_high"])
        primary = component == "G1"
        color = PURPLE if primary else BLUE
        ax.errorbar(
            x[i], mean,
            yerr=np.array([[mean - lo], [hi - mean]]),
            fmt="o",
            markersize=7,
            markerfacecolor=color if primary else "white",
            markeredgecolor=color,
            markeredgewidth=1.4,
            ecolor=color,
            elinewidth=1.5,
            capsize=3.5,
            capthick=1.2,
            zorder=4,
        )

    ax.axhline(0, color=MUTED, linewidth=0.9, linestyle=(0, (3, 2)), zorder=1)
    ax.set_xlim(-0.45, len(components) - 0.55)
    ax.set_ylim(-1.02, 1.02)
    ax.set_xticks(x, components)
    ax.set_ylabel("Participant Spearman $\\rho$")
    ax.set_xlabel("Functional-gradient component")
    clean_axis(ax, grid_axis="y")
    ax.text(
        -0.16, 1.04, "a", transform=ax.transAxes,
        fontsize=12, fontweight="bold", va="bottom",
    )

    table_ax.axis("off")
    table_ax.text(
        0.0, 1.04, "b", transform=table_ax.transAxes,
        fontsize=12, fontweight="bold", va="bottom",
    )
    table_ax.text(
        0.08, 0.97, "Effect-size and inference summary",
        transform=table_ax.transAxes, fontsize=9.5, fontweight="bold", va="top",
    )
    cell_rows = []
    for _, row in stats.iterrows():
        ci = (
            f"{row['mean_subject_spearman_rho']:+.2f} "
            f"[{row['participant_bootstrap_ci95_low']:+.2f}, "
            f"{row['participant_bootstrap_ci95_high']:+.2f}]"
        )
        cell_rows.append([
            row["component"],
            ci,
            format_p(float(row["holm_p_across_3_components"])),
            format_p(float(row["holm_spin_p_across_3_components"])),
        ])
    table = table_ax.table(
        cellText=cell_rows,
        colLabels=("Comp.", "Mean $\\rho$ [95% CI]", "Holm $P$", "Spin Holm $P$"),
        cellLoc="center",
        colLoc="center",
        colWidths=(0.15, 0.39, 0.21, 0.25),
        bbox=(0.02, 0.36, 0.98, 0.47),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.3)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.55)
        cell.set_facecolor("#F4F1FA" if row == 1 else "white")
        if row == 0:
            cell.set_text_props(fontweight="bold", color=INK)
            cell.set_facecolor("#F2F3F4")
    table_ax.text(
        0.04, 0.25,
        "Participant inference: two-sided exact sign flipping.\n"
        "Spatial inference: paired mirrored bilateral spin.\n"
        "Holm correction is applied across G1–G3.",
        transform=table_ax.transAxes, color=MUTED, fontsize=7.5,
        va="top", linespacing=1.5,
    )

    fig.suptitle(
        "Participant-level MPC–functional gradient alignment",
        x=0.06, y=0.95, ha="left", fontsize=15, fontweight="bold", color=INK,
    )
    fig.text(
        0.06, 0.885, reference_subtitle(mode, subject_df["subject"].nunique()),
        ha="left", va="top", fontsize=8.5, color=MUTED,
    )
    save_figure(
        fig,
        figure_dir / "mpc_functional_participant_alignment",
    )


def plot_group_map_correspondence(
    group_pred: np.ndarray,
    group_func: np.ndarray,
    roi_names: list[str],
    stats: pd.DataFrame,
    mode: str,
    figure_dir: Path,
) -> pd.DataFrame:
    components = stats["component"].tolist()
    n_components = len(components)
    all_values = np.concatenate([group_pred.ravel(), group_func.ravel()])
    bound = max(0.25, float(np.nanmax(np.abs(all_values))) * 1.08)
    norm = matplotlib.colors.Normalize(vmin=-bound, vmax=bound)
    cmap = matplotlib.colormaps["RdBu_r"]
    fig = plt.figure(figsize=(10.0, 4.25))
    grid = fig.add_gridspec(
        1, n_components + 1,
        width_ratios=([1.0] * n_components) + [0.055],
        left=0.08, right=0.95, bottom=0.19, top=0.74, wspace=0.28,
    )
    axes = [fig.add_subplot(grid[0, i]) for i in range(n_components)]
    color_ax = fig.add_subplot(grid[0, -1])
    rows = []
    anchors = {"V1", "V2", "V3", "7PC", "FEF", "55b", "SFL"}

    for i, (component, ax) in enumerate(zip(components, axes)):
        xx = np.asarray(group_func[:, i], float)
        yy = np.asarray(group_pred[:, i], float)
        colors = cmap(norm(xx))
        ax.scatter(
            xx, yy, s=30, c=colors, edgecolor="white",
            linewidth=0.65, alpha=0.88, zorder=3,
        )
        ax.plot(
            [-bound, bound], [-bound, bound],
            color=GRID, linewidth=0.9, linestyle=(0, (3, 2)), zorder=1,
        )
        if np.ptp(xx) > 1e-10:
            slope, intercept, _, _ = theilslopes(yy, xx, alpha=0.95)
            guide_x = np.array([-bound, bound])
            ax.plot(
                guide_x, intercept + slope * guide_x,
                color=INK, linewidth=1.15, zorder=2,
            )
        for roi, xv, yv in zip(roi_names, xx, yy):
            rows.append({
                "component": component,
                "roi": roi,
                "functional_gradient_score": xv,
                "predicted_mpc_score": yv,
            })
            if roi in anchors:
                ax.annotate(
                    roi, (xv, yv), xytext=(3, 3),
                    textcoords="offset points", fontsize=6.2,
                    color=INK, clip_on=True,
                )
        row = stats.iloc[i]
        ax.text(
            0.04, 0.96,
            (
                f"$\\rho$={row['group_map_spearman_rho']:+.2f}\n"
                f"spin $P$={format_p(float(row['paired_bilateral_spin_p_two_sided']))}\n"
                f"Holm $P$={format_p(float(row['holm_spin_p_across_3_components']))}"
            ),
            transform=ax.transAxes, va="top", fontsize=7.6,
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": GRID,
                "alpha": 0.92,
            },
        )
        ax.set_title(component, fontweight="bold", pad=7)
        ax.set_xlim(-bound, bound)
        ax.set_ylim(-bound, bound)
        ax.set_aspect("equal", adjustable="box")
        clean_axis(ax, grid_axis="both")
        ax.text(
            -0.18, 1.04, chr(ord("a") + i), transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="bottom",
        )

    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar = fig.colorbar(sm, cax=color_ax)
    colorbar.set_label("Functional-gradient score", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)
    fig.supxlabel("Group-mean functional-gradient score", y=0.08, fontsize=9)
    fig.supylabel("Group-mean MPC prediction", x=0.025, fontsize=9)
    fig.suptitle(
        "Group-map MPC–functional correspondence",
        x=0.06, y=0.95, ha="left", fontsize=15, fontweight="bold", color=INK,
    )
    status = (
        "LOSO sensitivity"
        if mode == "loso"
        else "full-sample functional reference; descriptive/secondary"
    )
    fig.text(
        0.06, 0.885,
        (
            f"{status}. Points are bilateral ROIs; solid lines are Theil–Sen "
            "guides; inference uses paired mirrored bilateral spins."
        ),
        ha="left", va="top", fontsize=8.5, color=MUTED,
    )
    save_figure(
        fig,
        figure_dir / "mpc_functional_group_map_correspondence",
    )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--functional-reference-mode",
        choices=("loso", "full_sample"),
        default="loso",
        help=(
            "loso is the strict cross-validated sensitivity analysis; "
            "full_sample uses the original Bianca/Serio mean reference and "
            "must be interpreted as descriptive/secondary"
        ),
    )
    p.add_argument(
        "--gradient-dir",
        type=Path,
        default=None,
        help="Defaults to the matching corrected functional-gradient output.",
    )
    p.add_argument("--mpc-base", type=Path, default=DEFAULT_MPC_BASE)
    p.add_argument("--lut-path", type=Path, default=DEFAULT_LUT)
    p.add_argument("--acq-tag", default="acq-T1map")
    p.add_argument("--atlas-tag", default="glasser-360")
    p.add_argument("--mpc-sparsity", type=float, default=0.0)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Defaults to a mode-specific output directory.",
    )
    p.add_argument("--annot-lh", type=Path, default=DEFAULT_ANNOT_DIR / "lh.HCPMMP1_fsLR32k.annot")
    p.add_argument("--annot-rh", type=Path, default=DEFAULT_ANNOT_DIR / "rh.HCPMMP1_fsLR32k.annot")
    p.add_argument("--sphere-lh", type=Path, default=DEFAULT_SURF_DIR / "fsLR-32k.L.sphere.surf.gii")
    p.add_argument("--sphere-rh", type=Path, default=DEFAULT_SURF_DIR / "fsLR-32k.R.sphere.surf.gii")
    p.add_argument("--n-components", type=int, default=3)
    p.add_argument("--n-bootstrap", type=int, default=10000)
    p.add_argument("--n-spin", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20260723)
    args = p.parse_args()
    if args.gradient_dir is None:
        args.gradient_dir = (
            DEFAULT_GRADIENT_LOSO
            if args.functional_reference_mode == "loso"
            else DEFAULT_GRADIENT_FULL
        )
    if args.out_dir is None:
        args.out_dir = (
            DEFAULT_OUT_LOSO
            if args.functional_reference_mode == "loso"
            else DEFAULT_OUT_FULL
        )
    return args


def norm_roi(x: object) -> str:
    s = str(x).strip()
    for prefix in ("L_", "R_", "lh_", "rh_"):
        if s.startswith(prefix):
            s = s[len(prefix):]
    for suffix in ("_ROI", "-ROI"):
        if s.endswith(suffix):
            s = s[:-len(suffix)]
    return NAME_MAP.get(s, s)


def zcols(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, float)
    mu = np.nanmean(a, axis=0, keepdims=True)
    sd = np.nanstd(a, axis=0, ddof=1, keepdims=True)
    if np.any(~np.isfinite(sd)) or np.any(sd < 1e-10):
        raise ValueError("A gradient component is degenerate")
    return (a - mu) / sd


def align(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    src = zcols(source)
    tar = zcols(target)
    rot, _ = orthogonal_procrustes(src, tar)
    return zcols(src @ rot), rot


def exact_signflip_p(x: np.ndarray) -> float:
    values = np.asarray(x, float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    obs = abs(float(np.mean(values)))
    extreme = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        stat = abs(float(np.mean(values * np.asarray(signs))))
        extreme += stat >= obs - 1e-15
        total += 1
    return extreme / total


def bootstrap_mean_ci(x: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    values = np.asarray(x, float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(values, (n_boot, len(values)), replace=True), axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def holm(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, float)
    out = np.full(p.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if not len(valid):
        return out
    order = valid[np.argsort(p[valid])]
    adjusted = np.maximum.accumulate(
        np.array([(len(order) - i) * p[idx] for i, idx in enumerate(order)])
    )
    adjusted = np.minimum(adjusted, 1.0)
    out[order] = adjusted
    return out


def parse_lut(lut_file: Path) -> tuple[dict[str, int], dict[str, int]]:
    if not lut_file.exists():
        raise FileNotFoundError(f"Missing HCP-MMP LUT: {lut_file}")
    records = []
    for line in lut_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) < 2 or line.lstrip().startswith("#"):
            continue
        try:
            records.append((int(parts[0]), parts[1]))
        except ValueError:
            continue
    left, right = {}, {}
    left_base = right_base = None
    for idx, name in records:
        if name.startswith("lh.L_") and name.endswith("_ROI.label"):
            roi = norm_roi(name[5:-10])
            left_base = idx if left_base is None else left_base
            left[roi] = idx - left_base
        elif name.startswith("rh.R_") and name.endswith("_ROI.label"):
            roi = norm_roi(name[5:-10])
            right_base = idx if right_base is None else right_base
            right[roi] = idx - right_base + 180
    if len(left) < 150 or len(right) < 150:
        raise RuntimeError(
            f"Could not parse a bilateral Glasser-360 LUT from {lut_file}: "
            f"L={len(left)}, R={len(right)}"
        )
    return left, right


def gifti_array(path: Path) -> np.ndarray:
    gii = nib.load(str(path))
    return np.squeeze(np.array([np.asarray(d.data) for d in gii.darrays]))


def load_raw_mpc(args: argparse.Namespace, sub: str) -> tuple[np.ndarray, dict[str, object]]:
    path = (
        args.mpc_base / f"sub-{sub}" / "mpc" / args.acq_tag
        / f"sub-{sub}_atlas-{args.atlas_tag}_desc-MPC.shape.gii"
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing raw micapipe MPC matrix: {path}")
    raw = np.asarray(gifti_array(path), float)
    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        flat = raw.ravel()
        n = int(round(np.sqrt(flat.size)))
        if n * n != flat.size:
            raise ValueError(f"Cannot interpret MPC GIFTI as square: {path}, {raw.shape}")
        raw = flat.reshape(n, n)

    upper = np.triu(raw, 1)
    lower = np.tril(raw, -1)
    denom = max(float(np.nanmax(np.abs(raw))), 1.0)
    symmetry_error = float(np.nanmax(np.abs(raw - raw.T)) / denom)
    n_upper = int(np.count_nonzero(np.isfinite(upper) & (np.abs(upper) > 0)))
    n_lower = int(np.count_nonzero(np.isfinite(lower) & (np.abs(lower) > 0)))
    if symmetry_error < 1e-6 and n_upper > 0 and n_lower > 0:
        matrix = (raw + raw.T) / 2
        reconstruction = "already_symmetric_average"
    elif n_upper >= n_lower:
        matrix = upper + upper.T
        reconstruction = "upper_triangle_mirrored"
    else:
        matrix = lower + lower.T
        reconstruction = "lower_triangle_mirrored"
    if matrix.shape == (362, 362):
        matrix = np.delete(np.delete(matrix, 0, axis=0), 0, axis=1)
        matrix = np.delete(np.delete(matrix, 180, axis=0), 180, axis=1)
        removed_background = True
    elif matrix.shape == (360, 360):
        removed_background = False
    else:
        raise ValueError(f"Unexpected raw MPC matrix size: {path}, {matrix.shape}")
    matrix = np.asarray(matrix, float)
    eps = np.finfo(float).eps
    matrix[~np.isfinite(matrix)] = eps
    matrix[matrix == 0] = eps
    matrix = (matrix + matrix.T) / 2
    return matrix, {
        "subject": f"sub-{sub}", "file": str(path),
        "raw_shape": str(tuple(raw.shape)),
        "final_shape": str(tuple(matrix.shape)),
        "reconstruction": reconstruction,
        "symmetry_error_raw": symmetry_error,
        "removed_background": removed_background,
    }


def embed_mpc(
    matrix: np.ndarray,
    roi_parcels: list[tuple[int, int]],
    n_components: int,
    sparsity: float,
) -> np.ndarray:
    try:
        from brainspace.gradient import GradientMaps
    except ImportError as exc:
        raise ImportError("BrainSpace is required: pip install brainspace") from exc
    gm = GradientMaps(
        kernel="normalized_angle", approach="dm",
        n_components=max(n_components, 10), random_state=0,
    )
    gm.fit(
        np.asarray(matrix, float),
        sparsity=sparsity,
        alpha=0.5,
        diffusion_time=0,
    )
    parcel = np.asarray(gm.gradients_, float)
    roi = np.full((len(roi_parcels), n_components), np.nan)
    for i, (li, ri) in enumerate(roi_parcels):
        vals = parcel[[li, ri], :n_components]
        roi[i] = np.nanmean(vals, axis=0)
    if not np.isfinite(roi).all():
        raise RuntimeError("Non-finite ROI MPC gradients after bilateral averaging")
    return zcols(roi)


def load_inputs(args: argparse.Namespace):
    grad_names_file = args.gradient_dir / "roi_names.npy"
    metadata_file = args.gradient_dir / "analysis_metadata.json"
    if not grad_names_file.exists() or not metadata_file.exists():
        raise FileNotFoundError(
            "Run gradient_bianca_reference.py with the requested reference "
            "mode first: "
            f"missing {grad_names_file} or {metadata_file}"
        )
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    saved_mode = metadata.get("reference_mode")
    if saved_mode != args.functional_reference_mode:
        raise ValueError(
            "Functional-gradient reference-mode mismatch:\n"
            f"  requested: {args.functional_reference_mode}\n"
            f"  metadata:  {saved_mode}\n"
            f"  directory: {args.gradient_dir}"
        )
    grad_names = [norm_roi(x) for x in np.load(grad_names_file, allow_pickle=True)]
    subjects = [
        str(x).replace("sub-", "")
        for x in metadata["subjects"]
    ]
    if len(set(grad_names)) != len(grad_names):
        raise ValueError("Duplicated ROI names after normalization")
    lh, rh = parse_lut(args.lut_path)
    common = [x for x in grad_names if x in lh and x in rh]
    if len(common) < 20:
        raise RuntimeError(f"Only {len(common)} functional-gradient ROIs map bilaterally to MPC parcels")
    gi = np.array([grad_names.index(x) for x in common])
    roi_parcels = [(lh[x], rh[x]) for x in common]
    raw_mpc = []
    mpc_qc = []
    for sub in subjects:
        matrix, row = load_raw_mpc(args, sub)
        raw_mpc.append(matrix)
        mpc_qc.append(row)

    func = []
    refs = []
    shared_reference_file = (
        args.gradient_dir / "references" / "full_sample_mean_reference.npy"
    )
    if (
        args.functional_reference_mode == "full_sample"
        and not shared_reference_file.exists()
    ):
        raise FileNotFoundError(
            "Missing corrected full-sample Bianca reference: "
            f"{shared_reference_file}"
        )
    for sub in subjects:
        sf = args.gradient_dir / "subject_gradients" / f"sub-{sub}.npy"
        rf = (
            args.gradient_dir / "references" / f"sub-{sub}_loso_reference.npy"
            if args.functional_reference_mode == "loso"
            else shared_reference_file
        )
        if not sf.exists() or not rf.exists():
            raise FileNotFoundError(
                f"Missing {args.functional_reference_mode} functional "
                f"gradient/reference for sub-{sub}: {sf} | {rf}"
            )
        func.append(np.asarray(np.load(sf), float)[gi, :args.n_components])
        refs.append(np.asarray(np.load(rf), float)[gi, :args.n_components])
    return (
        subjects, common, roi_parcels, np.stack(raw_mpc),
        np.stack(func), np.stack(refs), pd.DataFrame(mpc_qc),
    )


def annot_centroids(annot_file: Path, sphere_file: Path) -> dict[str, np.ndarray]:
    if not annot_file.exists() or not sphere_file.exists():
        raise FileNotFoundError(f"Missing annotation/sphere: {annot_file} | {sphere_file}")
    labels, _, names = read_annot(str(annot_file))
    coords = np.asarray(nib.load(str(sphere_file)).darrays[0].data, float)
    if len(labels) != len(coords):
        raise ValueError(f"Annotation and sphere vertex counts differ: {annot_file}")
    out = {}
    for label_idx, raw in enumerate(names):
        name = norm_roi(raw.decode() if isinstance(raw, bytes) else raw)
        if name.lower() in {"unknown", "medial_wall", "medialwall", "???"}:
            continue
        v = coords[labels == label_idx]
        if len(v):
            c = np.mean(v, axis=0)
            n = np.linalg.norm(c)
            if n > 0:
                out[name] = c / n
    return out


def bilateral_spin_p(
    x: np.ndarray,
    y: np.ndarray,
    roi_names: list[str],
    lh_centroids: dict[str, np.ndarray],
    rh_centroids: dict[str, np.ndarray],
    n_spin: int,
    seed: int,
) -> tuple[float, float, int]:
    usable = [i for i, r in enumerate(roi_names) if r in lh_centroids and r in rh_centroids]
    if len(usable) < 20:
        raise RuntimeError(
            f"Only {len(usable)} common ROIs have bilateral centroids; "
            "check annotation names and sphere files"
        )
    names = [roi_names[i] for i in usable]
    xx = np.concatenate([np.asarray(x)[usable], np.asarray(x)[usable]])
    yy_l = np.asarray(y)[usable]
    yy_r = np.asarray(y)[usable]
    lh = np.vstack([lh_centroids[r] for r in names])
    rh = np.vstack([rh_centroids[r] for r in names])
    observed = float(spearmanr(xx, np.concatenate([yy_l, yy_r])).statistic)

    rng = np.random.default_rng(seed)
    mirror = np.diag([-1.0, 1.0, 1.0])
    extreme = 0
    valid = 0
    for _ in range(n_spin):
        rot_l = Rotation.random(random_state=rng).as_matrix()
        rot_r = mirror @ rot_l @ mirror
        lh_rot = lh @ rot_l.T
        rh_rot = rh @ rot_r.T
        # Hungarian assignment preserves a one-to-one parcel permutation.
        row_l, col_l = linear_sum_assignment(-lh_rot @ lh.T)
        row_r, col_r = linear_sum_assignment(-rh_rot @ rh.T)
        perm_l = np.empty(len(names), int)
        perm_r = np.empty(len(names), int)
        perm_l[row_l] = col_l
        perm_r[row_r] = col_r
        null_rho = float(spearmanr(xx, np.concatenate([yy_l[perm_l], yy_r[perm_r]])).statistic)
        if np.isfinite(null_rho):
            valid += 1
            extreme += abs(null_rho) >= abs(observed) - 1e-15
    # Add-one convention protects against a zero Monte-Carlo p value.
    p = (extreme + 1) / (valid + 1)
    return observed, float(p), len(usable)


def main() -> None:
    args = parse_args()
    set_publication_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = args.out_dir / "figures"
    (
        subjects, roi_names, roi_parcels, raw_mpc,
        func, func_refs, mpc_qc,
    ) = load_inputs(args)
    n_sub = len(subjects)
    if n_sub < 3:
        raise RuntimeError("At least three participants are needed for alignment")
    mpc_qc.to_csv(args.out_dir / "raw_mpc_loading_qc.csv", index=False)

    rows = []
    predicted_mpc = np.full_like(func, np.nan)
    for si, sub in enumerate(subjects):
        train = np.arange(n_sub) != si
        # Both embeddings below are recomputed from raw MPC matrices.  The
        # held-out participant therefore never contributes to the MPC reference.
        mpc_ref = embed_mpc(
            np.mean(raw_mpc[train], axis=0), roi_parcels,
            args.n_components, args.mpc_sparsity,
        )
        heldout_native = embed_mpc(
            raw_mpc[si], roi_parcels,
            args.n_components, args.mpc_sparsity,
        )

        # The MPC anchor is training-only in both modes.  The functional anchor
        # is training-only in LOSO mode and is the shared Bianca/Serio
        # full-sample reference in full_sample mode.
        func_ref = zcols(func_refs[si])
        mpc_ref_to_func, cross_rot = align(mpc_ref, func_ref)
        del mpc_ref_to_func

        heldout_mpc, heldout_rot = align(heldout_native, mpc_ref)
        prediction = zcols(heldout_mpc @ cross_rot)
        predicted_mpc[si] = prediction

        for c in range(args.n_components):
            rho = float(spearmanr(prediction[:, c], func[si, :, c]).statistic)
            rows.append({
                "subject": f"sub-{sub}",
                "component": f"G{c + 1}",
                "n_rois": len(roi_names),
                "spearman_rho": rho,
            })
        np.save(args.out_dir / f"sub-{sub}_heldout_mpc_prediction.npy", prediction.astype(np.float32))
        np.save(args.out_dir / f"sub-{sub}_mpc_to_train_rotation.npy", heldout_rot)
        np.save(args.out_dir / f"sub-{sub}_train_crossmodal_rotation.npy", cross_rot)

    subject_df = pd.DataFrame(rows)
    subject_df.to_csv(
        args.out_dir / "subject_mpc_functional_alignment.csv", index=False
    )
    if args.functional_reference_mode == "loso":
        subject_df.to_csv(
            args.out_dir / "subject_crossvalidated_mpc_functional_alignment.csv",
            index=False,
        )
    np.save(
        args.out_dir / "mpc_predictions_common_functional_frame.npy",
        predicted_mpc.astype(np.float32),
    )
    if args.functional_reference_mode == "loso":
        np.save(
            args.out_dir / "crossvalidated_mpc_predictions.npy",
            predicted_mpc.astype(np.float32),
        )
    np.save(
        args.out_dir
        / f"functional_gradients_{args.functional_reference_mode}_common_rois.npy",
        func.astype(np.float32),
    )
    np.save(args.out_dir / "roi_names.npy", np.asarray(roi_names, object))

    stats_rows = []
    for c, sdf in subject_df.groupby("component", sort=True):
        values = sdf["spearman_rho"].to_numpy(float)
        lo, hi = bootstrap_mean_ci(values, args.n_bootstrap, args.seed + int(c[1:]))
        stats_rows.append({
            "component": c,
            "n_subjects": len(values),
            "mean_subject_spearman_rho": float(np.mean(values)),
            "sd_subject_spearman_rho": float(np.std(values, ddof=1)),
            "participant_bootstrap_ci95_low": lo,
            "participant_bootstrap_ci95_high": hi,
            "exact_signflip_p_two_sided": exact_signflip_p(values),
        })
    stats_df = pd.DataFrame(stats_rows)
    stats_df["holm_p_across_3_components"] = holm(
        stats_df["exact_signflip_p_two_sided"].to_numpy(float)
    )

    # Spatial null for cross-validated group maps.  This is deliberately
    # secondary to participant-level inference.
    lh = annot_centroids(args.annot_lh, args.sphere_lh)
    rh = annot_centroids(args.annot_rh, args.sphere_rh)
    group_pred = np.mean(predicted_mpc, axis=0)
    group_func = np.mean(func, axis=0)
    spin_rows = []
    for c in range(args.n_components):
        obs, p_spin, n_roi = bilateral_spin_p(
            group_pred[:, c], group_func[:, c], roi_names, lh, rh,
            args.n_spin, args.seed + 100 + c,
        )
        spin_rows.append({
            "component": f"G{c + 1}",
            "group_map_spearman_rho": obs,
            "paired_bilateral_spin_p_two_sided": p_spin,
            "n_bilateral_rois": n_roi,
            "n_spins": args.n_spin,
        })
    spin_df = pd.DataFrame(spin_rows)
    spin_df["holm_spin_p_across_3_components"] = holm(
        spin_df["paired_bilateral_spin_p_two_sided"].to_numpy(float)
    )
    spin_df.to_csv(args.out_dir / "group_map_bilateral_spin_statistics.csv", index=False)
    combined_stats = stats_df.merge(spin_df, on="component", how="left")
    combined_stats["analysis_role"] = np.where(
        combined_stats["component"].eq("G1"),
        "prespecified_primary",
        "exploratory",
    )
    combined_stats.to_csv(
        args.out_dir / "mpc_functional_alignment_primary_statistics.csv", index=False
    )
    group_map_df = plot_group_map_correspondence(
        group_pred, group_func, roi_names, combined_stats,
        args.functional_reference_mode, figure_dir,
    )
    group_map_df.to_csv(
        args.out_dir / "group_map_alignment_values.csv", index=False
    )
    plot_participant_summary(
        subject_df, combined_stats, args.functional_reference_mode, figure_dir
    )
    figure_manifest = pd.DataFrame([
        {
            "figure": "mpc_functional_participant_alignment",
            "role": "G1 prespecified primary; G2/G3 exploratory component summary",
            "uncertainty": "participant-bootstrap 95% CI",
            "inference": (
                "two-sided exact sign flipping; Holm across G1-G3; "
                "companion bilateral-spin Holm P shown"
            ),
            "formats": "png|pdf|svg",
        },
        {
            "figure": "mpc_functional_group_map_correspondence",
            "role": "group-map spatial correspondence",
            "uncertainty": "not applicable to individual ROI points",
            "inference": (
                "paired mirrored bilateral spin; Holm across G1-G3; "
                "Theil-Sen line is a visual guide"
            ),
            "formats": "png|pdf|svg",
        },
    ])
    figure_manifest.to_csv(args.out_dir / "figure_manifest.csv", index=False)

    metadata = {
        "analysis": (
            "training-only LOSO MPC-to-functional alignment"
            if args.functional_reference_mode == "loso"
            else (
                "training-only MPC alignment anchored to the original "
                "Bianca/Serio full-sample functional reference"
            )
        ),
        "functional_reference_mode": args.functional_reference_mode,
        "inferential_status": (
            "cross-validated sensitivity analysis"
            if args.functional_reference_mode == "loso"
            else (
                "descriptive/secondary; not fully held-out because every "
                "participant contributes to the shared functional reference"
            )
        ),
        "functional_input": str(args.gradient_dir),
        "mpc_input": str(args.mpc_base),
        "mpc_source": (
            "raw subject micapipe MPC matrices; the held-out participant is "
            "excluded from the MPC reference in every fold"
        ),
        "subjects": [f"sub-{s}" for s in subjects],
        "n_subjects": n_sub,
        "n_common_rois": len(roi_names),
        "inference_unit": "participant",
        "participant_inference": "two-sided exact sign flipping of participant Spearman rho",
        "uncertainty": "participant bootstrap 95% confidence interval",
        "spatial_null": "paired mirrored bilateral spins with one-to-one Hungarian reassignment",
        "bilateral_limitation": (
            "Input ROI values are bilateral averages; values are duplicated to "
            "homologous left/right parcels for the spatial null. No hemisphere-"
            "specific effect is claimed."
        ),
        "multiplicity": "Holm family-wise correction across G1-G3",
        "component_roles": {
            "G1": "prespecified_primary",
            "G2": "exploratory",
            "G3": "exploratory",
        },
        "figures": [
            "figures/mpc_functional_participant_alignment.png|pdf|svg",
            "figures/mpc_functional_group_map_correspondence.png|pdf|svg",
        ],
    }
    (args.out_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Saved corrected MPC-functional outputs: {args.out_dir}")


if __name__ == "__main__":
    main()
