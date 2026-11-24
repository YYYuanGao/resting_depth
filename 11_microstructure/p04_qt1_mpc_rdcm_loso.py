#!/usr/bin/env python3
"""
Layer-resolved qT1/MPC associations with the final outer-pial-5% EVC-cauPFC
rDCM.

Only the three layer-specific qT1 central-value columns are analysed:
``intensity_Deep``, ``intensity_Middle`` and ``intensity_Superficial``.

2026-11-10 (depthfix): the qT1 input is now produced by
compute_mpc_gradient_only_micapipe3.py. In the earlier input
(compute_mpc_gradient_only_micapipe2.py) the micapipe profile rows were
labelled in reverse: rows 0-4 lie on the pial side but were called "Deep".
v3 counts surfaces from the pial side and uses thirds of cortical thickness
(Superficial = surfaces 1-5, Middle = 6-10, Deep = 11-14). The analysis code
below is unchanged; only the input and output locations differ.
Depth differences, whole-depth skewness and layer-wise skewness are
deliberately excluded.

The analysis has three frozen families:

1. Four-test primary structure family (Holm corrected)
   - EVC superficial qT1-derived value versus feedforward (FF).
   - cauPFC deep qT1-derived value versus feedback (FB).
   - within-hemisphere EVC-cauPFC MPC versus FF.
   - within-hemisphere EVC-cauPFC MPC versus FB.

2. Four-test qT1 layer-diagnostic family (Holm corrected)
   - FF versus EVC deep and middle qT1-derived values.
   - FB versus cauPFC middle and superficial qT1-derived values.

3. One directional diagnostic
   - signed H versus EVC-cauPFC MPC.

The six qT1 results are also written to one layer-resolved table, with each
row retaining the correction family in which it was tested.

Definitions
-----------
EVC:    equal-ROI composite of V1, V2 and V3.
cauPFC: equal-ROI composite of 6d, 6v, 6mp, FEF, PEF, 55b and SFL.

The rDCM A matrix follows the SPM convention A(target, source):

FF = mean(EVC superficial to cauPFC deep,
          EVC superficial to cauPFC middle)

FB = mean(cauPFC deep to EVC deep,
          cauPFC deep to EVC superficial)

H = FB - FF

Thus, signed H and signed (FB - FF) are exactly the same derived contrast.
H is not a third independently estimated pathway.  This final script uses only
the signed FF, signed FB and signed H metrics.  It deliberately does not use
the historically ambiguous label ``absolute_H``.

MPC is computed directly from the raw micapipe MPC matrix.  The primary
bilateral feature is the equal-weight mean of the 3 x 7 EVC-cauPFC parcel
pairs within each hemisphere, followed by an equal mean across hemispheres.
Cross-hemisphere and all-four-combination summaries are saved for QC only.

The script never reads functional-gradient values.  This guarantees that a
qT1/MPC-rDCM association is structurally defined and is not circular with the
subsequent functional-gradient analysis.

Reviewer-facing leave-one-subject-out (LOSO) outputs are descriptive influence
analyses of the participant-level correlations.  The rDCM models and
structural features are not re-estimated within each LOSO deletion.  Formal
inference remains based on the complete N=15 sample.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

import os
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "/path/to/project")  # see config/paths.sh


DERIV = Path(PROJECT_ROOT + "/derivatives_0331")
SCRIPT_VERSION = "2026-11-10-depthfix-v2"
DEFAULT_RDCM_LONG = (
    DERIV
    / "S6_validation"
    / "pial5_only_update"
    / "rdcm_long"
    / "pial5_current"
    / "merged_layer_rdcm_connections_long.csv"
)
DEFAULT_QT1_LONG = (
    DERIV
    / "Step4_MPC_G1"
    / "Nov"
    / "mpc_gradient_only_micapipe3"
    / "subject_roi_intensity_moments_long.csv"
)
DEFAULT_MPC_BASE = Path(
    PROJECT_ROOT + "/derivatives/micapipe_v0.2.0"
)
DEFAULT_LUT = (
    Path(PROJECT_ROOT + "/derivatives/HCPMMP_parcellation")
    / "sub-01"
    / "LUT_HCPMMP1.txt"
)
DEFAULT_OUT = (
    DERIV
    / "Step4_MPC_G1"
    / "Nov"
    / "p04_qt1_mpc_rdcm_loso"
)

SUBJECTS = (
    "sub-01", "sub-02", "sub-03", "sub-04", "sub-05",
    "sub-06", "sub-07", "sub-08", "sub-09", "sub-10",
    "sub-11", "sub-12", "sub-13", "sub-14", "sub-16",
)
EVC_ROIS = ("V1", "V2", "V3")
CAUPFC_ROIS = ("6d", "6v", "6mp", "FEF", "PEF", "55b", "SFL")

REQUIRED_RDCM_COLUMNS = {
    "subject", "network", "source_group", "target_group",
    "source_layer", "target_layer", "A",
}
REQUIRED_QT1_COLUMNS = {
    "subject", "roi", "intensity_Deep", "intensity_Middle",
    "intensity_Superficial",
}
QT1_LAYER_METRICS = (
    "intensity_Deep", "intensity_Middle", "intensity_Superficial",
)
QT1_LAYER_LABELS = {
    "intensity_Deep": "deep",
    "intensity_Middle": "middle",
    "intensity_Superficial": "superficial",
}

INK = "#22252A"
MUTED = "#6B7178"
GRID = "#E2E5E9"
BLUE = "#1C6E9A"
VERMILLION = "#D65F3C"
PURPLE = "#6F559B"
TEAL = "#2A9D8F"
PALE_RANGE = "#E7EAED"

TEST_IDS = {
    "FF vs EVC superficial qT1": "qt1_ff_evc_superficial",
    "FB vs cauPFC deep qT1": "qt1_fb_caupfc_deep",
    "FF vs EVC-cauPFC MPC": "mpc_ff_evc_caupfc",
    "FB vs EVC-cauPFC MPC": "mpc_fb_evc_caupfc",
    "FF vs EVC deep qT1": "qt1_ff_evc_deep",
    "FF vs EVC middle qT1": "qt1_ff_evc_middle",
    "FB vs cauPFC middle qT1": "qt1_fb_caupfc_middle",
    "FB vs cauPFC superficial qT1": "qt1_fb_caupfc_superficial",
    "signed_H vs EVC-cauPFC MPC": "mpc_h_evc_caupfc",
}

DISPLAY_LABELS = {
    "FF vs EVC superficial qT1": "qT1: FF vs EVC superficial",
    "FB vs cauPFC deep qT1": "qT1: FB vs cauPFC deep",
    "FF vs EVC-cauPFC MPC": "MPC: FF vs EVC–cauPFC",
    "FB vs EVC-cauPFC MPC": "MPC: FB vs EVC–cauPFC",
    "FF vs EVC deep qT1": "EVC deep → FF",
    "FF vs EVC middle qT1": "EVC middle → FF",
    "FB vs cauPFC middle qT1": "cauPFC middle → FB",
    "FB vs cauPFC superficial qT1": "cauPFC superficial → FB",
    "signed_H vs EVC-cauPFC MPC": "MPC vs signed H (FB − FF)",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rdcm-long", type=Path, default=DEFAULT_RDCM_LONG)
    p.add_argument("--qt1-long", type=Path, default=DEFAULT_QT1_LONG)
    p.add_argument("--mpc-base", type=Path, default=DEFAULT_MPC_BASE)
    p.add_argument("--lut-path", type=Path, default=DEFAULT_LUT)
    p.add_argument("--acq-tag", default="acq-T1map")
    p.add_argument("--atlas-tag", default="glasser-360")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--subjects", nargs="+", default=list(SUBJECTS))
    p.add_argument("--expected-n", type=int, default=15)
    p.add_argument("--n-permutations", type=int, default=100_000)
    p.add_argument("--n-bootstrap", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=20260731)
    p.add_argument("--dpi", type=int, default=600)
    return p.parse_args()


def normalize_subject(value: object) -> str:
    text = str(value).strip()
    match = re.search(r"(\d+)", text)
    if match is None:
        raise ValueError(f"Could not parse participant identifier: {value!r}")
    return f"sub-{int(match.group(1)):02d}"


def normalize_roi(value: object) -> str:
    text = str(value).strip()
    mapping = {
        "7Al": "7AL",
        "8AD": "8Ad",
    }
    return mapping.get(text, text)


def group_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def stable_seed(base: int, *parts: object) -> int:
    payload = "|".join([str(base), *(str(x) for x in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def holm_adjust(values: np.ndarray) -> np.ndarray:
    p = np.asarray(values, float)
    if p.ndim != 1 or not np.isfinite(p).all():
        raise ValueError("Holm correction requires finite one-dimensional p-values")
    order = np.argsort(p)
    ranked = p[order]
    adjusted_ranked = np.maximum.accumulate(
        (len(ranked) - np.arange(len(ranked))) * ranked
    )
    adjusted_ranked = np.minimum(adjusted_ranked, 1.0)
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    return adjusted


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    return float(stats.spearmanr(x, y).statistic)


def permutation_p_spearman(
    x: np.ndarray,
    y: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
    chunk_size: int = 20_000,
) -> tuple[float, int]:
    """Two-sided label-permutation P with a +1 Monte Carlo correction."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    xr = stats.rankdata(x, method="average")
    yr = stats.rankdata(y, method="average")
    xr = xr - xr.mean()
    yr = yr - yr.mean()
    xnorm = np.linalg.norm(xr)
    ynorm = np.linalg.norm(yr)
    if xnorm == 0 or ynorm == 0:
        return np.nan, 0
    xr = xr / xnorm
    yr = yr / ynorm
    observed = float(xr @ yr)
    exceed = 0
    done = 0
    while done < n_permutations:
        m = min(chunk_size, n_permutations - done)
        indices = np.argsort(rng.random((m, len(xr))), axis=1)
        null = yr[indices] @ xr
        exceed += int(np.count_nonzero(np.abs(null) >= abs(observed) - 1e-15))
        done += m
    return float((exceed + 1) / (n_permutations + 1)), exceed


def bootstrap_spearman_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float, int]:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)
    values = np.full(n_bootstrap, np.nan)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        values[i] = spearman_rho(x[idx], y[idx])
    valid = values[np.isfinite(values)]
    if len(valid) < max(100, int(0.90 * n_bootstrap)):
        raise RuntimeError(
            f"Only {len(valid)}/{n_bootstrap} valid bootstrap estimates"
        )
    low, high = np.quantile(valid, [0.025, 0.975])
    return float(low), float(high), int(len(valid))


def loo_summary(
    x: np.ndarray,
    y: np.ndarray,
    observed: float,
) -> tuple[float, float, float]:
    values = []
    for omitted in range(len(x)):
        keep = np.arange(len(x)) != omitted
        values.append(spearman_rho(x[keep], y[keep]))
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, np.nan
    if observed > 0:
        same = values > 0
    elif observed < 0:
        same = values < 0
    else:
        same = values == 0
    return float(values.min()), float(values.max()), float(np.mean(same))


def one_edge(group: pd.DataFrame, mask: pd.Series, label: str) -> float:
    selected = group.loc[mask]
    if len(selected) != 1:
        subject = group["subject"].iloc[0]
        raise RuntimeError(
            f"{subject}: expected exactly one {label} edge, found {len(selected)}"
        )
    return float(selected.iloc[0]["A"])


def load_rdcm_features(
    path: Path,
    subjects: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    missing = REQUIRED_RDCM_COLUMNS - set(df.columns)
    if missing:
        raise RuntimeError(f"{path} missing rDCM columns: {sorted(missing)}")
    df = df.copy()
    df["subject"] = df["subject"].map(normalize_subject)
    df["network"] = df["network"].astype(str).str.strip()
    if "validation_condition" in df.columns:
        conditions = set(df["validation_condition"].astype(str).str.strip())
        if conditions != {"pial5_current"}:
            raise RuntimeError(
                "rDCM validation_condition must be exactly pial5_current; "
                f"found {sorted(conditions)}"
            )
    elif "pial5_current" not in str(path):
        raise RuntimeError(
            "Cannot verify final outer-pial-5% lineage: the table has no "
            "validation_condition and its path does not contain pial5_current"
        )
    df = df.loc[df["network"] == "cauPFC_EVC"].copy()
    if df.empty:
        raise RuntimeError("Missing cauPFC_EVC network in final rDCM long table")
    df["source_key"] = df["source_group"].map(group_key)
    df["target_key"] = df["target_group"].map(group_key)
    df["source_layer"] = df["source_layer"].astype(str).str.lower().str.strip()
    df["target_layer"] = df["target_layer"].astype(str).str.lower().str.strip()
    df["A"] = pd.to_numeric(df["A"], errors="raise")

    rows = []
    audit = []
    for subject in subjects:
        group = df.loc[df["subject"] == subject].copy()
        if group.empty:
            raise RuntimeError(f"Missing final EVC-cauPFC rDCM for {subject}")
        ff = (
            (group["source_key"] == "evc")
            & (group["target_key"] == "caupfc")
            & (group["source_layer"] == "sup")
        )
        fb = (
            (group["source_key"] == "caupfc")
            & (group["target_key"] == "evc")
            & (group["source_layer"] == "deep")
        )
        ff_deep = one_edge(
            group, ff & (group["target_layer"] == "deep"),
            "FF EVC-sup to cauPFC-deep",
        )
        ff_mid = one_edge(
            group, ff & (group["target_layer"] == "mid"),
            "FF EVC-sup to cauPFC-mid",
        )
        fb_deep = one_edge(
            group, fb & (group["target_layer"] == "deep"),
            "FB cauPFC-deep to EVC-deep",
        )
        fb_sup = one_edge(
            group, fb & (group["target_layer"] == "sup"),
            "FB cauPFC-deep to EVC-sup",
        )
        signed_ff = float(np.mean([ff_deep, ff_mid]))
        signed_fb = float(np.mean([fb_deep, fb_sup]))
        rows.append({
            "subject": subject,
            "signed_FF": signed_ff,
            "signed_FB": signed_fb,
            "signed_H": signed_fb - signed_ff,
            "FF_EVCsup_to_cauPFCdeep": ff_deep,
            "FF_EVCsup_to_cauPFCmid": ff_mid,
            "FB_cauPFCdeep_to_EVCdeep": fb_deep,
            "FB_cauPFCdeep_to_EVCsup": fb_sup,
        })
        audit.append({
            "subject": subject,
            "source": "final local 6-node rDCM",
            "network": "cauPFC_EVC",
            "EVC_definition": "V1|V2|V3 equal-ROI composite upstream",
            "cauPFC_definition": "|".join(CAUPFC_ROIS),
            "n_prespecified_edges": 4,
            "status": "PASS",
        })
    return pd.DataFrame(rows), pd.DataFrame(audit)


def load_qt1_features(
    path: Path,
    subjects: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    missing = REQUIRED_QT1_COLUMNS - set(df.columns)
    if missing:
        raise RuntimeError(f"{path} missing qT1 columns: {sorted(missing)}")
    df = df.copy()
    df["subject"] = df["subject"].map(normalize_subject)
    df["roi"] = df["roi"].map(normalize_roi)
    # Read only the three layer-specific central-value columns. Extra columns
    # in the source CSV (for example depth_gradient or skewness_*) are ignored.
    metric_columns = list(QT1_LAYER_METRICS)
    for column in metric_columns:
        df[column] = pd.to_numeric(df[column], errors="raise")

    rows = []
    audit = []
    definitions = {"EVC": EVC_ROIS, "cauPFC": CAUPFC_ROIS}
    for subject in subjects:
        subject_df = df.loc[df["subject"] == subject].copy()
        if subject_df.empty:
            raise RuntimeError(f"Missing qT1 rows for {subject}")
        rec: dict[str, object] = {"subject": subject}
        for stage, rois in definitions.items():
            selected = subject_df.loc[subject_df["roi"].isin(rois)].copy()
            counts = selected["roi"].value_counts()
            missing_rois = sorted(set(rois) - set(selected["roi"]))
            duplicated = sorted(counts[counts != 1].index.tolist())
            status = "PASS" if not missing_rois and not duplicated else "FAIL"
            audit.append({
                "subject": subject,
                "source": "qT1 subject ROI depth profiles",
                "stage": stage,
                "expected_rois": "|".join(rois),
                "n_expected_rois": len(rois),
                "n_rows": len(selected),
                "missing_rois": "|".join(missing_rois),
                "non_singleton_rois": "|".join(duplicated),
                "aggregation": "strict_equal_ROI_mean",
                "status": status,
            })
            if status != "PASS":
                raise RuntimeError(
                    f"{subject}/{stage} qT1 completeness failure: "
                    f"missing={missing_rois}, non-singleton={duplicated}"
                )
            for column in metric_columns:
                values = selected[column].to_numpy(float)
                if not np.isfinite(values).all():
                    raise RuntimeError(
                        f"{subject}/{stage}/{column} contains non-finite values"
                    )
                rec[f"qT1_{stage}_{column}"] = float(np.mean(values))
        rows.append(rec)
    return pd.DataFrame(rows), pd.DataFrame(audit)


def parse_lut(path: Path) -> tuple[dict[str, int], dict[str, int]]:
    if not path.exists():
        raise FileNotFoundError(path)
    records = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) < 2 or line.lstrip().startswith("#"):
            continue
        try:
            records.append((int(parts[0]), parts[1]))
        except ValueError:
            continue
    left: dict[str, int] = {}
    right: dict[str, int] = {}
    left_base = None
    right_base = None
    for idx, name in records:
        if name.startswith("lh.L_") and name.endswith("_ROI.label"):
            roi = normalize_roi(name[5:-10])
            left_base = idx if left_base is None else left_base
            left[roi] = idx - left_base
        elif name.startswith("rh.R_") and name.endswith("_ROI.label"):
            roi = normalize_roi(name[5:-10])
            right_base = idx if right_base is None else right_base
            right[roi] = idx - right_base + 180
    if len(left) < 150 or len(right) < 150:
        raise RuntimeError(
            f"Could not parse bilateral HCP-MMP LUT: L={len(left)}, R={len(right)}"
        )
    required = set(EVC_ROIS) | set(CAUPFC_ROIS)
    missing = sorted(required - (set(left) & set(right)))
    if missing:
        raise RuntimeError(f"LUT is missing required bilateral ROIs: {missing}")
    return left, right


def gifti_array(path: Path) -> np.ndarray:
    gii = nib.load(str(path))
    arrays = [np.asarray(d.data) for d in gii.darrays]
    if len(arrays) == 1:
        return np.squeeze(arrays[0])
    return np.squeeze(np.asarray(arrays))


def reconstruct_mpc(raw: np.ndarray) -> tuple[np.ndarray, str, bool, float]:
    raw = np.asarray(raw, float)
    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        flat = raw.ravel()
        n = int(round(np.sqrt(flat.size)))
        if n * n != flat.size:
            raise ValueError(f"Cannot interpret MPC as square: {raw.shape}")
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
    removed_background = False
    if matrix.shape == (362, 362):
        matrix = np.delete(np.delete(matrix, 0, axis=0), 0, axis=1)
        matrix = np.delete(np.delete(matrix, 180, axis=0), 180, axis=1)
        removed_background = True
    elif matrix.shape != (360, 360):
        raise ValueError(f"Unexpected reconstructed MPC size: {matrix.shape}")
    matrix = np.asarray(matrix, float)
    matrix = (matrix + matrix.T) / 2
    return matrix, reconstruction, removed_background, symmetry_error


def finite_mean(values: list[float], label: str) -> float:
    x = np.asarray(values, float)
    if not np.isfinite(x).all() or not len(x):
        raise RuntimeError(f"Non-finite or empty MPC feature: {label}")
    return float(np.mean(x))


def load_mpc_features(
    base: Path,
    lut_path: Path,
    subjects: tuple[str, ...],
    acq_tag: str,
    atlas_tag: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    left, right = parse_lut(lut_path)
    rows = []
    audit = []
    for subject in subjects:
        path = (
            base / subject / "mpc" / acq_tag
            / f"{subject}_atlas-{atlas_tag}_desc-MPC.shape.gii"
        )
        if not path.exists():
            raise FileNotFoundError(path)
        raw = gifti_array(path)
        matrix, reconstruction, removed_bg, symmetry_error = reconstruct_mpc(raw)
        left_values = []
        right_values = []
        cross_values = []
        for evc in EVC_ROIS:
            for frontal in CAUPFC_ROIS:
                left_values.append(float(matrix[left[evc], left[frontal]]))
                right_values.append(float(matrix[right[evc], right[frontal]]))
                cross_values.append(float(matrix[left[evc], right[frontal]]))
                cross_values.append(float(matrix[right[evc], left[frontal]]))
        primary_values = np.asarray(left_values + right_values, float)
        n_zero_primary = int(np.count_nonzero(primary_values == 0))
        if n_zero_primary:
            raise RuntimeError(
                f"{subject}: {n_zero_primary}/{len(primary_values)} selected "
                "within-hemisphere EVC-cauPFC MPC edges are exactly zero. "
                "Exact zeros are treated as missing/invalid MPC entries; "
                "inspect the raw matrix and ROI indices before analysis."
            )
        within_left = finite_mean(left_values, f"{subject}/MPC/left")
        within_right = finite_mean(right_values, f"{subject}/MPC/right")
        within = float(np.mean([within_left, within_right]))
        cross = finite_mean(cross_values, f"{subject}/MPC/cross")
        all_four = float(np.mean([within, cross]))
        rows.append({
            "subject": subject,
            "MPC_EVC_cauPFC_within_hemi": within,
            "MPC_EVC_cauPFC_left": within_left,
            "MPC_EVC_cauPFC_right": within_right,
            "MPC_EVC_cauPFC_cross_hemi": cross,
            "MPC_EVC_cauPFC_all_hemi_combinations": all_four,
        })
        audit.append({
            "subject": subject,
            "source": str(path),
            "raw_shape": str(tuple(np.asarray(raw).shape)),
            "final_shape": str(tuple(matrix.shape)),
            "reconstruction": reconstruction,
            "removed_background": removed_bg,
            "symmetry_error_raw": symmetry_error,
            "n_EVC_rois": len(EVC_ROIS),
            "n_cauPFC_rois": len(CAUPFC_ROIS),
            "n_within_hemi_edges": 2 * len(EVC_ROIS) * len(CAUPFC_ROIS),
            "n_zero_within_hemi_edges": n_zero_primary,
            "n_cross_hemi_edges": 2 * len(EVC_ROIS) * len(CAUPFC_ROIS),
            "primary_aggregation": "mean(LL and RR) across 3x7 parcel pairs",
            "status": "PASS",
        })
    return pd.DataFrame(rows), pd.DataFrame(audit)


def run_test(
    data: pd.DataFrame,
    family: str,
    label: str,
    structural_metric: str,
    rdcm_metric: str,
    n_permutations: int,
    n_bootstrap: int,
    seed: int,
) -> dict[str, object]:
    subset = data[["subject", structural_metric, rdcm_metric]].dropna()
    x = subset[structural_metric].to_numpy(float)
    y = subset[rdcm_metric].to_numpy(float)
    observed = spearman_rho(x, y)
    if not np.isfinite(observed):
        raise RuntimeError(f"Non-estimable test: {label}")
    rng_perm = np.random.default_rng(
        stable_seed(seed, family, label, "permutation")
    )
    rng_boot = np.random.default_rng(
        stable_seed(seed, family, label, "bootstrap")
    )
    p_perm, exceed = permutation_p_spearman(
        x, y, n_permutations, rng_perm
    )
    ci_low, ci_high, n_valid = bootstrap_spearman_ci(
        x, y, n_bootstrap, rng_boot
    )
    loo_min, loo_max, loo_same = loo_summary(x, y, observed)
    return {
        "family": family,
        "test": label,
        "structural_metric": structural_metric,
        "rdcm_metric": rdcm_metric,
        "n_subjects": len(subset),
        "spearman_rho": observed,
        "bootstrap_ci95_low": ci_low,
        "bootstrap_ci95_high": ci_high,
        "permutation_p_two_sided": p_perm,
        "permutation_exceedances": exceed,
        "n_permutations": n_permutations,
        "n_valid_bootstrap": n_valid,
        "loo_rho_min": loo_min,
        "loo_rho_max": loo_max,
        "loo_same_sign_fraction": loo_same,
    }


def analyze(
    data: pd.DataFrame,
    n_permutations: int,
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    primary_specs = [
        (
            "primary_qT1_source_layer",
            "FF vs EVC superficial qT1",
            "qT1_EVC_intensity_Superficial",
            "signed_FF",
        ),
        (
            "primary_qT1_source_layer",
            "FB vs cauPFC deep qT1",
            "qT1_cauPFC_intensity_Deep",
            "signed_FB",
        ),
        (
            "primary_MPC_interregional",
            "FF vs EVC-cauPFC MPC",
            "MPC_EVC_cauPFC_within_hemi",
            "signed_FF",
        ),
        (
            "primary_MPC_interregional",
            "FB vs EVC-cauPFC MPC",
            "MPC_EVC_cauPFC_within_hemi",
            "signed_FB",
        ),
    ]
    primary_rows = [
        run_test(
            data, family, label, structural, rdcm,
            n_permutations, n_bootstrap, seed,
        )
        for family, label, structural, rdcm in primary_specs
    ]
    primary = pd.DataFrame(primary_rows)
    primary["holm_p_across_4_primary"] = holm_adjust(
        primary["permutation_p_two_sided"].to_numpy(float)
    )

    layer_diagnostic_specs = [
        (
            "diagnostic_qT1_other_layers",
            "FF vs EVC deep qT1",
            "qT1_EVC_intensity_Deep",
            "signed_FF",
        ),
        (
            "diagnostic_qT1_other_layers",
            "FF vs EVC middle qT1",
            "qT1_EVC_intensity_Middle",
            "signed_FF",
        ),
        (
            "diagnostic_qT1_other_layers",
            "FB vs cauPFC middle qT1",
            "qT1_cauPFC_intensity_Middle",
            "signed_FB",
        ),
        (
            "diagnostic_qT1_other_layers",
            "FB vs cauPFC superficial qT1",
            "qT1_cauPFC_intensity_Superficial",
            "signed_FB",
        ),
    ]
    layer_diagnostic = pd.DataFrame([
        run_test(
            data, family, label, structural, rdcm,
            n_permutations, n_bootstrap, seed,
        )
        for family, label, structural, rdcm in layer_diagnostic_specs
    ])
    layer_diagnostic["holm_p_across_4_layer_diagnostics"] = holm_adjust(
        layer_diagnostic["permutation_p_two_sided"].to_numpy(float)
    )

    directional_specs = [
        (
            "diagnostic_MPC_directional_contrast",
            "signed_H vs EVC-cauPFC MPC",
            "MPC_EVC_cauPFC_within_hemi",
            "signed_H",
        )
    ]
    directional = pd.DataFrame([
        run_test(
            data, family, label, structural, rdcm,
            n_permutations, n_bootstrap, seed,
        )
        for family, label, structural, rdcm in directional_specs
    ])
    directional["adjusted_p_within_stated_family"] = (
        directional["permutation_p_two_sided"]
    )

    # A convenient six-row qT1-only table. The primary qT1 rows retain the
    # four-test structure-family Holm p values; diagnostic rows retain the
    # four-test layer-family Holm p values. They are never pooled into a new
    # post-hoc correction family.
    primary_qt1 = primary.loc[
        primary["family"] == "primary_qT1_source_layer"
    ].copy()
    primary_qt1["analysis_role"] = "primary"
    primary_qt1["multiplicity_family"] = (
        "four primary structure tests: 2 qT1 + 2 MPC"
    )
    primary_qt1["adjusted_p_within_stated_family"] = (
        primary_qt1["holm_p_across_4_primary"]
    )
    diagnostic_qt1 = layer_diagnostic.copy()
    diagnostic_qt1["analysis_role"] = "layer diagnostic"
    diagnostic_qt1["multiplicity_family"] = "four other-layer qT1 tests"
    diagnostic_qt1["adjusted_p_within_stated_family"] = (
        diagnostic_qt1["holm_p_across_4_layer_diagnostics"]
    )
    qt1_layer_resolved = pd.concat(
        [primary_qt1, diagnostic_qt1],
        ignore_index=True,
        sort=False,
    )
    metric_to_region_layer = {
        f"qT1_{region}_{metric}": (region, QT1_LAYER_LABELS[metric])
        for region in ("EVC", "cauPFC")
        for metric in QT1_LAYER_METRICS
    }
    parsed = qt1_layer_resolved["structural_metric"].map(
        metric_to_region_layer
    )
    if parsed.isna().any():
        bad = qt1_layer_resolved.loc[
            parsed.isna(), "structural_metric"
        ].tolist()
        raise RuntimeError(f"Could not parse qT1 layer metadata: {bad}")
    qt1_layer_resolved.insert(
        2, "qT1_region", [value[0] for value in parsed]
    )
    qt1_layer_resolved.insert(
        3, "qT1_layer", [value[1] for value in parsed]
    )
    qt1_layer_resolved.insert(
        4,
        "rDCM_pathway",
        qt1_layer_resolved["rdcm_metric"].str.replace(
            "signed_", "", regex=False
        ),
    )
    return primary, layer_diagnostic, directional, qt1_layer_resolved


def reviewer_test_inventory(
    primary: pd.DataFrame,
    layer_diagnostic: pd.DataFrame,
    directional: pd.DataFrame,
) -> pd.DataFrame:
    """Create one harmonized test inventory without changing multiplicity."""
    primary_review = primary.copy()
    primary_review["analysis_role"] = "primary"
    primary_review["multiplicity_family"] = (
        "four prespecified primary structure tests"
    )
    primary_review["adjusted_p_within_stated_family"] = (
        primary_review["holm_p_across_4_primary"]
    )

    layer_review = layer_diagnostic.copy()
    layer_review["analysis_role"] = "layer diagnostic"
    layer_review["multiplicity_family"] = (
        "four other-layer qT1 diagnostic tests"
    )
    layer_review["adjusted_p_within_stated_family"] = (
        layer_review["holm_p_across_4_layer_diagnostics"]
    )

    directional_review = directional.copy()
    directional_review["analysis_role"] = "directional diagnostic"
    directional_review["multiplicity_family"] = (
        "single directional H-versus-MPC diagnostic"
    )
    directional_review["adjusted_p_within_stated_family"] = (
        directional_review["permutation_p_two_sided"]
    )

    inventory = pd.concat(
        [primary_review, layer_review, directional_review],
        ignore_index=True,
        sort=False,
    )
    inventory.insert(0, "test_id", inventory["test"].map(TEST_IDS))
    inventory.insert(2, "display_label", inventory["test"].map(DISPLAY_LABELS))
    if inventory["test_id"].isna().any() or inventory["display_label"].isna().any():
        missing = inventory.loc[
            inventory["test_id"].isna() | inventory["display_label"].isna(),
            "test",
        ].tolist()
        raise RuntimeError(f"Reviewer test metadata are incomplete: {missing}")
    if inventory["test_id"].duplicated().any():
        raise RuntimeError("Reviewer test IDs are not unique")
    return inventory


def build_reviewer_loso_tables(
    data: pd.DataFrame,
    inventory: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return long, summary and wide participant-deletion LOSO tables.

    LOSO values quantify influence on the Spearman coefficient only.  They do
    not constitute 15 new hypothesis tests, and no deletion-specific P values
    are calculated.
    """
    rows: list[dict[str, object]] = []
    for _, test_row in inventory.iterrows():
        structural_metric = str(test_row["structural_metric"])
        rdcm_metric = str(test_row["rdcm_metric"])
        subset = data[["subject", structural_metric, rdcm_metric]].dropna().copy()
        subset = subset.sort_values("subject", kind="stable").reset_index(drop=True)
        full_rho = float(test_row["spearman_rho"])
        if len(subset) != int(test_row["n_subjects"]):
            raise RuntimeError(
                f"{test_row['test']}: test/data participant-count mismatch"
            )
        for omitted_index, omitted_row in subset.iterrows():
            retained = subset.index != omitted_index
            loso_rho = spearman_rho(
                subset.loc[retained, structural_metric].to_numpy(float),
                subset.loc[retained, rdcm_metric].to_numpy(float),
            )
            if not np.isfinite(loso_rho):
                raise RuntimeError(
                    f"{test_row['test']}: non-estimable LOSO coefficient after "
                    f"omitting {omitted_row['subject']}"
                )
            if full_rho > 0:
                same_sign = loso_rho > 0
            elif full_rho < 0:
                same_sign = loso_rho < 0
            else:
                same_sign = bool(np.isclose(loso_rho, 0.0, atol=1e-15))
            rows.append(
                {
                    "test_id": str(test_row["test_id"]),
                    "analysis_role": str(test_row["analysis_role"]),
                    "multiplicity_family": str(test_row["multiplicity_family"]),
                    "family": str(test_row["family"]),
                    "test": str(test_row["test"]),
                    "display_label": str(test_row["display_label"]),
                    "structural_metric": structural_metric,
                    "rdcm_metric": rdcm_metric,
                    "omitted_subject": str(omitted_row["subject"]),
                    "omitted_structural_value": float(
                        omitted_row[structural_metric]
                    ),
                    "omitted_rdcm_value": float(omitted_row[rdcm_metric]),
                    "n_full_sample": len(subset),
                    "n_retained": int(retained.sum()),
                    "full_sample_rho": full_rho,
                    "full_bootstrap_ci95_low": float(
                        test_row["bootstrap_ci95_low"]
                    ),
                    "full_bootstrap_ci95_high": float(
                        test_row["bootstrap_ci95_high"]
                    ),
                    "full_permutation_p_two_sided": float(
                        test_row["permutation_p_two_sided"]
                    ),
                    "full_adjusted_p_within_stated_family": float(
                        test_row["adjusted_p_within_stated_family"]
                    ),
                    "loso_rho": float(loso_rho),
                    "delta_rho_loso_minus_full": float(loso_rho - full_rho),
                    "abs_delta_rho": float(abs(loso_rho - full_rho)),
                    "same_sign_as_full": bool(same_sign),
                }
            )

    long_table = pd.DataFrame(rows)
    expected_rows = len(inventory) * data["subject"].nunique()
    if len(long_table) != expected_rows:
        raise RuntimeError(
            f"LOSO table has {len(long_table)} rows; expected {expected_rows}"
        )
    long_table["influence_rank_within_test"] = (
        long_table.groupby("test_id", observed=True)["abs_delta_rho"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    long_table["most_influential_for_test"] = (
        long_table["influence_rank_within_test"] == 1
    )

    summary_rows: list[dict[str, object]] = []
    for test_id, group in long_table.groupby("test_id", sort=False, observed=True):
        group = group.sort_values(
            ["abs_delta_rho", "omitted_subject"],
            ascending=[False, True],
            kind="stable",
        )
        first = group.iloc[0]
        loso = group["loso_rho"].to_numpy(float)
        same_sign_n = int(group["same_sign_as_full"].sum())
        most_influential = group.loc[
            np.isclose(
                group["abs_delta_rho"],
                float(group["abs_delta_rho"].max()),
                rtol=0.0,
                atol=1e-15,
            ),
            "omitted_subject",
        ].astype(str).tolist()
        summary_rows.append(
            {
                "test_id": test_id,
                "analysis_role": first["analysis_role"],
                "multiplicity_family": first["multiplicity_family"],
                "family": first["family"],
                "test": first["test"],
                "display_label": first["display_label"],
                "structural_metric": first["structural_metric"],
                "rdcm_metric": first["rdcm_metric"],
                "n_full_sample": int(first["n_full_sample"]),
                "n_loso_deletions": len(group),
                "n_retained_per_deletion": int(first["n_retained"]),
                "full_sample_rho": float(first["full_sample_rho"]),
                "full_bootstrap_ci95_low": float(
                    first["full_bootstrap_ci95_low"]
                ),
                "full_bootstrap_ci95_high": float(
                    first["full_bootstrap_ci95_high"]
                ),
                "full_permutation_p_two_sided": float(
                    first["full_permutation_p_two_sided"]
                ),
                "full_adjusted_p_within_stated_family": float(
                    first["full_adjusted_p_within_stated_family"]
                ),
                "loso_rho_min": float(np.min(loso)),
                "loso_rho_q25": float(np.quantile(loso, 0.25)),
                "loso_rho_median": float(np.median(loso)),
                "loso_rho_mean": float(np.mean(loso)),
                "loso_rho_q75": float(np.quantile(loso, 0.75)),
                "loso_rho_max": float(np.max(loso)),
                "loso_same_sign_n": same_sign_n,
                "loso_same_sign_fraction": float(same_sign_n / len(group)),
                "loso_range_includes_zero": bool(
                    np.min(loso) <= 0.0 <= np.max(loso)
                ),
                "max_abs_delta_rho": float(group["abs_delta_rho"].max()),
                "most_influential_subject": "|".join(most_influential),
                "rho_after_most_influential_omission": float(first["loso_rho"]),
                "loso_interpretation": (
                    "directionally stable across all deletions"
                    if same_sign_n == len(group)
                    else "direction changes in one or more deletions"
                ),
            }
        )
    summary_table = pd.DataFrame(summary_rows)

    wide_table = long_table.pivot(
        index="omitted_subject", columns="test_id", values="loso_rho"
    ).reset_index()
    wide_table.columns.name = None
    wide_table.insert(1, "n_retained", int(long_table["n_retained"].iloc[0]))
    return long_table, summary_table, wide_table


def validate_reviewer_loso_tables(
    inventory: pd.DataFrame,
    long_table: pd.DataFrame,
    summary_table: pd.DataFrame,
    expected_n: int,
) -> pd.DataFrame:
    failures: list[str] = []
    if len(inventory) != 9:
        failures.append(f"expected 9 tests; found {len(inventory)}")
    if len(long_table) != len(inventory) * expected_n:
        failures.append(
            f"expected {len(inventory) * expected_n} long rows; "
            f"found {len(long_table)}"
        )
    if len(summary_table) != len(inventory):
        failures.append(
            f"expected {len(inventory)} summary rows; found {len(summary_table)}"
        )
    if set(long_table["n_retained"].astype(int)) != {expected_n - 1}:
        failures.append("not every LOSO coefficient retains expected_n - 1 participants")
    expected_subjects = set(long_table["omitted_subject"].astype(str))
    for test_id, group in long_table.groupby("test_id", observed=True):
        if len(group) != expected_n:
            failures.append(f"{test_id}: expected {expected_n} deletions")
        if set(group["omitted_subject"].astype(str)) != expected_subjects:
            failures.append(f"{test_id}: omitted-participant set mismatch")
        if not np.isfinite(group["loso_rho"].to_numpy(float)).all():
            failures.append(f"{test_id}: non-finite LOSO coefficients")

    comparison = inventory.merge(
        summary_table,
        on="test_id",
        how="left",
        validate="one_to_one",
        suffixes=("_inventory", "_summary"),
    )
    for _, row in comparison.iterrows():
        checks = (
            ("loo_rho_min", "loso_rho_min"),
            ("loo_rho_max", "loso_rho_max"),
            ("loo_same_sign_fraction", "loso_same_sign_fraction"),
        )
        for legacy_column, detailed_column in checks:
            if not np.isclose(
                float(row[legacy_column]),
                float(row[detailed_column]),
                rtol=0.0,
                atol=1e-12,
            ):
                failures.append(
                    f"{row['test_id']}: {legacy_column} does not match "
                    f"detailed {detailed_column}"
                )

    return pd.DataFrame(
        [
            {
                "check": "reviewer-facing participant-deletion LOSO outputs",
                "n_tests": len(inventory),
                "n_expected_participants": expected_n,
                "n_long_rows": len(long_table),
                "n_summary_rows": len(summary_table),
                "legacy_summary_reconciliation": "PASS" if not failures else "FAIL",
                "failure_details": " | ".join(failures),
                "status": "PASS" if not failures else "FAIL",
            }
        ]
    )


def write_reviewer_loso_report(
    summary: pd.DataFrame,
    out_dir: Path,
) -> None:
    ordered_ids = [
        "qt1_ff_evc_superficial",
        "qt1_fb_caupfc_deep",
        "mpc_ff_evc_caupfc",
        "mpc_fb_evc_caupfc",
        "qt1_ff_evc_deep",
        "qt1_ff_evc_middle",
        "qt1_fb_caupfc_middle",
        "qt1_fb_caupfc_superficial",
        "mpc_h_evc_caupfc",
    ]
    frame = summary.set_index("test_id").loc[ordered_ids].reset_index()
    lines = [
        "# Reviewer-facing leave-one-subject-out influence analysis",
        "",
        (
            "LOSO is a descriptive influence analysis of participant-level "
            "Spearman correlations. Formal confidence intervals and P values "
            "come from the complete N=15 sample; deletion-specific P values "
            "were not calculated."
        ),
        "",
        (
            "| Analysis | Full ρ [95% CI] | Adjusted P | LOSO range "
            "(n=14) | Same direction | Largest |Δρ| |"
        ),
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in frame.iterrows():
        adjusted_p = float(row["full_adjusted_p_within_stated_family"])
        p_text = "<.001" if adjusted_p < 0.001 else f"{adjusted_p:.3f}".lstrip("0")
        lines.append(
            "| "
            + str(row["display_label"])
            + " | "
            + (
                f"{row['full_sample_rho']:+.2f} "
                f"[{row['full_bootstrap_ci95_low']:+.2f}, "
                f"{row['full_bootstrap_ci95_high']:+.2f}]"
            )
            + f" | {p_text}"
            + (
                f" | [{row['loso_rho_min']:+.2f}, "
                f"{row['loso_rho_max']:+.2f}]"
            )
            + (
                f" | {int(row['loso_same_sign_n'])}/"
                f"{int(row['n_loso_deletions'])}"
            )
            + (
                f" | omit {row['most_influential_subject']}: "
                f"|Δρ|={row['max_abs_delta_rho']:.2f} |"
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation rule",
            "",
            (
                "A stable LOSO direction indicates that no single participant "
                "reverses the sign of the complete-sample association. It does "
                "not create an independent replication and does not replace "
                "complete-sample multiplicity-adjusted inference."
            ),
            "",
        ]
    )
    (out_dir / "reviewer_loso_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def set_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.55, zorder=0)
    ax.tick_params(width=0.8, length=3, color=INK)


def theil_sen_line(ax: plt.Axes, x: np.ndarray, y: np.ndarray, color: str) -> None:
    if len(np.unique(x)) < 2:
        return
    slope, intercept, _, _ = stats.theilslopes(y, x, alpha=0.95)
    grid = np.linspace(float(np.min(x)), float(np.max(x)), 100)
    ax.plot(grid, intercept + slope * grid, color=color, linewidth=1.35, zorder=2)


def format_p(value: float) -> str:
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def plot_primary_scatter(
    data: pd.DataFrame,
    primary: pd.DataFrame,
    out_dir: Path,
) -> None:
    specs = [
        (
            "FF vs EVC superficial qT1",
            "qT1_EVC_intensity_Superficial", "signed_FF",
            "EVC superficial qT1-derived value", "Feedforward strength (Hz)",
            BLUE,
        ),
        (
            "FB vs cauPFC deep qT1",
            "qT1_cauPFC_intensity_Deep", "signed_FB",
            "cauPFC deep qT1-derived value", "Feedback strength (Hz)",
            VERMILLION,
        ),
        (
            "FF vs EVC-cauPFC MPC",
            "MPC_EVC_cauPFC_within_hemi", "signed_FF",
            "EVC-cauPFC MPC", "Feedforward strength (Hz)",
            BLUE,
        ),
        (
            "FB vs EVC-cauPFC MPC",
            "MPC_EVC_cauPFC_within_hemi", "signed_FB",
            "EVC-cauPFC MPC", "Feedback strength (Hz)",
            VERMILLION,
        ),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8))
    letters = "abcd"
    for ax, spec, letter in zip(axes.flat, specs, letters):
        test, xcol, ycol, xlabel, ylabel, color = spec
        row = primary.loc[primary["test"] == test].iloc[0]
        x = data[xcol].to_numpy(float)
        y = data[ycol].to_numpy(float)
        ax.scatter(
            x, y, s=23, color=color, alpha=0.72,
            edgecolor="white", linewidth=0.45, zorder=3,
        )
        theil_sen_line(ax, x, y, color)
        ax.axhline(0, color=MUTED, linewidth=0.75, linestyle=(0, (3, 2)))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        clean_axis(ax)
        ax.text(
            -0.16, 1.06, letter, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top",
        )
        ax.text(
            0.02, 0.98,
            (
                f"$\\rho$={row['spearman_rho']:+.2f} "
                f"[{row['bootstrap_ci95_low']:+.2f}, "
                f"{row['bootstrap_ci95_high']:+.2f}]\n"
                f"$P_{{\\mathrm{{Holm}}}}$="
                f"{format_p(float(row['holm_p_across_4_primary']))}"
            ),
            transform=ax.transAxes, ha="left", va="top",
            fontsize=7.7, color=INK,
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.88,
            },
        )
    fig.suptitle(
        "Local cortical structure and EVC-cauPFC effective connectivity",
        x=0.08, y=0.995, ha="left", fontsize=13, fontweight="bold",
    )
    fig.text(
        0.08, 0.955,
        (
            "N=15; points are participants; lines are Theil-Sen visual guides; "
            "inference uses Spearman correlations and label permutation"
        ),
        ha="left", va="top", fontsize=8, color=MUTED,
    )
    fig.subplots_adjust(
        left=0.11, right=0.98, bottom=0.10, top=0.86,
        hspace=0.42, wspace=0.34,
    )
    stem = out_dir / "fig_structure_rdcm_primary_scatter"
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    plt.close(fig)


def plot_primary_forest(primary: pd.DataFrame, out_dir: Path) -> None:
    order = [
        "FF vs EVC superficial qT1",
        "FB vs cauPFC deep qT1",
        "FF vs EVC-cauPFC MPC",
        "FB vs EVC-cauPFC MPC",
    ]
    labels = {
        "FF vs EVC superficial qT1": "qT1: FF vs EVC superficial",
        "FB vs cauPFC deep qT1": "qT1: FB vs cauPFC deep",
        "FF vs EVC-cauPFC MPC": "MPC: FF vs EVC-cauPFC",
        "FB vs EVC-cauPFC MPC": "MPC: FB vs EVC-cauPFC",
    }
    colors = {
        "signed_FF": BLUE,
        "signed_FB": VERMILLION,
    }
    frame = primary.set_index("test").loc[order].reset_index()
    y = np.arange(len(frame))[::-1]
    fig, ax = plt.subplots(figsize=(6.6, 3.15))
    for yi, (_, row) in zip(y, frame.iterrows()):
        rho = float(row["spearman_rho"])
        lo = float(row["bootstrap_ci95_low"])
        hi = float(row["bootstrap_ci95_high"])
        color = colors[str(row["rdcm_metric"])]
        significant = float(row["holm_p_across_4_primary"]) < 0.05
        ax.errorbar(
            rho, yi,
            xerr=np.array([[rho - lo], [hi - rho]]),
            fmt="o", markersize=5.7,
            markerfacecolor=color if significant else "white",
            markeredgecolor=color, markeredgewidth=1.2,
            ecolor=color, elinewidth=1.4, capsize=3.2, capthick=1.1,
            zorder=3,
        )
        ax.text(
            1.03, yi,
            (
                f"$P_{{\\mathrm{{Holm}}}}$="
                f"{format_p(float(row['holm_p_across_4_primary']))}"
            ),
            transform=ax.get_yaxis_transform(),
            ha="left", va="center", fontsize=7.7, color=INK,
        )
    ax.axvline(0, color=MUTED, linewidth=0.8, linestyle=(0, (3, 2)))
    ax.set_xlim(-1, 1)
    ax.set_ylim(-0.65, len(frame) - 0.35)
    ax.set_yticks(y, [labels[x] for x in order])
    ax.set_xlabel("Spearman $\\rho$ (participant-bootstrap 95% CI)")
    clean_axis(ax)
    ax.grid(False, axis="y")
    fig.suptitle(
        "Focused structure-rDCM associations",
        x=0.11, y=0.98, ha="left", fontsize=12.5, fontweight="bold",
    )
    fig.text(
        0.11, 0.91,
        "Holm correction is applied across all four prespecified primary tests",
        ha="left", va="top", fontsize=8, color=MUTED,
    )
    fig.subplots_adjust(left=0.39, right=0.85, bottom=0.20, top=0.80)
    stem = out_dir / "fig_structure_rdcm_primary_forest"
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    plt.close(fig)


def plot_qt1_layer_forest(
    qt1_layer_resolved: pd.DataFrame,
    out_dir: Path,
) -> None:
    order = [
        "FF vs EVC deep qT1",
        "FF vs EVC middle qT1",
        "FF vs EVC superficial qT1",
        "FB vs cauPFC deep qT1",
        "FB vs cauPFC middle qT1",
        "FB vs cauPFC superficial qT1",
    ]
    labels = {
        "FF vs EVC deep qT1": "EVC deep",
        "FF vs EVC middle qT1": "EVC middle",
        "FF vs EVC superficial qT1": "EVC superficial  [primary]",
        "FB vs cauPFC deep qT1": "cauPFC deep  [primary]",
        "FB vs cauPFC middle qT1": "cauPFC middle",
        "FB vs cauPFC superficial qT1": "cauPFC superficial",
    }
    frame = qt1_layer_resolved.set_index("test").loc[order].reset_index()
    y = np.arange(len(frame))[::-1]
    fig, ax = plt.subplots(figsize=(6.9, 4.15))
    for yi, (_, row) in zip(y, frame.iterrows()):
        rho = float(row["spearman_rho"])
        lo = float(row["bootstrap_ci95_low"])
        hi = float(row["bootstrap_ci95_high"])
        p_adj = float(row["adjusted_p_within_stated_family"])
        color = BLUE if str(row["rdcm_metric"]) == "signed_FF" else VERMILLION
        primary = str(row["analysis_role"]) == "primary"
        significant = p_adj < 0.05
        ax.errorbar(
            rho, yi,
            xerr=np.array([[rho - lo], [hi - rho]]),
            fmt="o",
            markersize=6.1 if primary else 5.2,
            markerfacecolor=color if significant else "white",
            markeredgecolor=color,
            markeredgewidth=1.35 if primary else 1.05,
            ecolor=color,
            elinewidth=1.45 if primary else 1.05,
            capsize=3.2,
            capthick=1.05,
            alpha=1.0 if primary else 0.78,
            zorder=3,
        )
        ax.text(
            1.03, yi,
            (
                f"$\\rho$={rho:+.2f}; "
                f"$P_{{\\mathrm{{adj}}}}$={format_p(p_adj)}"
            ),
            transform=ax.get_yaxis_transform(),
            ha="left", va="center", fontsize=7.5, color=INK,
        )
    ax.axvline(0, color=MUTED, linewidth=0.8, linestyle=(0, (3, 2)))
    ax.axhline(2.5, color=GRID, linewidth=0.8)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-0.7, len(frame) - 0.3)
    ax.set_yticks(y, [labels[x] for x in order])
    ax.set_xlabel("Spearman $\\rho$ (participant-bootstrap 95% CI)")
    clean_axis(ax)
    ax.grid(False, axis="y")
    ax.text(
        -0.99, 5.55, "Feedforward: EVC qT1 layers",
        color=BLUE, fontsize=8.2, fontweight="bold", va="bottom",
    )
    ax.text(
        -0.99, 2.55, "Feedback: cauPFC qT1 layers",
        color=VERMILLION, fontsize=8.2, fontweight="bold", va="bottom",
    )
    fig.suptitle(
        "Layer-resolved qT1-derived values and effective connectivity",
        x=0.09, y=0.985, ha="left", fontsize=12.3, fontweight="bold",
    )
    fig.text(
        0.09, 0.925,
        (
            "Only deep, middle and superficial central values are analysed; "
            "depth differences and skewness are excluded"
        ),
        ha="left", va="top", fontsize=8, color=MUTED,
    )
    fig.text(
        0.09, 0.035,
        (
            "Adjusted P values retain their stated families: four primary "
            "structure tests or four other-layer diagnostic tests"
        ),
        ha="left", va="bottom", fontsize=7.3, color=MUTED,
    )
    fig.subplots_adjust(left=0.31, right=0.77, bottom=0.18, top=0.83)
    stem = out_dir / "fig_qt1_layer_resolved_forest"
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    plt.close(fig)


def pathway_color(rdcm_metric: str) -> str:
    if rdcm_metric == "signed_FF":
        return BLUE
    if rdcm_metric == "signed_FB":
        return VERMILLION
    if rdcm_metric == "signed_H":
        return PURPLE
    raise KeyError(f"Unknown rDCM metric for plotting: {rdcm_metric}")


def save_publication_figure(fig: plt.Figure, stem: Path, dpi: int) -> None:
    for suffix in (".png", ".pdf", ".svg"):
        path = stem.with_suffix(suffix)
        kwargs: dict[str, object] = {
            "bbox_inches": "tight",
            "facecolor": "white",
        }
        if suffix == ".png":
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
    plt.close(fig)


def plot_reviewer_loso_influence(
    loso_long: pd.DataFrame,
    loso_summary: pd.DataFrame,
    ordered_test_ids: list[str],
    title: str,
    subtitle: str,
    stem: Path,
    dpi: int,
) -> None:
    """Plot complete-sample inference together with descriptive LOSO influence."""
    summary = loso_summary.set_index("test_id").loc[ordered_test_ids].reset_index()
    n_rows = len(summary)
    height = 3.45 if n_rows <= 4 else 4.55
    fig = plt.figure(figsize=(7.25, height))
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=(1.18, 0.82),
        left=0.27 if n_rows <= 4 else 0.29,
        right=0.98,
        bottom=0.20 if n_rows <= 4 else 0.16,
        top=0.72 if n_rows <= 4 else 0.76,
        wspace=0.08,
    )
    axis = fig.add_subplot(grid[0, 0])
    annotation_axis = fig.add_subplot(grid[0, 1], sharey=axis)
    y_positions = np.arange(n_rows, dtype=float)[::-1]

    for y_position, (_, row) in zip(y_positions, summary.iterrows()):
        test_id = str(row["test_id"])
        color = pathway_color(str(row["rdcm_metric"]))
        deletion_values = (
            loso_long.loc[loso_long["test_id"].eq(test_id)]
            .sort_values("omitted_subject", kind="stable")["loso_rho"]
            .to_numpy(float)
        )
        jitter = np.linspace(-0.105, 0.105, len(deletion_values))
        range_low = float(row["loso_rho_min"])
        range_high = float(row["loso_rho_max"])
        axis.fill_betweenx(
            [y_position - 0.15, y_position + 0.15],
            range_low,
            range_high,
            color=PALE_RANGE,
            alpha=0.95,
            linewidth=0,
            zorder=1,
        )
        axis.hlines(
            y_position,
            range_low,
            range_high,
            color=MUTED,
            linewidth=0.65,
            zorder=2,
        )
        axis.scatter(
            deletion_values,
            y_position + jitter,
            s=14,
            marker="o",
            facecolors="white",
            edgecolors=color,
            linewidths=0.65,
            alpha=0.72,
            zorder=3,
        )

        rho = float(row["full_sample_rho"])
        low = float(row["full_bootstrap_ci95_low"])
        high = float(row["full_bootstrap_ci95_high"])
        primary = str(row["analysis_role"]) == "primary"
        axis.errorbar(
            rho,
            y_position,
            xerr=np.asarray([[rho - low], [high - rho]]),
            fmt="D",
            markersize=6.1 if primary else 5.6,
            markerfacecolor=color if primary else "white",
            markeredgecolor=color,
            markeredgewidth=1.1,
            ecolor=color,
            elinewidth=1.35,
            capsize=3.4,
            capthick=1.05,
            zorder=5,
        )

        adjusted_p = float(row["full_adjusted_p_within_stated_family"])
        p_text = format_p(adjusted_p).replace("0.", ".")
        annotation_axis.text(
            0.02,
            y_position,
            (
                f"$\\rho$={rho:+.2f} [{low:+.2f}, {high:+.2f}]\n"
                f"$P_{{\\rm adj}}$={p_text};  "
                f"LOSO [{range_low:+.2f}, {range_high:+.2f}]\n"
                f"same direction {int(row['loso_same_sign_n'])}/"
                f"{int(row['n_loso_deletions'])};  "
                f"largest: {row['most_influential_subject']}"
            ),
            ha="left",
            va="center",
            fontsize=6.55,
            color=INK,
            linespacing=1.23,
        )

    axis.axvline(0, color=MUTED, linewidth=0.8, linestyle=(0, (3, 2)), zorder=0)
    axis.set_xlim(-1.0, 1.0)
    axis.set_ylim(-0.55, n_rows - 0.45)
    axis.set_yticks(y_positions, summary["display_label"].tolist())
    axis.set_xlabel("Spearman $\\rho$")
    clean_axis(axis)
    axis.grid(False, axis="y")
    annotation_axis.set_xlim(0, 1)
    annotation_axis.axis("off")

    fig.text(
        0.065,
        0.955,
        title,
        ha="left",
        va="top",
        fontsize=11.2,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.065,
        0.895,
        subtitle,
        ha="left",
        va="top",
        fontsize=7.1,
        color=MUTED,
    )
    has_diagnostic = bool((summary["analysis_role"] != "primary").any())
    legend_handles = [
        Line2D(
            [0], [0], marker="D", color=INK, markerfacecolor=INK,
            markersize=5.2, linewidth=1.0,
            label=(
                "Full N=15 primary (bootstrap 95% CI)"
                if has_diagnostic
                else "Full N=15 estimate (bootstrap 95% CI)"
            ),
        ),
    ]
    if has_diagnostic:
        legend_handles.append(
            Line2D(
                [0], [0], marker="D", color=INK, markerfacecolor="white",
                markeredgewidth=0.9, markersize=5.2, linewidth=1.0,
                label="Full N=15 diagnostic (bootstrap 95% CI)",
            )
        )
    legend_handles.extend(
        [
            Line2D(
                [0], [0], marker="o", linestyle="none", markerfacecolor="white",
                markeredgecolor=MUTED, markeredgewidth=0.7, markersize=4.4,
                label="15 LOSO estimates (n=14)",
            ),
            Patch(
                facecolor=PALE_RANGE, edgecolor="none",
                label="Complete LOSO range",
            ),
        ]
    )
    fig.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(0.065, 0.055 if n_rows <= 4 else 0.035),
        ncol=len(legend_handles),
        frameon=False,
        fontsize=6.3,
        handlelength=1.5,
        columnspacing=1.15,
    )
    fig.text(
        0.065,
        0.018,
        (
            "LOSO is descriptive influence analysis. Inferential statistics "
            "and multiplicity adjustment use the complete N=15 sample."
        ),
        ha="left",
        va="bottom",
        fontsize=6.15,
        color=MUTED,
    )
    save_publication_figure(fig, stem, dpi)


def plot_all_reviewer_loso_outputs(
    loso_long: pd.DataFrame,
    loso_summary: pd.DataFrame,
    out_dir: Path,
    dpi: int,
) -> None:
    plot_reviewer_loso_influence(
        loso_long,
        loso_summary,
        [
            "qt1_ff_evc_superficial",
            "qt1_fb_caupfc_deep",
            "mpc_ff_evc_caupfc",
            "mpc_fb_evc_caupfc",
        ],
        "Participant influence on primary structure–rDCM associations",
        (
            "Four prespecified tests; full-sample inference with descriptive "
            "leave-one-subject-out stability (N=15)."
        ),
        out_dir / "fig_structure_rdcm_primary_loso_influence",
        dpi,
    )
    plot_reviewer_loso_influence(
        loso_long,
        loso_summary,
        [
            "qt1_ff_evc_deep",
            "qt1_ff_evc_middle",
            "qt1_ff_evc_superficial",
            "qt1_fb_caupfc_deep",
            "qt1_fb_caupfc_middle",
            "qt1_fb_caupfc_superficial",
        ],
        "Participant influence on layer-resolved qT1–rDCM associations",
        (
            "Primary source-layer tests and other-layer diagnostics; LOSO "
            "quantifies single-participant influence on Spearman correlations."
        ),
        out_dir / "fig_qt1_layer_loso_influence",
        dpi,
    )


def write_readme(out_dir: Path) -> None:
    text = """# Layer-resolved qT1/MPC-rDCM bridge

## Primary tests

This is a focused, hypothesis-guided analysis; it is not described as
preregistered or confirmatory. The four primary structure tests are frozen
together:

1. **qT1 source-layer tests**: FF versus EVC superficial qT1-derived value;
   FB versus cauPFC deep qT1-derived value.
2. **MPC interregional tests**: EVC-cauPFC MPC versus FF and FB.

Holm correction is applied across this complete four-test primary family.

`qT1` is described as a qT1-derived depth-profile measure. Without B1+
correction it must not be interpreted as a direct or fully quantitative
measure of myelin.

## Layer-resolved qT1 tests

Only `intensity_Deep`, `intensity_Middle` and `intensity_Superficial` are used.
The source file may contain `depth_gradient`, `skewness_full` and
`skewness_*`, but this script ignores them and never exports derived
between-layer differences.

The four qT1 pairs not in the primary family are a separate diagnostic family:
FF versus EVC deep/middle and FB versus cauPFC middle/superficial. Holm
correction is applied across those four tests. The file
`qt1_layer_resolved_tests.csv` combines all six qT1 layer rows for reporting,
while retaining each row's original multiplicity family.

The only additional directional diagnostic is signed H versus MPC.

## Reviewer-facing LOSO influence analysis

For every one of the nine reported associations, the participant-level
Spearman coefficient is recalculated 15 times after omitting one participant.
The structural features and rDCM models are not re-estimated. This matches the
reviewer question about whether a participant drives the correlation while
avoiding the misleading presentation of LOSO deletions as independent tests.

- `structure_rdcm_loso_by_participant_long.csv` contains one row per test and
  omitted participant, including the omitted values, retained-sample rho,
  change from the full-sample rho and influence rank.
- `structure_rdcm_loso_by_participant_wide.csv` is a compact participant-by-test
  version of the same retained-sample coefficients.
- `structure_rdcm_loso_summary.csv` reports the complete LOSO range, quartiles,
  direction count and participant producing the largest coefficient change.
- `reviewer_loso_report.md` is a concise reviewer-facing numerical summary.
- `fig_structure_rdcm_primary_loso_influence.*` and
  `fig_qt1_layer_loso_influence.*` show the full-sample estimate and bootstrap
  CI together with all 15 LOSO estimates and their complete range.

LOSO is descriptive influence analysis. Formal bootstrap intervals,
permutation P values and Holm adjustments remain based on the full N=15
sample. A stable LOSO direction does not constitute independent replication.

## rDCM metric dictionary

- `signed_FF`: mean of the two prespecified feedforward edges.
- `signed_FB`: mean of the two prespecified feedback edges.
- `signed_H`: `signed_FB - signed_FF`.

`signed_H` and signed `FB - FF` are identical. H is a derived within-subject
contrast, not an independently estimated third pathway. This script does not
calculate or test an `absolute_H` variable. In older scripts that label denoted
`mean(abs(FB edges)) - mean(abs(FF edges))`, not `abs(FB - FF)`, and is
therefore excluded here to prevent a signed-versus-magnitude ambiguity.

## Key lineage

- rDCM is reconstructed only from the final `pial5_current` EVC-cauPFC model.
- EVC is exactly V1, V2 and V3, aggregated with an equal-ROI mean.
- cauPFC is exactly 6d, 6v, 6mp, FEF, PEF, 55b and SFL, aggregated with an
  equal-ROI mean.
- qT1 is retained separately for deep, middle and superficial cortex and
  aggregated across parcels with a strict equal-ROI mean within each stage.
  It is not averaged across cortical depth.
- MPC is extracted from raw micapipe MPC matrices and does not use a
  functional-gradient component.
- Structural qT1/MPC maps are anatomical inputs; outer-pial BOLD exclusion
  applies to the rDCM time-series lineage, not to the structural maps.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    subjects = tuple(normalize_subject(x) for x in args.subjects)
    if len(subjects) != args.expected_n or len(set(subjects)) != len(subjects):
        raise RuntimeError(
            f"Expected {args.expected_n} unique participants, got "
            f"{len(subjects)} rows / {len(set(subjects))} unique"
        )

    rdcm, rdcm_audit = load_rdcm_features(args.rdcm_long, subjects)
    qt1, qt1_audit = load_qt1_features(args.qt1_long, subjects)
    mpc, mpc_audit = load_mpc_features(
        args.mpc_base, args.lut_path, subjects,
        args.acq_tag, args.atlas_tag,
    )
    data = rdcm.merge(qt1, on="subject", validate="one_to_one")
    data = data.merge(mpc, on="subject", validate="one_to_one")
    if len(data) != args.expected_n:
        raise RuntimeError(
            f"Merged table has {len(data)} participants, expected {args.expected_n}"
        )
    numeric = data.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy(float)).all():
        bad = numeric.columns[~np.isfinite(numeric.to_numpy(float)).all(axis=0)]
        raise RuntimeError(f"Merged numeric table contains non-finite values: {list(bad)}")

    h_reconstructed = data["signed_FB"] - data["signed_FF"]
    h_max_abs_diff = float(
        np.max(np.abs(data["signed_H"].to_numpy(float) - h_reconstructed.to_numpy(float)))
    )
    if h_max_abs_diff > 1e-12:
        raise RuntimeError(
            "signed_H definition audit failed: max |H - (FB - FF)| = "
            f"{h_max_abs_diff:.16g}"
        )

    primary, layer_diagnostic, directional, qt1_layer_resolved = analyze(
        data, args.n_permutations, args.n_bootstrap, args.seed
    )
    reviewer_inventory = reviewer_test_inventory(
        primary, layer_diagnostic, directional
    )
    loso_long, loso_summary, loso_wide = build_reviewer_loso_tables(
        data, reviewer_inventory
    )
    loso_audit = validate_reviewer_loso_tables(
        reviewer_inventory,
        loso_long,
        loso_summary,
        args.expected_n,
    )
    loso_audit.to_csv(
        args.out_dir / "structure_rdcm_loso_audit.csv", index=False
    )
    if loso_audit.iloc[0]["status"] != "PASS":
        raise RuntimeError(
            "Reviewer-facing LOSO validation failed: "
            f"{loso_audit.iloc[0]['failure_details']}"
        )
    forbidden_qt1_tokens = ("depth_gradient", "skewness")
    forbidden_output_columns = [
        column for column in data.columns
        if any(token in column for token in forbidden_qt1_tokens)
    ]
    if forbidden_output_columns:
        raise RuntimeError(
            "Non-layer qT1 metrics leaked into the analysis table: "
            f"{forbidden_output_columns}"
        )
    data.to_csv(args.out_dir / "structure_rdcm_subjects.csv", index=False)
    primary.to_csv(args.out_dir / "structure_rdcm_primary_tests.csv", index=False)
    layer_diagnostic.to_csv(
        args.out_dir / "qt1_other_layer_diagnostic_tests.csv", index=False
    )
    directional.to_csv(
        args.out_dir / "mpc_directional_diagnostic_test.csv", index=False
    )
    qt1_layer_resolved.to_csv(
        args.out_dir / "qt1_layer_resolved_tests.csv", index=False
    )
    reviewer_inventory.to_csv(
        args.out_dir / "structure_rdcm_reviewer_test_inventory.csv", index=False
    )
    loso_long.to_csv(
        args.out_dir / "structure_rdcm_loso_by_participant_long.csv", index=False
    )
    loso_wide.to_csv(
        args.out_dir / "structure_rdcm_loso_by_participant_wide.csv", index=False
    )
    loso_summary.to_csv(
        args.out_dir / "structure_rdcm_loso_summary.csv", index=False
    )
    rdcm_audit.to_csv(args.out_dir / "rdcm_input_audit.csv", index=False)
    qt1_audit.to_csv(args.out_dir / "qt1_input_audit.csv", index=False)
    mpc_audit.to_csv(args.out_dir / "mpc_input_audit.csv", index=False)
    metric_dictionary = pd.DataFrame([
        {
            "metric": "signed_FF",
            "formula": (
                "mean(A[cauPFC_deep,EVC_sup], "
                "A[cauPFC_mid,EVC_sup])"
            ),
            "estimated_or_derived": "estimated-edge average",
            "analysis_role": "primary component",
        },
        {
            "metric": "signed_FB",
            "formula": (
                "mean(A[EVC_deep,cauPFC_deep], "
                "A[EVC_sup,cauPFC_deep])"
            ),
            "estimated_or_derived": "estimated-edge average",
            "analysis_role": "primary component",
        },
        {
            "metric": "signed_H",
            "formula": "signed_FB - signed_FF",
            "estimated_or_derived": "derived contrast",
            "analysis_role": "diagnostic directional asymmetry",
        },
    ])
    metric_dictionary.to_csv(
        args.out_dir / "rdcm_metric_dictionary.csv", index=False
    )
    pd.DataFrame([{
        "check": "signed_H equals signed_FB minus signed_FF",
        "n_subjects": len(data),
        "max_abs_diff": h_max_abs_diff,
        "tolerance": 1e-12,
        "status": "PASS",
    }]).to_csv(
        args.out_dir / "rdcm_metric_definition_audit.csv", index=False
    )
    pd.DataFrame([
        {
            "check": "only layer-specific qT1 central values are analysed",
            "included_columns": "|".join(QT1_LAYER_METRICS),
            "excluded_column_patterns": "depth_gradient|skewness*",
            "forbidden_output_columns": "|".join(forbidden_output_columns),
            "qT1_averaged_across_depth": False,
            "status": "PASS",
        }
    ]).to_csv(
        args.out_dir / "qt1_metric_definition_audit.csv", index=False
    )

    lineage = {
        "status": "PASS",
        "subjects": list(subjects),
        "expected_n": args.expected_n,
        "rdcm_long": str(args.rdcm_long),
        "rdcm_long_sha256": sha256_file(args.rdcm_long),
        "qt1_long": str(args.qt1_long),
        "qt1_long_sha256": sha256_file(args.qt1_long),
        "mpc_base": str(args.mpc_base),
        "lut_path": str(args.lut_path),
        "lut_sha256": sha256_file(args.lut_path),
        "EVC_ROIs": list(EVC_ROIS),
        "cauPFC_ROIs": list(CAUPFC_ROIS),
        "qT1_metrics_used": list(QT1_LAYER_METRICS),
        "qT1_metrics_excluded": [
            "depth_gradient", "skewness_full",
            "skewness_Deep", "skewness_Middle",
            "skewness_Superficial",
        ],
        "qT1_aggregation": (
            "separate deep/middle/superficial values; strict equal-ROI "
            "mean within EVC and cauPFC; no averaging across depth"
        ),
        "MPC_primary_feature": (
            "equal mean of within-hemisphere LL and RR MPC values across "
            "3 EVC x 7 cauPFC parcel pairs"
        ),
        "rdcm_convention": "A(target, source)",
        "FF_definition": (
            "mean(EVC sup to cauPFC deep, EVC sup to cauPFC mid)"
        ),
        "FB_definition": (
            "mean(cauPFC deep to EVC deep, cauPFC deep to EVC sup)"
        ),
        "H_definition": "signed_H = signed_FB - signed_FF",
        "H_is_identical_to_signed_FB_minus_FF": True,
        "absolute_H_used": False,
        "rdcm_metric_definition_audit_max_abs_diff": h_max_abs_diff,
        "primary_multiplicity": "Holm across all four primary tests",
        "qT1_layer_diagnostic_multiplicity": (
            "Holm across four other-layer qT1 tests"
        ),
        "directional_diagnostic_multiplicity": (
            "single H-versus-MPC test; adjusted p equals raw p"
        ),
        "n_permutations": args.n_permutations,
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "functional_gradient_used": False,
        "reviewer_loso_analysis": (
            "descriptive participant-deletion influence analysis of Spearman "
            "correlations; no feature or rDCM re-estimation"
        ),
        "reviewer_loso_n_deletions_per_test": args.expected_n,
        "reviewer_loso_n_retained_per_deletion": args.expected_n - 1,
        "reviewer_loso_formal_inference": (
            "none; formal bootstrap, permutation and Holm inference uses the "
            "complete N=15 sample"
        ),
    }
    (args.out_dir / "analysis_lineage.json").write_text(
        json.dumps(lineage, indent=2), encoding="utf-8"
    )
    write_readme(args.out_dir)
    write_reviewer_loso_report(loso_summary, args.out_dir)
    plot_primary_scatter(data, primary, args.out_dir)
    plot_primary_forest(primary, args.out_dir)
    plot_qt1_layer_forest(qt1_layer_resolved, args.out_dir)
    plot_all_reviewer_loso_outputs(
        loso_long, loso_summary, args.out_dir, args.dpi
    )

    reviewer_outputs = [
        "structure_rdcm_reviewer_test_inventory.csv",
        "structure_rdcm_loso_by_participant_long.csv",
        "structure_rdcm_loso_by_participant_wide.csv",
        "structure_rdcm_loso_summary.csv",
        "structure_rdcm_loso_audit.csv",
        "reviewer_loso_report.md",
        "fig_structure_rdcm_primary_loso_influence.png",
        "fig_structure_rdcm_primary_loso_influence.pdf",
        "fig_structure_rdcm_primary_loso_influence.svg",
        "fig_qt1_layer_loso_influence.png",
        "fig_qt1_layer_loso_influence.pdf",
        "fig_qt1_layer_loso_influence.svg",
    ]
    manifest = {
        "script_version": SCRIPT_VERSION,
        "analysis": "qT1_MPC_rDCM_with_reviewer_facing_LOSO_influence",
        "output_directory": str(args.out_dir.resolve()),
        "expected_n": args.expected_n,
        "loso_n_retained": args.expected_n - 1,
        "loso_is_descriptive": True,
        "formal_inference_sample": "complete N=15 sample",
        "inputs": {
            "rdcm_long": {
                "path": str(args.rdcm_long.resolve()),
                "sha256": sha256_file(args.rdcm_long),
            },
            "qt1_long": {
                "path": str(args.qt1_long.resolve()),
                "sha256": sha256_file(args.qt1_long),
            },
            "lut": {
                "path": str(args.lut_path.resolve()),
                "sha256": sha256_file(args.lut_path),
            },
        },
        "reviewer_outputs": reviewer_outputs,
        "n_permutations": args.n_permutations,
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "dpi": args.dpi,
    }
    (args.out_dir / "reviewer_loso_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    display_cols = [
        "family", "test", "n_subjects", "spearman_rho",
        "bootstrap_ci95_low", "bootstrap_ci95_high",
        "permutation_p_two_sided", "holm_p_across_4_primary",
    ]
    print("\nPrimary structure-rDCM tests")
    print(primary[display_cols].to_string(index=False))
    layer_display_cols = [
        "family", "test", "n_subjects", "spearman_rho",
        "bootstrap_ci95_low", "bootstrap_ci95_high",
        "permutation_p_two_sided",
        "holm_p_across_4_layer_diagnostics",
    ]
    print("\nOther-layer qT1 diagnostic tests")
    print(layer_diagnostic[layer_display_cols].to_string(index=False))
    print("\nDirectional MPC diagnostic")
    print(directional[[
        "family", "test", "n_subjects", "spearman_rho",
        "bootstrap_ci95_low", "bootstrap_ci95_high",
        "permutation_p_two_sided",
    ]].to_string(index=False))
    print("\nReviewer-facing participant-deletion LOSO summary")
    print(loso_summary[[
        "analysis_role", "test", "full_sample_rho",
        "loso_rho_min", "loso_rho_max", "loso_same_sign_n",
        "n_loso_deletions", "most_influential_subject",
        "max_abs_delta_rho",
    ]].to_string(index=False))
    print("\nLOSO validation status: PASS")
    print(f"\nSaved: {args.out_dir}")


if __name__ == "__main__":
    main()
