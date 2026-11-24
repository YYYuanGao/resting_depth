#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Depth-resolved qT1 summaries and MPC gradients from micapipe outputs (v3.1, 2026-11-10).

Changes from compute_mpc_gradient_only_micapipe2.py
---------------------------------------------------
1. Depth orientation. micapipe intensity profiles are ordered from the pial side:
   row 0 is the outermost retained equivolumetric surface and row 13 the innermost
   (micapipe generates 16 surfaces and drops the two boundary surfaces). v2 labelled
   rows 0-4 as "Deep", which is the pial side. Depth bins are now defined from the
   pial surface, and the orientation is checked against the data at run time
   (qT1 must decrease from the pial to the white-matter side).
2. Depth bins. Default "thirds": outer, middle and inner thirds of cortical thickness
   (surfaces 1-5, 6-10, 11-14 counted from the pial side; surface k lies at k/15 of
   thickness). "5-4-5" reproduces the v2 split with corrected labels, for sensitivity.
3. ROI names. v2 mapped "a9-46v" to "p9-46v"; both exist in HCP-MMP, so the LUT entry
   for a9-46v overwrote p9-46v. Only spelling variants are normalised now, and name
   collisions raise an error.
4. No fallback parcel order. v2 fell back to a hard-coded list that is not the HCP-MMP
   order and contains non-existent areas. The LUT is now required and every ROI must match.
5. Composites. The manuscript composites are defined explicitly (EVC 3, PPC 13,
   cauPFC 7 parcels). v2's "cau_pfc" scope contained 15 parcels, and its "PPC" group
   (47 parcels) is renamed "Parietal" so it is not confused with the PPC composite.
6. Per-bin skewness (4-5 points) is no longer computed; whole-profile moments
   (14 points) are kept.
7. Thickness defaults to parcel-wise values ("glasser"); "global" assigned one
   subject-wide value to every parcel.
8. MPC gradients use a fixed random seed and explicit sparsity (default 0.9).
9. v3.1: the slab ROI set is read from the 96-region Bianca/Serio functional reference
   (roi_names.npy); area H (column 119 / 299), absent from the LUT, is added explicitly;
   outputs go to Step4_MPC_G1/Nov.
"""

from pathlib import Path
import argparse
import warnings

import numpy as np
import pandas as pd
from scipy.stats import kurtosis as scipy_kurtosis
from scipy.stats import skew as scipy_skew

import os
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "/path/to/project")  # see config/paths.sh

warnings.filterwarnings("ignore")
import nibabel as nib
from brainspace.gradient import GradientMaps


BASE_ROOT = Path(PROJECT_ROOT + "")
MPC_BASE = BASE_ROOT / "derivatives" / "micapipe_v0.2.0"
FASTSURFER_BASE = BASE_ROOT / "derivatives" / "fastsurfer"
LUT_PATH = BASE_ROOT / "derivatives" / "HCPMMP_parcellation" / "sub-01" / "LUT_HCPMMP1.txt"
GRAD_DIR = BASE_ROOT / "derivatives_0331" / "S6_validation" / "pial5_only_update" / "functional_gradient_bianca_reference"
OUT_DIR = BASE_ROOT / "derivatives_0331" / "Step4_MPC_G1" / "Nov" / "mpc_gradient_only_micapipe3"

SUBJECTS = ["01", "02", "03", "04", "05", "06", "07",
            "08", "09", "10", "11", "12", "13", "14", "16"]

ACQ_TAG = "acq-T1map"
ATLAS_TAG = "glasser-360"
N_PARCELS = 360
N_HEMI = 180
N_DEPTHS = 14
N_GRADIENTS = 10

# Row index of the micapipe profile -> position from the pial side (0 = outermost retained surface).
DEPTH_SCHEMES = {
    # thirds of cortical thickness; surface k (1-based) lies at k/15 of thickness from the pial surface
    "thirds": {"Superficial": list(range(0, 5)), "Middle": list(range(5, 10)), "Deep": list(range(10, 14))},
    # v2 split (5/4/5 surfaces) with corrected labels
    "5-4-5": {"Superficial": list(range(0, 5)), "Middle": list(range(5, 9)), "Deep": list(range(9, 14))},
}

# Spelling variants only. Distinct HCP-MMP areas (a9-46v, p9-46v, 9-46d) are never merged.
ROI_NAME_MAP = {"7Al": "7AL", "8AD": "8Ad", "8AV": "8Av"}


def normalize_roi_name(roi):
    return ROI_NAME_MAP.get(str(roi), str(roi))


# Composites used in the manuscript
EVC_ROIS = {"V1", "V2", "V3"}
PPC_ROIS = {"LIPv", "LIPd", "VIP", "MIP", "IP0", "IP1", "IP2", "7PC", "7Am", "7AL", "7PL", "7Pm", "7m"}
CAUPFC_ROIS = {"6d", "6v", "6mp", "FEF", "PEF", "55b", "SFL"}

# Broad lobar groups (descriptive only; used for the MPC G1 sign convention)
OCCIPITAL_SET = {
    "V1", "V2", "V3", "V4", "V3A", "V3B", "V8", "MT", "MST", "V6", "V6A", "V7",
    "LO1", "LO2", "LO3", "V4t", "FST", "V3CD", "FFC", "VMV1", "VMV2", "PCV",
}
PARIETAL_SET = {
    "IPS1", "LIPv", "LIPd", "VIP", "AIP", "MIP", "IP0", "IP1", "IP2",
    "7PC", "7Am", "7AL", "7PL", "7Pm", "7m",
    "PGp", "PGi", "PGs", "PF", "PFm", "PFop", "PFcm", "PFt",
    "OP1", "OP2-3", "OP4",
    "POS1", "POS2", "RSC", "DVT", "31pd", "31pv", "v23ab", "d23ab", "23d", "23c",
    "TPOJ1", "TPOJ2", "TPOJ3", "STV", "PHA2", "PHA3",
    "5L", "5m", "5mv",
}
MOTOR_SET = {"1", "2", "3a", "3b", "4"}
FRONTAL_SET = {"6a", "6ma", "6d", "6v", "6mp", "6r", "i6-8", "SCEF", "FEF", "PEF", "55b",
               "SFL", "8Ad", "8C", "p9-46v"}
GROUP_DEFS = {"Occipital": OCCIPITAL_SET, "Parietal": PARIETAL_SET, "Motor": MOTOR_SET, "Frontal": FRONTAL_SET}

SCOPES = {
    "evc": EVC_ROIS,
    "ppc": PPC_ROIS,
    "caupfc": CAUPFC_ROIS,
    "evc_caupfc": EVC_ROIS | CAUPFC_ROIS,
    "v1": {"V1"},
}

PROFILE_LOAD_QC, MPC_LOAD_QC, THICKNESS_LOAD_QC = [], [], []


def get_group(name):
    name = normalize_roi_name(name)
    for group, rois in GROUP_DEFS.items():
        if name in rois:
            return group
    return "Other"


def composite_of(name):
    name = normalize_roi_name(name)
    for label, rois in (("EVC", EVC_ROIS), ("PPC", PPC_ROIS), ("cauPFC", CAUPFC_ROIS)):
        if name in rois:
            return label
    return ""


def profile_moments(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    out = dict(profile_mean=np.nan, profile_std=np.nan, profile_skewness=np.nan, profile_kurtosis=np.nan)
    if x.size == 0:
        return out
    out["profile_mean"] = float(np.mean(x))
    out["profile_std"] = float(np.std(x))
    if x.size >= 3 and out["profile_std"] > 1e-12:
        out["profile_skewness"] = float(scipy_skew(x, bias=False))
        out["profile_kurtosis"] = float(scipy_kurtosis(x, fisher=False, bias=False))
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Depth-resolved qT1 summaries and MPC gradients (v3).")
    p.add_argument("--mpc-base", type=Path, default=MPC_BASE)
    p.add_argument("--fastsurfer-base", type=Path, default=FASTSURFER_BASE)
    p.add_argument("--lut-path", type=Path, default=LUT_PATH)
    p.add_argument("--grad-dir", type=Path, default=GRAD_DIR)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--subjects", nargs="+", default=SUBJECTS)
    p.add_argument("--depth-scheme", choices=sorted(DEPTH_SCHEMES), default="thirds")
    p.add_argument("--thickness-mode", choices=["glasser", "global", "none"], default="glasser")
    p.add_argument("--mpc-sparsity", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def read_functional_roi_names(grad_dir):
    npy = grad_dir / "roi_names.npy"
    if npy.exists():
        return [normalize_roi_name(str(r)) for r in np.load(npy, allow_pickle=True)], npy
    for name in ("G1_scores_all_rois.csv", "gradient_scores_all_rois.csv", "G1_scores_by_subject_and_roi.csv"):
        path = grad_dir / name
        if path.exists():
            df = pd.read_csv(path)
            col = next((c for c in df.columns if c.lower() in {"roi", "roi_name", "region"}), None)
            if col is None:
                raise ValueError(f"No ROI column in {path}")
            return list(dict.fromkeys(df[col].map(normalize_roi_name).astype(str))), path
    raise FileNotFoundError(f"No functional gradient ROI table in {grad_dir}")


def parse_lut_file(lut_path):
    """HCP-MMP LUT -> {roi: column}; columns 0-179 left, 180-359 right, after background removal.
    Columns follow label value (1002 = V1 -> 0); H (1121/2121) is absent from this LUT and its
    column (119 / 299) is simply not addressed."""
    if not lut_path.exists():
        raise FileNotFoundError(f"LUT not found: {lut_path}")
    lh, rh = {}, {}
    lh_base = rh_base = None
    for line in open(lut_path):
        parts = line.split()
        if len(parts) < 2 or not parts[0].isdigit():
            continue
        idx, name = int(parts[0]), parts[1]
        for prefix, target, hemi in (("lh.L_", lh, "L"), ("rh.R_", rh, "R")):
            if name.startswith(prefix) and name.endswith("_ROI.label"):
                short = normalize_roi_name(name[len(prefix):-10])
                if short in target:
                    raise ValueError(f"ROI name collision in LUT after normalisation: {name} -> {short}")
                if hemi == "L":
                    lh_base = idx if lh_base is None else lh_base
                    target[short] = idx - lh_base
                else:
                    rh_base = idx if rh_base is None else rh_base
                    target[short] = idx - rh_base + N_HEMI
    if lh_base != 1002 or rh_base != 2002:
        raise ValueError(f"Unexpected LUT bases {lh_base}, {rh_base}; expected 1002 (L_V1) and 2002 (R_V1)")
    # H (HCP-MMP area 120; LUT 1121/2121) is missing from this LUT but present in the micapipe
    # Glasser-360 annotation and matrices, at column 119 (left) and 299 (right).
    if "H" not in lh and 119 not in lh.values() and 299 not in rh.values():
        lh["H"], rh["H"] = 119, 299
    if len(lh) != N_HEMI or sorted(lh.values()) != list(range(N_HEMI)):
        raise ValueError("Left-hemisphere LUT mapping does not cover columns 0-179 exactly once")
    return lh, rh


def gifti_to_array(fp):
    return np.squeeze(np.array([np.asarray(d.data) for d in nib.load(str(fp)).darrays]))


def load_parcel_profiles(args, sub):
    fp = args.mpc_base / f"sub-{sub}" / "mpc" / ACQ_TAG / f"sub-{sub}_atlas-{ATLAS_TAG}_desc-intensity_profiles.shape.gii"
    if not fp.exists():
        print(f"[WARN] missing profiles: {fp}")
        return None
    data = np.asarray(gifti_to_array(fp), dtype=np.float32)
    if data.shape == (N_DEPTHS, N_PARCELS + 2):
        prof = data
    elif data.shape == (N_PARCELS + 2, N_DEPTHS):
        prof = data.T
    else:
        raise ValueError(f"sub-{sub}: unexpected profile shape {data.shape}")
    if not (np.all(np.isnan(prof[:, 0]) | (prof[:, 0] == 0)) and np.all(np.isnan(prof[:, N_HEMI + 1]) | (prof[:, N_HEMI + 1] == 0))):
        raise ValueError(f"sub-{sub}: background columns 0 and {N_HEMI + 1} are not empty")
    prof = np.delete(prof, [0, N_HEMI + 1], axis=1)
    PROFILE_LOAD_QC.append(dict(subject=f"sub-{sub}", file=str(fp), raw_shape=str(data.shape), final_shape=str(prof.shape)))
    return prof


def check_depth_orientation(profiles_by_sub):
    """qT1 must be longer at the pial side than at the white-matter side in (almost) all parcels."""
    rows = []
    for sub, p in profiles_by_sub.items():
        frac = float(np.nanmean(p[0] > p[-1]))
        rows.append(dict(subject=f"sub-{sub}", qT1_row0=float(np.nanmean(p[0])), qT1_row13=float(np.nanmean(p[-1])),
                         frac_parcels_row0_gt_row13=frac))
        if frac < 0.9:
            raise RuntimeError(f"sub-{sub}: only {frac:.0%} of parcels have qT1(row 0) > qT1(row 13); "
                               "profile orientation differs from the expected pial-to-white order")
    return pd.DataFrame(rows)


def load_parcel_thickness(args, sub, lh, rh):
    if args.thickness_mode == "none":
        return None
    maps_dir = args.mpc_base / f"sub-{sub}" / "maps"
    if args.thickness_mode == "global":
        files = sorted(maps_dir.glob("*surf-fsnative*label-thickness.func.gii"))
        vals = [np.nanmean(v[v > 0]) for v in (np.asarray(gifti_to_array(f)).ravel() for f in files[:2])]
        return np.full(N_PARCELS, float(np.nanmean(vals)), dtype=np.float32) if vals else None
    from nibabel.freesurfer.io import read_annot
    full = np.full(N_PARCELS, np.nan, dtype=np.float32)
    for hemi, prefix, lut in (("L", "lh", lh), ("R", "rh", rh)):
        tf = next(iter(sorted(maps_dir.glob(f"*hemi-{hemi}*surf-fsnative*label-thickness.func.gii"))), None)
        af = args.fastsurfer_base / f"sub-{sub}" / "label" / f"{prefix}.glasser-360_mics.annot"
        if tf is None or not af.exists():
            continue
        th = np.asarray(gifti_to_array(tf), dtype=np.float32).ravel()
        labels, ctab, names = read_annot(str(af))
        if len(th) != len(labels):
            raise ValueError(f"sub-{sub} hemi-{hemi}: thickness/annot length mismatch")
        for li, raw in enumerate(names):
            s = (raw.decode() if isinstance(raw, bytes) else str(raw)).replace("_ROI", "").replace(".label", "")
            for pre in ("L_", "R_", "lh.", "rh."):
                s = s[len(pre):] if s.startswith(pre) else s
            s = normalize_roi_name(s)
            if s in lut:
                v = th[(labels == li) & np.isfinite(th)]
                if v.size:
                    full[lut[s]] = float(v.mean())
        THICKNESS_LOAD_QC.append(dict(subject=f"sub-{sub}", hemi=hemi, thickness=str(tf), annot=str(af)))
    return full if np.isfinite(full).any() else None


def load_mpc_matrix(args, sub):
    fp = args.mpc_base / f"sub-{sub}" / "mpc" / ACQ_TAG / f"sub-{sub}_atlas-{ATLAS_TAG}_desc-MPC.shape.gii"
    if not fp.exists():
        print(f"[WARN] missing MPC: {fp}")
        return None
    raw = np.asarray(gifti_to_array(fp), dtype=np.float64)
    if raw.shape != (N_PARCELS + 2, N_PARCELS + 2):
        raise ValueError(f"sub-{sub}: unexpected MPC shape {raw.shape}")
    up = np.triu(raw, 1)
    mat = up + up.T                                   # micapipe stores the upper triangle
    keep = [i for i in range(N_PARCELS + 2) if i not in (0, N_HEMI + 1)]
    mat = mat[np.ix_(keep, keep)]
    np.fill_diagonal(mat, 0.0)
    MPC_LOAD_QC.append(dict(subject=f"sub-{sub}", file=str(fp), n_offdiag_zero=int((mat == 0).sum() - N_PARCELS),
                            min=float(mat.min()), max=float(mat.max())))
    return mat


def embed(mat, args, reference=None):
    gm = GradientMaps(n_components=N_GRADIENTS, kernel="normalized_angle", approach="dm",
                      random_state=args.seed, alignment="procrustes" if reference is not None else None)
    gm.fit(mat, sparsity=args.mpc_sparsity, alpha=0.5, diffusion_time=0, reference=reference)
    return (gm.aligned_ if reference is not None else gm.gradients_), gm.lambdas_


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    layer_map = DEPTH_SCHEMES[args.depth_scheme]

    roi_names, grad_table = read_functional_roi_names(args.grad_dir)
    lh, rh = parse_lut_file(args.lut_path)
    missing = [r for r in roi_names if r not in lh or r not in rh]
    if missing:
        raise ValueError(f"ROIs not found in LUT: {missing}")
    for comp in (EVC_ROIS, PPC_ROIS, CAUPFC_ROIS):
        absent = sorted(comp - set(lh))
        if absent:
            raise ValueError(f"Composite parcels not found in LUT: {absent}")
    all_rois = sorted(lh, key=lambda r: lh[r])          # all 180 HCP-MMP areas, in column order

    pd.DataFrame({"roi": all_rois, "lh_col": [lh[r] for r in all_rois], "rh_col": [rh[r] for r in all_rois],
                  "in_functional_table": [r in roi_names for r in all_rois],
                  "composite": [composite_of(r) for r in all_rois], "group": [get_group(r) for r in all_rois]}
                 ).to_csv(args.out_dir / "atlas_label_check.csv", index=False)

    profiles = {s: p for s in args.subjects if (p := load_parcel_profiles(args, s)) is not None}
    orient = check_depth_orientation(profiles)
    orient.to_csv(args.out_dir / "depth_orientation_check.csv", index=False)
    print(f"[INFO] depth orientation OK: qT1 row0 {orient.qT1_row0.mean():.3f} > row13 {orient.qT1_row13.mean():.3f}")

    thickness = {s: load_parcel_thickness(args, s, lh, rh) for s in profiles}

    rows = []
    for sub, prof in profiles.items():
        for roi in all_rois:
            bil = np.nanmean(np.vstack([prof[:, lh[roi]], prof[:, rh[roi]]]), axis=0)   # 14 depths, pial -> white
            th = thickness.get(sub)
            row = dict(subject=f"sub-{sub}", roi=roi, composite=composite_of(roi), group=get_group(roi),
                       in_functional_table=roi in roi_names, depth_scheme=args.depth_scheme,
                       thickness=float(np.nanmean([th[lh[roi]], th[rh[roi]]])) if th is not None else np.nan)
            row.update(profile_moments(bil))
            for layer, idx in layer_map.items():
                row[f"intensity_{layer}"] = float(np.nanmean(bil[idx]))
                row[f"std_{layer}"] = float(np.nanstd(bil[idx]))
            row["depth_gradient_sup_minus_deep"] = row["intensity_Superficial"] - row["intensity_Deep"]
            for d in range(N_DEPTHS):
                row[f"qT1_surface{d + 1:02d}_from_pial"] = float(bil[d])
            rows.append(row)
    long = pd.DataFrame(rows)
    long.to_csv(args.out_dir / "subject_roi_intensity_moments_long.csv", index=False)

    value_cols = [c for c in long.columns if c.startswith(("intensity_", "std_", "profile_", "depth_gradient", "thickness"))]
    summ = []
    for scope, rois in SCOPES.items():
        sdf = long[long.roi.isin(rois)]
        g = sdf.groupby("subject")[value_cols].mean().reset_index()      # equal-ROI weighting
        g.insert(1, "roi_scope", scope)
        g.insert(2, "n_rois", sdf.roi.nunique())
        summ.append(g)
    pd.concat(summ).to_csv(args.out_dir / "subject_region_qt1_summary.csv", index=False)

    # ---- MPC gradients (whole cortex, native; summarised for the functional-table ROIs) ----
    mpcs = {s: m for s in args.subjects if (m := load_mpc_matrix(args, s)) is not None}
    pd.DataFrame(MPC_LOAD_QC).to_csv(args.out_dir / "mpc_loading_qc.csv", index=False)
    pd.DataFrame(PROFILE_LOAD_QC).to_csv(args.out_dir / "profile_loading_qc.csv", index=False)
    if THICKNESS_LOAD_QC:
        pd.DataFrame(THICKNESS_LOAD_QC).to_csv(args.out_dir / "thickness_loading_qc.csv", index=False)

    group = np.mean(np.stack(list(mpcs.values())), axis=0)
    g, lam = embed(group, args)
    pd.DataFrame({"component": np.arange(1, N_GRADIENTS + 1), "lambda": lam,
                  "variance_explained": lam / lam.sum()}).to_csv(args.out_dir / "mpc_lambdas.csv", index=False)

    def bilateral(grad):
        return {r: (grad[lh[r]] + grad[rh[r]]) / 2 for r in all_rois}

    gb = [bilateral(g[:, k]) for k in range(3)]
    occ = np.mean([gb[0][r] for r in all_rois if get_group(r) == "Occipital"])
    fro = np.mean([gb[0][r] for r in all_rois if get_group(r) == "Frontal"])
    if fro < occ:                                   # sign convention: frontal > occipital on G1
        g[:, 0] *= -1
        gb[0] = {k: -v for k, v in gb[0].items()}
    pd.DataFrame({"roi": all_rois, "composite": [composite_of(r) for r in all_rois],
                  "in_functional_table": [r in roi_names for r in all_rois],
                  **{f"mpc_g{k + 1}": [gb[k][r] for r in all_rois] for k in range(3)}}
                 ).to_csv(args.out_dir / "mpc_gradient_summary.csv", index=False)

    subj_rows = []
    for sub, m in mpcs.items():
        gi, _ = embed(m, args, reference=g)
        for k in range(3):
            b = bilateral(gi[:, k])
            subj_rows += [dict(subject=f"sub-{sub}", roi=r, component=f"G{k + 1}", score=b[r]) for r in all_rois]
    pd.DataFrame(subj_rows).to_csv(args.out_dir / "subject_mpc_gradient_scores_long.csv", index=False)

    with open(args.out_dir / "README.txt", "w") as fh:
        fh.write(f"Depth scheme: {args.depth_scheme} -> {layer_map} (row 0 = pial side)\n"
                 f"MPC sparsity {args.mpc_sparsity}, seed {args.seed}; functional ROI table: {grad_table}\n")
    print(f"Done: {args.out_dir}")


if __name__ == "__main__":
    main()
