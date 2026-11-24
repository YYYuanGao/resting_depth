#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 3: tSNR-based mask cleaning + time series extraction
==========================================================
Combines the original Script 5 (tSNR cleaning) and Script 6 (re-extraction).

Pipeline position:
  Script 4 → Step2_ROI_LayerMasks_AllROI_ColumnNonOverlap
  Script 5+6 → Step3_TimeSeries_tSNR_clean   ← this script

For each subject:
  1. Load 4D preprocessed fMRI time series
  2. Compute whole-brain tSNR map  (mean / std across time)
  3. For each ROI × layer combination:
       a. Start with the non-overlap mask from Script 4
       b. Remove voxels with tSNR below  mean - SIGMA × std  (within that mask)
       c. Skip mask if surviving voxels < MIN_VOX
       d. Extract mean time series from surviving voxels
  4. Save per-subject outputs (see Output below)

Input:
  4D func: BIDS/sub-XX/func/derivatives/fsl_afni_fsl_preproc/AP_dc_st_mc.nii.gz
  Masks:   Step2_ROI_LayerMasks_AllROI_ColumnNonOverlap/<tag>/sub-XX/

Output  Step3_TimeSeries_tSNR_clean/<tag>/sub-XX/:
  ts_matrix.npy          float32  (n_rois, n_layers, n_timepoints)
                         NaN slice = ROI-layer had too few clean voxels
  n_voxels_clean.npy     int32    (n_rois, n_layers) voxels AFTER tSNR cleaning
  n_voxels_raw.npy       int32    (n_rois, n_layers) voxels BEFORE cleaning
  roi_names.npy          str      (n_rois,)  ROI name list
  tsnr.nii.gz            float32  whole-brain tSNR map
  clean_masks/
    layer{L}_clean.nii.gz  int16  per-layer mask after tSNR cleaning (label = ROI idx)

Usage:
  python clean_extract_ts_tsnr.py --subject 01
  python clean_extract_ts_tsnr.py --parallel 4
  python clean_extract_ts_tsnr.py --tsnr-sigma 2.0 --min-vox 5 --force
"""

import argparse
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import nibabel as nib

import os
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "/path/to/project")  # see config/paths.sh

# ── global paths ──────────────────────────────────────────────────────────────
BIDS_ROOT  = Path(PROJECT_ROOT + "/BIDS")
ROOT       = Path(PROJECT_ROOT + "/derivatives_0331")
MASKS_BASE = ROOT / "Step2_ROI_LayerMasks_AllROI_ColumnNonOverlap"
OUT_BASE   = ROOT / "Step3_TimeSeries_tSNR_clean"

SUBJECTS = ["01","02","03","04","05","06","07","08","09","10","11","12","13","14","16"]

# Preprocessed 4-D fMRI filename (relative to per-subject func/derivatives/fsl_afni_fsl_preproc/)
FUNC_FNAME = "AP_dc_st_mc.nii.gz"


# ── argument parsing ──────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="tSNR cleaning + time series extraction (Scripts 5+6 combined)")
    p.add_argument("--subject",     default=None,
                   help="Single subject (e.g. 01). Omit to run all.")
    p.add_argument("--ld",          default="1000")
    p.add_argument("--nr-layers",   default=3, type=int)
    p.add_argument("--layer-method",default="equivol")
    p.add_argument("--tsnr-sigma",  default=2.0, type=float,
                   help="tSNR threshold = mean - SIGMA×std (default 2.0)")
    p.add_argument("--min-vox",     default=5, type=int,
                   help="Minimum voxels after cleaning (default 5); "
                        "ROI-layer gets NaN if below.")
    p.add_argument("--parallel",    default=1, type=int)
    p.add_argument("--force",       action="store_true")
    return p.parse_args()


def tag_from_args(args):
    return "ld%s_%dlayers_%s" % (args.ld, args.nr_layers, args.layer_method)


# ── tSNR helpers ──────────────────────────────────────────────────────────────
def compute_tsnr(func4d):
    """
    Compute tSNR = mean(t) / std(t) per voxel.
    Returns float32 3-D array; zero where std==0.
    """
    mean_t = func4d.mean(axis=-1)
    std_t  = func4d.std(axis=-1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        tsnr = np.where(std_t > 1e-9, mean_t / std_t, 0.0)
    return tsnr.astype(np.float32)


def clean_mask(roi_vox_mask, tsnr_vol, sigma=2.0, min_vox=5):
    """
    Given a boolean mask of voxels belonging to one ROI-layer:
      - Restrict to voxels with positive tSNR
      - Remove voxels where tSNR < mean - sigma * std  (within mask)
      - Return (clean_bool_mask, n_surviving)
    Returns (None, 0) if too few voxels survive.
    """
    valid = roi_vox_mask & (tsnr_vol > 0)
    n_raw = int(valid.sum())
    if n_raw < min_vox:
        return None, 0

    tsnr_vals = tsnr_vol[valid]
    thresh    = tsnr_vals.mean() - sigma * tsnr_vals.std()
    clean     = valid & (tsnr_vol >= thresh)
    n_clean   = int(clean.sum())

    if n_clean < min_vox:
        return None, 0
    return clean, n_clean


def extract_mean_ts(func4d, mask):
    """Average time series over mask voxels → 1-D float32 array."""
    return func4d[mask].mean(axis=0).astype(np.float32)


# ── per-subject worker ────────────────────────────────────────────────────────
def process_subject(sub, args_dict):
    import argparse as _ap
    args = _ap.Namespace(**args_dict)

    tag     = tag_from_args(args)
    out_dir = OUT_BASE / tag / ("sub-%s" % sub)
    sentinel = out_dir / "ts_matrix.npy"

    if sentinel.exists() and not args.force:
        print("[SKIP] sub-%s — output exists (--force to redo)" % sub)
        return sub, True

    # ── locate func time series ───────────────────────────────────────────────
    func_path = (BIDS_ROOT / ("sub-%s" % sub) / "func" / "derivatives"
                 / "fsl_afni_fsl_preproc" / FUNC_FNAME)
    if not func_path.exists():
        print("[ERROR] sub-%s: func not found:\n  %s" % (sub, func_path))
        return sub, False

    # ── locate Script 4 masks ─────────────────────────────────────────────────
    mask_dir  = MASKS_BASE / tag / ("sub-%s" % sub)
    roi_names_path = mask_dir / "roi_names.npy"
    if not mask_dir.exists() or not roi_names_path.exists():
        print("[ERROR] sub-%s: mask dir not found:\n  %s" % (sub, mask_dir))
        return sub, False

    # ── load ──────────────────────────────────────────────────────────────────
    print("[sub-%s] Loading func %s ..." % (sub, FUNC_FNAME))
    func_img  = nib.load(str(func_path))
    func4d    = np.asarray(func_img.dataobj, dtype=np.float32)
    if func4d.ndim != 4:
        print("[ERROR] sub-%s: func is not 4D (shape=%s)" % (sub, func4d.shape))
        return sub, False

    n_tp = func4d.shape[-1]
    print("[sub-%s] Shape: %s  (%d timepoints)" % (sub, func4d.shape, n_tp))

    roi_names = np.load(str(roi_names_path), allow_pickle=True)
    n_rois    = len(roi_names)
    n_layers  = args.nr_layers

    # ── compute tSNR ──────────────────────────────────────────────────────────
    print("[sub-%s] Computing tSNR ..." % sub)
    tsnr = compute_tsnr(func4d)

    # ── allocate output arrays ────────────────────────────────────────────────
    ts_matrix   = np.full((n_rois, n_layers, n_tp), np.nan, dtype=np.float32)
    n_vox_clean = np.zeros((n_rois, n_layers), dtype=np.int32)
    n_vox_raw   = np.zeros((n_rois, n_layers), dtype=np.int32)

    # ── per-layer clean masks (for saving) ────────────────────────────────────
    clean_mask_vols = {
        L: np.zeros(tsnr.shape, dtype=np.int16)
        for L in range(1, n_layers + 1)
    }

    # ── load reference layer label image for header ───────────────────────────
    ref_img = nib.load(str(mask_dir / "layer1_roi_label.nii.gz"))

    # ── iterate over layers and ROIs ──────────────────────────────────────────
    for L in range(1, n_layers + 1):
        layer_label_path = mask_dir / ("layer%d_roi_label.nii.gz" % L)
        if not layer_label_path.exists():
            print("[WARN]  sub-%s layer%d label file missing" % (sub, L))
            continue

        layer_labels = np.asarray(
            nib.load(str(layer_label_path)).dataobj, dtype=np.int16).squeeze()

        for roi_idx, roi_name in enumerate(roi_names, start=1):
            # boolean mask for this ROI within this layer
            roi_mask = layer_labels == roi_idx

            n_raw = int(roi_mask.sum())
            n_vox_raw[roi_idx - 1, L - 1] = n_raw

            if n_raw == 0:
                continue

            # tSNR cleaning
            cmask, n_clean = clean_mask(roi_mask, tsnr,
                                        sigma=args.tsnr_sigma,
                                        min_vox=args.min_vox)
            n_vox_clean[roi_idx - 1, L - 1] = n_clean

            if cmask is None:
                continue

            # extract mean time series
            ts_matrix[roi_idx - 1, L - 1, :] = extract_mean_ts(func4d, cmask)

            # record in clean mask volume
            clean_mask_vols[L][cmask] = roi_idx

        n_present = int((n_vox_clean[:, L-1] > 0).sum())
        print("[sub-%s] layer %d: %d / %d ROIs have clean voxels"
              % (sub, L, n_present, n_rois))

    # ── save ──────────────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    clean_dir = out_dir / "clean_masks"
    clean_dir.mkdir(exist_ok=True)

    np.save(str(out_dir / "ts_matrix.npy"),       ts_matrix)
    np.save(str(out_dir / "n_voxels_clean.npy"),  n_vox_clean)
    np.save(str(out_dir / "n_voxels_raw.npy"),    n_vox_raw)
    np.save(str(out_dir / "roi_names.npy"),        roi_names)

    # tSNR map
    tsnr_img = nib.Nifti1Image(tsnr, func_img.affine, func_img.header)
    tsnr_img.header.set_data_dtype(np.float32)
    nib.save(tsnr_img, str(out_dir / "tsnr.nii.gz"))

    # per-layer clean masks
    for L, vol in clean_mask_vols.items():
        out_mask = nib.Nifti1Image(vol, ref_img.affine, ref_img.header)
        out_mask.header.set_data_dtype(np.int16)
        nib.save(out_mask, str(clean_dir / ("layer%d_clean.nii.gz" % L)))

    # ── summary ───────────────────────────────────────────────────────────────
    total_ok  = int((n_vox_clean > 0).sum())
    total_all = n_rois * n_layers
    frac_nan  = np.isnan(ts_matrix).all(axis=-1).sum()
    print("[DONE] sub-%s  %d / %d ROI-layer pairs have time series  "
          "(%d NaN pairs)  → %s"
          % (sub, total_ok, total_all, frac_nan, out_dir))

    return sub, True


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    args     = parse_args()
    subjects = [args.subject] if args.subject else SUBJECTS
    args_dict = vars(args)
    tag = tag_from_args(args)

    print("=" * 64)
    print(" Step 3: tSNR cleaning + time series extraction")
    print(" Tag:         %s" % tag)
    print(" tSNR sigma:  %.1f" % args.tsnr_sigma)
    print(" Min voxels:  %d" % args.min_vox)
    print(" Subjects:    %d" % len(subjects))
    print(" Parallel:    %d" % args.parallel)
    print(" Output:      %s" % (OUT_BASE / tag))
    print("=" * 64)

    results = {}

    if args.parallel > 1:
        with ProcessPoolExecutor(max_workers=args.parallel) as ex:
            futures = {
                ex.submit(process_subject, sub, args_dict): sub
                for sub in subjects
            }
            for fut in as_completed(futures):
                sub_id, ok = fut.result()
                results[sub_id] = ok
    else:
        for sub in subjects:
            _, ok = process_subject(sub, args_dict)
            results[sub] = ok

    # ── summary ───────────────────────────────────────────────────────────────
    ok_list   = [s for s in subjects if results.get(s)]
    fail_list = [s for s in subjects if not results.get(s)]
    print("\n" + "=" * 64)
    print(" OK:   %d — %s" % (len(ok_list),   ok_list))
    print(" FAIL: %d — %s" % (len(fail_list), fail_list))
    print("=" * 64)

    if fail_list:
        sys.exit(1)


if __name__ == "__main__":
    main()