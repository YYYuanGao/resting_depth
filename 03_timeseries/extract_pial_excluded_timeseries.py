#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create a parity-preserving pial-depth sensitivity branch.

The script starts from the *existing* tSNR-clean ROI x layer masks and removes
only voxels in the functional-space outer-pial exclusion band.

Inputs per subject
------------------
1. Existing tSNR-clean labelled masks, one per layer.
2. A binary functional-space exclusion band from
   ``warp_pial_exclusion_band_to_func.sh`` (1=exclude, 0=retain).
3. The same raw preprocessed resting-state BOLD used by the primary analysis.

Outputs retain the Step3 array layout so the existing composite exporter can be
reused after changing only its input/output base paths.

Usage
-----
python extract_pial_excluded_timeseries.py --subject 01 --force
python extract_pial_excluded_timeseries.py --force
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

import os
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "/path/to/project")  # see config/paths.sh


BIDS_ROOT = Path(PROJECT_ROOT + "/BIDS")
DERIV_ROOT = Path(PROJECT_ROOT + "/derivatives_0331")
PRIMARY_STEP3 = DERIV_ROOT / "Step3_TimeSeries_tSNR_clean"
PIAL_ROOT = (
    DERIV_ROOT
    / "S6_validation"
    / "pial_depth_exclusion"
    / "pial_exclusion_bands_func"
)
OUT_BASE = (
    DERIV_ROOT
    / "S6_validation"
    / "pial_depth_exclusion"
    / "Step3_TimeSeries_tSNRclean_pial5pct"
)

SUBJECTS = ["01", "02", "03", "04", "05", "06", "07", "08",
            "09", "10", "11", "12", "13", "14", "16"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Subtract outer-pial band from existing tSNR-clean masks.")
    p.add_argument("--subject", default=None, help="One subject, e.g. 01 or sub-01.")
    p.add_argument("--ld", default="1000")
    p.add_argument("--nr-layers", default=3, type=int)
    p.add_argument("--layer-method", default="equivol")
    p.add_argument("--pct", default=5, type=int)
    p.add_argument("--min-vox", default=5, type=int)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def norm_sub(value: str) -> str:
    return value.replace("sub-", "").zfill(2)


def tag(args) -> str:
    return f"ld{args.ld}_{args.nr_layers}layers_{args.layer_method}"


def assert_same_grid(ref_img, other_img, ref_name, other_name, atol=1e-4):
    if tuple(ref_img.shape[:3]) != tuple(other_img.shape[:3]):
        raise ValueError(
            f"Grid shape mismatch: {ref_name} {ref_img.shape[:3]} vs "
            f"{other_name} {other_img.shape[:3]}")
    if not np.allclose(ref_img.affine, other_img.affine, atol=atol, rtol=0):
        diff = float(np.max(np.abs(ref_img.affine - other_img.affine)))
        raise ValueError(
            f"Affine mismatch: {ref_name} vs {other_name}; max_abs_diff={diff:.6g}")


def save_label_image(data, ref_img, path):
    out = nib.Nifti1Image(data.astype(np.int16), ref_img.affine, ref_img.header)
    out.header.set_data_dtype(np.int16)
    nib.save(out, str(path))


def write_csv(path: Path, records: list[dict]):
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def process_subject(sub: str, args) -> tuple[bool, list[dict]]:
    sub = norm_sub(sub)
    subject = f"sub-{sub}"
    base_tag = tag(args)
    in_dir = PRIMARY_STEP3 / base_tag / subject
    pial_dir = PIAL_ROOT / f"{base_tag}_pial{args.pct}pct" / subject
    pial_file = pial_dir / f"{subject}_pial_outer{args.pct}pct_exclusion_func.nii.gz"
    func_file = (
        BIDS_ROOT / subject / "func" / "derivatives" /
        "fsl_afni_fsl_preproc" / "AP_dc_st_mc.nii.gz"
    )
    out_dir = OUT_BASE / base_tag / subject
    sentinel = out_dir / "ts_matrix.npy"

    if sentinel.exists() and not args.force:
        print(f"[SKIP] {subject}: output exists; use --force to regenerate")
        qc_path = out_dir / "pial_exclusion_voxel_qc.csv"
        if qc_path.exists():
            with qc_path.open(newline="", encoding="utf-8") as f:
                return True, list(csv.DictReader(f))
        return True, []

    required = [
        func_file,
        pial_file,
        in_dir / "roi_names.npy",
        in_dir / "n_voxels_raw.npy",
    ] + [
        in_dir / "clean_masks" / f"layer{layer}_clean.nii.gz"
        for layer in range(1, args.nr_layers + 1)
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print(f"[ERROR] {subject}: missing required inputs:", file=sys.stderr)
        for p in missing:
            print(f"  {p}", file=sys.stderr)
        return False, []

    print(f"\n[{subject}] Loading the same primary resting-state BOLD")
    func_img = nib.load(str(func_file))
    if len(func_img.shape) != 4:
        raise ValueError(f"Expected 4D BOLD, got {func_img.shape}: {func_file}")
    func = np.asarray(func_img.dataobj, dtype=np.float32)

    pial_img = nib.load(str(pial_file))
    assert_same_grid(func_img, pial_img, "BOLD", "pial exclusion band")
    pial_data = np.asarray(pial_img.dataobj, dtype=np.float32).squeeze()
    if not np.all(np.isfinite(pial_data)):
        raise ValueError(f"Non-finite values in pial band: {pial_file}")
    if np.max(np.abs(pial_data - np.rint(pial_data))) > 1e-5:
        raise ValueError(f"Pial band is not integer/binary: {pial_file}")
    unique = set(np.unique(np.rint(pial_data).astype(int)).tolist())
    if not unique.issubset({0, 1}) or 1 not in unique:
        raise ValueError(f"Pial band values must be 0/1 with some 1s; got {unique}")
    exclude = pial_data > 0.5

    roi_names = np.load(in_dir / "roi_names.npy", allow_pickle=True).astype(str)
    n_rois = len(roi_names)
    n_tp = func.shape[-1]
    n_vox_raw = np.load(in_dir / "n_voxels_raw.npy")
    if n_vox_raw.shape != (n_rois, args.nr_layers):
        raise ValueError(
            f"n_voxels_raw shape {n_vox_raw.shape} != {(n_rois, args.nr_layers)}")

    ts_matrix = np.full((n_rois, args.nr_layers, n_tp), np.nan, dtype=np.float32)
    n_before = np.zeros((n_rois, args.nr_layers), dtype=np.int32)
    n_after_unthresholded = np.zeros_like(n_before)
    n_final = np.zeros_like(n_before)
    n_removed = np.zeros_like(n_before)
    qc_records: list[dict] = []

    out_dir.mkdir(parents=True, exist_ok=True)
    clean_out = out_dir / "clean_masks"
    clean_out.mkdir(exist_ok=True)

    for layer in range(1, args.nr_layers + 1):
        mask_file = in_dir / "clean_masks" / f"layer{layer}_clean.nii.gz"
        mask_img = nib.load(str(mask_file))
        assert_same_grid(func_img, mask_img, "BOLD", f"primary layer{layer} clean mask")
        labels_float = np.asarray(mask_img.dataobj, dtype=np.float32).squeeze()
        if np.max(np.abs(labels_float - np.rint(labels_float))) > 1e-5:
            raise ValueError(f"Non-integer ROI labels: {mask_file}")
        labels = np.rint(labels_float).astype(np.int32)
        if labels.min() < 0 or labels.max() > n_rois:
            raise ValueError(
                f"ROI label range [{labels.min()}, {labels.max()}] incompatible "
                f"with n_rois={n_rois}: {mask_file}")

        out_labels = np.zeros(labels.shape, dtype=np.int16)
        for roi_index, roi_name in enumerate(roi_names, start=1):
            primary_mask = labels == roi_index
            sensitivity_mask = primary_mask & (~exclude)
            before = int(primary_mask.sum())
            after = int(sensitivity_mask.sum())
            removed = before - after
            usable = after >= args.min_vox

            n_before[roi_index - 1, layer - 1] = before
            n_after_unthresholded[roi_index - 1, layer - 1] = after
            n_removed[roi_index - 1, layer - 1] = removed

            if usable:
                n_final[roi_index - 1, layer - 1] = after
                ts_matrix[roi_index - 1, layer - 1] = (
                    func[sensitivity_mask].mean(axis=0).astype(np.float32)
                )
                out_labels[sensitivity_mask] = roi_index

            qc_records.append({
                "subject": subject,
                "roi_index": roi_index,
                "roi_name": roi_name,
                "layer": layer,
                "layer_name": {1: "deep", 2: "mid", 3: "sup"}.get(layer, str(layer)),
                "n_primary_tsnr_clean": before,
                "n_after_pial_exclusion": after,
                "n_removed_pial": removed,
                "fraction_removed_pial": (removed / before) if before else np.nan,
                "n_final_for_timeseries": after if usable else 0,
                "timeseries_valid": int(usable),
            })

        save_label_image(out_labels, mask_img, clean_out / f"layer{layer}_clean.nii.gz")
        layer_before = int(n_before[:, layer - 1].sum())
        layer_removed = int(n_removed[:, layer - 1].sum())
        pct_removed = 100.0 * layer_removed / layer_before if layer_before else np.nan
        print(
            f"  layer {layer}: primary={layer_before}, removed={layer_removed} "
            f"({pct_removed:.3f}%), valid ROI={int((n_final[:, layer-1] > 0).sum())}/{n_rois}")

    np.save(out_dir / "ts_matrix.npy", ts_matrix)
    np.save(out_dir / "roi_names.npy", roi_names)
    np.save(out_dir / "n_voxels_raw.npy", n_vox_raw)
    np.save(out_dir / "n_voxels_before_pial.npy", n_before)
    np.save(out_dir / "n_voxels_after_pial_unthresholded.npy", n_after_unthresholded)
    np.save(out_dir / "n_voxels_removed_pial.npy", n_removed)
    np.save(out_dir / "n_voxels_clean.npy", n_final)
    write_csv(out_dir / "pial_exclusion_voxel_qc.csv", qc_records)

    metadata = {
        "subject": subject,
        "analysis_role": "pial-depth exclusion sensitivity; not literal vein removal",
        "exclusion_semantics": "functional pial band value 1 was subtracted",
        "primary_clean_masks": str(in_dir / "clean_masks"),
        "pial_exclusion_band": str(pial_file),
        "bold_file": str(func_file),
        "minimum_voxels": args.min_vox,
        "n_rois": n_rois,
        "n_layers": args.nr_layers,
        "n_timepoints": n_tp,
        "grid_and_affine_checks_passed": True,
    }
    with (out_dir / "pial_exclusion_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    deep_mid_removed = int(n_removed[:, : min(2, args.nr_layers)].sum())
    status = "PASS" if deep_mid_removed == 0 else "WARN_DEEP_OR_MID_REMOVAL"
    valid_pairs = int((n_final > 0).sum())
    print(f"  status={status}; valid ROI-layer pairs={valid_pairs}/{n_rois*args.nr_layers}")
    return True, qc_records


def main():
    args = parse_args()
    subjects = [norm_sub(args.subject)] if args.subject else SUBJECTS
    all_records: list[dict] = []
    failed = []
    for sub in subjects:
        try:
            ok, records = process_subject(sub, args)
        except Exception as exc:
            ok, records = False, []
            print(f"[ERROR] sub-{norm_sub(sub)}: {exc}", file=sys.stderr)
        if ok:
            all_records.extend(records)
        else:
            failed.append(norm_sub(sub))

    group_dir = OUT_BASE / tag(args) / "group"
    write_csv(group_dir / "pial_exclusion_voxel_qc_all_subjects.csv", all_records)

    print("\n" + "=" * 78)
    print(f"Finished: {len(subjects) - len(failed)}/{len(subjects)} subjects")
    print(f"Group QC: {group_dir / 'pial_exclusion_voxel_qc_all_subjects.csv'}")
    if failed:
        print("Failed: " + ", ".join(f"sub-{s}" for s in failed))
        raise SystemExit(1)
    print("=" * 78)


if __name__ == "__main__":
    main()
