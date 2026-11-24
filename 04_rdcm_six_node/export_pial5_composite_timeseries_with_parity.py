#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export pial5 composite-first rDCM inputs only after exact parity succeeds.

The script reconstructs every original composite input from the original Step3
arrays and compares it with the saved primary ``*_ts_merged_layered.mat``.
It tests two plausible composite rules:

1. ``equal_roi_mean``: equal-weight mean of component ROI mean time series.
2. ``voxel_weighted``: mean weighted by each component ROI's clean voxel count.

Only an aggregation rule that reproduces all available original files within
tolerance is allowed to generate pial5 inputs.  This prevents a change in node
construction from being misattributed to pial exclusion.

Usage:
    python export_pial5_composite_timeseries_with_parity.py --parity-only
    python export_pial5_composite_timeseries_with_parity.py --force
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

import os
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "/path/to/project")  # see config/paths.sh


DERIV_ROOT = Path(PROJECT_ROOT + "/derivatives_0331")
TAG = "ld1000_3layers_equivol"
PRIMARY_STEP3 = DERIV_ROOT / "Step3_TimeSeries_tSNR_clean" / TAG
PIAL_STEP3 = (
    DERIV_ROOT / "S6_validation" / "pial_depth_exclusion" /
    "Step3_TimeSeries_tSNRclean_pial5pct" / TAG
)
PRIMARY_COMPOSITE = (
    DERIV_ROOT / "S4_rDCM_com_first" /
    "timeseries_merged_cauPFC_EVC_tSNRclean_layered"
)
OUT_BASE = (
    DERIV_ROOT / "S6_validation" / "pial_depth_exclusion" /
    "timeseries_merged_cauPFC_EVC_tSNRclean_pial5pct"
)
QC_DIR = (
    DERIV_ROOT / "S6_validation" / "pial_depth_exclusion" /
    "composite_input_parity"
)

SUBJECTS = ["01", "02", "03", "04", "05", "06", "07", "08",
            "09", "10", "11", "12", "13", "14", "16"]
NETWORKS = {
    "cauPFC_V1": ["V1"],
    "cauPFC_EVC": ["V1", "V2", "V3"],
}
CAU_PFC_ROIS = ["6d", "6v", "6mp", "FEF", "PEF", "55b", "SFL"]
LAYER_NAMES = ["deep", "mid", "sup"]
AGGREGATIONS = ["equal_roi_mean", "voxel_weighted"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subject", default=None, help="One subject, e.g. 01.")
    p.add_argument("--network", default=None, choices=sorted(NETWORKS))
    p.add_argument(
        "--aggregation", default="auto",
        choices=["auto"] + AGGREGATIONS,
        help="auto requires one rule to pass all original parity checks.")
    p.add_argument("--atol", type=float, default=1e-5)
    p.add_argument("--rtol", type=float, default=1e-7)
    p.add_argument("--tr", type=float, default=2.2)
    p.add_argument("--te", type=float, default=0.026)
    p.add_argument("--parity-only", action="store_true")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def norm_sub(value):
    return str(value).replace("sub-", "").zfill(2)


def normalize_name(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def load_step3(base: Path, subject: str):
    directory = base / subject
    required = [
        directory / "ts_matrix.npy",
        directory / "roi_names.npy",
        directory / "n_voxels_clean.npy",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing Step3 inputs: " + " | ".join(missing))

    ts = np.load(directory / "ts_matrix.npy")
    names_raw = np.load(directory / "roi_names.npy", allow_pickle=True)
    names = [normalize_name(x) for x in names_raw.tolist()]
    nvox = np.load(directory / "n_voxels_clean.npy")

    if ts.ndim != 3 or ts.shape[1] != 3:
        raise ValueError(f"Expected ts_matrix (ROI,3,time), got {ts.shape}: {directory}")
    if ts.shape[0] != len(names) or nvox.shape != ts.shape[:2]:
        raise ValueError(
            f"Step3 shape mismatch: ts={ts.shape}, names={len(names)}, nvox={nvox.shape}")
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate ROI names in {directory}")
    return np.asarray(ts, dtype=float), names, np.asarray(nvox, dtype=float)


def roi_indices(names, required):
    lookup = {name: i for i, name in enumerate(names)}
    missing = [name for name in required if name not in lookup]
    if missing:
        raise ValueError("Missing required ROI names: " + ", ".join(missing))
    return np.array([lookup[name] for name in required], dtype=int)


def aggregate_component(ts, nvox, indices, layer, method):
    values = ts[indices, layer, :]
    counts = nvox[indices, layer]
    valid_rows = np.all(np.isfinite(values), axis=1) & np.isfinite(counts) & (counts > 0)
    if not np.all(valid_rows):
        bad = indices[~valid_rows].tolist()
        raise ValueError(f"Invalid component ROI rows at layer {layer+1}: {bad}")

    if method == "equal_roi_mean":
        out = np.mean(values, axis=0)
    elif method == "voxel_weighted":
        out = np.average(values, axis=0, weights=counts)
    else:
        raise ValueError(method)
    if not np.all(np.isfinite(out)) or np.std(out) < 1e-12:
        raise ValueError(f"Invalid composite at layer {layer+1}, method={method}")
    return out


def construct_composite(ts, names, nvox, network, method):
    source_rois = NETWORKS[network]
    source_idx = roi_indices(names, source_rois)
    target_idx = roi_indices(names, CAU_PFC_ROIS)

    nodes = []
    for layer in range(3):
        nodes.append(aggregate_component(ts, nvox, source_idx, layer, method))
    for layer in range(3):
        nodes.append(aggregate_component(ts, nvox, target_idx, layer, method))
    matrix = np.column_stack(nodes)

    source_name = "V1" if network == "cauPFC_V1" else "EVC"
    labels = [f"{source_name}_{name}" for name in LAYER_NAMES]
    labels += [f"cau_PFC_{name}" for name in LAYER_NAMES]
    return matrix, labels


def normalize_ts_shape(array, n_timepoints):
    arr = np.asarray(array, dtype=float).squeeze()
    if arr.shape == (n_timepoints, 6):
        return arr
    if arr.shape == (6, n_timepoints):
        return arr.T
    raise ValueError(f"Saved ts shape {arr.shape} is not {(n_timepoints, 6)} or {(6, n_timepoints)}")


def load_saved_ts(path, n_timepoints):
    try:
        mat = loadmat(path, squeeze_me=True, struct_as_record=False)
        if "ts" not in mat:
            raise KeyError(f"No variable 'ts' in {path}")
        return normalize_ts_shape(mat["ts"], n_timepoints)
    except NotImplementedError:
        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError(f"MAT v7.3 requires h5py: {path}") from exc
        with h5py.File(path, "r") as f:
            if "ts" not in f:
                raise KeyError(f"No dataset 'ts' in {path}")
            return normalize_ts_shape(np.array(f["ts"]), n_timepoints)


def similarity_metrics(reconstructed, saved, atol, rtol):
    diff = reconstructed - saved
    max_abs = float(np.max(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    x = reconstructed.ravel()
    y = saved.ravel()
    pearson = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else np.nan
    passed = bool(np.allclose(reconstructed, saved, atol=atol, rtol=rtol, equal_nan=False))
    return passed, max_abs, rmse, pearson


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    args = parse_args()
    subjects = [norm_sub(args.subject)] if args.subject else SUBJECTS
    networks = [args.network] if args.network else list(NETWORKS)
    pairs = [(f"sub-{sub}", net) for net in networks for sub in subjects]

    parity_rows = []
    cache = {}
    for subject, network in pairs:
        primary = load_step3(PRIMARY_STEP3, subject)
        cache[(subject, "primary")] = primary
        ts, names, nvox = primary
        saved_path = (
            PRIMARY_COMPOSITE / network /
            f"{subject}_{network}_ts_merged_layered.mat"
        )
        if not saved_path.exists():
            raise FileNotFoundError(f"Missing primary composite reference: {saved_path}")
        saved = load_saved_ts(saved_path, ts.shape[-1])

        for method in AGGREGATIONS:
            reconstructed, _ = construct_composite(ts, names, nvox, network, method)
            passed, max_abs, rmse, pearson = similarity_metrics(
                reconstructed, saved, args.atol, args.rtol)
            parity_rows.append({
                "subject": subject,
                "network": network,
                "aggregation": method,
                "n_timepoints": reconstructed.shape[0],
                "pass": passed,
                "max_abs_diff": max_abs,
                "rmse": rmse,
                "pearson_r": pearson,
                "reference_file": str(saved_path),
            })

    QC_DIR.mkdir(parents=True, exist_ok=True)
    parity_csv = QC_DIR / "composite_input_parity_candidates.csv"
    write_csv(parity_csv, parity_rows)

    counts = {
        method: sum(r["pass"] for r in parity_rows if r["aggregation"] == method)
        for method in AGGREGATIONS
    }
    total = len(pairs)
    eligible = [method for method, count in counts.items() if count == total]

    print("=" * 78)
    print("Composite-first input parity")
    print(f"Subject-network pairs: {total}")
    for method in AGGREGATIONS:
        method_rows = [r for r in parity_rows if r["aggregation"] == method]
        print(
            f"  {method:18s}: PASS {counts[method]}/{total}; "
            f"max_abs_diff={max(r['max_abs_diff'] for r in method_rows):.9g}"
        )
    print(f"CSV: {parity_csv}")

    if args.aggregation == "auto":
        if len(eligible) != 1:
            print(
                "[STOP] Expected exactly one aggregation rule to reproduce all "
                f"primary inputs; eligible={eligible}", file=sys.stderr)
            raise SystemExit(1)
        selected = eligible[0]
    else:
        selected = args.aggregation
        if counts[selected] != total:
            print(
                f"[STOP] Requested rule {selected} passed only "
                f"{counts[selected]}/{total} parity checks.", file=sys.stderr)
            raise SystemExit(1)

    print(f"Selected aggregation: {selected}")
    if args.parity_only:
        print("Parity-only mode: no pial5 MAT files written.")
        return

    output_rows = []
    for subject, network in pairs:
        pial = load_step3(PIAL_STEP3, subject)
        ts, names, nvox = pial
        composite, labels = construct_composite(ts, names, nvox, network, selected)

        out_dir = OUT_BASE / network
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{subject}_{network}_ts_merged_layered.mat"
        if out_path.exists() and not args.force:
            action = "SKIP_EXISTS"
        else:
            savemat(
                out_path,
                {
                    "ts": composite,
                    "labels": np.asarray(labels, dtype=object),
                    "TR": float(args.tr),
                    "TE": float(args.te),
                    "network_name": network,
                    "aggregation_rule": selected,
                    "source_component_rois": np.asarray(NETWORKS[network], dtype=object),
                    "target_component_rois": np.asarray(CAU_PFC_ROIS, dtype=object),
                    "sensitivity": "outermost 5% continuous-depth-coordinate exclusion",
                },
                do_compression=True,
            )
            action = "WROTE"
        output_rows.append({
            "subject": subject,
            "network": network,
            "aggregation": selected,
            "n_timepoints": composite.shape[0],
            "n_nodes": composite.shape[1],
            "all_finite": bool(np.all(np.isfinite(composite))),
            "min_node_sd": float(np.min(np.std(composite, axis=0, ddof=1))),
            "action": action,
            "output_file": str(out_path),
        })

    output_csv = QC_DIR / "pial5_composite_input_export.csv"
    write_csv(output_csv, output_rows)
    metadata = {
        "selected_aggregation": selected,
        "parity_passed_pairs": counts[selected],
        "parity_total_pairs": total,
        "parity_atol": args.atol,
        "parity_rtol": args.rtol,
        "primary_step3": str(PRIMARY_STEP3),
        "pial_step3": str(PIAL_STEP3),
        "primary_composite_reference": str(PRIMARY_COMPOSITE),
        "pial_composite_output": str(OUT_BASE),
    }
    with (QC_DIR / "pial5_composite_export_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Exported/checked pial5 pairs: {len(output_rows)}")
    print(f"Export QC: {output_csv}")
    print(f"Output base: {OUT_BASE}")
    print("=" * 78)


if __name__ == "__main__":
    main()
