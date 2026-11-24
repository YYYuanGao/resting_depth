#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build matched composite-first inputs for a pial-depth sensitivity analysis.

This analysis deliberately does *not* compare the pial5 inputs directly with
the historical primary inputs, because the ROI-level Step3 arrays used in June
were later overwritten.  Instead it creates two branches from the currently
available, matched Step3 products:

  baseline_current : current tSNR-clean ROI x layer time series
  pial5_current    : the same current pipeline plus outer-pial 5% exclusion

The composite-node construction is copied from the original exporter:
strict validity of every component ROI and an equal-ROI mean within each layer.
The old six-node primary files are read only as an external replication
reference and are never overwritten.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

import os
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "/path/to/project")  # see config/paths.sh


ROOT = Path(PROJECT_ROOT + "/derivatives_0331")
TAG = "ld1000_3layers_equivol"
STEP3_BRANCHES = {
    "baseline_current": ROOT / "Step3_TimeSeries_tSNR_clean" / TAG,
    "pial5_current": (
        ROOT / "S6_validation" / "pial_depth_exclusion" /
        "Step3_TimeSeries_tSNRclean_pial5pct" / TAG
    ),
}
OUT_BASE = (
    ROOT / "S6_validation" / "pial_depth_exclusion" /
    "matched_current_inputs"
)
OLD_PRIMARY = (
    ROOT / "S4_rDCM_com_first" /
    "timeseries_merged_cauPFC_EVC_tSNRclean_layered"
)

SUBJECTS = [
    "01", "02", "03", "04", "05", "06", "07", "08",
    "09", "10", "11", "12", "13", "14", "16",
]
N_LAYERS = 3
LAYER_NAMES = ("deep", "mid", "sup")
TR = 2.2
TE = 0.026
MIN_STD = 1e-6

ROI_GROUPS = {
    "cau_PFC": ("6d", "6v", "6mp", "FEF", "PEF", "55b", "SFL"),
    "V1": ("V1",),
    "EVC": ("V1", "V2", "V3"),
}
NETWORKS = {
    "cauPFC_V1": ("V1", "cau_PFC"),
    "cauPFC_EVC": ("EVC", "cau_PFC"),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", nargs="+", default=SUBJECTS)
    p.add_argument("--networks", nargs="+", choices=tuple(NETWORKS),
                   default=list(NETWORKS))
    p.add_argument("--out-base", type=Path, default=OUT_BASE)
    p.add_argument("--tr", type=float, default=TR)
    p.add_argument("--te", type=float, default=TE)
    p.add_argument("--min-std", type=float, default=MIN_STD)
    p.add_argument("--force", action="store_true",
                   help="Replace existing matched validation inputs.")
    return p.parse_args()


def norm_sub(value):
    return str(value).replace("sub-", "").zfill(2)


def str_array(values):
    return np.array([str(x) for x in values], dtype=object)


def load_subject_ts(base: Path, sub: str):
    sub_dir = base / f"sub-{sub}"
    ts_file = sub_dir / "ts_matrix.npy"
    roi_file = sub_dir / "roi_names.npy"
    if not ts_file.exists() or not roi_file.exists():
        raise FileNotFoundError(
            f"Missing Step3 input for sub-{sub}: {ts_file} or {roi_file}")

    ts = np.load(ts_file)
    names = str_array(np.load(roi_file, allow_pickle=True))
    if ts.ndim != 3 or ts.shape[1] != N_LAYERS:
        raise ValueError(f"Expected ROI x 3 x time, got {ts.shape}: {ts_file}")
    if ts.shape[0] != len(names):
        raise ValueError(f"ROI count mismatch: {ts_file}")
    if len(set(map(str, names))) != len(names):
        raise ValueError(f"Duplicate ROI names: {roi_file}")
    return ts, names


def roi_indices(names, requested, sub):
    lookup = {str(name): i for i, name in enumerate(names)}
    missing = [name for name in requested if name not in lookup]
    if missing:
        raise RuntimeError(f"sub-{sub} missing requested ROIs: {missing}")
    return np.array([lookup[name] for name in requested], dtype=int)


def valid_signal(signal, min_std):
    return np.isfinite(signal).all() and np.nanstd(signal) >= min_std


def build_group(ts, names, group_name, sub, branch, min_std):
    members = ROI_GROUPS[group_name]
    indices = roi_indices(names, members, sub)
    group_ts = ts[indices, :, :]
    merged = []
    qc = []

    for li, layer_name in enumerate(LAYER_NAMES):
        layer_signals = group_ts[:, li, :]
        member_valid = np.array(
            [valid_signal(row, min_std) for row in layer_signals], dtype=bool)
        if not member_valid.all():
            bad = [members[i] for i in np.flatnonzero(~member_valid)]
            raise RuntimeError(
                f"sub-{sub} {branch} {group_name}_{layer_name} "
                f"has invalid members: {bad}")

        # Preserve the original operation order and source dtype.
        signal = np.mean(layer_signals, axis=0)
        if not valid_signal(signal, min_std):
            raise RuntimeError(
                f"sub-{sub} {branch} merged node "
                f"{group_name}_{layer_name} is invalid")
        merged.append(signal)
        qc.append({
            "subject": f"sub-{sub}",
            "branch": branch,
            "group": group_name,
            "layer": layer_name,
            "members": ",".join(members),
            "n_members": len(members),
            "n_valid_members": int(member_valid.sum()),
            "strict_group": True,
            "merged_mean": float(np.mean(signal)),
            "merged_std": float(np.std(signal)),
            "merged_has_nonfinite": bool(~np.isfinite(signal).all()),
        })
    return np.vstack(merged), qc


def build_network(ts, names, network, sub, branch, min_std):
    source_group, target_group = NETWORKS[network]
    source, qc1 = build_group(
        ts, names, source_group, sub, branch, min_std)
    target, qc2 = build_group(
        ts, names, target_group, sub, branch, min_std)
    nodes = np.vstack([source, target])
    labels = np.array(
        [f"{source_group}_{layer}" for layer in LAYER_NAMES] +
        [f"{target_group}_{layer}" for layer in LAYER_NAMES],
        dtype=object,
    )
    return nodes.T.astype(np.float64), labels, qc1 + qc2


def save_input(out_base, branch, sub, network, ts, labels, source_base,
               args):
    out_dir = out_base / branch / network
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"sub-{sub}_{network}_ts_merged_layered"
    mat_file = out_dir / f"{stem}.mat"
    npy_file = out_dir / f"{stem}.npy"
    existing = [path for path in (mat_file, npy_file) if path.exists()]
    if existing and not args.force:
        raise FileExistsError(
            "Validation output exists; inspect it or rerun with --force: " +
            " | ".join(map(str, existing)))

    source_group, target_group = NETWORKS[network]
    payload = {
        "ts": ts,
        "labels": labels,
        "TR": float(args.tr),
        "TE": float(args.te),
        "Subject": f"sub-{sub}",
        "network_name": network,
        "source_group": source_group,
        "target_group": target_group,
        "n_nodes": float(ts.shape[1]),
        "n_layers": float(N_LAYERS),
        "layer_order": "source_deep_mid_sup_then_target_deep_mid_sup",
        "roi_groups": str(ROI_GROUPS),
        "validation_branch": branch,
        "aggregation_rule": "strict_equal_roi_mean",
        "signal_definition": (
            "current tSNR-clean BOLD ROI time series; equal-ROI mean within "
            "predefined groups and layer before rDCM"
        ),
        "source_ts_base": str(source_base),
    }
    sio.savemat(mat_file, payload)
    np.save(npy_file, ts)
    return mat_file, npy_file


def metrics(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    delta = y - x
    xf = x.ravel()
    yf = y.ravel()
    r = np.corrcoef(xf, yf)[0, 1] if np.std(xf) and np.std(yf) else np.nan
    xsd = np.std(xf)
    return {
        "max_abs_diff": float(np.max(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta ** 2))),
        "pearson_r": float(r),
        "mean_delta_pial_minus_baseline": float(np.mean(delta)),
        "baseline_std": float(xsd),
        "pial5_std": float(np.std(yf)),
        "pial5_to_baseline_std_ratio": (
            float(np.std(yf) / xsd) if xsd > 0 else np.nan),
        "exact_equal": bool(np.array_equal(x, y)),
    }


def load_old_primary(sub, network, n_timepoints):
    path = OLD_PRIMARY / network / f"sub-{sub}_{network}_ts_merged_layered.mat"
    if not path.exists():
        return path, None
    mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    if "ts" not in mat:
        raise KeyError(f"No ts variable in old primary file: {path}")
    ts = np.asarray(mat["ts"], dtype=float).squeeze()
    if ts.shape == (6, n_timepoints):
        ts = ts.T
    if ts.shape != (n_timepoints, 6):
        raise ValueError(f"Unexpected old primary shape {ts.shape}: {path}")
    return path, ts


def main():
    args = parse_args()
    subjects = [norm_sub(value) for value in args.subjects]
    args.out_base.mkdir(parents=True, exist_ok=True)

    export_rows = []
    group_qc_rows = []
    paired_rows = []
    old_rows = []

    for sub in subjects:
        loaded = {
            branch: load_subject_ts(base, sub)
            for branch, base in STEP3_BRANCHES.items()
        }
        bts, bnames = loaded["baseline_current"]
        pts, pnames = loaded["pial5_current"]
        if bts.shape != pts.shape or not np.array_equal(bnames, pnames):
            raise RuntimeError(
                f"Matched-branch shape/name mismatch for sub-{sub}: "
                f"{bts.shape} vs {pts.shape}")

        for network in args.networks:
            built = {}
            for branch, base in STEP3_BRANCHES.items():
                ts, names = loaded[branch]
                six_node, labels, qc = build_network(
                    ts, names, network, sub, branch, args.min_std)
                mat_file, npy_file = save_input(
                    args.out_base, branch, sub, network, six_node, labels,
                    base, args)
                built[branch] = (six_node, labels)
                group_qc_rows.extend(qc)
                export_rows.append({
                    "subject": f"sub-{sub}",
                    "network": network,
                    "branch": branch,
                    "source_ts_base": str(base),
                    "mat_file": str(mat_file),
                    "npy_file": str(npy_file),
                    "n_timepoints": six_node.shape[0],
                    "n_nodes": six_node.shape[1],
                    "labels": "|".join(map(str, labels)),
                    "has_nonfinite": bool(~np.isfinite(six_node).all()),
                    "low_variance_nodes": int(
                        (np.std(six_node, axis=0) < args.min_std).sum()),
                })

            baseline, labels = built["baseline_current"]
            pial5, labels2 = built["pial5_current"]
            if not np.array_equal(labels, labels2):
                raise RuntimeError(f"Node-label mismatch: sub-{sub} {network}")

            row = {
                "subject": f"sub-{sub}", "network": network,
                "node_index": 0, "node_label": "ALL_NODES",
            }
            row.update(metrics(baseline, pial5))
            paired_rows.append(row)
            for node in range(6):
                row = {
                    "subject": f"sub-{sub}", "network": network,
                    "node_index": node + 1,
                    "node_label": str(labels[node]),
                }
                row.update(metrics(baseline[:, node], pial5[:, node]))
                paired_rows.append(row)

            old_path, old = load_old_primary(sub, network, baseline.shape[0])
            old_row = {
                "subject": f"sub-{sub}",
                "network": network,
                "reference_file": str(old_path),
                "reference_available": old is not None,
            }
            if old is not None:
                # Here x is old and y is current baseline.
                old_row.update(metrics(old, baseline))
            old_rows.append(old_row)

    export_df = pd.DataFrame(export_rows)
    group_df = pd.DataFrame(group_qc_rows)
    paired_df = pd.DataFrame(paired_rows)
    old_df = pd.DataFrame(old_rows)
    export_df.to_csv(args.out_base / "matched_export_qc.csv", index=False)
    group_df.to_csv(args.out_base / "matched_group_layer_qc.csv", index=False)
    paired_df.to_csv(args.out_base / "matched_input_differences_long.csv", index=False)
    old_df.to_csv(args.out_base / "current_baseline_vs_old_primary.csv", index=False)

    overall = paired_df[paired_df["node_index"] == 0]
    summary = overall.groupby("network", as_index=False).agg(
        n_subjects=("subject", "nunique"),
        pearson_r_mean=("pearson_r", "mean"),
        pearson_r_min=("pearson_r", "min"),
        rmse_mean=("rmse", "mean"),
        max_abs_diff_max=("max_abs_diff", "max"),
        n_exact_equal=("exact_equal", "sum"),
    )
    summary.to_csv(args.out_base / "matched_input_differences_summary.csv", index=False)

    print("=" * 88)
    print("Matched current baseline / pial5 composite export complete")
    print(f"Subjects: {len(subjects)} | networks: {len(args.networks)}")
    print(f"Output:   {args.out_base}")
    print("Construction: strict equal-ROI mean (same as original exporter)")
    print("\nWhole-network input comparison:")
    print(summary.to_string(index=False))
    print("\nOld primary is reference-only; it is not the pial contrast baseline.")
    print("=" * 88)


if __name__ == "__main__":
    main()
