#!/usr/bin/env python3
"""Export a three-stage, nine-node outer-pial-5% rDCM input.

The frozen stages are:

    EVC (equal ROI mean of V1, V2 and V3) -> PPC -> cauPFC

Each stage contributes deep, middle and superficial nodes.  This script
deliberately rebuilds the EVC composite from the ROI-level time series; it
does not average parameters from the earlier V1 / V2-V3 rDCM.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import savemat

import os
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "/path/to/project")  # see config/paths.sh


DERIV = Path(PROJECT_ROOT + "/derivatives_0331")
TAG = "ld1000_3layers_equivol"
DEFAULT_TS = (
    DERIV
    / "S6_validation"
    / "pial_depth_exclusion"
    / "Step3_TimeSeries_tSNRclean_pial5pct"
    / TAG
)
POOLED_ROOT = (
    DERIV
    / "S6_validation"
    / "pial5_multistage_bridge_evc_pooled"
)
DEFAULT_OUT = (
    POOLED_ROOT / "timeseries"
)
PROTECTED_OUTPUT_ROOTS = (
    DERIV / "S6_validation" / "pial5_multistage_bridge",
    DERIV
    / "S6_validation"
    / "pial_depth_exclusion"
    / "matched_current_rdcm",
    DERIV
    / "S6_validation"
    / "pial5_only_update"
    / "functional_gradient_bianca_reference",
)
SUBJECTS = tuple(f"{value:02d}" for value in [*range(1, 15), 16])
LAYERS = ("deep", "mid", "sup")
STAGE_ORDER = ("EVC", "PPC", "cauPFC")
TR = 2.2
TE = 0.026
MIN_STD = 1e-6

# EVC now matches the endpoint model exactly: V1, V2 and V3 enter one
# equal-ROI composite.  PPC and cauPFC retain the frozen definitions used by
# the four-stage analysis.
STAGE_CANDIDATES = {
    "EVC": ("V1", "V2", "V3"),
    "PPC": (
        "IPS1",
        "LIPv",
        "LIPd",
        "VIP",
        "AIP",
        "MIP",
        "IP0",
        "IP1",
        "IP2",
        "7PC",
        "7Am",
        "7AL",
        "7PL",
        "7Pm",
        "7m",
    ),
    "cauPFC": ("6d", "6v", "6mp", "FEF", "PEF", "55b", "SFL"),
}
STRICT_STAGES = {"EVC", "cauPFC"}
MIN_RETAINED = {"EVC": 3, "PPC": 5, "cauPFC": 7}
ROI_NAME_MAP = {"7Al": "7AL", "8AD": "8Ad"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ts-base", type=Path, default=DEFAULT_TS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--subjects", nargs="+", default=list(SUBJECTS))
    parser.add_argument("--tr", type=float, default=TR)
    parser.add_argument("--te", type=float, default=TE)
    parser.add_argument("--min-std", type=float, default=MIN_STD)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def normalize_subject(value: object) -> str:
    return str(value).replace("sub-", "").zfill(2)


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


def validate_output_isolation(ts_base: Path, out_dir: Path) -> None:
    if paths_overlap(ts_base, out_dir):
        raise RuntimeError(
            f"Unsafe input/output overlap: {ts_base} versus {out_dir}"
        )
    for protected in PROTECTED_OUTPUT_ROOTS:
        if paths_overlap(out_dir, protected):
            raise RuntimeError(
                "Refusing to write into or around a protected prior-analysis "
                f"path: {out_dir} versus {protected}"
            )


def valid_signal(signal: np.ndarray, min_std: float) -> bool:
    return bool(np.isfinite(signal).all() and np.std(signal) >= min_std)


def load_subject(base: Path, subject: str) -> tuple[np.ndarray, np.ndarray]:
    folder = base / f"sub-{subject}"
    ts_file = folder / "ts_matrix.npy"
    roi_file = folder / "roi_names.npy"
    if not ts_file.exists() or not roi_file.exists():
        raise FileNotFoundError(f"Missing outer-pial-5% input: {ts_file} or {roi_file}")
    ts = np.load(ts_file)
    names = np.asarray(
        [canonical_roi(value) for value in np.load(roi_file, allow_pickle=True)],
        dtype=object,
    )
    if ts.ndim != 3 or ts.shape[1] != 3 or ts.shape[0] != len(names):
        raise RuntimeError(f"Unexpected ROI x layer x time input {ts.shape}: {ts_file}")
    if len(set(names.tolist())) != len(names):
        raise RuntimeError(f"Duplicate canonical ROI names: {roi_file}")
    return ts, names


def freeze_membership(
    loaded: dict[str, tuple[np.ndarray, np.ndarray]], min_std: float
) -> tuple[dict[str, tuple[str, ...]], pd.DataFrame]:
    retained: dict[str, tuple[str, ...]] = {}
    rows: list[dict[str, object]] = []
    for stage, candidates in STAGE_CANDIDATES.items():
        keep: list[str] = []
        for candidate in candidates:
            roi = canonical_roi(candidate)
            invalid_cells: list[str] = []
            present_all = True
            valid_all = True
            for subject, (ts, names) in loaded.items():
                locations = np.flatnonzero(names == roi)
                if len(locations) != 1:
                    present_all = False
                    valid_all = False
                    invalid_cells.append(f"sub-{subject}:missing")
                    continue
                roi_index = int(locations[0])
                for layer_index, layer in enumerate(LAYERS):
                    if not valid_signal(ts[roi_index, layer_index], min_std):
                        valid_all = False
                        invalid_cells.append(f"sub-{subject}:{layer}")
            use_roi = present_all and valid_all
            if use_roi:
                keep.append(roi)
            rows.append(
                {
                    "stage": stage,
                    "candidate_roi": roi,
                    "present_all_subjects": present_all,
                    "valid_all_subjects_layers": valid_all,
                    "retained": use_roi,
                    "invalid_cells": ";".join(invalid_cells),
                    "selection_rule": "predefined_membership_plus_complete_QC_only",
                }
            )
        if stage in STRICT_STAGES and len(keep) != len(candidates):
            lost = sorted(set(map(canonical_roi, candidates)) - set(keep))
            raise RuntimeError(
                f"Strict stage {stage} lost frozen members {lost}; "
                "do not silently redefine the composite"
            )
        if len(keep) < MIN_RETAINED[stage]:
            raise RuntimeError(
                f"Stage {stage}: retained {len(keep)}, require >= {MIN_RETAINED[stage]}"
            )
        retained[stage] = tuple(keep)
    return retained, pd.DataFrame(rows)


def build_subject(
    subject: str,
    ts: np.ndarray,
    names: np.ndarray,
    members: dict[str, tuple[str, ...]],
    min_std: float,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    lookup = {str(name): index for index, name in enumerate(names)}
    signals: list[np.ndarray] = []
    labels: list[str] = []
    qc_rows: list[dict[str, object]] = []
    for stage in STAGE_ORDER:
        rois = members[stage]
        indices = np.asarray([lookup[roi] for roi in rois], dtype=int)
        for layer_index, layer in enumerate(LAYERS):
            constituent = ts[indices, layer_index, :]
            if not all(valid_signal(row, min_std) for row in constituent):
                raise RuntimeError(f"sub-{subject} {stage}_{layer}: invalid member")
            composite = constituent.mean(axis=0)
            if not valid_signal(composite, min_std):
                raise RuntimeError(f"sub-{subject} {stage}_{layer}: invalid composite")
            signals.append(composite)
            labels.append(f"{stage}_{layer}")
            qc_rows.append(
                {
                    "subject": f"sub-{subject}",
                    "stage": stage,
                    "layer": layer,
                    "members": "|".join(rois),
                    "n_members": len(rois),
                    "mean": float(composite.mean()),
                    "sd": float(composite.std()),
                    "has_nonfinite": bool(~np.isfinite(composite).all()),
                }
            )
    node_ts = np.vstack(signals).T.astype(np.float64)
    return node_ts, np.asarray(labels, dtype=object), qc_rows


def main() -> None:
    args = parse_args()
    validate_output_isolation(args.ts_base, args.out_dir)
    subjects = [normalize_subject(value) for value in args.subjects]
    if len(set(subjects)) != len(subjects):
        raise RuntimeError("Duplicate participants")

    loaded = {subject: load_subject(args.ts_base, subject) for subject in subjects}
    shapes = {ts.shape for ts, _ in loaded.values()}
    if len(shapes) != 1:
        raise RuntimeError(f"Participants do not share one input shape: {shapes}")

    members, membership_qc = freeze_membership(loaded, args.min_std)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    membership_qc.to_csv(
        args.out_dir / "evc_pooled_multistage_membership_qc.csv", index=False
    )

    export_rows: list[dict[str, object]] = []
    node_qc: list[dict[str, object]] = []
    for subject in subjects:
        ts, names = loaded[subject]
        node_ts, labels, subject_qc = build_subject(
            subject, ts, names, members, args.min_std
        )
        stem = args.out_dir / f"sub-{subject}_pial5_evc_pooled_multistage_layered"
        mat_file = stem.with_suffix(".mat")
        npy_file = stem.with_suffix(".npy")
        if (mat_file.exists() or npy_file.exists()) and not args.force:
            raise FileExistsError(f"Output exists; inspect or add --force: {mat_file}")

        savemat(
            mat_file,
            {
                "ts": node_ts,
                "labels": labels,
                "TR": float(args.tr),
                "TE": float(args.te),
                "Subject": f"sub-{subject}",
                "n_nodes": 9.0,
                "n_layers": 3.0,
                "stage_order": np.asarray(STAGE_ORDER, dtype=object),
                "layer_order": np.asarray(LAYERS, dtype=object),
                "source_ts_base": str(args.ts_base),
                "outer_pial_exclusion": "outer-pial 5% final input",
                "aggregation_rule": "equal ROI mean within frozen stage and layer",
                "stage_members_json": json.dumps(members),
                "model_definition": "EVC(V1,V2,V3)->PPC->cauPFC",
            },
        )
        np.save(npy_file, node_ts)
        node_qc.extend(subject_qc)
        export_rows.append(
            {
                "subject": f"sub-{subject}",
                "mat_file": str(mat_file),
                "n_timepoints": node_ts.shape[0],
                "n_nodes": node_ts.shape[1],
                "labels": "|".join(map(str, labels)),
                "source_ts_base": str(args.ts_base),
                "has_nonfinite": bool(~np.isfinite(node_ts).all()),
                "min_node_sd": float(np.std(node_ts, axis=0).min()),
                "EVC_members": "|".join(members["EVC"]),
            }
        )

    pd.DataFrame(node_qc).to_csv(
        args.out_dir / "evc_pooled_multistage_node_qc.csv", index=False
    )
    pd.DataFrame(export_rows).to_csv(
        args.out_dir / "evc_pooled_multistage_export_audit.csv", index=False
    )
    (args.out_dir / "evc_pooled_multistage_stage_definition.json").write_text(
        json.dumps(
            {
                "stage_order": STAGE_ORDER,
                "candidate_members": STAGE_CANDIDATES,
                "retained_members": members,
                "strict_stages": sorted(STRICT_STAGES),
                "selection_rule": (
                    "predefined membership; complete finite/variance QC only"
                ),
                "EVC_definition": "equal ROI mean of V1, V2 and V3",
                "source_ts_base": str(args.ts_base.resolve()),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("=" * 82)
    print("Outer-pial-5% EVC-pooled three-stage export complete")
    print(f"Input : {args.ts_base}")
    print(f"Output: {args.out_dir}")
    for stage in STAGE_ORDER:
        print(f"  {stage:7s}: {len(members[stage]):2d} ROI(s): {', '.join(members[stage])}")
    print(f"Participants: {len(subjects)} | nodes: 9")
    print("=" * 82)


if __name__ == "__main__":
    main()
