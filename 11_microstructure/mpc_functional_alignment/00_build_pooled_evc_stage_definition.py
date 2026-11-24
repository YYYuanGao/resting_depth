#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a pooled-EVC stage definition from the final pial5 common-valid ROIs.

This utility applies the same finite-data, missing-data and non-constant checks
used by ``01_gradient_bianca_loso_pooled_evc.py``.  EVC is fixed as
V1/V2/V3. PPC is the subset of the frozen PPC candidates that passes QC in
every participant. The cauPFC set is fixed and must pass intact.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import os
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "/path/to/project")  # see config/paths.sh


ROOT = Path(PROJECT_ROOT + "/derivatives_0331")
DEFAULT_TS_BASE = (
    ROOT / "S6_validation" / "pial_depth_exclusion"
    / "Step3_TimeSeries_tSNRclean_pial5pct"
)
DEFAULT_OUT = (
    ROOT / "S6_validation" / "pial5_only_update"
    / "functional_gradient_loso_stage_definition_pooled_evc.json"
)
DEFAULT_SUBJECTS = [
    "01", "02", "03", "04", "05", "06", "07", "08",
    "09", "10", "11", "12", "13", "14", "16",
]

STAGE_CANDIDATES = {
    "EVC": ("V1", "V2", "V3"),
    "PPC": (
        "IPS1", "LIPv", "LIPd", "VIP", "AIP", "MIP",
        "IP0", "IP1", "IP2", "7PC", "7Am", "7AL", "7PL", "7Pm", "7m",
    ),
    "cauPFC": ("6d", "6v", "6mp", "FEF", "PEF", "55b", "SFL"),
}
NAME_MAP = {"7Al": "7AL", "8AD": "8Ad"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ts-base", type=Path, default=DEFAULT_TS_BASE)
    parser.add_argument("--tag", default="ld1000_3layers_equivol")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--subjects", nargs="+", default=DEFAULT_SUBJECTS)
    parser.add_argument("--max-nan-frac", type=float, default=0.20)
    return parser.parse_args()


def normalize_roi(value: object) -> str:
    return NAME_MAP.get(str(value).strip(), str(value).strip())


def layer_qc(ts: np.ndarray, max_nan_frac: float) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(ts, float)
    if array.ndim != 3 or array.shape[1] != 3:
        raise ValueError(f"Expected ROI x 3 layers x time, got {array.shape}")
    valid_layer = np.zeros(array.shape[:2], dtype=bool)
    grid = np.arange(array.shape[2])
    for roi_index in range(array.shape[0]):
        for layer_index in range(array.shape[1]):
            row = array[roi_index, layer_index].copy()
            finite = np.isfinite(row)
            nan_fraction = 1.0 - float(finite.mean())
            if nan_fraction > max_nan_frac or int(finite.sum()) < 3:
                continue
            if not finite.all():
                row[~finite] = np.interp(grid[~finite], grid[finite], row[finite])
            if float(np.std(row)) < 1e-8:
                continue
            valid_layer[roi_index, layer_index] = True
    return valid_layer.all(axis=1), valid_layer


def main() -> None:
    args = parse_args()
    source_dir = (args.ts_base / args.tag).resolve()
    if "Step3_TimeSeries_tSNRclean_pial5pct" not in str(source_dir):
        raise ValueError(f"Refusing non-final input branch: {source_dir}")

    canonical_names: list[str] | None = None
    participant_validity: list[np.ndarray] = []
    audit_rows: list[dict[str, object]] = []

    for subject in [str(s).replace("sub-", "") for s in args.subjects]:
        subject_dir = source_dir / f"sub-{subject}"
        ts_path = subject_dir / "ts_matrix.npy"
        names_path = subject_dir / "roi_names.npy"
        if not ts_path.is_file() or not names_path.is_file():
            raise FileNotFoundError(f"Missing final pial5 input: {subject_dir}")
        ts = np.load(ts_path)
        names = [normalize_roi(x) for x in np.load(names_path, allow_pickle=True)]
        if len(names) != ts.shape[0] or len(set(names)) != len(names):
            raise ValueError(f"Invalid or duplicated ROI names: {names_path}")
        if canonical_names is None:
            canonical_names = names
        elif names != canonical_names:
            raise ValueError(f"ROI order differs for sub-{subject}")
        roi_valid, layer_valid = layer_qc(ts, float(args.max_nan_frac))
        participant_validity.append(roi_valid)
        audit_rows.append(
            {
                "subject": f"sub-{subject}",
                "n_rois": len(names),
                "n_valid_roi_layers": int(layer_valid.sum()),
                "n_common_candidate_rois_passed_subject_qc": int(roi_valid.sum()),
            }
        )

    if canonical_names is None:
        raise RuntimeError("No participants were loaded")
    common_valid = np.logical_and.reduce(participant_validity)
    common_names = {
        roi for roi, keep in zip(canonical_names, common_valid) if bool(keep)
    }

    retained: dict[str, list[str]] = {}
    excluded: dict[str, list[str]] = {}
    for stage, candidates in STAGE_CANDIDATES.items():
        retained[stage] = [roi for roi in candidates if roi in common_names]
        excluded[stage] = [roi for roi in candidates if roi not in common_names]

    for strict_stage in ("EVC", "cauPFC"):
        if tuple(retained[strict_stage]) != STAGE_CANDIDATES[strict_stage]:
            raise RuntimeError(
                f"Strict stage {strict_stage} is incomplete: "
                f"missing {excluded[strict_stage]}"
            )
    if len(retained["PPC"]) < 5:
        raise RuntimeError(
            f"Only {len(retained['PPC'])} PPC candidates passed common-valid QC"
        )

    definition = {
        "source_ts_base": str(source_dir),
        "subjects": [f"sub-{str(s).replace('sub-', '')}" for s in args.subjects],
        "stage_order": ["EVC", "PPC", "cauPFC"],
        "retained_members": retained,
        "excluded_by_common_valid_qc": excluded,
        "stage_scheme": "pooled EVC (V1/V2/V3) -> PPC -> cauPFC",
        "max_nan_fraction_per_roi_layer": float(args.max_nan_frac),
        "n_common_valid_rois": int(common_valid.sum()),
        "definition_role": (
            "Frozen pooled-EVC stage definition derived without reference to "
            "MPC-functional association results."
        ),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(definition, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    audit_path = args.out_json.with_name(args.out_json.stem + "_qc.csv")
    with audit_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)

    print(f"Saved stage definition: {args.out_json}")
    print(f"Saved QC audit:        {audit_path}")
    for stage in ("EVC", "PPC", "cauPFC"):
        print(
            f"{stage}: retained={len(retained[stage])} {retained[stage]} | "
            f"excluded={excluded[stage]}"
        )


if __name__ == "__main__":
    main()
