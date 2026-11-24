#!/usr/bin/env python3
"""Export matched rDCM A matrices to the long table used downstream.

The collector for the matched pial analysis writes one row per A(target,source)
parameter.  The task and qT1 scripts expect one row per *interregional* edge
with explicit source/target groups and layers.  This script performs only that
lossless schema conversion and writes one frozen table per condition.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import os
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "/path/to/project")  # see config/paths.sh


DERIV = Path(PROJECT_ROOT + "/derivatives_0331")
DEFAULT_EDGE_CSV = (
    DERIV / "S6_validation" / "pial_depth_exclusion" /
    "matched_current_rdcm" / "group_audit" /
    "matched_pial5_A_edges_long.csv"
)
DEFAULT_OUT_BASE = (
    DERIV / "S6_validation" / "pial5_only_update" / "rdcm_long"
)
DEFAULT_CONDITIONS = ("pial5_current",)
LAYER_ORDER = {"deep": 0, "mid": 1, "sup": 2}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--edge-csv", type=Path, default=DEFAULT_EDGE_CSV)
    p.add_argument("--out-base", type=Path, default=DEFAULT_OUT_BASE)
    p.add_argument("--conditions", nargs="+", default=list(DEFAULT_CONDITIONS))
    p.add_argument("--expected-subjects", type=int, default=15)
    return p.parse_args()


def split_node_label(label: object) -> tuple[str, str]:
    text = str(label)
    for layer in LAYER_ORDER:
        suffix = f"_{layer}"
        if text.endswith(suffix):
            return text[: -len(suffix)], layer
    raise ValueError(f"Node label does not end in _deep/_mid/_sup: {text!r}")


def normalize_subject(value: object) -> str:
    text = str(value).replace("sub-", "").strip()
    return f"sub-{int(text):02d}"


def convert_condition(edges: pd.DataFrame, condition: str, expected_subjects: int) -> pd.DataFrame:
    part = edges.loc[edges["condition"].astype(str) == condition].copy()
    if part.empty:
        raise RuntimeError(f"No rows found for condition={condition!r}")

    target_parts = part["target_label"].map(split_node_label)
    source_parts = part["source_label"].map(split_node_label)
    part["target_group"] = [x[0] for x in target_parts]
    part["target_layer"] = [x[1] for x in target_parts]
    part["source_group"] = [x[0] for x in source_parts]
    part["source_layer"] = [x[1] for x in source_parts]
    part["subject"] = part["subject"].map(normalize_subject)
    part["A"] = pd.to_numeric(part["A"], errors="raise")

    # The downstream tables contain the two 3x3 interregional blocks only.
    part = part.loc[part["source_group"] != part["target_group"]].copy()
    part["direction"] = part["source_group"] + "_to_" + part["target_group"]
    part["matrix_convention"] = "A(target,source): directed influence from source to target"
    part["validation_condition"] = condition

    keep = [
        "subject", "network", "direction", "source_group", "target_group",
        "source_layer", "target_layer", "A", "matrix_convention",
        "validation_condition", "output_file",
    ]
    for col in keep:
        if col not in part.columns:
            part[col] = ""
    out = part[keep].copy()

    # Fail closed on incomplete/duplicated matrices.
    key = ["subject", "network", "direction", "source_layer", "target_layer"]
    if out.duplicated(key).any():
        bad = out.loc[out.duplicated(key, keep=False), key]
        raise RuntimeError(f"Duplicate directed edges detected:\n{bad.to_string(index=False)}")
    counts = out.groupby(["subject", "network"], sort=False).size()
    if not counts.eq(18).all():
        raise RuntimeError(
            "Expected 18 interregional edges per subject-network pair; got:\n" +
            counts[counts != 18].to_string()
        )
    networks = sorted(out["network"].astype(str).unique())
    for network in networks:
        n_sub = out.loc[out["network"].astype(str) == network, "subject"].nunique()
        if n_sub != expected_subjects:
            raise RuntimeError(
                f"{condition} {network}: expected {expected_subjects} subjects, got {n_sub}"
            )
    if not np.isfinite(out["A"].to_numpy(float)).all():
        raise RuntimeError(f"Non-finite A values found for {condition}")

    out["_source_order"] = out["source_layer"].map(LAYER_ORDER)
    out["_target_order"] = out["target_layer"].map(LAYER_ORDER)
    out = out.sort_values(
        ["network", "subject", "direction", "_target_order", "_source_order"]
    ).drop(columns=["_source_order", "_target_order"])
    return out.reset_index(drop=True)


def main() -> None:
    args = parse_args()
    if not args.edge_csv.exists():
        raise FileNotFoundError(args.edge_csv)
    edges = pd.read_csv(args.edge_csv)
    required = {
        "subject", "network", "condition", "target_label", "source_label", "A"
    }
    missing = required - set(edges.columns)
    if missing:
        raise RuntimeError(f"{args.edge_csv} missing columns: {sorted(missing)}")

    audit_rows = []
    for condition in args.conditions:
        out = convert_condition(edges, condition, args.expected_subjects)
        out_dir = args.out_base / condition
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "merged_layer_rdcm_connections_long.csv"
        out.to_csv(out_file, index=False)
        for network, part in out.groupby("network", sort=False):
            audit_rows.append({
                "condition": condition,
                "network": network,
                "n_subjects": part["subject"].nunique(),
                "n_rows": len(part),
                "edges_per_subject_network": int(
                    part.groupby(["subject", "network"]).size().iloc[0]
                ),
                "A_min": float(part["A"].min()),
                "A_max": float(part["A"].max()),
                "status": "PASS",
                "output_csv": str(out_file),
            })
        print(f"Saved: {out_file}")

    args.out_base.mkdir(parents=True, exist_ok=True)
    audit_file = args.out_base / "rdcm_downstream_export_audit.csv"
    pd.DataFrame(audit_rows).to_csv(audit_file, index=False)
    print(f"Audit: {audit_file}")


if __name__ == "__main__":
    main()
