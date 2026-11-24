#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate the pooled-EVC LOSO gradient and MPC-functional outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import os
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "/path/to/project")  # see config/paths.sh


ROOT = Path(PROJECT_ROOT + "/derivatives_0331")
DEFAULT_GRADIENT = (
    ROOT / "S6_validation" / "pial5_only_update"
    / "functional_gradient_bianca_loso_pooled_evc"
)
DEFAULT_ALIGNMENT = (
    ROOT / "S6_validation" / "pial5_only_update"
    / "mpc_functional_alignment_loso_pooled_evc"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gradient-dir", type=Path, default=DEFAULT_GRADIENT)
    parser.add_argument("--alignment-dir", type=Path, default=DEFAULT_ALIGNMENT)
    parser.add_argument("--expected-n", type=int, default=15)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def finite_number(row: dict[str, str], key: str) -> float:
    if key not in row:
        raise KeyError(f"Missing column {key!r}")
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {key}: {row[key]!r}")
    return value


def main() -> None:
    args = parse_args()
    failures: list[str] = []
    checks: dict[str, object] = {}

    gradient_meta = load_json(args.gradient_dir / "analysis_metadata.json")
    if gradient_meta.get("reference_mode") != "loso":
        failures.append("Gradient metadata reference_mode is not 'loso'")
    if int(gradient_meta.get("n_subjects", -1)) != args.expected_n:
        failures.append("Gradient metadata participant count mismatch")

    stage_used = load_json(args.gradient_dir / "stage_definition_used.json")
    retained = stage_used.get("retained_members", {})
    if retained.get("EVC") != ["V1", "V2", "V3"]:
        failures.append(f"EVC is not pooled V1/V2/V3: {retained.get('EVC')}")
    if "V1" in retained:
        failures.append("A separate V1 stage is still present")

    references = sorted((args.gradient_dir / "references").glob("sub-*_loso_reference.npy"))
    gradients = sorted((args.gradient_dir / "subject_gradients").glob("sub-*.npy"))
    gradients = [p for p in gradients if not p.name.endswith("_native.npy")]
    if len(references) != args.expected_n:
        failures.append(f"Expected {args.expected_n} LOSO references, found {len(references)}")
    if len(gradients) != args.expected_n:
        failures.append(f"Expected {args.expected_n} aligned gradients, found {len(gradients)}")

    alignment_meta = load_json(args.alignment_dir / "analysis_metadata.json")
    if alignment_meta.get("functional_reference_mode") != "loso":
        failures.append("Alignment metadata functional_reference_mode is not 'loso'")
    if int(alignment_meta.get("n_subjects", -1)) != args.expected_n:
        failures.append("Alignment metadata participant count mismatch")

    subject_rows = load_csv(args.alignment_dir / "subject_mpc_functional_alignment.csv")
    counts = Counter(row.get("component") for row in subject_rows)
    expected_counts = {f"G{i}": args.expected_n for i in range(1, 4)}
    if dict(counts) != expected_counts:
        failures.append(f"Participant/component counts mismatch: {dict(counts)}")
    if len({row.get("subject") for row in subject_rows}) != args.expected_n:
        failures.append("Participant CSV unique-subject count mismatch")

    stats_rows = load_csv(
        args.alignment_dir / "mpc_functional_alignment_primary_statistics.csv"
    )
    if {row.get("component") for row in stats_rows} != {"G1", "G2", "G3"}:
        failures.append("Statistics CSV does not contain exactly G1/G2/G3")

    summary_rows: list[dict[str, object]] = []
    for row in sorted(stats_rows, key=lambda item: item["component"]):
        component = row["component"]
        mean = finite_number(row, "mean_subject_spearman_rho")
        low = finite_number(row, "participant_bootstrap_ci95_low")
        high = finite_number(row, "participant_bootstrap_ci95_high")
        if not low <= mean <= high:
            failures.append(f"{component}: bootstrap CI does not contain mean")
        for p_column in (
            "exact_signflip_p_two_sided",
            "holm_p_across_3_components",
            "paired_bilateral_spin_p_two_sided",
            "holm_spin_p_across_3_components",
        ):
            p_value = finite_number(row, p_column)
            if not 0.0 <= p_value <= 1.0:
                failures.append(f"{component}: invalid {p_column}={p_value}")
        expected_role = "prespecified_primary" if component == "G1" else "exploratory"
        if row.get("analysis_role") != expected_role:
            failures.append(
                f"{component}: analysis_role={row.get('analysis_role')!r}; "
                f"expected {expected_role!r}"
            )
        summary_rows.append(
            {
                "component": component,
                "role": expected_role,
                "mean_rho": mean,
                "ci95_low": low,
                "ci95_high": high,
                "participant_p_holm": finite_number(row, "holm_p_across_3_components"),
                "spatial_spin_p_holm": finite_number(row, "holm_spin_p_across_3_components"),
            }
        )

    checks.update(
        {
            "gradient_reference_mode": gradient_meta.get("reference_mode"),
            "alignment_reference_mode": alignment_meta.get("functional_reference_mode"),
            "n_loso_references": len(references),
            "n_aligned_gradients": len(gradients),
            "n_participant_rows": len(subject_rows),
            "retained_stage_members": retained,
            "component_summary": summary_rows,
        }
    )
    report = {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "checks": checks,
    }
    report_path = args.alignment_dir / "loso_validation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Validation status: {report['status']}")
    print(f"Validation report: {report_path}")
    print("component  role                  mean rho   95% CI              participant P_Holm  spin P_Holm")
    for row in summary_rows:
        print(
            f"{row['component']:<9} {row['role']:<21} "
            f"{row['mean_rho']:+.3f}     "
            f"[{row['ci95_low']:+.3f}, {row['ci95_high']:+.3f}]   "
            f"{row['participant_p_holm']:.6g}             "
            f"{row['spatial_spin_p_holm']:.6g}"
        )
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
