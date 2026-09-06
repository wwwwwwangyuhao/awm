"""Development-only audit for Risk Contract v1.

This CLI consumes the canonical 2000-2022 agricultural-baseline JSONL sweep,
verifies the frozen Y_ref table, normalizes every yield by same-year reference,
and reports empirical lower-CVaR values for the registered eta levels.
It cannot consume 2023-2025 station final-test rows.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Mapping, Sequence

from awm.baselines.sweep_summary import load_result_rows, validate_canonical_results

from .contract import (
    DEFAULT_ALPHA,
    REGISTERED_ETA_LEVELS,
    build_development_reference_map,
    evaluate_retention_risk,
    normalize_rows_by_reference,
)


def load_frozen_reference_table(path: str | Path) -> dict[int, float]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("final_test_reference_values") is not None:
        raise ValueError("development reference table must not contain station final-test values")
    merged: dict[int, float] = {}
    for section in ("training_years", "validation_years"):
        values = payload.get(section)
        if not isinstance(values, dict):
            raise TypeError(f"reference config {section} must be an object")
        for year_text, value in values.items():
            year = int(year_text)
            if year in merged:
                raise ValueError(f"duplicate frozen Y_ref for year {year}")
            merged[year] = float(value)
    return dict(sorted(merged.items()))


def build_development_risk_audit(
    rows: Sequence[Mapping[str, object]],
    *,
    frozen_reference: Mapping[int, float],
    alpha: float = DEFAULT_ALPHA,
    eta_levels: Sequence[float] = REGISTERED_ETA_LEVELS,
) -> dict[str, object]:
    integrity = validate_canonical_results(rows)
    derived_reference = build_development_reference_map(rows)
    frozen = {int(year): float(value) for year, value in frozen_reference.items()}
    if derived_reference != frozen:
        mismatches = {
            year: {"derived": derived_reference.get(year), "frozen": frozen.get(year)}
            for year in sorted(set(derived_reference) | set(frozen))
            if derived_reference.get(year) != frozen.get(year)
        }
        raise RuntimeError(f"frozen Y_ref table mismatch: {mismatches}")

    normalized = normalize_rows_by_reference(rows, frozen)
    groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in normalized:
        groups[(str(row["split"]), str(row["method"]), str(row["water_treatment"]))].append(
            float(row["yield_retention"])
        )

    group_reports: list[dict[str, object]] = []
    for (split, method, treatment), values in sorted(groups.items()):
        evaluations = [
            evaluate_retention_risk(values, eta=float(eta), alpha=alpha)
            for eta in eta_levels
        ]
        group_reports.append(
            {
                "split": split,
                "method": method,
                "water_treatment": treatment,
                "n": len(values),
                "alpha": float(alpha),
                "tail_mass_observations": float(alpha) * len(values),
                "mean_retention": evaluations[0].mean_retention,
                "minimum_retention": evaluations[0].minimum_retention,
                "lcvar": evaluations[0].lcvar,
                "eta_results": [
                    {
                        "eta": item.eta,
                        "margin": item.margin,
                        "feasible": item.feasible,
                    }
                    for item in evaluations
                ],
            }
        )

    reference_groups = [
        item
        for item in group_reports
        if item["method"] == "conventional" and item["water_treatment"] == "W100"
    ]
    if len(reference_groups) != 2 or any(abs(float(item["lcvar"]) - 1.0) > 1e-12 for item in reference_groups):
        raise RuntimeError("conventional-W100 normalized reference must have LCVaR exactly 1")

    return {
        "status": "passed",
        "risk_contract_id": "awm-risk-contract-v1",
        "integrity": integrity,
        "alpha": float(alpha),
        "eta_levels": [float(value) for value in eta_levels],
        "reference_year_count": len(frozen),
        "final_test_station_results_present": False,
        "group_reports": group_reports,
    }


def audit_files(
    results_jsonl: str | Path,
    *,
    reference_config: str | Path,
    output: str | Path | None = None,
) -> dict[str, object]:
    rows = load_result_rows(results_jsonl)
    reference = load_frozen_reference_table(reference_config)
    report = build_development_risk_audit(rows, frozen_reference=reference)
    if output is not None:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Risk Contract v1 on the canonical 2000-2022 development sweep"
    )
    parser.add_argument("--results", required=True)
    parser.add_argument(
        "--reference-config",
        default="configs/yield_reference_v1_development.json",
    )
    parser.add_argument("--output")
    args = parser.parse_args()
    report = audit_files(
        args.results,
        reference_config=args.reference_config,
        output=args.output,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "audit_files",
    "build_development_risk_audit",
    "load_frozen_reference_table",
]
