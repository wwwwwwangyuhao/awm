"""Integrity validation and descriptive summary for development baseline sweeps.

This module intentionally reports descriptive agricultural-baseline statistics
only.  It does not define the paper's Y_ref, eta, alpha, model-selection rule,
or any final-test quantity.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Iterable, Mapping, Sequence

from .development_sweep import (
    DEVELOPMENT_YEARS,
    METHODS,
    TRAIN_YEARS,
    TREATMENTS,
    VALIDATION_YEARS,
)


EXPECTED_CANONICAL_EPISODES = len(DEVELOPMENT_YEARS) * len(METHODS) * len(TREATMENTS)
LOCKED_FINAL_TEST_YEARS = (2023, 2024, 2025)


def load_result_rows(path: str | Path) -> list[dict[str, object]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"development sweep results not found: {source}")
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid results JSONL at line {line_number}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"results JSONL line {line_number} is not an object")
        rows.append(payload)
    return rows


def validate_canonical_results(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Require exactly one passed row for every 2000--2022 × method × treatment."""

    expected_keys = {
        (year, method, treatment)
        for year in DEVELOPMENT_YEARS
        for method in METHODS
        for treatment in TREATMENTS
    }
    observed_keys: list[tuple[int, str, str]] = []
    problems: list[str] = []

    for idx, row in enumerate(rows, 1):
        try:
            year = int(row["year"])
            method = str(row["method"])
            treatment = str(row["water_treatment"])
        except KeyError as exc:
            problems.append(f"row {idx} missing key {exc.args[0]}")
            continue
        observed_keys.append((year, method, treatment))

        if row.get("status") != "passed":
            problems.append(f"row {idx} status is not passed")
        if row.get("weather_source") != "era5":
            problems.append(f"row {idx} weather_source is not era5")
        if year in LOCKED_FINAL_TEST_YEARS:
            problems.append(f"row {idx} illegally contains locked final-test year {year}")
        if year not in DEVELOPMENT_YEARS:
            problems.append(f"row {idx} contains non-development year {year}")
        expected_split = "train" if year in TRAIN_YEARS else "validation"
        if row.get("split") != expected_split:
            problems.append(
                f"row {idx} split mismatch for {year}: {row.get('split')!r} != {expected_split!r}"
            )
        if method not in METHODS:
            problems.append(f"row {idx} has unsupported method {method!r}")
        if treatment not in TREATMENTS:
            problems.append(f"row {idx} has unsupported treatment {treatment!r}")
        if row.get("irrigation_accounting_passed") is not True:
            problems.append(f"row {idx} irrigation accounting did not pass")

        for field in ("HWAM_kg_ha", "IRCM_mm", "policy_irrigation_mm", "IWP_kg_m3"):
            value = row.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                problems.append(f"row {idx} field {field} is not finite numeric")

    duplicate_keys = sorted({key for key in observed_keys if observed_keys.count(key) > 1})
    observed_set = set(observed_keys)
    missing_keys = sorted(expected_keys - observed_set)
    extra_keys = sorted(observed_set - expected_keys)

    if len(rows) != EXPECTED_CANONICAL_EPISODES:
        problems.append(
            f"row count {len(rows)} != canonical {EXPECTED_CANONICAL_EPISODES}"
        )
    if duplicate_keys:
        problems.append(f"duplicate episode keys: {duplicate_keys}")
    if missing_keys:
        problems.append(f"missing episode keys: {missing_keys}")
    if extra_keys:
        problems.append(f"unexpected episode keys: {extra_keys}")

    report = {
        "status": "passed" if not problems else "failed",
        "row_count": len(rows),
        "expected_row_count": EXPECTED_CANONICAL_EPISODES,
        "unique_episode_key_count": len(observed_set),
        "expected_unique_episode_key_count": len(expected_keys),
        "train_year_count": len(TRAIN_YEARS),
        "validation_year_count": len(VALIDATION_YEARS),
        "locked_final_test_years": list(LOCKED_FINAL_TEST_YEARS),
        "duplicate_episode_keys": [list(key) for key in duplicate_keys],
        "missing_episode_keys": [list(key) for key in missing_keys],
        "unexpected_episode_keys": [list(key) for key in extra_keys],
        "problems": problems,
    }
    if problems:
        raise RuntimeError(
            "canonical development sweep integrity validation failed: "
            + "; ".join(problems[:10])
            + (f"; ... ({len(problems)} total problems)" if len(problems) > 10 else "")
        )
    return report


def summarize_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Return descriptive statistics by split/method/treatment.

    The summary is intentionally distributional.  No row or aggregate is
    elevated into a formal yield reference or risk threshold here.
    """

    groups: dict[tuple[str, str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["split"]), str(row["method"]), str(row["water_treatment"]))].append(row)

    summaries: list[dict[str, object]] = []
    for key in sorted(groups):
        split, method, treatment = key
        group = groups[key]
        yields = _finite_values(group, "HWAM_kg_ha")
        irrigation = _finite_values(group, "IRCM_mm")
        policy_irrigation = _finite_values(group, "policy_irrigation_mm")
        iwp = _finite_values(group, "IWP_kg_m3")
        events = _finite_values(group, "irrigation_event_count")
        adjusted = _finite_values(group, "baseline_constraint_adjusted_event_count")
        adapter_projected = _finite_values(group, "adapter_projected_event_count")

        summaries.append(
            {
                "split": split,
                "method": method,
                "water_treatment": treatment,
                "n": len(group),
                "years": sorted(int(row["year"]) for row in group),
                "yield_kg_ha": _stats(yields),
                "ircm_mm": _stats(irrigation),
                "policy_irrigation_mm": _stats(policy_irrigation),
                "iwp_kg_m3": _stats(iwp),
                "irrigation_event_count": _stats(events),
                "baseline_constraint_adjusted_event_count": _stats(adjusted),
                "adapter_projected_event_count": _stats(adapter_projected),
                "accounting_pass_count": sum(
                    1 for row in group if row.get("irrigation_accounting_passed") is True
                ),
            }
        )

    per_year_conventional_w100 = [
        {
            "year": int(row["year"]),
            "split": str(row["split"]),
            "HWAM_kg_ha": float(row["HWAM_kg_ha"]),
            "IRCM_mm": float(row["IRCM_mm"]),
        }
        for row in sorted(rows, key=lambda row: int(row["year"]))
        if row["method"] == "conventional" and row["water_treatment"] == "W100"
    ]

    return {
        "status": "descriptive_summary_only",
        "formal_y_ref_defined": False,
        "eta_defined": False,
        "alpha_defined": False,
        "row_count": len(rows),
        "group_count": len(summaries),
        "groups": summaries,
        "conventional_w100_yield_trace_not_y_ref": per_year_conventional_w100,
    }


def build_summary_document(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    integrity = validate_canonical_results(rows)
    descriptive = summarize_rows(rows)
    return {
        "integrity": integrity,
        "descriptive": descriptive,
    }


def write_summary(
    results_jsonl: str | Path,
    *,
    output: str | Path,
) -> dict[str, object]:
    rows = load_result_rows(results_jsonl)
    summary = build_summary_document(rows)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _finite_values(rows: Iterable[Mapping[str, object]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"field {field} contains non-finite/non-numeric value")
        values.append(float(value))
    return values


def _stats(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty value sequence")
    return {
        "n": len(values),
        "mean": mean(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and descriptively summarize a canonical 207-row development sweep"
    )
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summary = write_summary(args.results, output=args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "EXPECTED_CANONICAL_EPISODES",
    "LOCKED_FINAL_TEST_YEARS",
    "build_summary_document",
    "load_result_rows",
    "summarize_rows",
    "validate_canonical_results",
    "write_summary",
]
