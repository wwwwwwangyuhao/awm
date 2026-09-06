import pytest

from awm.baselines.development_sweep import DEVELOPMENT_YEARS, METHODS, TREATMENTS
from awm.baselines.sweep_summary import (
    EXPECTED_CANONICAL_EPISODES,
    build_summary_document,
    validate_canonical_results,
)


def canonical_rows():
    rows = []
    for year in DEVELOPMENT_YEARS:
        split = "train" if year <= 2017 else "validation"
        for method_idx, method in enumerate(METHODS):
            for treatment_idx, treatment in enumerate(TREATMENTS):
                total = {"W100": 540.0, "W80": 432.0, "W60": 324.0}[treatment]
                policy = total - 45.0
                rows.append(
                    {
                        "status": "passed",
                        "year": year,
                        "split": split,
                        "method": method,
                        "baseline_name": method,
                        "water_treatment": treatment,
                        "weather_source": "era5",
                        "HWAM_kg_ha": 5000.0 + (year - 2000) * 10 + method_idx * 100,
                        "IRCM_mm": total,
                        "policy_irrigation_mm": policy,
                        "IWP_kg_m3": 1.0 + treatment_idx * 0.1,
                        "baseline_constraint_adjusted_event_count": method_idx,
                        "adapter_projected_event_count": 0,
                        "irrigation_event_count": 10 - treatment_idx,
                        "irrigation_accounting_passed": True,
                    }
                )
    return rows


def test_canonical_matrix_is_exactly_207_unique_passed_rows():
    rows = canonical_rows()
    assert len(rows) == EXPECTED_CANONICAL_EPISODES == 207
    report = validate_canonical_results(rows)
    assert report["status"] == "passed"
    assert report["unique_episode_key_count"] == 207
    assert report["train_year_count"] == 18
    assert report["validation_year_count"] == 5
    assert report["locked_final_test_years"] == [2023, 2024, 2025]


def test_summary_is_descriptive_and_does_not_define_risk_contract():
    summary = build_summary_document(canonical_rows())
    descriptive = summary["descriptive"]
    assert descriptive["formal_y_ref_defined"] is False
    assert descriptive["eta_defined"] is False
    assert descriptive["alpha_defined"] is False
    assert descriptive["group_count"] == 18  # 2 splits × 3 methods × 3 treatments
    assert len(descriptive["conventional_w100_yield_trace_not_y_ref"]) == 23

    train_conventional_w100 = next(
        group
        for group in descriptive["groups"]
        if group["split"] == "train"
        and group["method"] == "conventional"
        and group["water_treatment"] == "W100"
    )
    assert train_conventional_w100["n"] == 18
    assert train_conventional_w100["accounting_pass_count"] == 18


def test_integrity_rejects_missing_duplicate_and_final_test_rows():
    rows = canonical_rows()
    with pytest.raises(RuntimeError, match="row count"):
        validate_canonical_results(rows[:-1])

    duplicated = rows + [dict(rows[0])]
    with pytest.raises(RuntimeError, match="duplicate episode keys"):
        validate_canonical_results(duplicated)

    contaminated = canonical_rows()
    contaminated[0] = dict(contaminated[0], year=2023, split="validation")
    with pytest.raises(RuntimeError, match="locked final-test year"):
        validate_canonical_results(contaminated)


def test_integrity_rejects_failed_irrigation_accounting():
    rows = canonical_rows()
    rows[0] = dict(rows[0], irrigation_accounting_passed=False)
    with pytest.raises(RuntimeError, match="irrigation accounting did not pass"):
        validate_canonical_results(rows)
