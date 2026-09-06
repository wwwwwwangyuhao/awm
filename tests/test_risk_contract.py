from __future__ import annotations

import json
from pathlib import Path

import pytest

from awm.risk import (
    DEFAULT_ALPHA,
    LOCKED_FINAL_TEST_YEARS,
    REGISTERED_ETA_LEVELS,
    build_development_reference_map,
    empirical_lower_cvar,
    evaluate_retention_risk,
    normalize_rows_by_reference,
    yield_retention,
)


ROOT = Path(__file__).resolve().parents[1]


def test_registered_risk_levels_are_frozen() -> None:
    assert DEFAULT_ALPHA == 0.20
    assert REGISTERED_ETA_LEVELS == (0.90, 0.95, 0.98)


def test_yield_retention_is_not_clipped_at_one() -> None:
    assert yield_retention(110.0, 100.0) == pytest.approx(1.10)
    assert yield_retention(90.0, 100.0) == pytest.approx(0.90)


def test_yield_retention_rejects_invalid_reference() -> None:
    with pytest.raises(ValueError, match="reference_yield_kg_ha must be > 0"):
        yield_retention(10.0, 0.0)


def test_lower_cvar_integer_tail_mass() -> None:
    values = [0.7, 0.8, 0.9, 1.0, 1.1]
    assert empirical_lower_cvar(values, alpha=0.20) == pytest.approx(0.7)
    assert empirical_lower_cvar(values, alpha=0.40) == pytest.approx(0.75)
    assert empirical_lower_cvar(values, alpha=1.0) == pytest.approx(0.9)


def test_lower_cvar_fractional_tail_mass() -> None:
    # n=4, alpha=0.375 -> 1.5 observations of lower-tail mass.
    # (0.7 + 0.5*0.8) / 1.5 = 0.733333...
    values = [1.0, 0.8, 0.7, 0.9]
    assert empirical_lower_cvar(values, alpha=0.375) == pytest.approx(11.0 / 15.0)


def test_lower_cvar_alpha_n_below_one_is_minimum() -> None:
    assert empirical_lower_cvar([0.82, 0.91, 1.03], alpha=0.20) == pytest.approx(0.82)


def test_lower_cvar_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="at least one"):
        empirical_lower_cvar([], alpha=0.20)
    with pytest.raises(ValueError, match="alpha"):
        empirical_lower_cvar([1.0], alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        empirical_lower_cvar([1.0], alpha=1.01)


def test_risk_evaluation_uses_zero_scientific_slack() -> None:
    result = evaluate_retention_risk([0.90, 0.95, 1.0, 1.05, 1.10], eta=0.90)
    assert result.lcvar == pytest.approx(0.90)
    assert result.margin == pytest.approx(0.0)
    assert result.feasible is True

    failed = evaluate_retention_risk([0.89, 0.95, 1.0, 1.05, 1.10], eta=0.90)
    assert failed.lcvar == pytest.approx(0.89)
    assert failed.feasible is False


def _canonical_reference_rows() -> list[dict[str, object]]:
    config = json.loads(
        (ROOT / "configs" / "yield_reference_v1_development.json").read_text(
            encoding="utf-8"
        )
    )
    values = {**config["training_years"], **config["validation_years"]}
    return [
        {
            "status": "passed",
            "year": int(year),
            "method": "conventional",
            "water_treatment": "W100",
            "HWAM_kg_ha": float(value),
        }
        for year, value in values.items()
    ]


def test_frozen_development_reference_table_matches_extractor() -> None:
    rows = _canonical_reference_rows()
    derived = build_development_reference_map(rows)
    assert len(derived) == 23
    assert derived[2000] == 6831.0
    assert derived[2022] == 8005.0


def test_reference_extractor_rejects_station_final_test() -> None:
    rows = _canonical_reference_rows()
    rows.append(
        {
            "status": "passed",
            "year": LOCKED_FINAL_TEST_YEARS[0],
            "method": "conventional",
            "water_treatment": "W100",
            "HWAM_kg_ha": 7000.0,
        }
    )
    with pytest.raises(ValueError, match="final-test year"):
        build_development_reference_map(rows)


def test_normalization_rejects_station_final_test_and_preserves_above_one() -> None:
    normalized = normalize_rows_by_reference(
        [{"year": 2000, "HWAM_kg_ha": 7000.0}],
        {2000: 6831.0},
    )
    assert normalized[0]["yield_retention"] > 1.0

    with pytest.raises(ValueError, match="final-test year"):
        normalize_rows_by_reference(
            [{"year": 2023, "HWAM_kg_ha": 7000.0}],
            {2023: 7000.0},
        )


def test_machine_readable_contract_matches_code_constants() -> None:
    contract = json.loads(
        (ROOT / "configs" / "risk_contract_v1.json").read_text(encoding="utf-8")
    )
    assert contract["risk_measure"]["alpha"] == DEFAULT_ALPHA
    assert tuple(contract["protection_targets"]["eta_levels"]) == REGISTERED_ETA_LEVELS
    assert contract["yield_retention"]["clip_at_one"] is False
    assert contract["weather_measure"]["locked_final_test_station_years"] == [2023, 2024, 2025]
    assert contract["development_freeze_provenance"]["learned_policy_results_seen_before_freeze"] is False
    assert contract["development_freeze_provenance"]["station_final_test_results_seen_before_freeze"] is False


def test_reference_config_contains_no_station_reference_values() -> None:
    reference = json.loads(
        (ROOT / "configs" / "yield_reference_v1_development.json").read_text(
            encoding="utf-8"
        )
    )
    assert reference["locked_final_test_station_years"] == [2023, 2024, 2025]
    assert reference["final_test_reference_values"] is None
    years = {int(y) for y in reference["training_years"]} | {
        int(y) for y in reference["validation_years"]
    }
    assert years == set(range(2000, 2023))
    assert years.isdisjoint({2023, 2024, 2025})
