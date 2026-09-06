from __future__ import annotations

import json
from pathlib import Path

import pytest

from awm.baselines.agricultural import (
    ConventionalScheduleBaseline,
    PotentialETWaterBalanceBaseline,
    RootZoneREWThresholdBaseline,
)
from awm.baselines.smoke import _validate_budget_accounting, build_baseline


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "agricultural_baselines_v1.json"
CONVENTIONAL = ROOT / "configs" / "formal_baseline_conventional_w100_2000.json"
ET = ROOT / "configs" / "formal_baseline_et_w100_2000.json"
REW = ROOT / "configs" / "formal_baseline_rew_w100_2000.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_common_water_accounting_is_total_45_fixed_plus_postplant_quota():
    common = _load(PROTOCOL)["common"]
    assert common["fixed_preplant_irrigation_mm"] == 45.0
    assert common["total_budget_mm"] == {"W100": 540.0, "W80": 432.0, "W60": 324.0}
    assert common["postplant_policy_budget_mm"] == {"W100": 495.0, "W80": 387.0, "W60": 279.0}
    for treatment in ("W100", "W80", "W60"):
        assert common["total_budget_mm"][treatment] == (
            common["fixed_preplant_irrigation_mm"]
            + common["postplant_policy_budget_mm"][treatment]
        )


def test_conventional_schedule_uses_field_timing_and_exact_quota_normalization():
    spec = _load(PROTOCOL)["local_conventional"]
    assert spec["source_file"] == "data/COX/XJHX2301.COX"
    assert spec["source_planting_doy"] == 124
    assert spec["source_postplant_irrigation_doys"] == [160, 170, 180, 190, 195, 202, 209, 216, 223, 230, 237]
    assert spec["action_daps"] == [36, 46, 56, 66, 71, 78, 85, 92, 99, 106, 113]

    expected_totals = {"W100": 495.0, "W80": 387.0, "W60": 279.0}
    for treatment, expected in expected_totals.items():
        schedule = spec["schedules_mm_by_action_dap"][treatment]
        assert list(map(int, schedule.keys())) == spec["action_daps"]
        assert sum(schedule.values()) == pytest.approx(expected)
        assert max(schedule.values()) <= 45.0
        assert all(round(value * 10) == pytest.approx(value * 10) for value in schedule.values())


def test_et_rule_is_preregistered_without_validation_tuning():
    protocol = _load(PROTOCOL)
    spec = protocol["potential_et_water_balance"]
    assert spec["trigger_deficit_mm"] == 40.5
    assert spec["irrigation_efficiency"] == 0.90
    assert spec["effective_rain_fraction"] == 1.0
    assert spec["refill_fraction"] == 1.0
    assert protocol["selection_policy"]["tuned_on_validation"] is False

    policy = build_baseline({
        "type": "potential_et_water_balance",
        "trigger_deficit_mm": spec["trigger_deficit_mm"],
        "irrigation_efficiency": spec["irrigation_efficiency"],
        "effective_rain_fraction": spec["effective_rain_fraction"],
        "refill_fraction": spec["refill_fraction"],
    })
    assert isinstance(policy, PotentialETWaterBalanceBaseline)


def test_rew_rule_uses_cotton_p065_mapping_and_shallow_fallback():
    spec = _load(PROTOCOL)["root_zone_rew_threshold"]
    assert spec["trigger_rew"] == 0.35
    assert spec["event_depth_mm"] == 45.0
    assert spec["fallback_rew_layers_1_based"] == [1, 2, 3, 4]

    policy = build_baseline({
        "type": "root_zone_rew_threshold",
        "trigger_rew": 0.35,
        "event_depth_mm": 45.0,
        "fallback_rew_layers_1_based": [1, 2, 3, 4],
    })
    assert isinstance(policy, RootZoneREWThresholdBaseline)


def test_w100_smoke_configs_share_identical_hard_constraints():
    configs = [_load(path) for path in (CONVENTIONAL, ET, REW)]
    for config in configs:
        assert config["status"] == "formal_agricultural_baseline_smoke"
        assert config["water_treatment"] == "W100"
        assert config["total_seasonal_budget_mm"] == 540.0
        assert config["seasonal_budget_mm"] == 495.0
        assert config["nonpolicy_irrigation_mm"] == 45.0
        assert config["max_event_mm"] == 45.0
        assert config["min_event_mm"] == 0.1
        assert config["execution_resolution_mm"] == 0.1
        assert config["min_interval_days"] == 0
        assert config["ircm_tolerance_mm"] == 0.1
        _validate_budget_accounting(config)

    assert isinstance(build_baseline(configs[0]["baseline"]), ConventionalScheduleBaseline)
    assert isinstance(build_baseline(configs[1]["baseline"]), PotentialETWaterBalanceBaseline)
    assert isinstance(build_baseline(configs[2]["baseline"]), RootZoneREWThresholdBaseline)


def test_budget_accounting_rejects_total_policy_fixed_mismatch():
    config = _load(CONVENTIONAL)
    config["total_seasonal_budget_mm"] = 539.9
    with pytest.raises(ValueError, match="must equal"):
        _validate_budget_accounting(config)
