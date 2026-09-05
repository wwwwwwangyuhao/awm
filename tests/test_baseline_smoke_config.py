import pytest

from awm.baselines.agricultural import (
    ConventionalScheduleBaseline,
    PotentialETWaterBalanceBaseline,
    RootZoneREWThresholdBaseline,
)
from awm.baselines.smoke import build_baseline


def test_build_conventional_baseline_from_json_style_keys():
    policy = build_baseline(
        {
            "type": "local_conventional",
            "schedule_mm_by_action_dap": {"1": 10.0, "3": 20.0},
        }
    )
    assert isinstance(policy, ConventionalScheduleBaseline)
    assert policy.schedule_mm == {1: 10.0, 3: 20.0}


def test_build_potential_et_baseline_requires_explicit_parameters():
    policy = build_baseline(
        {
            "type": "potential_et_water_balance",
            "trigger_deficit_mm": 5.0,
            "irrigation_efficiency": 0.9,
            "effective_rain_fraction": 0.8,
            "refill_fraction": 1.0,
        }
    )
    assert isinstance(policy, PotentialETWaterBalanceBaseline)
    with pytest.raises(KeyError):
        build_baseline({"type": "potential_et_water_balance"})


def test_build_rew_baseline_requires_explicit_layers():
    policy = build_baseline(
        {
            "type": "root_zone_rew_threshold",
            "trigger_rew": 0.5,
            "event_depth_mm": 15.0,
            "fallback_rew_layers_1_based": [1, 2],
        }
    )
    assert isinstance(policy, RootZoneREWThresholdBaseline)
    assert policy.fallback_rew_layers == (1, 2)


def test_unknown_baseline_type_is_rejected():
    with pytest.raises(ValueError, match="unsupported agricultural baseline"):
        build_baseline({"type": "unknown"})
