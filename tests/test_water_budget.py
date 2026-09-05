from __future__ import annotations

import math

import pytest

from awm.envs import IrrigationSystemSpec, WaterBudgetController


def spec(**overrides: object) -> IrrigationSystemSpec:
    values: dict[str, object] = {
        "seasonal_budget_mm": 100.0,
        "min_event_mm": 10.0,
        "max_event_mm": 30.0,
        "min_interval_days": 3,
        "horizon_days": 20,
    }
    values.update(overrides)
    return IrrigationSystemSpec(**values)  # type: ignore[arg-type]


def test_system_spec_has_no_silent_invalid_values() -> None:
    with pytest.raises(ValueError):
        spec(seasonal_budget_mm=0.0)
    with pytest.raises(ValueError):
        spec(min_event_mm=0.0)
    with pytest.raises(ValueError):
        spec(max_event_mm=5.0)
    with pytest.raises(ValueError):
        spec(min_interval_days=-1)
    with pytest.raises(ValueError):
        spec(horizon_days=0)


def test_exact_no_irrigation_is_preserved() -> None:
    controller = WaterBudgetController(spec())
    decision = controller.step(day=0, irrigate=False, amount_fraction=0.73)

    assert decision.applied_mm == 0.0
    assert not decision.event_applied
    assert not decision.was_projected
    assert decision.reasons == ("policy_no_irrigation",)
    assert controller.remaining_mm == 100.0


def test_amount_fraction_maps_to_minimum_and_maximum_event() -> None:
    low = WaterBudgetController(spec())
    high = WaterBudgetController(spec())

    low_decision = low.step(day=0, irrigate=True, amount_fraction=0.0)
    high_decision = high.step(day=0, irrigate=True, amount_fraction=1.0)

    assert low_decision.applied_mm == 10.0
    assert high_decision.applied_mm == 30.0


def test_amount_fraction_is_clipped_and_audited() -> None:
    controller = WaterBudgetController(spec())
    decision = controller.step(day=0, irrigate=True, amount_fraction=2.0)

    assert decision.applied_mm == 30.0
    assert decision.was_projected
    assert "amount_fraction_clipped" in decision.reasons


def test_minimum_interval_forces_exact_zero() -> None:
    controller = WaterBudgetController(spec())
    controller.step(day=0, irrigate=True, amount_fraction=0.0)

    blocked = controller.step(day=2, irrigate=True, amount_fraction=1.0)
    open_again = controller.step(day=3, irrigate=True, amount_fraction=0.0)

    assert blocked.applied_mm == 0.0
    assert blocked.was_projected
    assert blocked.reasons == ("minimum_interval_active",)
    assert open_again.applied_mm == 10.0


def test_remaining_budget_caps_event_capacity() -> None:
    controller = WaterBudgetController(
        spec(seasonal_budget_mm=35.0, min_interval_days=0)
    )
    first = controller.step(day=0, irrigate=True, amount_fraction=1.0)
    second = controller.step(day=1, irrigate=True, amount_fraction=1.0)

    assert first.applied_mm == 30.0
    assert second.applied_mm == 0.0
    assert second.was_projected
    assert second.reasons == ("remaining_budget_below_minimum_event",)
    assert controller.remaining_mm == 5.0


def test_budget_is_never_borrowed_from_future() -> None:
    controller = WaterBudgetController(
        spec(
            seasonal_budget_mm=45.0,
            min_event_mm=5.0,
            max_event_mm=30.0,
            min_interval_days=0,
        )
    )

    first = controller.step(day=0, irrigate=True, amount_fraction=1.0)
    second = controller.step(day=1, irrigate=True, amount_fraction=1.0)

    assert first.applied_mm == 30.0
    assert second.feasible_max_mm == 15.0
    assert second.applied_mm == 15.0
    assert controller.used_mm == 45.0
    assert controller.remaining_mm == 0.0


def test_budget_observation_updates_after_irrigation() -> None:
    controller = WaterBudgetController(spec())
    before = controller.observe(day=0, yield_target_fraction=0.95)
    controller.step(day=0, irrigate=True, amount_fraction=0.5)
    after = controller.observe(day=1, yield_target_fraction=0.95)

    assert before.remaining_budget_fraction == 1.0
    assert before.cumulative_irrigation_fraction == 0.0
    assert before.remaining_horizon_fraction == 1.0
    assert before.days_since_last_irrigation == 1

    assert math.isclose(after.remaining_budget_fraction, 0.8)
    assert math.isclose(after.cumulative_irrigation_fraction, 0.2)
    assert math.isclose(after.remaining_horizon_fraction, 19 / 20)
    assert after.days_since_last_irrigation == 1


def test_decision_days_must_be_strictly_increasing() -> None:
    controller = WaterBudgetController(spec())
    controller.step(day=0, irrigate=False, amount_fraction=0.0)

    with pytest.raises(ValueError, match="strictly increasing"):
        controller.step(day=0, irrigate=False, amount_fraction=0.0)


def test_invalid_target_and_nonfinite_amount_fail_fast() -> None:
    controller = WaterBudgetController(spec())
    with pytest.raises(ValueError):
        controller.observe(day=0, yield_target_fraction=1.01)
    with pytest.raises(ValueError):
        controller.step(day=0, irrigate=True, amount_fraction=float("nan"))
