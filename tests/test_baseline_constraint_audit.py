from types import SimpleNamespace

import pytest

from awm.baselines.agricultural import IrrigationRequest
from awm.baselines.runner import _baseline_constraint_audit


class Controller:
    def __init__(
        self,
        *,
        remaining_mm=100.0,
        min_event_mm=0.1,
        max_event_mm=45.0,
        min_interval_days=0,
        last_irrigation_day=None,
    ):
        self._remaining_mm = float(remaining_mm)
        self.spec = SimpleNamespace(
            min_event_mm=float(min_event_mm),
            max_event_mm=float(max_event_mm),
            min_interval_days=int(min_interval_days),
        )
        self.last_irrigation_day = last_irrigation_day

    @property
    def remaining_mm(self):
        return self._remaining_mm

    def feasible_max_mm(self, *, day):
        if (
            self.last_irrigation_day is not None
            and day - self.last_irrigation_day < self.spec.min_interval_days
        ):
            return 0.0
        return min(self.spec.max_event_mm, self.remaining_mm)


def request_for_depth(*, desired_mm, constrained_mm, controller):
    feasible = controller.feasible_max_mm(day=10)
    if constrained_mm <= 0.0:
        return IrrigationRequest(False, 0.0, float(desired_mm), "test")
    if feasible <= controller.spec.min_event_mm:
        fraction = 0.0
    else:
        fraction = (
            float(constrained_mm) - controller.spec.min_event_mm
        ) / (feasible - controller.spec.min_event_mm)
    return IrrigationRequest(True, fraction, float(desired_mm), "test")


def test_remaining_budget_clip_is_visible_before_adapter_projection():
    controller = Controller(remaining_mm=9.0)
    request = request_for_depth(
        desired_mm=45.0,
        constrained_mm=9.0,
        controller=controller,
    )
    audit = _baseline_constraint_audit(
        policy_day=10,
        request=request,
        controller=controller,
    )
    assert audit.desired_mm == 45.0
    assert audit.constrained_request_mm == pytest.approx(9.0)
    assert audit.adjusted is True
    assert "remaining_seasonal_budget" in audit.reasons


def test_event_capacity_clip_is_distinct_from_seasonal_quota_clip():
    controller = Controller(remaining_mm=200.0, max_event_mm=45.0)
    request = request_for_depth(
        desired_mm=60.0,
        constrained_mm=45.0,
        controller=controller,
    )
    audit = _baseline_constraint_audit(
        policy_day=10,
        request=request,
        controller=controller,
    )
    assert audit.constrained_request_mm == 45.0
    assert audit.reasons == ("maximum_event_depth",)


def test_minimum_interval_can_turn_positive_desire_into_zero_request():
    controller = Controller(
        remaining_mm=100.0,
        min_interval_days=5,
        last_irrigation_day=8,
    )
    request = IrrigationRequest(False, 0.0, 45.0, "test:infeasible")
    audit = _baseline_constraint_audit(
        policy_day=10,
        request=request,
        controller=controller,
    )
    assert audit.constrained_request_mm == 0.0
    assert audit.adjusted is True
    assert "minimum_interval_active" in audit.reasons


def test_unconstrained_request_is_not_marked_adjusted():
    controller = Controller(remaining_mm=100.0)
    request = request_for_depth(
        desired_mm=30.0,
        constrained_mm=30.0,
        controller=controller,
    )
    audit = _baseline_constraint_audit(
        policy_day=10,
        request=request,
        controller=controller,
    )
    assert audit.desired_mm == 30.0
    assert audit.constrained_request_mm == pytest.approx(30.0)
    assert audit.adjusted is False
    assert audit.reasons == ()
