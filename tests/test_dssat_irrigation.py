from dataclasses import dataclass

import pytest

from awm.envs.dssat_irrigation import DSSATDecisionCalendar, DSSATIrrigationAdapter


@dataclass
class Spec:
    seasonal_budget_mm: float = 20.0
    min_event_mm: float = 5.0
    max_event_mm: float = 10.0
    min_interval_days: int = 2
    horizon_days: int = 5


@dataclass
class Decision:
    requested_event: bool
    requested_amount_fraction: float
    applied_mm: float
    event_applied: bool
    remaining_before_mm: float
    remaining_after_mm: float
    feasible_max_mm: float
    was_projected: bool
    reasons: tuple[str, ...]


class FakeController:
    def __init__(self):
        self.spec = Spec()
        self.reset()

    def reset(self):
        self.used = 0.0
        self.last_event = None
        self.last_day = None

    @property
    def used_mm(self):
        return self.used

    @property
    def remaining_mm(self):
        return max(0.0, self.spec.seasonal_budget_mm - self.used)

    def feasible_max_mm(self, *, day):
        if self.last_day is not None and day <= self.last_day:
            raise ValueError("decision days must be strictly increasing")
        if self.last_event is not None and day - self.last_event < self.spec.min_interval_days:
            return 0.0
        return min(self.spec.max_event_mm, self.remaining_mm)

    def step(self, *, day, irrigate, amount_fraction):
        if self.last_day is not None and day <= self.last_day:
            raise ValueError("decision days must be strictly increasing")
        before = self.remaining_mm
        feasible = self.feasible_max_mm(day=day)
        projected = False
        reasons = []
        if not irrigate:
            amount = 0.0
            reasons.append("policy_no_irrigation")
        elif feasible < self.spec.min_event_mm:
            amount = 0.0
            projected = True
            reasons.append("minimum_interval_or_budget")
        else:
            clipped = min(1.0, max(0.0, amount_fraction))
            amount = self.spec.min_event_mm + clipped * (
                feasible - self.spec.min_event_mm
            )
            reasons.append("irrigation_applied")
            self.used += amount
            self.last_event = day
        self.last_day = day
        return Decision(
            requested_event=irrigate,
            requested_amount_fraction=amount_fraction,
            applied_mm=amount,
            event_applied=amount > 0,
            remaining_before_mm=before,
            remaining_after_mm=self.remaining_mm,
            feasible_max_mm=feasible,
            was_projected=projected,
            reasons=tuple(reasons),
        )


class FakeBackend:
    def __init__(self):
        self.reset_count = 0
        self.writes = []
        self.reruns = 0
        self.fail_write = False
        self.fail_rerun = False

    def reset_episode(self):
        self.reset_count += 1
        self.writes.clear()
        self.reruns = 0
        self.fail_write = False
        self.fail_rerun = False

    def write_irrigation(self, action_yrdoy, amount_mm):
        if self.fail_write:
            raise OSError("write failed")
        self.writes.append((action_yrdoy, amount_mm))

    def rerun_and_refresh(self):
        if self.fail_rerun:
            raise RuntimeError("DSSAT failed")
        self.reruns += 1


def adapter():
    return DSSATIrrigationAdapter(
        controller=FakeController(),
        backend=FakeBackend(),
        calendar=DSSATDecisionCalendar(
            calendar_year=2023,
            planting_doy=119,
            horizon_days=5,
        ),
        execution_resolution_mm=0.01,
        nonpolicy_irrigation_mm=0.0,
        summary_tolerance_mm=0.02,
    )


def accounting_adapter(*, nonpolicy_irrigation_mm=45.0, tolerance_mm=0.1):
    return DSSATIrrigationAdapter(
        controller=FakeController(),
        backend=FakeBackend(),
        calendar=DSSATDecisionCalendar(
            calendar_year=2023,
            planting_doy=119,
            horizon_days=5,
        ),
        execution_resolution_mm=0.1,
        nonpolicy_irrigation_mm=nonpolicy_irrigation_mm,
        summary_tolerance_mm=tolerance_mm,
    )


def test_calendar_preserves_d_plus_one_semantics():
    cal = DSSATDecisionCalendar(calendar_year=2023, planting_doy=119, horizon_days=5)
    d0 = cal.action_date(0)
    assert d0.action_day == 1
    assert d0.action_yrdoy == "23120"
    assert d0.action_iso_date == "2023-04-30"


def test_no_irrigation_does_not_touch_dssat():
    a = adapter()
    audit = a.apply(policy_day=0, irrigate=False, amount_fraction=0.8)
    assert audit.applied_irrigation_mm == 0.0
    assert audit.dssat_management_written is False
    assert audit.dssat_rerun is False
    assert a.backend.writes == []


def test_positive_event_is_quantized_written_and_rerun():
    a = adapter()
    audit = a.apply(policy_day=0, irrigate=True, amount_fraction=0.333)
    assert audit.applied_irrigation_mm == pytest.approx(6.67)
    assert audit.execution_quantized is True
    assert a.backend.writes == [("23120", pytest.approx(6.67))]
    assert a.backend.reruns == 1


def test_interval_projection_never_writes_or_reruns():
    a = adapter()
    a.apply(policy_day=0, irrigate=True, amount_fraction=0.0)
    audit = a.apply(policy_day=1, irrigate=True, amount_fraction=1.0)
    assert audit.applied_irrigation_mm == 0.0
    assert audit.water_budget_projected is True
    assert len(a.backend.writes) == 1
    assert a.backend.reruns == 1


def test_raw_fraction_is_retained_when_execution_fraction_is_canonicalized():
    a = adapter()
    audit = a.apply(policy_day=0, irrigate=True, amount_fraction=1.25)
    assert audit.requested_amount_fraction == 1.25
    assert audit.canonical_amount_fraction == 1.0
    assert "amount_fraction_clipped_before_dssat" in audit.projection_reasons


def test_dssat_failure_faults_episode_and_blocks_continuation():
    a = adapter()
    a.backend.fail_rerun = True
    with pytest.raises(RuntimeError, match="must be discarded"):
        a.apply(policy_day=0, irrigate=True, amount_fraction=0.0)
    assert a.faulted is True
    with pytest.raises(RuntimeError, match="faulted"):
        a.apply(policy_day=1, irrigate=False, amount_fraction=0.0)


def test_reset_clears_fault_only_after_backend_reset():
    a = adapter()
    a.backend.fail_write = True
    with pytest.raises(RuntimeError):
        a.apply(policy_day=0, irrigate=True, amount_fraction=0.0)
    a.reset_episode()
    assert a.faulted is False
    assert a.controller.used_mm == 0.0
    assert a.backend.reset_count == 1


def test_terminal_irrigation_reconciliation_uses_executed_policy_water():
    a = adapter()
    a.apply(policy_day=0, irrigate=True, amount_fraction=0.0)
    audit = a.reconcile_terminal_irrigation(dssat_ircm_mm=5.01)
    assert audit.policy_irrigation_mm == pytest.approx(5.0)
    assert audit.expected_summary_ircm_mm == pytest.approx(5.0)
    assert audit.summary_difference_mm == pytest.approx(0.01)
    assert audit.passed is True


def test_terminal_irrigation_reconciliation_supports_fixed_nonpolicy_water():
    a = DSSATIrrigationAdapter(
        controller=FakeController(),
        backend=FakeBackend(),
        calendar=DSSATDecisionCalendar(
            calendar_year=2023,
            planting_doy=119,
            horizon_days=5,
        ),
        execution_resolution_mm=0.01,
        nonpolicy_irrigation_mm=20.0,
        summary_tolerance_mm=0.01,
    )
    a.apply(policy_day=0, irrigate=True, amount_fraction=0.0)
    audit = a.reconcile_terminal_irrigation(dssat_ircm_mm=25.0)
    assert audit.expected_ircm_mm == 25.0
    assert audit.expected_summary_ircm_mm == 25.0


def test_summary_ircm_rounding_accepts_539_7_reported_as_540():
    a = accounting_adapter()
    a.controller.used = 494.7
    audit = a.reconcile_terminal_irrigation(dssat_ircm_mm=540.0)
    assert audit.expected_ircm_mm == pytest.approx(539.7)
    assert audit.expected_summary_ircm_mm == pytest.approx(540.0)
    assert audit.difference_mm == pytest.approx(0.3)
    assert audit.summary_difference_mm == pytest.approx(0.0)
    assert audit.passed is True


def test_summary_ircm_rounding_accepts_539_4_reported_as_539():
    a = accounting_adapter()
    a.controller.used = 494.4
    audit = a.reconcile_terminal_irrigation(dssat_ircm_mm=539.0)
    assert audit.expected_ircm_mm == pytest.approx(539.4)
    assert audit.expected_summary_ircm_mm == pytest.approx(539.0)
    assert audit.difference_mm == pytest.approx(-0.4)
    assert audit.summary_difference_mm == pytest.approx(0.0)
    assert audit.passed is True


def test_summary_ircm_rounding_rejects_539_4_reported_as_540():
    a = accounting_adapter()
    a.controller.used = 494.4
    with pytest.raises(RuntimeError, match="summary_difference=1.000000"):
        a.reconcile_terminal_irrigation(dssat_ircm_mm=540.0)


def test_summary_ircm_half_up_tie_matches_real_dssat_74_5_to_75():
    a = accounting_adapter()
    a.controller.used = 29.5
    audit = a.reconcile_terminal_irrigation(dssat_ircm_mm=75.0)
    assert audit.expected_ircm_mm == pytest.approx(74.5)
    assert audit.expected_summary_ircm_mm == pytest.approx(75.0)
    assert audit.summary_difference_mm == pytest.approx(0.0)
    assert audit.passed is True


def test_terminal_irrigation_mismatch_fails_fast():
    a = adapter()
    a.apply(policy_day=0, irrigate=True, amount_fraction=0.0)
    with pytest.raises(RuntimeError, match="seasonal irrigation audit failed"):
        a.reconcile_terminal_irrigation(dssat_ircm_mm=7.0)
