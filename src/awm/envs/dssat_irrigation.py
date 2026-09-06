"""DSSAT irrigation adapter with strict action and accounting audits.

The adapter preserves the canonical DSSAT-RL timing contract used by this
project: a policy observation on decision day ``d`` selects management that is
applied on biological day ``d + 1``. Positive irrigation events are written
to the mutable DSSAT management state and trigger a full DSSAT rerun; exact
no-irrigation actions do neither.

This module deliberately owns no reward logic and exposes no future DSSAT
outputs. Seasonal Summary.OUT reconciliation is terminal-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal, ROUND_FLOOR, ROUND_HALF_UP
import math
from typing import Protocol, runtime_checkable


class IrrigationSpecLike(Protocol):
    min_event_mm: float


class IrrigationDecisionLike(Protocol):
    requested_event: bool
    requested_amount_fraction: float
    applied_mm: float
    event_applied: bool
    remaining_before_mm: float
    remaining_after_mm: float
    feasible_max_mm: float
    was_projected: bool
    reasons: tuple[str, ...]


class WaterBudgetControllerLike(Protocol):
    spec: IrrigationSpecLike

    @property
    def used_mm(self) -> float: ...

    @property
    def remaining_mm(self) -> float: ...

    def reset(self) -> None: ...

    def feasible_max_mm(self, *, day: int) -> float: ...

    def step(
        self,
        *,
        day: int,
        irrigate: bool,
        amount_fraction: float,
    ) -> IrrigationDecisionLike: ...


@runtime_checkable
class DSSATIrrigationBackend(Protocol):
    """Minimal mutable-DSSAT operations required by the adapter.

    ``reset_episode`` must restore a clean worker state and execute/refresh the
    baseline season so that a failed management write never leaks into the next
    episode. Any write/rerun failure is therefore terminal for the current
    episode; callers must discard that trajectory and reset the worker.
    """

    def reset_episode(self) -> None: ...

    def write_irrigation(self, action_yrdoy: str, amount_mm: float) -> None: ...

    def rerun_and_refresh(self) -> None: ...


@dataclass(frozen=True, slots=True)
class DSSATActionDate:
    policy_day: int
    action_day: int
    action_yrdoy: str
    action_iso_date: str


@dataclass(frozen=True, slots=True)
class DSSATDecisionCalendar:
    """Map zero-indexed policy days to DSSAT YYDDD management dates."""

    calendar_year: int
    planting_doy: int
    horizon_days: int

    def __post_init__(self) -> None:
        if not isinstance(self.calendar_year, int) or self.calendar_year < 1:
            raise ValueError("calendar_year must be a positive integer")
        if not isinstance(self.planting_doy, int) or self.planting_doy <= 0:
            raise ValueError("planting_doy must be a positive integer")
        if not isinstance(self.horizon_days, int) or self.horizon_days <= 0:
            raise ValueError("horizon_days must be a positive integer")
        try:
            planting = date(self.calendar_year, 1, 1) + timedelta(
                days=self.planting_doy - 1
            )
        except OverflowError as exc:
            raise ValueError("invalid planting date") from exc
        if planting.year != self.calendar_year:
            raise ValueError("planting_doy is outside calendar_year")

    @classmethod
    def from_yrdoy(cls, plant_yrdoy: str, *, horizon_days: int) -> "DSSATDecisionCalendar":
        text = str(plant_yrdoy).strip()
        if len(text) != 5 or not text.isdigit():
            raise ValueError("plant_yrdoy must be DSSAT YYDDD")
        yy = int(text[:2])
        calendar_year = 2000 + yy if yy <= 69 else 1900 + yy
        return cls(
            calendar_year=calendar_year,
            planting_doy=int(text[2:]),
            horizon_days=horizon_days,
        )

    def action_date(self, policy_day: int) -> DSSATActionDate:
        if not isinstance(policy_day, int) or policy_day < 0:
            raise ValueError("policy_day must be an integer >= 0")
        if policy_day >= self.horizon_days:
            raise ValueError("policy_day must be < horizon_days")

        planting = date(self.calendar_year, 1, 1) + timedelta(
            days=self.planting_doy - 1
        )
        management_date = planting + timedelta(days=policy_day + 1)
        doy = management_date.timetuple().tm_yday
        return DSSATActionDate(
            policy_day=policy_day,
            action_day=policy_day + 1,
            action_yrdoy=f"{management_date.year % 100:02d}{doy:03d}",
            action_iso_date=management_date.isoformat(),
        )


@dataclass(frozen=True, slots=True)
class IrrigationStepAudit:
    """Canonical rollout record for one irrigation decision."""

    policy_day: int
    action_day: int
    action_yrdoy: str
    action_iso_date: str
    requested_event: bool
    requested_amount_fraction: float
    canonical_amount_fraction: float
    applied_irrigation_mm: float
    remaining_before_mm: float
    remaining_after_mm: float
    event_applied: bool
    water_budget_projected: bool
    execution_quantized: bool
    projection_reasons: tuple[str, ...]
    dssat_management_written: bool
    dssat_rerun: bool

    def as_info_dict(self) -> dict[str, object]:
        return {
            "policy_day": self.policy_day,
            "decision_action_day": self.action_day,
            "decision_action_yrdoy": self.action_yrdoy,
            "decision_action_date": self.action_iso_date,
            "requested_irrigation_event": self.requested_event,
            "requested_irrigation_amount_fraction": self.requested_amount_fraction,
            "canonical_irrigation_amount_fraction": self.canonical_amount_fraction,
            "applied_irrigation_mm": self.applied_irrigation_mm,
            "remaining_irrigation_before_mm": self.remaining_before_mm,
            "remaining_irrigation_after_mm": self.remaining_after_mm,
            "irrigation_event_applied": self.event_applied,
            "irrigation_action_projected": self.water_budget_projected,
            "irrigation_execution_quantized": self.execution_quantized,
            "irrigation_projection_reasons": self.projection_reasons,
            "dssat_management_written": self.dssat_management_written,
            "dssat_rerun": self.dssat_rerun,
        }


@dataclass(frozen=True, slots=True)
class TerminalIrrigationAudit:
    """Terminal reconciliation between controller accounting and DSSAT IRCM."""

    dssat_ircm_mm: float
    policy_irrigation_mm: float
    nonpolicy_irrigation_mm: float
    expected_ircm_mm: float
    difference_mm: float
    tolerance_mm: float
    passed: bool


class DSSATIrrigationAdapter:
    """Bridge hierarchical policy actions to executable DSSAT irrigation.

    The controller remains the authority on seasonal budget and interval
    feasibility. This adapter adds DSSAT date semantics, execution-resolution
    canonicalization, worker I/O, failure containment, and terminal water
    accounting.
    """

    def __init__(
        self,
        *,
        controller: WaterBudgetControllerLike,
        backend: DSSATIrrigationBackend,
        calendar: DSSATDecisionCalendar,
        execution_resolution_mm: float,
        nonpolicy_irrigation_mm: float,
        summary_tolerance_mm: float,
    ) -> None:
        _require_positive("execution_resolution_mm", execution_resolution_mm)
        _require_nonnegative("nonpolicy_irrigation_mm", nonpolicy_irrigation_mm)
        _require_nonnegative("summary_tolerance_mm", summary_tolerance_mm)
        if calendar.horizon_days != getattr(
            controller.spec, "horizon_days", calendar.horizon_days
        ):
            raise ValueError("calendar and water-budget horizons must match")

        self.controller = controller
        self.backend = backend
        self.calendar = calendar
        self.execution_resolution_mm = float(execution_resolution_mm)
        self.nonpolicy_irrigation_mm = float(nonpolicy_irrigation_mm)
        self.summary_tolerance_mm = float(summary_tolerance_mm)
        self._faulted = False

    @property
    def faulted(self) -> bool:
        return self._faulted

    def reset_episode(self) -> None:
        """Reset DSSAT first, then reset water accounting after successful reset."""

        self.backend.reset_episode()
        self.controller.reset()
        self._faulted = False

    def apply(
        self,
        *,
        policy_day: int,
        irrigate: bool,
        amount_fraction: float,
    ) -> IrrigationStepAudit:
        """Apply one policy request and return the exact rollout audit record."""

        if self._faulted:
            raise RuntimeError(
                "DSSAT irrigation adapter is faulted; discard the episode and call reset_episode()"
            )
        _require_finite("amount_fraction", amount_fraction)
        action_date = self.calendar.action_date(policy_day)
        requested_fraction = float(amount_fraction)

        canonical_fraction, quantized, adapter_reasons = self._canonical_fraction(
            policy_day=policy_day,
            irrigate=bool(irrigate),
            requested_fraction=requested_fraction,
        )

        decision = self.controller.step(
            day=policy_day,
            irrigate=bool(irrigate),
            amount_fraction=canonical_fraction,
        )

        reasons = list(decision.reasons)
        for reason in adapter_reasons:
            if reason not in reasons:
                reasons.append(reason)

        written = False
        rerun = False
        if decision.applied_mm > 0.0:
            try:
                self.backend.write_irrigation(
                    action_date.action_yrdoy,
                    float(decision.applied_mm),
                )
                written = True
                self.backend.rerun_and_refresh()
                rerun = True
            except Exception as exc:
                self._faulted = True
                raise RuntimeError(
                    "DSSAT irrigation execution failed; current episode must be discarded"
                ) from exc

        projected = bool(decision.was_projected or quantized or adapter_reasons)
        return IrrigationStepAudit(
            policy_day=policy_day,
            action_day=action_date.action_day,
            action_yrdoy=action_date.action_yrdoy,
            action_iso_date=action_date.action_iso_date,
            requested_event=bool(irrigate),
            requested_amount_fraction=requested_fraction,
            canonical_amount_fraction=canonical_fraction,
            applied_irrigation_mm=float(decision.applied_mm),
            remaining_before_mm=float(decision.remaining_before_mm),
            remaining_after_mm=float(decision.remaining_after_mm),
            event_applied=bool(decision.event_applied),
            water_budget_projected=projected,
            execution_quantized=quantized,
            projection_reasons=tuple(reasons),
            dssat_management_written=written,
            dssat_rerun=rerun,
        )

    def reconcile_terminal_irrigation(
        self, *, dssat_ircm_mm: float
    ) -> TerminalIrrigationAudit:
        """Reconcile seasonal irrigation after termination only.

        ``dssat_ircm_mm`` must come from terminal Summary.OUT. Calling code is
        responsible for ensuring that no seasonal summary field is exposed to
        the policy before termination.
        """

        if self._faulted:
            raise RuntimeError("cannot reconcile a faulted/discarded episode")
        _require_nonnegative("dssat_ircm_mm", dssat_ircm_mm)
        policy_mm = float(self.controller.used_mm)
        expected = self.nonpolicy_irrigation_mm + policy_mm
        difference = float(dssat_ircm_mm) - expected
        passed = abs(difference) <= self.summary_tolerance_mm
        audit = TerminalIrrigationAudit(
            dssat_ircm_mm=float(dssat_ircm_mm),
            policy_irrigation_mm=policy_mm,
            nonpolicy_irrigation_mm=self.nonpolicy_irrigation_mm,
            expected_ircm_mm=expected,
            difference_mm=difference,
            tolerance_mm=self.summary_tolerance_mm,
            passed=passed,
        )
        if not passed:
            raise RuntimeError(
                "DSSAT seasonal irrigation audit failed: "
                f"IRCM={audit.dssat_ircm_mm:.6f}, "
                f"expected={audit.expected_ircm_mm:.6f}, "
                f"difference={audit.difference_mm:.6f}, "
                f"tolerance={audit.tolerance_mm:.6f}"
            )
        return audit

    def _canonical_fraction(
        self,
        *,
        policy_day: int,
        irrigate: bool,
        requested_fraction: float,
    ) -> tuple[float, bool, tuple[str, ...]]:
        if not irrigate:
            return requested_fraction, False, ()

        feasible_max = float(self.controller.feasible_max_mm(day=policy_day))
        min_event = float(self.controller.spec.min_event_mm)
        if feasible_max + 1e-12 < min_event:
            return requested_fraction, False, ()

        clipped = min(1.0, max(0.0, requested_fraction))
        reasons: list[str] = []
        if clipped != requested_fraction:
            reasons.append("amount_fraction_clipped_before_dssat")

        if feasible_max <= min_event + 1e-12:
            canonical_fraction = 0.0
        else:
            target = min_event + clipped * (feasible_max - min_event)
            quantized_amount = _quantize_amount(
                target,
                resolution=self.execution_resolution_mm,
                lower=min_event,
                upper=feasible_max,
            )
            canonical_fraction = (quantized_amount - min_event) / (
                feasible_max - min_event
            )

        reconstructed = min_event + canonical_fraction * max(
            0.0, feasible_max - min_event
        )
        requested_amount = min_event + clipped * max(
            0.0, feasible_max - min_event
        )
        quantized = abs(reconstructed - requested_amount) > 1e-12
        if quantized:
            reasons.append("execution_resolution_quantized")
        return float(canonical_fraction), quantized, tuple(reasons)


def _quantize_amount(
    value: float, *, resolution: float, lower: float, upper: float
) -> float:
    """Round to execution resolution without crossing a real feasible bound.

    The controller tracks water in binary floats, while execution resolution is
    a decimal field quantity. Repeated 0.1-mm-grid events can therefore leave a
    feasible upper bound such as 25.29999999999998 when the intended remaining
    quota is exactly 25.3 mm. Treat a candidate that exceeds the bound only by
    microscopic float drift as the bound itself; otherwise floor to the largest
    executable grid point below the true bound.
    """

    q = Decimal(str(resolution))
    value_d = Decimal(str(value))
    lower_d = Decimal(str(lower))
    upper_d = Decimal(str(upper))
    drift_tol = max(Decimal("1e-9"), q * Decimal("1e-9"))

    units = (value_d / q).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    candidate = units * q
    if candidate > upper_d:
        if candidate - upper_d <= drift_tol:
            candidate = upper_d
        else:
            candidate = (
                (upper_d / q).quantize(Decimal("1"), rounding=ROUND_FLOOR) * q
            )
    if candidate < lower_d:
        candidate = lower_d
    if candidate > upper_d:
        candidate = upper_d
    return float(candidate)


def _require_finite(name: str, value: float) -> None:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")


def _require_positive(name: str, value: float) -> None:
    _require_finite(name, value)
    if float(value) <= 0.0:
        raise ValueError(f"{name} must be > 0")


def _require_nonnegative(name: str, value: float) -> None:
    _require_finite(name, value)
    if float(value) < 0.0:
        raise ValueError(f"{name} must be >= 0")


__all__ = [
    "DSSATActionDate",
    "DSSATDecisionCalendar",
    "DSSATIrrigationAdapter",
    "DSSATIrrigationBackend",
    "IrrigationStepAudit",
    "TerminalIrrigationAudit",
]
