"""Hard seasonal-water constraints for irrigation decisions.

This module is deliberately independent of DSSAT and any RL algorithm.  Its
single responsibility is to convert a policy's hierarchical irrigation
request (event/no-event + normalized amount) into an executable irrigation
depth while enforcing the agricultural water-allocation contract.

All agronomic/system quantities are explicit constructor arguments.  There
are intentionally no hidden defaults for seasonal water allocation, minimum
irrigation event depth, maximum event capacity, or minimum event interval.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class IrrigationSystemSpec:
    """Frozen operational constraints for one experiment.

    Parameters are in millimetres of applied irrigation water and days.
    Numerical values must be justified externally by the field system,
    agronomic recommendation, or preregistered treatment design.
    """

    seasonal_budget_mm: float
    min_event_mm: float
    max_event_mm: float
    min_interval_days: int
    horizon_days: int

    def __post_init__(self) -> None:
        _require_finite_positive("seasonal_budget_mm", self.seasonal_budget_mm)
        _require_finite_positive("min_event_mm", self.min_event_mm)
        _require_finite_positive("max_event_mm", self.max_event_mm)
        if self.max_event_mm < self.min_event_mm:
            raise ValueError("max_event_mm must be >= min_event_mm")
        if not isinstance(self.min_interval_days, int) or self.min_interval_days < 0:
            raise ValueError("min_interval_days must be an integer >= 0")
        if not isinstance(self.horizon_days, int) or self.horizon_days <= 0:
            raise ValueError("horizon_days must be a positive integer")


@dataclass(frozen=True, slots=True)
class BudgetObservation:
    """Policy-facing water-allocation state, excluding DSSAT crop state."""

    remaining_budget_fraction: float
    cumulative_irrigation_fraction: float
    remaining_horizon_fraction: float
    days_since_last_irrigation: int
    yield_target_fraction: float

    def as_vector(self) -> tuple[float, float, float, float, float]:
        """Return a simple numeric representation for an RL observation.

        ``days_since_last_irrigation`` is normalized by the caller's horizon
        only when desired; the raw integer is retained here to avoid silently
        choosing a representation that belongs in model preprocessing.
        """

        return (
            self.remaining_budget_fraction,
            self.cumulative_irrigation_fraction,
            self.remaining_horizon_fraction,
            float(self.days_since_last_irrigation),
            self.yield_target_fraction,
        )


@dataclass(frozen=True, slots=True)
class IrrigationDecision:
    """Auditable result of projecting one policy request into feasibility."""

    day: int
    requested_event: bool
    requested_amount_fraction: float
    applied_mm: float
    event_applied: bool
    remaining_before_mm: float
    remaining_after_mm: float
    feasible_max_mm: float
    was_projected: bool
    reasons: tuple[str, ...]


class WaterBudgetController:
    """Stateful hard-constraint controller for one growing season.

    The controller is intentionally small enough to unit test independently.
    DSSAT should receive only ``decision.applied_mm``.  Reward shaping must not
    be used as a substitute for any constraint enforced here.
    """

    def __init__(self, spec: IrrigationSystemSpec) -> None:
        self.spec = spec
        self.reset()

    def reset(self) -> None:
        self._used_mm = 0.0
        self._last_irrigation_day: int | None = None
        self._last_decision_day: int | None = None

    @property
    def used_mm(self) -> float:
        return self._used_mm

    @property
    def remaining_mm(self) -> float:
        return max(0.0, self.spec.seasonal_budget_mm - self._used_mm)

    @property
    def last_irrigation_day(self) -> int | None:
        return self._last_irrigation_day

    def observe(self, *, day: int, yield_target_fraction: float) -> BudgetObservation:
        """Build water-allocation features available at decision time.

        Days are zero-indexed: ``day=0`` is the first management decision.
        Before the first irrigation event, days-since-irrigation is ``day + 1``.
        This convention is deterministic and distinguishes the start of season
        without inventing a fictitious previous irrigation event.
        """

        self._validate_day(day)
        _validate_yield_target(yield_target_fraction)
        if day >= self.spec.horizon_days:
            raise ValueError("day must be < horizon_days")

        if self._last_irrigation_day is None:
            days_since = day + 1
        else:
            days_since = day - self._last_irrigation_day

        budget = self.spec.seasonal_budget_mm
        return BudgetObservation(
            remaining_budget_fraction=self.remaining_mm / budget,
            cumulative_irrigation_fraction=self._used_mm / budget,
            remaining_horizon_fraction=(self.spec.horizon_days - day) / self.spec.horizon_days,
            days_since_last_irrigation=days_since,
            yield_target_fraction=yield_target_fraction,
        )

    def feasible_max_mm(self, *, day: int) -> float:
        """Maximum positive event depth allowed on ``day`` before actor mapping."""

        self._validate_decision_day(day, mutate=False)
        if not self._interval_is_open(day):
            return 0.0
        return min(self.spec.max_event_mm, self.remaining_mm)

    def step(
        self,
        *,
        day: int,
        irrigate: bool,
        amount_fraction: float,
    ) -> IrrigationDecision:
        """Project and commit one hierarchical irrigation request.

        ``amount_fraction`` is interpreted only when ``irrigate`` is true. It
        is clipped to [0, 1] and mapped to [min_event_mm, feasible_max_mm].
        If the remaining budget cannot support a minimum effective event, or
        if the minimum interval is closed, the only feasible action is exactly
        zero irrigation.
        """

        self._validate_decision_day(day, mutate=True)
        _require_finite("amount_fraction", amount_fraction)

        remaining_before = self.remaining_mm
        reasons: list[str] = []
        feasible_max = 0.0
        projected = False

        if not irrigate:
            applied = 0.0
            reasons.append("policy_no_irrigation")
        elif not self._interval_is_open(day):
            applied = 0.0
            projected = True
            reasons.append("minimum_interval_active")
        else:
            feasible_max = min(self.spec.max_event_mm, remaining_before)
            if feasible_max + _EPS < self.spec.min_event_mm:
                applied = 0.0
                projected = True
                reasons.append("remaining_budget_below_minimum_event")
            else:
                clipped = min(1.0, max(0.0, amount_fraction))
                if clipped != amount_fraction:
                    projected = True
                    reasons.append("amount_fraction_clipped")
                applied = self.spec.min_event_mm + clipped * (
                    feasible_max - self.spec.min_event_mm
                )
                # Numerical guard: never allow floating-point drift above budget.
                applied = min(applied, feasible_max, remaining_before)
                reasons.append("irrigation_applied")

        if applied > 0.0:
            self._used_mm += applied
            if self._used_mm > self.spec.seasonal_budget_mm + _EPS:
                raise AssertionError("seasonal irrigation budget was exceeded")
            # Snap microscopic accumulated error to the exact budget boundary.
            if abs(self._used_mm - self.spec.seasonal_budget_mm) <= _EPS:
                self._used_mm = self.spec.seasonal_budget_mm
            self._last_irrigation_day = day

        remaining_after = self.remaining_mm
        return IrrigationDecision(
            day=day,
            requested_event=bool(irrigate),
            requested_amount_fraction=float(amount_fraction),
            applied_mm=applied,
            event_applied=applied > 0.0,
            remaining_before_mm=remaining_before,
            remaining_after_mm=remaining_after,
            feasible_max_mm=feasible_max,
            was_projected=projected,
            reasons=tuple(reasons),
        )

    def _interval_is_open(self, day: int) -> bool:
        if self._last_irrigation_day is None:
            return True
        return (day - self._last_irrigation_day) >= self.spec.min_interval_days

    def _validate_decision_day(self, day: int, *, mutate: bool) -> None:
        self._validate_day(day)
        if day >= self.spec.horizon_days:
            raise ValueError("day must be < horizon_days")
        if self._last_decision_day is not None and day <= self._last_decision_day:
            raise ValueError("decision days must be strictly increasing")
        if mutate:
            self._last_decision_day = day

    @staticmethod
    def _validate_day(day: int) -> None:
        if not isinstance(day, int) or day < 0:
            raise ValueError("day must be an integer >= 0")


_EPS = 1e-9


def _require_finite(name: str, value: float) -> None:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")


def _require_finite_positive(name: str, value: float) -> None:
    _require_finite(name, value)
    if float(value) <= 0.0:
        raise ValueError(f"{name} must be > 0")


def _validate_yield_target(value: float) -> None:
    _require_finite("yield_target_fraction", value)
    if not 0.0 < float(value) <= 1.0:
        raise ValueError("yield_target_fraction must lie in (0, 1]")
