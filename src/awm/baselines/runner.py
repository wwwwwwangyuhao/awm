"""Episode runner and canonical outcome record for agricultural baselines."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Protocol

from .agricultural import AgriculturalBaseline, IrrigationRequest


class EnvStepLike(Protocol):
    observation: Any
    terminated: bool
    irrigation_audit: Any
    info: Mapping[str, Any]


class CottonWaterEnvLike(Protocol):
    current_day: int
    adapter: Any

    def reset(self) -> tuple[Any, Mapping[str, Any]]: ...

    def step(self, *, irrigate: bool, amount_fraction: float) -> EnvStepLike: ...


@dataclass(frozen=True, slots=True)
class BaselineConstraintAudit:
    """Rule-level desired water after the common hard feasibility envelope.

    This audit sits *before* DSSAT execution canonicalization.  It therefore
    distinguishes three quantities that must not be conflated in experiments:

    1. the agricultural rule's raw desired depth;
    2. the depth represented by the hierarchical request after hard constraints;
    3. the amount finally executed by DSSAT after adapter quantization/projection.
    """

    desired_mm: float
    constrained_request_mm: float
    adjusted: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BaselineEpisodeResult:
    baseline_name: str
    yield_hwam_kg_ha: float
    dssat_ircm_mm: float
    policy_irrigation_mm: float
    irrigation_event_count: int
    irrigation_accounting_passed: bool
    desired_event_count: int
    requested_event_count: int
    baseline_constraint_adjusted_event_count: int
    projected_event_count: int
    step_audits: tuple[Mapping[str, object], ...]
    terminal_info: Mapping[str, Any]

    @property
    def irrigation_water_productivity_kg_m3(self) -> float:
        if self.dssat_ircm_mm <= 0.0:
            return float("nan")
        return self.yield_hwam_kg_ha / (10.0 * self.dssat_ircm_mm)


def _baseline_constraint_audit(
    *,
    policy_day: int,
    request: IrrigationRequest,
    controller: Any,
) -> BaselineConstraintAudit:
    """Recover the hard-constraint-adjusted depth encoded by a baseline request.

    Agricultural baselines call ``request_depth`` before the environment step,
    so their amount fraction may already reflect seasonal quota, event-capacity,
    minimum-event, or interval feasibility.  The DSSAT adapter cannot infer that
    earlier adjustment from the fraction alone.  This helper records it without
    changing the request or mutating the controller.
    """

    desired = float(request.desired_mm)
    if not math.isfinite(desired) or desired < 0.0:
        raise ValueError("baseline desired irrigation must be finite and >= 0")

    spec = controller.spec
    min_event = float(spec.min_event_mm)
    max_event = float(spec.max_event_mm)
    remaining = float(controller.remaining_mm)
    if desired <= 0.0:
        return BaselineConstraintAudit(0.0, 0.0, False, ())

    reasons: list[str] = []
    if desired > max_event + _AUDIT_EPS:
        reasons.append("maximum_event_depth")
    if desired > remaining + _AUDIT_EPS:
        reasons.append("remaining_seasonal_budget")
    if 0.0 < desired < min_event - _AUDIT_EPS:
        reasons.append("minimum_event_depth")

    last_irrigation_day = getattr(controller, "last_irrigation_day", None)
    min_interval_days = int(getattr(spec, "min_interval_days", 0))
    interval_closed = (
        last_irrigation_day is not None
        and int(policy_day) - int(last_irrigation_day) < min_interval_days
    )
    if interval_closed:
        reasons.append("minimum_interval_active")
    if remaining + _AUDIT_EPS < min_event:
        reasons.append("remaining_budget_below_minimum_event")

    if not request.irrigate:
        constrained = 0.0
    else:
        feasible_max = float(controller.feasible_max_mm(day=int(policy_day)))
        if feasible_max + _AUDIT_EPS < min_event:
            constrained = 0.0
        elif feasible_max <= min_event + _AUDIT_EPS:
            constrained = min_event
        else:
            fraction = float(request.amount_fraction)
            if not math.isfinite(fraction):
                raise ValueError("baseline amount fraction must be finite")
            constrained = min_event + fraction * (feasible_max - min_event)
            constrained = min(constrained, feasible_max, remaining)

    adjusted = abs(constrained - desired) > _AUDIT_EPS
    if adjusted and not reasons:
        reasons.append("baseline_request_adjusted")
    return BaselineConstraintAudit(
        desired_mm=desired,
        constrained_request_mm=float(constrained),
        adjusted=adjusted,
        reasons=tuple(reasons),
    )


def run_baseline_episode(
    env: CottonWaterEnvLike,
    policy: AgriculturalBaseline,
) -> BaselineEpisodeResult:
    """Run one complete deterministic/non-learning management episode.

    Every step records raw agricultural desire, hard-constraint-adjusted request,
    and final DSSAT execution separately.  All actions still pass through the
    same WaterBudgetController/DSSAT adapter path used by learned policies.
    """

    observation, _ = env.reset()
    policy.reset()
    audits: list[Mapping[str, object]] = []
    desired_events = 0
    requested_events = 0
    baseline_adjusted_events = 0
    projected_events = 0
    applied_events = 0
    terminal_info: Mapping[str, Any] | None = None

    while True:
        policy_day = int(env.current_day)
        request = policy.act(
            policy_day=policy_day,
            observation=observation,
            controller=env.adapter.controller,
        )
        constraint_audit = _baseline_constraint_audit(
            policy_day=policy_day,
            request=request,
            controller=env.adapter.controller,
        )
        desired_events += int(constraint_audit.desired_mm > 0.0)
        requested_events += int(request.irrigate)
        baseline_adjusted_events += int(constraint_audit.adjusted)

        step = env.step(
            irrigate=bool(request.irrigate),
            amount_fraction=float(request.amount_fraction),
        )
        audit = step.irrigation_audit
        audit_dict = dict(audit.as_info_dict())
        audit_dict.update(
            {
                "baseline_name": policy.name,
                "baseline_desired_mm": constraint_audit.desired_mm,
                "baseline_constrained_request_mm": constraint_audit.constrained_request_mm,
                "baseline_constraint_adjusted": constraint_audit.adjusted,
                "baseline_constraint_reasons": constraint_audit.reasons,
                "baseline_reason": request.reason,
            }
        )
        audits.append(audit_dict)
        projected_events += int(audit.water_budget_projected)
        applied_events += int(audit.event_applied)
        policy.observe_execution(
            applied_mm=float(audit.applied_irrigation_mm)
        )
        observation = step.observation
        if step.terminated:
            terminal_info = dict(step.info)
            break

    if terminal_info is None:
        raise AssertionError("baseline episode ended without terminal info")
    required = (
        "HWAM",
        "IRCM",
        "policy_irrigation_mm",
        "irrigation_accounting_passed",
    )
    missing = [key for key in required if key not in terminal_info]
    if missing:
        raise KeyError(
            "terminal baseline episode missing fields: " + ", ".join(missing)
        )
    if not bool(terminal_info["irrigation_accounting_passed"]):
        raise RuntimeError("baseline episode failed irrigation accounting audit")

    return BaselineEpisodeResult(
        baseline_name=policy.name,
        yield_hwam_kg_ha=float(terminal_info["HWAM"]),
        dssat_ircm_mm=float(terminal_info["IRCM"]),
        policy_irrigation_mm=float(terminal_info["policy_irrigation_mm"]),
        irrigation_event_count=applied_events,
        irrigation_accounting_passed=True,
        desired_event_count=desired_events,
        requested_event_count=requested_events,
        baseline_constraint_adjusted_event_count=baseline_adjusted_events,
        projected_event_count=projected_events,
        step_audits=tuple(audits),
        terminal_info=terminal_info,
    )


_AUDIT_EPS = 1e-9


__all__ = [
    "BaselineConstraintAudit",
    "BaselineEpisodeResult",
    "_baseline_constraint_audit",
    "run_baseline_episode",
]
