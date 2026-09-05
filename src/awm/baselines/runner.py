"""Episode runner and canonical outcome record for agricultural baselines."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from .agricultural import AgriculturalBaseline


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
class BaselineEpisodeResult:
    baseline_name: str
    yield_hwam_kg_ha: float
    dssat_ircm_mm: float
    policy_irrigation_mm: float
    irrigation_event_count: int
    irrigation_accounting_passed: bool
    requested_event_count: int
    projected_event_count: int
    step_audits: tuple[Mapping[str, object], ...]
    terminal_info: Mapping[str, Any]

    @property
    def irrigation_water_productivity_kg_m3(self) -> float:
        if self.dssat_ircm_mm <= 0.0:
            return float("nan")
        return self.yield_hwam_kg_ha / (10.0 * self.dssat_ircm_mm)


def run_baseline_episode(
    env: CottonWaterEnvLike,
    policy: AgriculturalBaseline,
) -> BaselineEpisodeResult:
    """Run one complete deterministic/non-learning management episode.

    The function records executed irrigation rather than desired baseline depth.
    All actions pass through the same hard constraint and DSSAT execution path
    used by RL policies.
    """
    observation, _ = env.reset()
    policy.reset()
    audits: list[Mapping[str, object]] = []
    requested_events = 0
    projected_events = 0
    applied_events = 0
    terminal_info: Mapping[str, Any] | None = None

    while True:
        request = policy.act(
            policy_day=int(env.current_day),
            observation=observation,
            controller=env.adapter.controller,
        )
        requested_events += int(request.irrigate)
        step = env.step(
            irrigate=bool(request.irrigate),
            amount_fraction=float(request.amount_fraction),
        )
        audit = step.irrigation_audit
        audit_dict = dict(audit.as_info_dict())
        audit_dict.update(
            {
                "baseline_name": policy.name,
                "baseline_desired_mm": float(request.desired_mm),
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
        requested_event_count=requested_events,
        projected_event_count=projected_events,
        step_audits=tuple(audits),
        terminal_info=terminal_info,
    )


__all__ = ["BaselineEpisodeResult", "run_baseline_episode"]
