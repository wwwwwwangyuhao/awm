"""Agricultural irrigation baselines for AWM management comparisons.

These policies are intentionally non-learning. Every agronomic threshold or
schedule quantity is an explicit constructor argument so formal experiments
cannot inherit hidden tuning defaults.

All policies return a hierarchical irrigation request and rely on the same
WaterBudgetController/DSSATIrrigationAdapter used by RL policies for seasonal
quota, event-depth, interval and execution-resolution enforcement.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Protocol, Sequence


class ObservationLike(Protocol):
    feature_names: tuple[str, ...]
    def flat(self) -> tuple[float, ...]: ...


class ControllerSpecLike(Protocol):
    min_event_mm: float


class ControllerLike(Protocol):
    spec: ControllerSpecLike
    def feasible_max_mm(self, *, day: int) -> float: ...


@dataclass(frozen=True, slots=True)
class IrrigationRequest:
    irrigate: bool
    amount_fraction: float
    desired_mm: float
    reason: str


class AgriculturalBaseline:
    """Base protocol with optional executed-action feedback."""

    name = "agricultural_baseline"

    def reset(self) -> None:
        pass

    def act(
        self,
        *,
        policy_day: int,
        observation: ObservationLike,
        controller: ControllerLike,
    ) -> IrrigationRequest:
        raise NotImplementedError

    def observe_execution(self, *, applied_mm: float) -> None:
        del applied_mm


def observation_dict(observation: ObservationLike) -> dict[str, float]:
    names = tuple(observation.feature_names)
    values = tuple(float(x) for x in observation.flat())
    if len(names) != len(values):
        raise ValueError("observation feature_names/value length mismatch")
    if len(set(names)) != len(names):
        raise ValueError("observation feature_names contain duplicates")
    result = dict(zip(names, values, strict=True))
    if not all(math.isfinite(v) for v in result.values()):
        raise FloatingPointError("baseline observation contains NaN/Inf")
    return result


def request_depth(
    *,
    desired_mm: float,
    controller: ControllerLike,
    policy_day: int,
    reason: str,
) -> IrrigationRequest:
    """Encode an agronomic desired depth into the common hierarchical action.

    This helper never bypasses hard constraints. If the controller says a
    positive event is currently infeasible, the request is an exact no-op.
    Otherwise desired depth is clipped to the feasible [I_min, I_max_today]
    interval before conversion to the actor-compatible amount fraction.
    """
    desired = float(desired_mm)
    if not math.isfinite(desired) or desired < 0.0:
        raise ValueError("desired_mm must be finite and >= 0")
    if desired == 0.0:
        return IrrigationRequest(False, 0.0, 0.0, reason)

    feasible_max = float(controller.feasible_max_mm(day=int(policy_day)))
    min_event = float(controller.spec.min_event_mm)
    if feasible_max + 1e-12 < min_event:
        return IrrigationRequest(False, 0.0, desired, f"{reason}:infeasible")

    executable = min(feasible_max, max(min_event, desired))
    if feasible_max <= min_event + 1e-12:
        fraction = 0.0
    else:
        fraction = (executable - min_event) / (feasible_max - min_event)
    return IrrigationRequest(True, float(fraction), desired, reason)


class ConventionalScheduleBaseline(AgriculturalBaseline):
    """Pre-registered local schedule indexed by biological action DAP.

    ``schedule_mm`` maps action DAP 1..T to desired irrigation depth in mm.
    Policy day d controls biological day d+1, so a schedule entry at DAP 1 is
    requested from policy_day=0. No schedule is supplied by default.
    """

    name = "local_conventional"

    def __init__(self, schedule_mm: Mapping[int, float]) -> None:
        if not schedule_mm:
            raise ValueError("conventional schedule must not be empty")
        cleaned: dict[int, float] = {}
        for action_day, amount in schedule_mm.items():
            day = int(action_day)
            value = float(amount)
            if day <= 0:
                raise ValueError("conventional action DAP must be >= 1")
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError("conventional event depths must be > 0")
            if day in cleaned:
                raise ValueError(f"duplicate conventional action DAP {day}")
            cleaned[day] = value
        self.schedule_mm = dict(sorted(cleaned.items()))

    def act(self, *, policy_day, observation, controller) -> IrrigationRequest:
        del observation
        action_day = int(policy_day) + 1
        desired = self.schedule_mm.get(action_day, 0.0)
        return request_depth(
            desired_mm=desired,
            controller=controller,
            policy_day=policy_day,
            reason=f"conventional_dap_{action_day}",
        )


class PotentialETWaterBalanceBaseline(AgriculturalBaseline):
    """Causal ET-based replenishment using DSSAT daily potential ET.

    DSSAT ``EOAA`` is potential evapotranspiration (mm d-1). The policy keeps a
    simple observed water-demand ledger:

        deficit <- max(0, deficit + EOAA - f_eff * PRED)

    and triggers irrigation when the accumulated deficit reaches an explicitly
    pre-registered threshold. Executed irrigation reduces the ledger by
    ``irrigation_efficiency * applied_mm``.

    This is intentionally called a DSSAT-potential-ET baseline, not FAO-56,
    because the canonical weather files do not contain all variables required
    for a full FAO-56 Penman-Monteith ET0 calculation.
    """

    name = "potential_et_water_balance"

    def __init__(
        self,
        *,
        trigger_deficit_mm: float,
        irrigation_efficiency: float,
        effective_rain_fraction: float,
        refill_fraction: float,
    ) -> None:
        self.trigger_deficit_mm = _positive(
            "trigger_deficit_mm", trigger_deficit_mm
        )
        self.irrigation_efficiency = _unit_interval_open_closed(
            "irrigation_efficiency", irrigation_efficiency
        )
        self.effective_rain_fraction = _unit_interval_closed(
            "effective_rain_fraction", effective_rain_fraction
        )
        self.refill_fraction = _unit_interval_open_closed(
            "refill_fraction", refill_fraction
        )
        self.reset()

    def reset(self) -> None:
        self.deficit_mm = 0.0

    def act(self, *, policy_day, observation, controller) -> IrrigationRequest:
        state = observation_dict(observation)
        for key in ("EOAA", "PRED"):
            if key not in state:
                raise KeyError(f"ET baseline requires observation field {key}")
        pet = max(0.0, state["EOAA"])
        rainfall = max(0.0, state["PRED"])
        self.deficit_mm = max(
            0.0,
            self.deficit_mm
            + pet
            - self.effective_rain_fraction * rainfall,
        )
        if self.deficit_mm + 1e-12 < self.trigger_deficit_mm:
            return IrrigationRequest(
                False, 0.0, 0.0, "potential_et_deficit_below_trigger"
            )
        desired = (
            self.refill_fraction
            * self.deficit_mm
            / self.irrigation_efficiency
        )
        return request_depth(
            desired_mm=desired,
            controller=controller,
            policy_day=policy_day,
            reason="potential_et_replenishment",
        )

    def observe_execution(self, *, applied_mm: float) -> None:
        applied = float(applied_mm)
        if not math.isfinite(applied) or applied < 0.0:
            raise ValueError("applied_mm must be finite and >= 0")
        self.deficit_mm = max(
            0.0,
            self.deficit_mm - self.irrigation_efficiency * applied,
        )


class RootZoneREWThresholdBaseline(AgriculturalBaseline):
    """Root-weighted relative-extractable-water threshold irrigation.

    REW1..REW10 are weighted by DSSAT RL1D..RL10D root-length information.
    Before roots are numerically available, an explicitly supplied set of
    1-based fallback REW layers is averaged. The trigger and event depth are
    experiment parameters and deliberately have no defaults.
    """

    name = "root_zone_rew_threshold"

    def __init__(
        self,
        *,
        trigger_rew: float,
        event_depth_mm: float,
        fallback_rew_layers: Sequence[int],
    ) -> None:
        trigger = float(trigger_rew)
        if not math.isfinite(trigger):
            raise ValueError("trigger_rew must be finite")
        self.trigger_rew = trigger
        self.event_depth_mm = _positive("event_depth_mm", event_depth_mm)
        layers = tuple(int(x) for x in fallback_rew_layers)
        if not layers or any(x < 1 or x > 10 for x in layers):
            raise ValueError("fallback_rew_layers must contain values in 1..10")
        if len(set(layers)) != len(layers):
            raise ValueError("fallback_rew_layers must not contain duplicates")
        self.fallback_rew_layers = layers
        self.last_root_zone_rew: float | None = None

    def reset(self) -> None:
        self.last_root_zone_rew = None

    def _root_zone_rew(self, state: Mapping[str, float]) -> float:
        rew = []
        roots = []
        for idx in range(1, 11):
            rew_key = f"REW{idx}"
            root_key = f"RL{idx}D"
            if rew_key not in state or root_key not in state:
                raise KeyError(
                    "REW baseline requires REW1..REW10 and RL1D..RL10D"
                )
            rew.append(float(state[rew_key]))
            roots.append(max(0.0, float(state[root_key])))
        root_total = sum(roots)
        if root_total > 1e-12:
            return sum(
                r * w for r, w in zip(rew, roots, strict=True)
            ) / root_total
        return sum(rew[i - 1] for i in self.fallback_rew_layers) / len(
            self.fallback_rew_layers
        )

    def act(self, *, policy_day, observation, controller) -> IrrigationRequest:
        state = observation_dict(observation)
        root_rew = self._root_zone_rew(state)
        self.last_root_zone_rew = root_rew
        if root_rew > self.trigger_rew:
            return IrrigationRequest(
                False, 0.0, 0.0, "root_zone_rew_above_trigger"
            )
        return request_depth(
            desired_mm=self.event_depth_mm,
            controller=controller,
            policy_day=policy_day,
            reason="root_zone_rew_threshold_triggered",
        )


def _positive(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return result


def _unit_interval_closed(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must lie in [0,1]")
    return result


def _unit_interval_open_closed(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result <= 1.0:
        raise ValueError(f"{name} must lie in (0,1]")
    return result


__all__ = [
    "AgriculturalBaseline",
    "ConventionalScheduleBaseline",
    "IrrigationRequest",
    "PotentialETWaterBalanceBaseline",
    "RootZoneREWThresholdBaseline",
    "observation_dict",
    "request_depth",
]
