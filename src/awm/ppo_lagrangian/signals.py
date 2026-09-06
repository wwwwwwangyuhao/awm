"""Training signals for the expected-retention PPO-Lagrangian baseline.

The environment remains reward/cost free.  This module defines an algorithm-only
water reward and a signed terminal constraint cost

    c_eta = eta - Y/Y_ref

so the formal training constraint is E[c_eta | eta] <= 0.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from awm.ppo.reward import POLICY_BUDGET_MM
from awm.risk import REGISTERED_ETA_LEVELS, TRAIN_YEARS, yield_retention

_EPS = 1e-12


@dataclass(frozen=True, slots=True)
class LagrangianSignalBreakdown:
    weather_year: int
    eta: float
    policy_irrigation_mm: float
    yield_kg_ha: float
    reference_yield_kg_ha: float
    yield_retention: float
    water_return: float
    signed_constraint_cost: float
    per_year_expected_constraint_satisfied: bool


class LagrangianEpisodeSignals:
    """Track water reward and signed terminal cost without future leakage."""

    def __init__(
        self,
        *,
        weather_year: int,
        eta: float,
        reference_yield_by_year: Mapping[int, float],
        policy_budget_mm: float = POLICY_BUDGET_MM,
    ) -> None:
        self.weather_year = int(weather_year)
        if self.weather_year not in TRAIN_YEARS:
            raise ValueError("PPO-Lagrangian training signals are restricted to ERA5 2000-2017")
        self.eta = float(eta)
        if not any(abs(self.eta - float(x)) <= _EPS for x in REGISTERED_ETA_LEVELS):
            raise ValueError(f"eta must be one of registered levels {REGISTERED_ETA_LEVELS}")
        if self.weather_year not in reference_yield_by_year:
            raise KeyError(f"missing Y_ref for training year {self.weather_year}")
        self.reference_yield_kg_ha = _positive(
            "reference_yield_kg_ha", reference_yield_by_year[self.weather_year]
        )
        self.policy_budget_mm = _positive("policy_budget_mm", policy_budget_mm)
        self._policy_irrigation_mm = 0.0
        self._finished = False

    @property
    def policy_irrigation_mm(self) -> float:
        return self._policy_irrigation_mm

    def step_reward(self, executed_policy_irrigation_mm: float) -> float:
        if self._finished:
            raise RuntimeError("signal tracker already finished")
        amount = _nonnegative("executed_policy_irrigation_mm", executed_policy_irrigation_mm)
        if self._policy_irrigation_mm + amount > self.policy_budget_mm + 1e-9:
            raise ValueError("executed policy irrigation exceeds the frozen policy budget")
        self._policy_irrigation_mm += amount
        if abs(self._policy_irrigation_mm - self.policy_budget_mm) <= 1e-9:
            self._policy_irrigation_mm = self.policy_budget_mm
        return -amount / self.policy_budget_mm

    def finish(
        self,
        *,
        yield_kg_ha: float,
        irrigation_accounting_passed: bool,
    ) -> LagrangianSignalBreakdown:
        if self._finished:
            raise RuntimeError("signal tracker already finished")
        if irrigation_accounting_passed is not True:
            raise RuntimeError("invalid DSSAT irrigation accounting cannot enter training")
        y = _nonnegative("yield_kg_ha", yield_kg_ha)
        retention = yield_retention(y, self.reference_yield_kg_ha)
        signed_cost = self.eta - retention
        water_return = -self._policy_irrigation_mm / self.policy_budget_mm
        self._finished = True
        return LagrangianSignalBreakdown(
            weather_year=self.weather_year,
            eta=self.eta,
            policy_irrigation_mm=self._policy_irrigation_mm,
            yield_kg_ha=y,
            reference_yield_kg_ha=self.reference_yield_kg_ha,
            yield_retention=retention,
            water_return=water_return,
            signed_constraint_cost=signed_cost,
            per_year_expected_constraint_satisfied=signed_cost <= _EPS,
        )


def _nonnegative(name: str, value: float) -> float:
    result = _finite(name, value)
    if result < 0.0:
        raise ValueError(f"{name} must be >= 0")
    return result


def _positive(name: str, value: float) -> float:
    result = _finite(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be > 0")
    return result


def _finite(name: str, value: float) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite numeric")
    return float(value)


__all__ = ["LagrangianEpisodeSignals", "LagrangianSignalBreakdown"]
