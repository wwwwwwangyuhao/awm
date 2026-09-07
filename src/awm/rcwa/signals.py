"""Per-episode water objective and terminal retention bookkeeping for RCWA-RL."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from awm.risk import yield_retention


@dataclass(frozen=True, slots=True)
class RCWAEpisodeBreakdown:
    weather_year: int
    eta: float
    reference_yield_kg_ha: float
    yield_kg_ha: float
    yield_retention: float
    policy_irrigation_mm: float
    water_return: float


class RCWAEpisodeSignals:
    def __init__(
        self,
        *,
        weather_year: int,
        eta: float,
        reference_yield_by_year: Mapping[int, float],
        policy_budget_mm: float = 495.0,
    ) -> None:
        if int(weather_year) not in reference_yield_by_year:
            raise KeyError(f"missing Y_ref for weather year {weather_year}")
        if not math.isfinite(float(eta)):
            raise ValueError("eta must be finite")
        if not math.isfinite(float(policy_budget_mm)) or float(policy_budget_mm) <= 0.0:
            raise ValueError("policy_budget_mm must be finite and > 0")
        self.weather_year = int(weather_year)
        self.eta = float(eta)
        self.reference_yield = float(reference_yield_by_year[self.weather_year])
        self.policy_budget_mm = float(policy_budget_mm)
        self._policy_irrigation_mm = 0.0

    @property
    def policy_irrigation_mm(self) -> float:
        return self._policy_irrigation_mm

    def step_reward(self, applied_irrigation_mm: float) -> float:
        value = float(applied_irrigation_mm)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("applied_irrigation_mm must be finite and >= 0")
        self._policy_irrigation_mm += value
        if self._policy_irrigation_mm > self.policy_budget_mm + 1e-8:
            raise RuntimeError("RCWA water ledger exceeded policy budget")
        return -value / self.policy_budget_mm

    def finish(
        self,
        *,
        yield_kg_ha: float,
        irrigation_accounting_passed: bool,
    ) -> RCWAEpisodeBreakdown:
        if irrigation_accounting_passed is not True:
            raise RuntimeError("RCWA episode failed irrigation accounting")
        y = float(yield_kg_ha)
        retention = yield_retention(y, self.reference_yield)
        return RCWAEpisodeBreakdown(
            weather_year=self.weather_year,
            eta=self.eta,
            reference_yield_kg_ha=self.reference_yield,
            yield_kg_ha=y,
            yield_retention=retention,
            policy_irrigation_mm=self._policy_irrigation_mm,
            water_return=-self._policy_irrigation_mm / self.policy_budget_mm,
        )


__all__ = ["RCWAEpisodeBreakdown", "RCWAEpisodeSignals"]
