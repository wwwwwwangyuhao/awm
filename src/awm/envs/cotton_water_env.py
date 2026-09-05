"""DSSAT cotton irrigation environment with finite seasonal water allocation."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Mapping, Protocol

from .cotton_state import CottonObservationBuilder, OBSERVATION_FEATURE_NAMES
from .dssat_irrigation import DSSATIrrigationAdapter, IrrigationStepAudit


class BackendStateLike(Protocol):
    def daily_state(self, yrdoy: str) -> dict: ...
    def season_summary(self) -> dict: ...


@dataclass(frozen=True, slots=True)
class CottonWaterObservation:
    """Structured policy observation before model-specific encoding."""

    dssat_features: tuple[float, ...]
    water_features: tuple[float, float, float, float, float]
    feature_names: tuple[str, ...]

    def flat(self) -> tuple[float, ...]:
        return self.dssat_features + self.water_features


@dataclass(frozen=True, slots=True)
class CottonWaterStep:
    observation: CottonWaterObservation
    terminated: bool
    irrigation_audit: IrrigationStepAudit
    info: Mapping[str, Any]


class CottonWaterEnv:
    """Reward-free irrigation environment contract.

    Nitrogen is intentionally not an action. It remains fixed in the external
    DSSAT experiment template. Summary.OUT is read only at termination.
    """

    WATER_FEATURE_NAMES = (
        "remaining_budget_fraction",
        "cumulative_irrigation_fraction",
        "remaining_horizon_fraction",
        "days_since_last_irrigation",
        "yield_target_fraction",
    )

    def __init__(
        self,
        *,
        backend: BackendStateLike,
        adapter: DSSATIrrigationAdapter,
        plant_yrdoy: str,
        yield_target_fraction: float,
    ) -> None:
        self.backend = backend
        self.adapter = adapter
        self.calendar = adapter.calendar
        expected_plant = (
            f"{self.calendar.calendar_year % 100:02d}"
            f"{self.calendar.planting_doy:03d}"
        )
        if str(plant_yrdoy).strip() != expected_plant:
            raise ValueError("plant_yrdoy and adapter calendar disagree")
        self.plant_yrdoy = str(plant_yrdoy).strip()
        self.yield_target_fraction = float(yield_target_fraction)
        if not 0.0 < self.yield_target_fraction <= 1.0:
            raise ValueError("yield_target_fraction must lie in (0,1]")
        self.observation_builder = CottonObservationBuilder(
            self.calendar.horizon_days
        )
        self.current_day = 0
        self._terminated = False

    def _state_yrdoy(self, day: int) -> str:
        planting = date(self.calendar.calendar_year, 1, 1) + timedelta(
            days=self.calendar.planting_doy - 1
        )
        current = planting + timedelta(days=day)
        return f"{current.year % 100:02d}{current.timetuple().tm_yday:03d}"

    def _water_features(self) -> tuple[float, float, float, float, float]:
        controller = self.adapter.controller
        if self.current_day < self.calendar.horizon_days:
            return controller.observe(
                day=self.current_day,
                yield_target_fraction=self.yield_target_fraction,
            ).as_vector()

        total = float(controller.spec.seasonal_budget_mm)
        used = float(controller.used_mm)
        last = controller.last_irrigation_day
        days_since = (
            self.current_day + 1
            if last is None
            else self.current_day - last
        )
        return (
            max(0.0, total - used) / total,
            used / total,
            0.0,
            float(days_since),
            self.yield_target_fraction,
        )

    def _observation(self) -> CottonWaterObservation:
        raw_state = self.backend.daily_state(
            self._state_yrdoy(self.current_day)
        )
        dssat = self.observation_builder.build(
            raw_state,
            self.current_day,
        )
        water = self._water_features()
        return CottonWaterObservation(
            dssat_features=dssat,
            water_features=water,
            feature_names=(
                OBSERVATION_FEATURE_NAMES + self.WATER_FEATURE_NAMES
            ),
        )

    def reset(self) -> tuple[CottonWaterObservation, dict[str, object]]:
        self.adapter.reset_episode()
        self.current_day = 0
        self._terminated = False
        return self._observation(), {
            "current_day": 0,
            "policy_irrigation_mm": 0.0,
            "seasonal_summary_exposed": False,
        }

    def step(
        self,
        *,
        irrigate: bool,
        amount_fraction: float,
    ) -> CottonWaterStep:
        if self._terminated:
            raise RuntimeError("episode has ended; call reset()")

        audit = self.adapter.apply(
            policy_day=self.current_day,
            irrigate=irrigate,
            amount_fraction=amount_fraction,
        )
        self.current_day += 1
        self._terminated = self.current_day >= self.calendar.horizon_days

        info: dict[str, object] = dict(audit.as_info_dict())
        info.update(
            {
                "current_day": self.current_day,
                "policy_irrigation_mm": float(
                    self.adapter.controller.used_mm
                ),
                "seasonal_summary_exposed": False,
            }
        )

        if self._terminated:
            summary = self.backend.season_summary()
            if "HWAM" not in summary or "IRCM" not in summary:
                raise KeyError(
                    "terminal Summary.OUT must contain HWAM and IRCM"
                )
            terminal_audit = self.adapter.reconcile_terminal_irrigation(
                dssat_ircm_mm=float(summary["IRCM"])
            )
            info.update(
                {
                    "seasonal_summary_exposed": True,
                    "HWAM": float(summary["HWAM"]),
                    "IRCM": float(summary["IRCM"]),
                    "irrigation_accounting_passed": terminal_audit.passed,
                    "irrigation_accounting_difference_mm": (
                        terminal_audit.difference_mm
                    ),
                }
            )

        # Terminal Summary.OUT fields are never appended to this observation.
        return CottonWaterStep(
            observation=self._observation(),
            terminated=self._terminated,
            irrigation_audit=audit,
            info=info,
        )


__all__ = [
    "CottonWaterEnv",
    "CottonWaterObservation",
    "CottonWaterStep",
]
