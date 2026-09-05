"""History-free additive reward for cotton water--nitrogen management."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ...configs.reward import REWARD_CONFIG, validate_reward_config


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    yield_reward: float
    irrigation_penalty: float
    nitrogen_penalty: float
    operation_penalty: float
    management_event: int
    previous_gwad: float
    next_gwad: float
    final_hwam: float | None
    pre_weather_affine_total: float
    weather_normalization_enabled: bool
    weather_reference_year: int | None
    weather_yield_low_kg_ha: float | None
    weather_yield_sufficient_kg_ha: float | None
    weather_yield_span_kg_ha: float | None
    weather_reward_multiplier: float
    weather_yield_offset: float

    def as_dict(self) -> dict[str, float | int | bool | None]:
        return {
            "reward_total": self.total,
            "reward_yield": self.yield_reward,
            "reward_irrigation_penalty": self.irrigation_penalty,
            "reward_nitrogen_penalty": self.nitrogen_penalty,
            "reward_operation_penalty": self.operation_penalty,
            "management_event": self.management_event,
            "reward_previous_gwad": self.previous_gwad,
            "reward_next_gwad": self.next_gwad,
            "reward_final_hwam": self.final_hwam,
            "reward_pre_weather_affine_total": self.pre_weather_affine_total,
            "weather_normalization_enabled": self.weather_normalization_enabled,
            "weather_reference_year": self.weather_reference_year,
            "weather_yield_low_kg_ha": self.weather_yield_low_kg_ha,
            "weather_yield_sufficient_kg_ha": (
                self.weather_yield_sufficient_kg_ha
            ),
            "weather_yield_span_kg_ha": self.weather_yield_span_kg_ha,
            "weather_reward_multiplier": self.weather_reward_multiplier,
            "weather_yield_offset": self.weather_yield_offset,
        }


class CottonReward:
    """Compute one transition reward with optional weather-year normalization.

    Step-6 raw seasonal objective::

        R_raw = HWAM / 1000 - I / 6000 - N / 3000 - M / 1200.

    Step-7 does not introduce a new within-year water/yield trade-off.  For a
    weather year y with two frozen standard-treatment yields Y_low and Y_suff,
    D_y = Y_suff - Y_low > 0 and the complete Step-6 objective is transformed
    by the positive affine map

        R_norm = (1000 / D_y) * (R_raw - Y_low / 1000).

    Therefore policy ordering within one weather realization is unchanged;
    only cross-year location/scale are normalized.  The corresponding yield
    component telescopes to (HWAM - Y_low) / D_y and may exceed 1 when learned
    dynamic management outperforms the sufficient-input standard treatment.
    """

    def __init__(self, config: Mapping[str, Any] | None = None):
        cfg = validate_reward_config(REWARD_CONFIG if config is None else config)
        self.config = cfg
        self.yield_scale = float(cfg["yield-scale-kg-ha"])
        self.irrigation_scale = float(cfg["irrigation-scale-mm"])
        self.nitrogen_scale = float(cfg["nitrogen-scale-kg-ha"])
        self.management_event_scale = float(cfg["management-event-scale"])
        self.daily_yield_field = str(cfg["daily-yield-field"])
        self.final_yield_field = str(cfg["final-yield-field"])

        self.weather_normalization_enabled = bool(
            cfg.get("weather-normalization-enabled", False)
        )
        self.weather_reference_year: int | None = None
        self.weather_yield_low: float | None = None
        self.weather_yield_sufficient: float | None = None
        self.weather_yield_span: float | None = None
        self.weather_reward_multiplier = 1.0
        if self.weather_normalization_enabled:
            self.weather_reference_year = int(cfg["weather-reference-year"])
            self.weather_yield_low = self._finite_nonnegative(
                cfg["weather-yield-low-kg-ha"], "weather-yield-low-kg-ha"
            )
            self.weather_yield_sufficient = self._finite_nonnegative(
                cfg["weather-yield-sufficient-kg-ha"],
                "weather-yield-sufficient-kg-ha",
            )
            self.weather_yield_span = float(
                self.weather_yield_sufficient - self.weather_yield_low
            )
            if not np.isfinite(self.weather_yield_span) or self.weather_yield_span <= 0:
                raise ValueError(
                    "Weather normalization requires sufficient yield > low yield; "
                    f"low={self.weather_yield_low}, "
                    f"sufficient={self.weather_yield_sufficient}."
                )
            configured_span = float(
                cfg.get("weather-yield-span-kg-ha", self.weather_yield_span)
            )
            if not np.isclose(
                configured_span,
                self.weather_yield_span,
                rtol=0.0,
                atol=1e-9,
            ):
                raise ValueError(
                    "Weather-yield span is inconsistent with reference yields: "
                    f"configured={configured_span}, actual={self.weather_yield_span}."
                )
            self.weather_reward_multiplier = float(
                self.yield_scale / self.weather_yield_span
            )
            if not np.isfinite(self.weather_reward_multiplier) or (
                self.weather_reward_multiplier <= 0.0
            ):
                raise ValueError("Weather reward multiplier must be positive.")

    @staticmethod
    def _finite_nonnegative(value: Any, name: str) -> float:
        try:
            result = float(np.asarray(value).reshape(-1)[0])
        except (TypeError, ValueError, IndexError) as exc:
            raise TypeError(f"{name} must be numeric, got {value!r}.") from exc
        if not np.isfinite(result):
            raise FloatingPointError(f"{name} must be finite, got {result!r}.")
        if result < 0.0:
            raise ValueError(f"{name} must be non-negative, got {result!r}.")
        return result

    @staticmethod
    def _require_state_value(state: Mapping[str, Any], key: str) -> float:
        if key not in state:
            raise KeyError(f"Reward state is missing required field {key!r}.")
        try:
            result = float(np.asarray(state[key]).reshape(-1)[0])
        except (TypeError, ValueError, IndexError) as exc:
            raise TypeError(
                f"Reward field {key!r} must be numeric, got {state[key]!r}."
            ) from exc
        if not np.isfinite(result):
            raise FloatingPointError(
                f"Reward field {key!r} must be finite, got {result!r}."
            )
        return result

    @staticmethod
    def _applied_action(action: Sequence[float]) -> tuple[float, float]:
        values = np.asarray(action, dtype=np.float64).reshape(-1)
        if values.shape != (2,):
            raise ValueError(
                "Applied action must have shape (2,) = [irrigation, nitrogen], "
                f"got {values.shape}."
            )
        if not np.isfinite(values).all():
            raise FloatingPointError(f"Applied action contains NaN/Inf: {values}.")
        if np.any(values < 0.0):
            raise ValueError(f"Applied action must be non-negative: {values}.")
        return float(values[0]), float(values[1])

    def normalized_yield_response(self, final_hwam: float) -> float:
        hwam = self._finite_nonnegative(final_hwam, "final_hwam")
        if not self.weather_normalization_enabled:
            return float(hwam / self.yield_scale)
        assert self.weather_yield_low is not None
        assert self.weather_yield_span is not None
        return float((hwam - self.weather_yield_low) / self.weather_yield_span)

    def compute(
        self,
        *,
        previous_state: Mapping[str, Any],
        next_state: Mapping[str, Any],
        applied_action: Sequence[float],
        terminated: bool,
        season_summary: Mapping[str, Any] | None = None,
    ) -> RewardBreakdown:
        previous_gwad = self._require_state_value(
            previous_state, self.daily_yield_field
        )
        next_gwad = self._require_state_value(next_state, self.daily_yield_field)
        irrigation, nitrogen = self._applied_action(applied_action)

        final_hwam: float | None = None
        if terminated:
            if season_summary is None:
                raise ValueError(
                    "Terminal reward requires the mature DSSAT season summary."
                )
            final_hwam = self._require_state_value(
                season_summary, self.final_yield_field
            )
            yield_increment = final_hwam - previous_gwad
        else:
            yield_increment = next_gwad - previous_gwad

        # Preserve negative GWAD increments so the mature-yield telescope is exact.
        raw_yield_reward = yield_increment / self.yield_scale
        raw_irrigation_penalty = irrigation / self.irrigation_scale
        raw_nitrogen_penalty = nitrogen / self.nitrogen_scale
        management_event = int(irrigation > 0.0 or nitrogen > 0.0)
        raw_operation_penalty = management_event / self.management_event_scale
        raw_total = (
            raw_yield_reward
            - raw_irrigation_penalty
            - raw_nitrogen_penalty
            - raw_operation_penalty
        )

        multiplier = self.weather_reward_multiplier
        yield_offset = 0.0
        if self.weather_normalization_enabled and terminated:
            assert self.weather_yield_low is not None
            assert self.weather_yield_span is not None
            yield_offset = -self.weather_yield_low / self.weather_yield_span

        yield_reward = raw_yield_reward * multiplier + yield_offset
        irrigation_penalty = raw_irrigation_penalty * multiplier
        nitrogen_penalty = raw_nitrogen_penalty * multiplier
        operation_penalty = raw_operation_penalty * multiplier
        total = (
            yield_reward
            - irrigation_penalty
            - nitrogen_penalty
            - operation_penalty
        )
        if not np.isfinite(total):
            raise FloatingPointError("Reward computation produced NaN/Inf.")

        return RewardBreakdown(
            total=float(total),
            yield_reward=float(yield_reward),
            irrigation_penalty=float(irrigation_penalty),
            nitrogen_penalty=float(nitrogen_penalty),
            operation_penalty=float(operation_penalty),
            management_event=management_event,
            previous_gwad=float(previous_gwad),
            next_gwad=float(next_gwad),
            final_hwam=(None if final_hwam is None else float(final_hwam)),
            pre_weather_affine_total=float(raw_total),
            weather_normalization_enabled=bool(self.weather_normalization_enabled),
            weather_reference_year=self.weather_reference_year,
            weather_yield_low_kg_ha=self.weather_yield_low,
            weather_yield_sufficient_kg_ha=self.weather_yield_sufficient,
            weather_yield_span_kg_ha=self.weather_yield_span,
            weather_reward_multiplier=float(multiplier),
            weather_yield_offset=float(yield_offset),
        )

    def seasonal_objective(
        self,
        *,
        final_hwam: float,
        total_irrigation: float,
        total_nitrogen: float,
        management_events: int,
    ) -> float:
        """Direct undiscounted objective used for terminal telescope audit."""
        hwam = self._finite_nonnegative(final_hwam, "final_hwam")
        irrigation = self._finite_nonnegative(
            total_irrigation, "total_irrigation"
        )
        nitrogen = self._finite_nonnegative(total_nitrogen, "total_nitrogen")
        events = int(management_events)
        if events < 0:
            raise ValueError("management_events must be non-negative.")

        raw = float(
            hwam / self.yield_scale
            - irrigation / self.irrigation_scale
            - nitrogen / self.nitrogen_scale
            - events / self.management_event_scale
        )
        if not self.weather_normalization_enabled:
            return raw
        assert self.weather_yield_low is not None
        return float(
            self.weather_reward_multiplier
            * (raw - self.weather_yield_low / self.yield_scale)
        )


__all__ = ["CottonReward", "RewardBreakdown"]
