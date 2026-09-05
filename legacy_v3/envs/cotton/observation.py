"""Canonical cotton observation construction and Gym wrapper."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .state_schema import (
    DIRECT_RAW_FEATURES,
    DRAINED_UPPER_LIMIT_FEATURES,
    EXPECTED_STATE_DIM,
    LOWER_LIMIT_FEATURES,
    OBSERVATION_FEATURE_NAMES,
    REW_FEATURES,
    SOIL_WATER_FEATURES,
)


class CottonObservationBuilder:
    """Convert one DSSAT daily-state mapping to the canonical observation.

    The builder deliberately does not accept reward/action-history features.
    ``dap_frac`` describes the current DSSAT state, not a discrete phenology
    class: state at DAP ``d`` contains ``dap_frac=d/decision_horizon``.
    """

    def __init__(self, decision_horizon: int = 125):
        self.decision_horizon = int(decision_horizon)
        if self.decision_horizon <= 0:
            raise ValueError("decision_horizon must be positive.")

    @staticmethod
    def _finite_float(value: Any, name: str) -> float:
        try:
            result = float(np.asarray(value).reshape(-1)[0])
        except (TypeError, ValueError, IndexError) as exc:
            raise TypeError(
                f"DSSAT state field {name!r} cannot be converted to float: "
                f"{value!r}"
            ) from exc
        if not np.isfinite(result):
            raise FloatingPointError(
                f"DSSAT state field {name!r} is NaN/Inf: {result!r}"
            )
        return result

    @staticmethod
    def _require_keys(raw_state: Mapping[str, Any], keys: tuple[str, ...]) -> None:
        missing = [name for name in keys if name not in raw_state]
        if missing:
            raise KeyError(
                "DSSAT daily state is missing fields required by the "
                "observation: " + ", ".join(missing)
            )

    def _build_rew(self, raw_state: Mapping[str, Any]) -> dict[str, float]:
        result: dict[str, float] = {}
        for index, (sw_name, ll_name, dul_name, rew_name) in enumerate(
            zip(
                SOIL_WATER_FEATURES,
                LOWER_LIMIT_FEATURES,
                DRAINED_UPPER_LIMIT_FEATURES,
                REW_FEATURES,
                strict=True,
            ),
            start=1,
        ):
            sw = self._finite_float(raw_state[sw_name], sw_name)
            ll = self._finite_float(raw_state[ll_name], ll_name)
            dul = self._finite_float(raw_state[dul_name], dul_name)
            denominator = dul - ll
            if not np.isfinite(denominator) or denominator <= 0.0:
                raise ValueError(
                    "Invalid fixed soil-water limits for REW layer "
                    f"{index}: LL={ll}, DUL={dul}. Expected DUL > LL."
                )
            # Intentionally un-clipped. Values below 0 or above 1 preserve
            # physically useful information about states below LL or above DUL.
            result[rew_name] = (sw - ll) / denominator
        return result

    def build_dict(
        self,
        raw_state: Mapping[str, Any],
        current_day: int | float,
    ) -> dict[str, float]:
        if not isinstance(raw_state, Mapping):
            raise TypeError(
                "raw_state must be a mapping, got "
                f"{type(raw_state).__name__}."
            )

        required_rew_sources = (
            SOIL_WATER_FEATURES
            + LOWER_LIMIT_FEATURES
            + DRAINED_UPPER_LIMIT_FEATURES
        )
        self._require_keys(raw_state, DIRECT_RAW_FEATURES)
        self._require_keys(raw_state, required_rew_sources)

        day = self._finite_float(current_day, "current_day")
        if day < 0.0 or day > float(self.decision_horizon):
            raise ValueError(
                "current_day must remain inside the decision horizon: "
                f"day={day}, horizon={self.decision_horizon}."
            )

        values = {
            name: self._finite_float(raw_state[name], name)
            for name in DIRECT_RAW_FEATURES
        }
        values.update(self._build_rew(raw_state))
        values["dap_frac"] = day / float(self.decision_horizon)

        ordered = {name: values[name] for name in OBSERVATION_FEATURE_NAMES}
        if len(ordered) != EXPECTED_STATE_DIM:
            raise RuntimeError(
                "Observation builder produced wrong dimension: "
                f"{len(ordered)} != {EXPECTED_STATE_DIM}."
            )
        return ordered

    def build(
        self,
        raw_state: Mapping[str, Any],
        current_day: int | float,
    ) -> np.ndarray:
        ordered = self.build_dict(raw_state, current_day)
        observation = np.fromiter(
            ordered.values(),
            dtype=np.float32,
            count=EXPECTED_STATE_DIM,
        )
        if observation.shape != (EXPECTED_STATE_DIM,):
            raise RuntimeError(
                "Observation shape mismatch: "
                f"{observation.shape} != ({EXPECTED_STATE_DIM},)."
            )
        if not np.isfinite(observation).all():
            bad = np.flatnonzero(~np.isfinite(observation)).tolist()
            raise FloatingPointError(
                f"Observation contains NaN/Inf at indices {bad}."
            )
        return observation

    @property
    def feature_names(self) -> tuple[str, ...]:
        return OBSERVATION_FEATURE_NAMES


class CottonObservationWrapper(gym.Wrapper):
    """Expose the canonical observation from a raw cotton DSSAT environment."""

    ACTION_FEATURE_KEYS: tuple[str, ...] = tuple()

    def __init__(self, env: gym.Env, *, decision_horizon: int = 125):
        super().__init__(env)
        self.total_days = int(decision_horizon)
        if self.total_days <= 0:
            raise ValueError("decision_horizon must be positive.")

        raw_env = getattr(env, "unwrapped", env)
        env_total_days = getattr(raw_env, "total_days", None)
        if env_total_days is not None and int(env_total_days) != self.total_days:
            raise RuntimeError(
                "Observation decision horizon differs from environment: "
                f"wrapper={self.total_days}, env={env_total_days}."
            )

        self.builder = CottonObservationBuilder(
            decision_horizon=self.total_days
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(EXPECTED_STATE_DIM,),
            dtype=np.float32,
        )

    @property
    def observation_feature_names(self) -> tuple[str, ...]:
        return OBSERVATION_FEATURE_NAMES

    def _build_observation(
        self,
        raw_state_dict: Mapping[str, Any],
        info: Mapping[str, Any],
    ) -> np.ndarray:
        if not isinstance(info, Mapping):
            raise TypeError("Environment info must be a mapping.")
        if "current_day" not in info:
            raise KeyError("Environment info must contain current_day.")
        observation = self.builder.build(
            raw_state=raw_state_dict,
            current_day=info["current_day"],
        )
        if observation.shape != self.observation_space.shape:
            raise RuntimeError(
                "Observation shape changed unexpectedly: "
                f"{observation.shape} != {self.observation_space.shape}."
            )
        return observation

    def reset(self, **kwargs):
        raw_state_dict, info = self.env.reset(**kwargs)
        return self._build_observation(raw_state_dict, info), info

    def step(self, action):
        raw_state_dict, reward, terminated, truncated, info = self.env.step(action)
        observation = self._build_observation(raw_state_dict, info)
        return observation, reward, terminated, truncated, info


__all__ = ["CottonObservationBuilder", "CottonObservationWrapper"]
