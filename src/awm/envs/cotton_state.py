"""Leakage-safe 74-feature DSSAT cotton observation builder."""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

PLANT_FEATURES = (
    "CWAD", "LAID", "PWAD", "P#AD", "GWAD",
    "WSPD", "WSGD", "NSTD", "LN%D", "RDPD",
)
ROOT_FEATURES = tuple(f"RL{i}D" for i in range(1, 11))
REW_FEATURES = tuple(f"REW{i}" for i in range(1, 16))
NO3_FEATURES = tuple(f"NI{i}D" for i in range(1, 10)) + ("NI10",)
NH4_FEATURES = tuple(f"NH{i}D" for i in range(1, 10)) + ("NH10",)
UREA_FEATURES = ("NURTD",)
SOIL_TEMPERATURE_FEATURES = tuple(f"TS{i}D" for i in range(1, 10)) + ("TS10",)
CROP_WATER_DEMAND_FEATURES = ("EOPA",)
ET_FEATURES = ("EOAA",)
WEATHER_FEATURES = ("PRED", "SRAD", "TMXD", "TMND", "WDSD")
TIME_FEATURES = ("dap_frac",)

OBSERVATION_FEATURE_NAMES = (
    PLANT_FEATURES
    + ROOT_FEATURES
    + REW_FEATURES
    + NO3_FEATURES
    + NH4_FEATURES
    + UREA_FEATURES
    + SOIL_TEMPERATURE_FEATURES
    + CROP_WATER_DEMAND_FEATURES
    + ET_FEATURES
    + WEATHER_FEATURES
    + TIME_FEATURES
)
EXPECTED_STATE_DIM = len(OBSERVATION_FEATURE_NAMES)

SOIL_WATER_FEATURES = tuple(f"SW{i}D" for i in range(1, 16))
LOWER_LIMIT_FEATURES = tuple(f"LL{i}D" for i in range(1, 16))
DRAINED_UPPER_LIMIT_FEATURES = tuple(f"DUL{i}D" for i in range(1, 16))
DIRECT_RAW_FEATURES = (
    PLANT_FEATURES
    + ROOT_FEATURES
    + NO3_FEATURES
    + NH4_FEATURES
    + UREA_FEATURES
    + SOIL_TEMPERATURE_FEATURES
    + CROP_WATER_DEMAND_FEATURES
    + ET_FEATURES
    + WEATHER_FEATURES
)
FORBIDDEN_ACTOR_KEYS = frozenset(
    {
        "HWAM", "HWAH", "EYLDH", "LIWAM", "HDAT", "MDAT",
        "BWAH", "FCWAM", "FHWAM", "HWAHF", "FBWAH", "FPWAM",
        "HWUM", "H#AM", "H#UM", "HIAM", "LAIX",
        "IRCM", "NICM", "ETCM", "EPCM", "ESCM", "PRCM", "DRCM",
    }
)

if EXPECTED_STATE_DIM != 74:
    raise RuntimeError(
        f"cotton observation schema must remain 74D, got {EXPECTED_STATE_DIM}"
    )


def _finite(raw: Any, name: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"DSSAT state field {name!r} cannot be converted to float"
        ) from exc
    if not math.isfinite(value):
        raise FloatingPointError(f"DSSAT state field {name!r} is NaN/Inf")
    return value


class CottonObservationBuilder:
    def __init__(self, decision_horizon: int = 125) -> None:
        self.decision_horizon = int(decision_horizon)
        if self.decision_horizon <= 0:
            raise ValueError("decision_horizon must be positive")

    def build_dict(
        self,
        raw_state: Mapping[str, Any],
        current_day: int,
    ) -> dict[str, float]:
        missing = [name for name in DIRECT_RAW_FEATURES if name not in raw_state]
        rew_sources = (
            SOIL_WATER_FEATURES
            + LOWER_LIMIT_FEATURES
            + DRAINED_UPPER_LIMIT_FEATURES
        )
        missing += [name for name in rew_sources if name not in raw_state]
        if missing:
            raise KeyError(
                "DSSAT daily state missing observation fields: "
                + ", ".join(dict.fromkeys(missing))
            )
        if not 0 <= int(current_day) <= self.decision_horizon:
            raise ValueError("current_day outside decision horizon")

        values = {
            name: _finite(raw_state[name], name)
            for name in DIRECT_RAW_FEATURES
        }
        for sw, ll, dul, rew in zip(
            SOIL_WATER_FEATURES,
            LOWER_LIMIT_FEATURES,
            DRAINED_UPPER_LIMIT_FEATURES,
            REW_FEATURES,
            strict=True,
        ):
            sw_value = _finite(raw_state[sw], sw)
            ll_value = _finite(raw_state[ll], ll)
            dul_value = _finite(raw_state[dul], dul)
            if dul_value <= ll_value:
                raise ValueError(f"invalid soil water limits: {dul} <= {ll}")
            values[rew] = (sw_value - ll_value) / (dul_value - ll_value)

        values["dap_frac"] = int(current_day) / self.decision_horizon
        return {name: values[name] for name in OBSERVATION_FEATURE_NAMES}

    def build(
        self,
        raw_state: Mapping[str, Any],
        current_day: int,
    ) -> tuple[float, ...]:
        return tuple(self.build_dict(raw_state, current_day).values())


__all__ = [
    "CottonObservationBuilder",
    "DIRECT_RAW_FEATURES",
    "DRAINED_UPPER_LIMIT_FEATURES",
    "EXPECTED_STATE_DIM",
    "FORBIDDEN_ACTOR_KEYS",
    "LOWER_LIMIT_FEATURES",
    "OBSERVATION_FEATURE_NAMES",
    "SOIL_WATER_FEATURES",
]
