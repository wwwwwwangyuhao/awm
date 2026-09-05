"""Canonical cotton observation schema.

The actor/critic observation is an explicit allow-list of decision-time state.
Seasonal/harvest summaries never enter this schema. Soil water is represented
by un-clipped relative extractable water (REW), derived layer-wise from
SW/LL/DUL rather than exposing those raw values directly.
"""
from __future__ import annotations

PLANT_FEATURES = (
    "CWAD", "LAID", "PWAD", "P#AD", "GWAD",
    "WSPD", "WSGD", "NSTD", "LN%D", "RDPD",
)
ROOT_FEATURES = tuple(f"RL{i}D" for i in range(1, 11))
REW_FEATURES = tuple(f"REW{i}" for i in range(1, 16))
NO3_FEATURES = tuple(f"NI{i}D" for i in range(1, 10)) + ("NI10",)
NH4_FEATURES = tuple(f"NH{i}D" for i in range(1, 10)) + ("NH10",)
UREA_FEATURES = ("NURTD",)
SOIL_TEMPERATURE_FEATURES = (
    tuple(f"TS{i}D" for i in range(1, 10)) + ("TS10",)
)
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
NON_TIME_FEATURE_NAMES = OBSERVATION_FEATURE_NAMES[:-1]
EXPECTED_STATE_DIM = len(OBSERVATION_FEATURE_NAMES)

# Raw fields required only to derive REW. They are intentionally absent from
# OBSERVATION_FEATURE_NAMES because the site/soil profile is fixed.
SOIL_WATER_FEATURES = tuple(f"SW{i}D" for i in range(1, 16))
LOWER_LIMIT_FEATURES = tuple(f"LL{i}D" for i in range(1, 16))
DRAINED_UPPER_LIMIT_FEATURES = tuple(f"DUL{i}D" for i in range(1, 16))
REW_SOURCE_FEATURES = (
    SOIL_WATER_FEATURES
    + LOWER_LIMIT_FEATURES
    + DRAINED_UPPER_LIMIT_FEATURES
)
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
RAW_REQUIRED_KEYS = DIRECT_RAW_FEATURES + REW_SOURCE_FEATURES

# The observation allow-list is the primary protection. This set documents
# especially dangerous seasonal/future variables and is asserted as a second
# line of defence.
FORBIDDEN_ACTOR_KEYS = frozenset(
    {
        "HWAM", "HWAH", "EYLDH", "LIWAM", "HDAT", "MDAT",
        "BWAH", "FCWAM", "FHWAM", "HWAHF", "FBWAH", "FPWAM",
        "HWUM", "H#AM", "H#UM", "HIAM", "LAIX",
        "IRCM", "NICM", "ETCM", "EPCM", "ESCM", "PRCM", "DRCM",
    }
)
ACTION_HISTORY_FEATURES = frozenset(
    {
        "continuous_irrigate_days",
        "continuous_fertilize_days",
        "irrigate_risk_level",
        "fertilize_risk_level",
        "last_3_irrigation_avg",
        "last_3_fertilizer_avg",
        "irrigate_trend",
        "fertilize_trend",
    }
)


def validate_schema() -> None:
    if len(set(OBSERVATION_FEATURE_NAMES)) != EXPECTED_STATE_DIM:
        raise RuntimeError("Observation schema contains duplicate names.")
    forbidden = FORBIDDEN_ACTOR_KEYS.intersection(OBSERVATION_FEATURE_NAMES)
    if forbidden:
        raise RuntimeError(
            "Oracle/seasonal fields leaked into observation schema: "
            + ", ".join(sorted(forbidden))
        )
    history = ACTION_HISTORY_FEATURES.intersection(OBSERVATION_FEATURE_NAMES)
    if history:
        raise RuntimeError(
            "Action-history fields leaked into observation schema: "
            + ", ".join(sorted(history))
        )
    if TIME_FEATURES != ("dap_frac",):
        raise RuntimeError("Time schema must contain exactly one dap_frac.")
    expected_non_time = EXPECTED_STATE_DIM - len(TIME_FEATURES)
    if len(NON_TIME_FEATURE_NAMES) != expected_non_time:
        raise RuntimeError(
            "Non-time observation schema size mismatch: "
            f"{len(NON_TIME_FEATURE_NAMES)} != {expected_non_time}."
        )


validate_schema()

__all__ = [
    "PLANT_FEATURES",
    "ROOT_FEATURES",
    "REW_FEATURES",
    "NO3_FEATURES",
    "NH4_FEATURES",
    "UREA_FEATURES",
    "SOIL_TEMPERATURE_FEATURES",
    "CROP_WATER_DEMAND_FEATURES",
    "ET_FEATURES",
    "WEATHER_FEATURES",
    "TIME_FEATURES",
    "OBSERVATION_FEATURE_NAMES",
    "NON_TIME_FEATURE_NAMES",
    "EXPECTED_STATE_DIM",
    "SOIL_WATER_FEATURES",
    "LOWER_LIMIT_FEATURES",
    "DRAINED_UPPER_LIMIT_FEATURES",
    "RAW_REQUIRED_KEYS",
    "FORBIDDEN_ACTOR_KEYS",
]
