"""Canonical DSSAT environment configuration shared by all policy roles."""
from __future__ import annotations

import os

from .base_settings import BASE_DIR
from .reward import REWARD_CONFIG
from ..envs.cotton.state_schema import NON_TIME_FEATURE_NAMES

EXP_YEAR = "25"
CALENDAR_YEAR = 2025
DECISION_HORIZON = 125
TOTAL_DAYS = DECISION_HORIZON  # compatibility alias

SITE = "XJHX"
FIELD_CODE = "0001"
FIELD_NAME = f"{SITE}{FIELD_CODE}"
EXP_DUR_YEAR = "01"
PLANT_DATE = "119"
EMERGENCE_DATE = "133"

EXP_DIR = BASE_DIR
DATA_DIR = os.path.join(EXP_DIR, "data")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")
DSSAT_EXEC = os.path.join(EXP_DIR, "dscsm048")

COX_NAME = f"{SITE}{EXP_YEAR}{EXP_DUR_YEAR}"
WEATHER_NAME = COX_NAME
IRRI_NAME = f"IRRI{EXP_YEAR}{EXP_DUR_YEAR}"
FERT_NAME = f"FERT{EXP_YEAR}{EXP_DUR_YEAR}"

COX_PATH_FILE = os.path.join(DATA_DIR, "COX", f"{COX_NAME}.COX")
WEATHER_PATH_FILE = os.path.join(DATA_DIR, "weather", f"{WEATHER_NAME}.WTH")
SOIL_PATH_FILE = os.path.join(DATA_DIR, "soil", "SOIL.SOL")
IRRI_PATH_FILE = os.path.join(DATA_DIR, "irrigation", f"{IRRI_NAME}.csv")
FERT_PATH_FILE = os.path.join(DATA_DIR, "fertilizer", f"{FERT_NAME}.csv")

SUMMARY_OUT = os.path.join(OUTPUT_DIR, "Summary.OUT")
PLANTGRO_OUT = os.path.join(OUTPUT_DIR, "PlantGro.OUT")
SOILWAT_OUT = os.path.join(OUTPUT_DIR, "SoilWat.OUT")
WEATHER_OUT = os.path.join(OUTPUT_DIR, "Weather.OUT")
ET_OUT = os.path.join(OUTPUT_DIR, "ET.OUT")
GHG_OUT = os.path.join(OUTPUT_DIR, "GHG.OUT")
MGMT_OPS_OUT = os.path.join(OUTPUT_DIR, "MgmtOps.OUT")
MULCH_OUT = os.path.join(OUTPUT_DIR, "Mulch.OUT")
N2O_OUT = os.path.join(OUTPUT_DIR, "N2O.OUT")
PLANT_C_OUT = os.path.join(OUTPUT_DIR, "PlantC.OUT")
PLANT_N_OUT = os.path.join(OUTPUT_DIR, "PlantN.OUT")
SOIL_TEMP_OUT = os.path.join(OUTPUT_DIR, "SoilTemp.OUT")
SOIL_WATER_OUT = os.path.join(OUTPUT_DIR, "SoilWater.OUT")
SOIL_N_OUT = os.path.join(OUTPUT_DIR, "SoilNi.OUT")

# Phase 1/2 keeps the broad output inventory for diagnostics. The cached
# reader parses it once per DSSAT execution rather than once per RL step.
OUT_LIST = [
    PLANTGRO_OUT,
    SOILWAT_OUT,
    WEATHER_OUT,
    ET_OUT,
    GHG_OUT,
    MGMT_OPS_OUT,
    MULCH_OUT,
    N2O_OUT,
    PLANT_C_OUT,
    PLANT_N_OUT,
    SOIL_TEMP_OUT,
    SOIL_WATER_OUT,
    SOIL_N_OUT,
]

STR_FIELDS = {"CR", "MODEL", "EXNAME", "TNAM", "FNAM", "NSTA", "SOIL_ID"}
DATE_FIELDS = {"SDAT", "PDAT", "EDAT", "ADAT", "MDAT", "HDAT"}
DELETE_DIR = ["COX", "fertilizer", "irrigation", "output"]

# Compatibility export while rollout/evaluation still construct the wrapper
# through the pre-refactor arguments. CottonObservationBuilder is authoritative.
FILTERED_KEYS = list(NON_TIME_FEATURE_NAMES)

ENV_CONFIG = {
    "exp-year": EXP_YEAR,
    "calendar-year": CALENDAR_YEAR,
    "dssat-exec": DSSAT_EXEC,
    "data-dir": DATA_DIR,
    "output-dir": OUTPUT_DIR,
    "cox-file": COX_PATH_FILE,
    "weather-file": WEATHER_PATH_FILE,
    "soil-file": SOIL_PATH_FILE,
    "irrigation-file": IRRI_PATH_FILE,
    "fertilizer-file": FERT_PATH_FILE,
    "summary-out": SUMMARY_OUT,
    "out-list": OUT_LIST,
    "plant-date": f"{EXP_YEAR}{PLANT_DATE}",
    "emergence-date": f"{EXP_YEAR}{EMERGENCE_DATE}",
    "total-days": DECISION_HORIZON,
    "decision-horizon": DECISION_HORIZON,
    "is-phosphorus": 0,
    "is-potassium": 0,
    "field-name": FIELD_NAME,
    "weather-name": WEATHER_NAME,
    "cox-name": COX_NAME,
    "delete-dir": DELETE_DIR,
    "verbose": 0,
    "str-fields": STR_FIELDS,
    "date-fields": DATE_FIELDS,
    "action-type": "continuous",
    "max-irri": 50.0,
    "max-fert": 30.0,
    # Legacy <1 -> 0 application is disabled. Exact no-op is now a policy
    # decision, while positive actor outputs are already canonicalized to the
    # 0.01 COX resolution before reaching the environment.
    "action-application-threshold": 0.0,
    "reward-config": dict(REWARD_CONFIG),
}

__all__ = [
    "DECISION_HORIZON",
    "TOTAL_DAYS",
    "FILTERED_KEYS",
    "ENV_CONFIG",
]
