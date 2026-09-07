"""Balanced training/validation cells for PPO baseline v1."""
from __future__ import annotations

from dataclasses import dataclass
import random

from awm.risk import REGISTERED_ETA_LEVELS, TRAIN_YEARS, VALIDATION_YEARS


@dataclass(frozen=True, slots=True, order=True)
class WeatherEtaCell:
    weather_year: int
    eta: float


def training_cells() -> tuple[WeatherEtaCell, ...]:
    cells = tuple(
        WeatherEtaCell(int(year), float(eta))
        for year in TRAIN_YEARS
        for eta in REGISTERED_ETA_LEVELS
    )
    if len(cells) != 54:
        raise AssertionError("PPO v1 training grid must contain exactly 54 cells")
    return cells


def balanced_training_cycle(*, seed: int, update_index: int) -> tuple[WeatherEtaCell, ...]:
    """Return every train weather×eta cell exactly once in deterministic shuffle."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be an integer >= 0")
    if isinstance(update_index, bool) or not isinstance(update_index, int) or update_index <= 0:
        raise ValueError("update_index must be a positive integer")
    cells = list(training_cells())
    # Stable integer mixing, independent of Python's randomized object hash.
    mixed_seed = ((seed & 0xFFFFFFFF) << 32) ^ (update_index & 0xFFFFFFFF) ^ 0xA57C0DE
    random.Random(mixed_seed).shuffle(cells)
    return tuple(cells)


def validation_cells() -> tuple[WeatherEtaCell, ...]:
    cells = tuple(
        WeatherEtaCell(int(year), float(eta))
        for year in VALIDATION_YEARS
        for eta in REGISTERED_ETA_LEVELS
    )
    if len(cells) != 15:
        raise AssertionError("PPO v1 validation grid must contain exactly 15 cells")
    return cells


__all__ = [
    "WeatherEtaCell",
    "balanced_training_cycle",
    "training_cells",
    "validation_cells",
]
