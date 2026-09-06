"""Deterministic learned-method checkpoint selection for AWM evaluation v1.

Selection is performed independently for each training seed.  A single
checkpoint must serve all registered eta levels.  Validation years are used
only for evaluation/selection and never for gradient updates.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import mean
from typing import Iterable, Mapping, Sequence

from awm.risk import DEFAULT_ALPHA, REGISTERED_ETA_LEVELS


EVALUATION_CONTRACT_ID = "awm-learned-method-evaluation-v1"
VALIDATION_YEARS = (2018, 2019, 2020, 2021, 2022)
LOCKED_FINAL_TEST_YEARS = (2023, 2024, 2025)
REQUIRED_VALIDATION_CELL_COUNT = len(VALIDATION_YEARS) * len(REGISTERED_ETA_LEVELS)
_EPS = 1e-12


@dataclass(frozen=True, slots=True)
class EtaValidationMetrics:
    eta: float
    validation_years: tuple[int, ...]
    lcvar_retention: float
    mean_total_irrigation_mm: float
    minimum_retention: float

    @property
    def margin(self) -> float:
        return self.lcvar_retention - self.eta

    @property
    def shortfall(self) -> float:
        return max(0.0, -self.margin)

    @property
    def feasible(self) -> bool:
        return self.margin >= -_EPS


@dataclass(frozen=True, slots=True)
class CheckpointValidation:
    checkpoint_id: str
    training_seed: int
    training_step: int
    eta_metrics: tuple[EtaValidationMetrics, ...]

    @property
    def jointly_feasible(self) -> bool:
        return all(item.feasible for item in self.eta_metrics)

    @property
    def mean_total_irrigation_mm(self) -> float:
        return mean(item.mean_total_irrigation_mm for item in self.eta_metrics)

    @property
    def minimum_margin(self) -> float:
        return min(item.margin for item in self.eta_metrics)

    @property
    def mean_margin(self) -> float:
        return mean(item.margin for item in self.eta_metrics)

    @property
    def worst_shortfall(self) -> float:
        return max(item.shortfall for item in self.eta_metrics)

    @property
    def mean_shortfall(self) -> float:
        return mean(item.shortfall for item in self.eta_metrics)


@dataclass(frozen=True, slots=True)
class CheckpointSelection:
    training_seed: int
    selected_checkpoint_id: str
    selected_training_step: int
    selection_status: str
    jointly_feasible: bool
    mean_total_irrigation_mm: float
    minimum_risk_margin: float
    mean_risk_margin: float
    worst_shortfall: float
    mean_shortfall: float
    candidate_count: int


def select_checkpoint(
    candidates: Sequence[CheckpointValidation],
    *,
    expected_seed: int | None = None,
) -> CheckpointSelection:
    """Select exactly one checkpoint using the frozen lexicographic contract."""

    validated = tuple(_validate_candidate(candidate) for candidate in candidates)
    if not validated:
        raise ValueError("at least one checkpoint candidate is required")

    seeds = {candidate.training_seed for candidate in validated}
    if len(seeds) != 1:
        raise ValueError("checkpoint selection must operate on exactly one training seed")
    seed = next(iter(seeds))
    if expected_seed is not None and seed != int(expected_seed):
        raise ValueError(f"candidate seed {seed} does not match expected seed {expected_seed}")

    checkpoint_ids = [candidate.checkpoint_id for candidate in validated]
    if len(checkpoint_ids) != len(set(checkpoint_ids)):
        raise ValueError("checkpoint_id values must be unique within one seed")

    feasible = [candidate for candidate in validated if candidate.jointly_feasible]
    if feasible:
        selected = min(feasible, key=_feasible_sort_key)
        status = "selected_feasible"
    else:
        selected = min(validated, key=_fallback_sort_key)
        status = "selected_infeasible"

    return CheckpointSelection(
        training_seed=seed,
        selected_checkpoint_id=selected.checkpoint_id,
        selected_training_step=selected.training_step,
        selection_status=status,
        jointly_feasible=selected.jointly_feasible,
        mean_total_irrigation_mm=selected.mean_total_irrigation_mm,
        minimum_risk_margin=selected.minimum_margin,
        mean_risk_margin=selected.mean_margin,
        worst_shortfall=selected.worst_shortfall,
        mean_shortfall=selected.mean_shortfall,
        candidate_count=len(validated),
    )


def build_candidate_from_report(payload: Mapping[str, object]) -> CheckpointValidation:
    """Parse and validate a machine-readable checkpoint validation report."""

    checkpoint_id = str(payload.get("checkpoint_id", "")).strip()
    if not checkpoint_id:
        raise ValueError("checkpoint_id must be non-empty")
    training_seed = _require_int("training_seed", payload.get("training_seed"), minimum=0)
    training_step = _require_int("training_step", payload.get("training_step"), minimum=0)

    raw_metrics = payload.get("eta_metrics")
    if not isinstance(raw_metrics, list):
        raise TypeError("eta_metrics must be a list")

    metrics: list[EtaValidationMetrics] = []
    for raw in raw_metrics:
        if not isinstance(raw, Mapping):
            raise TypeError("each eta_metrics entry must be an object")
        years_raw = raw.get("validation_years")
        if not isinstance(years_raw, list):
            raise TypeError("validation_years must be a list")
        metrics.append(
            EtaValidationMetrics(
                eta=_finite("eta", raw.get("eta")),
                validation_years=tuple(int(year) for year in years_raw),
                lcvar_retention=_finite("lcvar_retention", raw.get("lcvar_retention")),
                mean_total_irrigation_mm=_finite(
                    "mean_total_irrigation_mm", raw.get("mean_total_irrigation_mm")
                ),
                minimum_retention=_finite("minimum_retention", raw.get("minimum_retention")),
            )
        )
    return _validate_candidate(
        CheckpointValidation(
            checkpoint_id=checkpoint_id,
            training_seed=training_seed,
            training_step=training_step,
            eta_metrics=tuple(metrics),
        )
    )


def select_checkpoints_by_seed(
    reports: Iterable[Mapping[str, object]],
) -> dict[int, CheckpointSelection]:
    """Select one checkpoint per seed while retaining every seed as a replicate."""

    grouped: dict[int, list[CheckpointValidation]] = {}
    for payload in reports:
        candidate = build_candidate_from_report(payload)
        grouped.setdefault(candidate.training_seed, []).append(candidate)
    if not grouped:
        raise ValueError("no checkpoint validation reports were provided")
    return {
        seed: select_checkpoint(candidates, expected_seed=seed)
        for seed, candidates in sorted(grouped.items())
    }


def _validate_candidate(candidate: CheckpointValidation) -> CheckpointValidation:
    if not isinstance(candidate.checkpoint_id, str) or not candidate.checkpoint_id.strip():
        raise ValueError("checkpoint_id must be non-empty")
    _require_int("training_seed", candidate.training_seed, minimum=0)
    _require_int("training_step", candidate.training_step, minimum=0)

    expected_etas = tuple(float(value) for value in REGISTERED_ETA_LEVELS)
    observed_etas = tuple(sorted(float(item.eta) for item in candidate.eta_metrics))
    if observed_etas != expected_etas:
        raise ValueError(
            f"eta_metrics must contain exactly registered levels {expected_etas}, got {observed_etas}"
        )

    for item in candidate.eta_metrics:
        if tuple(item.validation_years) != VALIDATION_YEARS:
            if any(year in LOCKED_FINAL_TEST_YEARS for year in item.validation_years):
                raise ValueError("station final-test years are forbidden in checkpoint selection")
            raise ValueError(
                f"validation_years must be exactly {VALIDATION_YEARS}, got {item.validation_years}"
            )
        if not 0.0 < float(item.eta) <= 1.0:
            raise ValueError("eta must lie in (0, 1]")
        _finite("lcvar_retention", item.lcvar_retention)
        if item.lcvar_retention < 0.0:
            raise ValueError("lcvar_retention must be >= 0")
        _finite("minimum_retention", item.minimum_retention)
        if item.minimum_retention < 0.0:
            raise ValueError("minimum_retention must be >= 0")
        _finite("mean_total_irrigation_mm", item.mean_total_irrigation_mm)
        if item.mean_total_irrigation_mm < 45.0 - _EPS:
            raise ValueError("mean_total_irrigation_mm cannot be below fixed 45-mm preplant water")
    return candidate


def _feasible_sort_key(candidate: CheckpointValidation) -> tuple[float, float, float, int, str]:
    return (
        candidate.mean_total_irrigation_mm,
        -candidate.minimum_margin,
        -candidate.mean_margin,
        candidate.training_step,
        candidate.checkpoint_id,
    )


def _fallback_sort_key(candidate: CheckpointValidation) -> tuple[float, float, float, float, int, str]:
    return (
        candidate.worst_shortfall,
        candidate.mean_shortfall,
        candidate.mean_total_irrigation_mm,
        -candidate.minimum_margin,
        candidate.training_step,
        candidate.checkpoint_id,
    )


def _finite(name: str, value: object) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite numeric")
    return float(value)


def _require_int(name: str, value: object, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


__all__ = [
    "EVALUATION_CONTRACT_ID",
    "EtaValidationMetrics",
    "CheckpointValidation",
    "CheckpointSelection",
    "LOCKED_FINAL_TEST_YEARS",
    "REQUIRED_VALIDATION_CELL_COUNT",
    "VALIDATION_YEARS",
    "build_candidate_from_report",
    "select_checkpoint",
    "select_checkpoints_by_seed",
]
