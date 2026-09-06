"""Frozen yield-retention risk semantics for the first AWM learned methods.

The formal random variable is weather-normalized yield retention

    R_pi(w) = Y_pi(w) / Y_ref(w)

where Y_ref(w) is the frozen conventional-W100 reference yield under the same
weather year.  The lower-tail risk measure uses alpha as *tail probability
mass*, not as a confidence level.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import mean
from typing import Iterable, Mapping, Sequence


RISK_CONTRACT_ID = "awm-risk-contract-v1"
DEFAULT_ALPHA = 0.20
REGISTERED_ETA_LEVELS = (0.90, 0.95, 0.98)
TRAIN_YEARS = tuple(range(2000, 2018))
VALIDATION_YEARS = tuple(range(2018, 2023))
DEVELOPMENT_YEARS = TRAIN_YEARS + VALIDATION_YEARS
LOCKED_FINAL_TEST_YEARS = (2023, 2024, 2025)
REFERENCE_METHOD = "conventional"
REFERENCE_TREATMENT = "W100"
_EPS = 1e-12


@dataclass(frozen=True, slots=True)
class RiskEvaluation:
    """Auditable lower-tail evaluation for one eta on one weather split."""

    n: int
    alpha: float
    eta: float
    mean_retention: float
    minimum_retention: float
    lcvar: float
    margin: float
    feasible: bool


def yield_retention(yield_kg_ha: float, reference_yield_kg_ha: float) -> float:
    """Return Y/Y_ref without clipping values above one."""

    y = _finite("yield_kg_ha", yield_kg_ha)
    ref = _finite("reference_yield_kg_ha", reference_yield_kg_ha)
    if y < 0.0:
        raise ValueError("yield_kg_ha must be >= 0")
    if ref <= 0.0:
        raise ValueError("reference_yield_kg_ha must be > 0")
    return y / ref


def empirical_lower_cvar(
    values: Iterable[float],
    *,
    alpha: float = DEFAULT_ALPHA,
) -> float:
    """Average the worst ``alpha`` probability mass of equally weighted values.

    For n observations sorted x_(1) <= ... <= x_(n), let m=alpha*n,
    k=floor(m), and f=m-k.  The estimator is

        (sum_{i=1}^k x_(i) + f*x_(k+1)) / m,

    with the fractional term omitted when f=0.  This exact convention avoids
    ambiguity from quantile interpolation libraries and remains well-defined
    when alpha*n < 1 (the result is then the minimum observation).
    """

    a = _validate_alpha(alpha)
    ordered = sorted(_finite("retention", value) for value in values)
    if not ordered:
        raise ValueError("at least one retention value is required")

    mass = a * len(ordered)
    full_count = int(math.floor(mass + _EPS))
    full_count = min(full_count, len(ordered))
    fractional = mass - full_count
    if abs(fractional) <= _EPS:
        fractional = 0.0

    total = sum(ordered[:full_count])
    if fractional > 0.0:
        if full_count >= len(ordered):
            raise AssertionError("fractional lower-tail weight exceeds sample")
        total += fractional * ordered[full_count]
    return total / mass


def evaluate_retention_risk(
    retentions: Sequence[float],
    *,
    eta: float,
    alpha: float = DEFAULT_ALPHA,
) -> RiskEvaluation:
    """Evaluate the formal constraint LCVaR_alpha(R) >= eta."""

    target = _validate_eta(eta)
    values = tuple(_finite("retention", value) for value in retentions)
    if not values:
        raise ValueError("at least one retention value is required")
    lcvar = empirical_lower_cvar(values, alpha=alpha)
    margin = lcvar - target
    return RiskEvaluation(
        n=len(values),
        alpha=float(alpha),
        eta=target,
        mean_retention=mean(values),
        minimum_retention=min(values),
        lcvar=lcvar,
        margin=margin,
        feasible=margin >= -_EPS,
    )


def build_development_reference_map(
    rows: Sequence[Mapping[str, object]],
) -> dict[int, float]:
    """Extract exactly one conventional-W100 reference for every 2000-2022 year.

    This helper deliberately rejects final-test years.  Station references are
    generated only by the later locked final-evaluation path.
    """

    result: dict[int, float] = {}
    for row in rows:
        year = int(row["year"])
        if year in LOCKED_FINAL_TEST_YEARS:
            raise ValueError(f"final-test year {year} is forbidden in development reference data")
        if year not in DEVELOPMENT_YEARS:
            continue
        if row.get("method") != REFERENCE_METHOD:
            continue
        if row.get("water_treatment") != REFERENCE_TREATMENT:
            continue
        if row.get("status") != "passed":
            raise ValueError(f"reference row for {year} did not pass")
        if year in result:
            raise ValueError(f"duplicate reference row for {year}")
        value = _finite("HWAM_kg_ha", row["HWAM_kg_ha"])
        if value <= 0.0:
            raise ValueError(f"reference yield for {year} must be > 0")
        result[year] = value

    missing = sorted(set(DEVELOPMENT_YEARS) - set(result))
    if missing:
        raise ValueError(f"missing development Y_ref rows: {missing}")
    return dict(sorted(result.items()))


def normalize_rows_by_reference(
    rows: Sequence[Mapping[str, object]],
    reference_by_year: Mapping[int, float],
) -> list[dict[str, object]]:
    """Copy rows and append un-clipped ``yield_retention`` values."""

    normalized: list[dict[str, object]] = []
    for row in rows:
        year = int(row["year"])
        if year in LOCKED_FINAL_TEST_YEARS:
            raise ValueError(f"final-test year {year} is forbidden in development normalization")
        if year not in reference_by_year:
            raise KeyError(f"missing Y_ref for year {year}")
        payload = dict(row)
        payload["Y_ref_kg_ha"] = float(reference_by_year[year])
        payload["yield_retention"] = yield_retention(
            float(row["HWAM_kg_ha"]),
            float(reference_by_year[year]),
        )
        normalized.append(payload)
    return normalized


def _validate_alpha(alpha: float) -> float:
    value = _finite("alpha", alpha)
    if not 0.0 < value <= 1.0:
        raise ValueError("alpha must lie in (0, 1]")
    return value


def _validate_eta(eta: float) -> float:
    value = _finite("eta", eta)
    if not 0.0 < value <= 1.0:
        raise ValueError("eta must lie in (0, 1]")
    return value


def _finite(name: str, value: float) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite numeric")
    return float(value)


__all__ = [
    "DEFAULT_ALPHA",
    "DEVELOPMENT_YEARS",
    "LOCKED_FINAL_TEST_YEARS",
    "REFERENCE_METHOD",
    "REFERENCE_TREATMENT",
    "REGISTERED_ETA_LEVELS",
    "RISK_CONTRACT_ID",
    "RiskEvaluation",
    "TRAIN_YEARS",
    "VALIDATION_YEARS",
    "build_development_reference_map",
    "empirical_lower_cvar",
    "evaluate_retention_risk",
    "normalize_rows_by_reference",
    "yield_retention",
]
