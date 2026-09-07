"""DSSAT Summary.OUT IRCM reporting semantics.

Management irrigation is written on a decimal execution grid, while DSSAT
4.8.5 Summary.OUT exposes seasonal IRCM as an integer-mm field. Real formal
runs show that an exact half-mm COX total can be printed as either adjacent
integer because DSSAT accumulates the decimal events in binary real arithmetic
before formatting. Values away from the half boundary remain unambiguous.
"""
from __future__ import annotations

from decimal import Decimal, ROUND_FLOOR, ROUND_HALF_UP


def _decimal(value: float) -> Decimal:
    return Decimal(str(float(value)))


def snap_to_execution_grid(value: float, *, resolution_mm: float) -> Decimal:
    if value < 0 or resolution_mm <= 0:
        raise ValueError("value must be nonnegative and resolution_mm positive")
    q = _decimal(resolution_mm)
    units = (_decimal(value) / q).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    return units * q


def accepted_summary_ircm_values(
    value: float,
    *,
    execution_resolution_mm: float,
    reporting_resolution_mm: float = 1.0,
) -> tuple[float, ...]:
    """Return the strict set of valid integer-domain Summary IRCM reports.

    The exact ledger is first restored to the management execution grid. Below
    a reporting half-bin only the lower integer is valid; above it only the
    upper integer is valid. At an exact half-bin both neighbors are accepted,
    matching observed formal DSSAT runs (e.g. 494.5 -> 495 and 539.5 -> 539).
    """
    if reporting_resolution_mm <= 0:
        raise ValueError("reporting_resolution_mm must be positive")
    exact = snap_to_execution_grid(value, resolution_mm=execution_resolution_mm)
    q = _decimal(reporting_resolution_mm)
    scaled = exact / q
    lower_units = scaled.to_integral_value(rounding=ROUND_FLOOR)
    fraction = scaled - lower_units
    half = Decimal("0.5")
    if fraction < half:
        candidates = (lower_units * q,)
    elif fraction > half:
        candidates = ((lower_units + 1) * q,)
    else:
        candidates = (lower_units * q, (lower_units + 1) * q)
    return tuple(float(item) for item in candidates)


def nearest_summary_ircm_value(actual: float, accepted: tuple[float, ...]) -> float:
    if not accepted:
        raise ValueError("accepted summary values cannot be empty")
    return min(accepted, key=lambda candidate: (abs(float(actual) - candidate), candidate))


__all__ = [
    "accepted_summary_ircm_values",
    "nearest_summary_ircm_value",
    "snap_to_execution_grid",
]
