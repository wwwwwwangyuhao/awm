"""DSSAT date helpers with explicit calendar-year semantics."""
from __future__ import annotations

from datetime import date, datetime, timedelta


def is_leap_year(year: int) -> bool:
    year = int(year)
    return year % 400 == 0 or (year % 4 == 0 and year % 100 != 0)


def doy_to_date(doy: int | float | str, calendar_year: int) -> str:
    year = int(calendar_year)
    day = int(float(doy))
    max_doy = 366 if is_leap_year(year) else 365
    if not 1 <= day <= max_doy:
        raise ValueError(
            f"DOY {day} is invalid for calendar year {year}; "
            f"expected 1..{max_doy}."
        )
    return (datetime(year, 1, 1) + timedelta(days=day - 1)).strftime(
        "%Y-%m-%d"
    )


def yy_doy(calendar_year: int, doy: int) -> str:
    year = int(calendar_year)
    day = int(doy)
    max_doy = 366 if is_leap_year(year) else 365
    if not 1 <= day <= max_doy:
        raise ValueError(
            f"DOY {day} is invalid for calendar year {year}; "
            f"expected 1..{max_doy}."
        )
    return f"{year % 100:02d}{day:03d}"


def parse_yyddd(value: str) -> tuple[int, int]:
    text = str(value).strip()
    if len(text) != 5 or not text.isdigit():
        raise ValueError(f"Expected DSSAT YYDDD date, got {value!r}.")
    year = 2000 + int(text[:2])
    doy = int(text[2:])
    max_doy = 366 if is_leap_year(year) else 365
    if not 1 <= doy <= max_doy:
        raise ValueError(f"Invalid DSSAT YYDDD date: {value!r}.")
    return year, doy


__all__ = ["is_leap_year", "doy_to_date", "yy_doy", "parse_yyddd"]
