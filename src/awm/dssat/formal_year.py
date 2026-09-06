"""Strict year materialization for the frozen AWM agricultural COX protocol.

The canonical protocol is stored once for year 2000.  Development experiments
must preserve agronomic day-of-year management while changing only calendar-
year-dependent DSSAT fields.  In particular, ``ID_SOIL`` remains XJHX0001
because that is the profile identifier in SOIL.SOL; only the weather-station
field (WSTA) follows the year-specific WTH filename.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


FORMAL_SOURCE_YEAR = 2000
FORMAL_SOURCE_STATION = "XJHX0001"
FORMAL_IRRIGATION_MARKER = "{{AWM_IRRIGATION_EVENTS}}"
DEVELOPMENT_YEARS = tuple(range(2000, 2023))
AVAILABLE_ERA5_YEARS = tuple(range(2000, 2026))

_DATE_TOKEN_RE = re.compile(r"(?<!\d)(?P<yy>\d{2})(?P<doy>\d{3})(?!\d)")
_FIELD_ROW_RE = re.compile(
    r"^(?P<prefix>\s*1\s+\S+\s+)(?P<wsta>\S+)(?P<suffix>\s+.*)$"
)


@dataclass(frozen=True, slots=True)
class FormalYearMaterialization:
    target_year: int
    station_id: str
    weather_filename: str
    plant_yrdoy: str
    text: str
    date_token_count: int
    field_wsta_replacement_count: int
    note_replacement_count: int
    label_year_replacement_count: int


def weather_station_id(year: int) -> str:
    _validate_available_year(year)
    return f"XJHX{year % 100:02d}01"


def weather_filename(year: int) -> str:
    return weather_station_id(year) + ".WTH"


def materialize_formal_cox_year(
    source_text: str,
    *,
    target_year: int,
    source_year: int = FORMAL_SOURCE_YEAR,
    source_station: str = FORMAL_SOURCE_STATION,
) -> FormalYearMaterialization:
    """Instantiate one calendar year without changing protocol DOYs.

    Safeguards:
    - the irrigation insertion marker must occur exactly once;
    - only valid standalone YYDDD date tokens carrying ``source_year`` are
      rewritten;
    - in ``*FIELDS`` only WSTA changes; ID_FIELD and ID_SOIL are untouched;
    - human-readable year labels are changed only in EXP.DETAILS and the GE
      simulation-control row, never by a global numeric replacement.
    """

    _validate_available_year(target_year)
    if source_year < 2000 or source_year > 2099:
        raise ValueError("source_year must lie in 2000..2099")
    if source_text.count(FORMAL_IRRIGATION_MARKER) != 1:
        raise ValueError("formal COX must contain exactly one irrigation marker")

    source_yy = f"{source_year % 100:02d}"
    target_yy = f"{target_year % 100:02d}"
    target_station = weather_station_id(target_year)

    date_count = 0

    def replace_date(match: re.Match[str]) -> str:
        nonlocal date_count
        if match.group("yy") != source_yy:
            return match.group(0)
        doy = int(match.group("doy"))
        if not 1 <= doy <= 366:
            raise ValueError(
                f"invalid source YYDDD token {match.group(0)!r} in formal COX"
            )
        date_count += 1
        return f"{target_yy}{doy:03d}"

    date_materialized = _DATE_TOKEN_RE.sub(replace_date, source_text)

    lines = date_materialized.splitlines(keepends=True)
    in_fields = False
    field_wsta_count = 0
    note_count = 0
    label_year_count = 0
    expect_note_value = False
    output: list[str] = []

    for raw_line in lines:
        newline = "\n" if raw_line.endswith("\n") else ""
        line = raw_line[:-1] if newline else raw_line
        stripped = line.strip()

        if line.startswith("*FIELDS"):
            in_fields = True
        elif line.startswith("*") and not line.startswith("*FIELDS"):
            in_fields = False

        if in_fields and stripped.startswith("1 "):
            match = _FIELD_ROW_RE.match(line)
            if match is not None and match.group("wsta") == source_station:
                line = (
                    match.group("prefix")
                    + target_station
                    + match.group("suffix")
                )
                field_wsta_count += 1

        if expect_note_value and stripped:
            parts = line.split()
            if parts and parts[0] == source_station:
                parts[0] = target_station
                prefix_len = len(line) - len(line.lstrip())
                line = line[:prefix_len] + " ".join(parts)
                note_count += 1
            expect_note_value = False
        if line.startswith("@NOTE"):
            expect_note_value = True

        if line.startswith("*EXP.DETAILS:") or re.match(r"^\s*1\s+GE\s", line):
            replaced, count = re.subn(
                rf"\b{source_year}\b",
                str(target_year),
                line,
            )
            line = replaced
            label_year_count += count

        output.append(line + newline)

    result = "".join(output)
    if field_wsta_count != 1:
        raise ValueError(
            "formal COX must contain exactly one *FIELDS WSTA replacement; "
            f"found {field_wsta_count}"
        )
    if note_count != 1:
        raise ValueError(
            "formal COX must contain exactly one @NOTE station replacement; "
            f"found {note_count}"
        )
    if label_year_count != 2:
        raise ValueError(
            "formal COX must contain exactly two human-readable source-year "
            f"labels; found {label_year_count}"
        )
    if date_count == 0:
        raise ValueError("formal COX contains no source-year YYDDD date tokens")

    if target_year != source_year:
        leftovers = [
            token
            for token in _standalone_date_tokens(result)
            if token[:2] == source_yy and 1 <= int(token[2:]) <= 366
        ]
        if leftovers:
            raise ValueError(
                "source-year date tokens remained after materialization: "
                + ", ".join(sorted(set(leftovers)))
            )

    plant_yrdoy = f"{target_yy}119"
    return FormalYearMaterialization(
        target_year=target_year,
        station_id=target_station,
        weather_filename=target_station + ".WTH",
        plant_yrdoy=plant_yrdoy,
        text=result,
        date_token_count=date_count,
        field_wsta_replacement_count=field_wsta_count,
        note_replacement_count=note_count,
        label_year_replacement_count=label_year_count,
    )


def _standalone_date_tokens(text: str) -> Iterable[str]:
    return (match.group(0) for match in _DATE_TOKEN_RE.finditer(text))


def _validate_available_year(year: int) -> None:
    if not isinstance(year, int) or year not in AVAILABLE_ERA5_YEARS:
        raise ValueError(
            "target year must be one of the committed ERA5 years 2000..2025"
        )


__all__ = [
    "AVAILABLE_ERA5_YEARS",
    "DEVELOPMENT_YEARS",
    "FORMAL_IRRIGATION_MARKER",
    "FORMAL_SOURCE_STATION",
    "FORMAL_SOURCE_YEAR",
    "FormalYearMaterialization",
    "materialize_formal_cox_year",
    "weather_filename",
    "weather_station_id",
]
