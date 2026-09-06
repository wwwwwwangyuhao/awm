"""Structural and provenance audit for AWM DSSAT COX templates.

This module deliberately separates two questions:

1. Is a COX file structurally usable by the AWM irrigation renderer?
2. Is its agronomic content sufficiently reviewed to become a formal protocol?

A structurally valid engineering-smoke candidate is not automatically a
protocol-locked AWM experiment template.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from .freeze_template import sha256_file
from .management import IRRIGATION_MARKER


def _tokens(line: str) -> tuple[str, ...]:
    return tuple(line.strip().split())


def _find_unique_header(lines: list[str], prefix: Iterable[str]) -> int:
    expected = tuple(prefix)
    matches = [
        idx
        for idx, line in enumerate(lines)
        if _tokens(line)[: len(expected)] == expected
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one COX header starting with {expected}, got {len(matches)}"
        )
    return matches[0]


def _table_rows(lines: list[str], header_index: int) -> list[str]:
    rows: list[str] = []
    for line in lines[header_index + 1 :]:
        stripped = line.strip()
        if not stripped:
            if rows:
                break
            continue
        if stripped.startswith(("*", "@")):
            break
        if stripped == IRRIGATION_MARKER:
            continue
        if stripped.startswith("!"):
            continue
        rows.append(line.rstrip())
    return rows


def _value_after_label(lines: list[str], label: str) -> str | None:
    for idx, line in enumerate(lines):
        if line.strip() != label:
            continue
        for candidate in lines[idx + 1 :]:
            stripped = candidate.strip()
            if not stripped:
                continue
            if stripped.startswith(("*", "@")):
                return None
            return stripped
    return None


def _safe_rows(lines: list[str], prefix: tuple[str, ...]) -> list[str]:
    try:
        return _table_rows(lines, _find_unique_header(lines, prefix))
    except ValueError:
        return []


def _fertilizer_n_total(rows: list[str]) -> float | None:
    if not rows:
        return None
    values: list[float] = []
    for row in rows:
        parts = row.split()
        if len(parts) < 6:
            return None
        try:
            values.append(float(parts[5]))
        except ValueError:
            return None
    return float(sum(values))


def _management_switches(rows: list[str]) -> dict[str, str]:
    """Parse the DSSAT @N MANAGEMENT switch row.

    DSSAT's SIMULATION.CDE defines the five switches after the row code as
    PLANT, IRRIG, FERTI, RESID and HARVS.  For this AWM protocol, IRRIG=R and
    FERTI=R mean that irrigation and fertilizer are taken only from reported
    explicit management rows.  The AUTOMATIC MANAGEMENT parameter tables may
    remain present in FileX but are inert unless an automatic mode is selected.
    """
    if len(rows) != 1:
        return {}
    parts = rows[0].split()
    if len(parts) < 7:
        return {}
    return {
        "planting": parts[2].upper(),
        "irrigation": parts[3].upper(),
        "fertilization": parts[4].upper(),
        "residue": parts[5].upper(),
        "harvest": parts[6].upper(),
    }


def audit_cox_template(path: str | Path) -> dict[str, object]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    text = source.read_text(encoding="utf-8")
    lines = text.splitlines()

    cultivar_rows = _safe_rows(lines, ("@C", "CR", "INGENO", "CNAME"))
    field_rows = _safe_rows(lines, ("@L", "ID_FIELD", "WSTA...."))
    mulch_rows = _safe_rows(lines, ("@L", "PMALB", "PMWD"))
    planting_rows = _safe_rows(lines, ("@P", "PDATE", "EDATE"))
    irrigation_rows = _safe_rows(lines, ("@I", "IDATE", "IROP", "IRVAL"))
    fertilizer_rows = _safe_rows(lines, ("@F", "FDATE", "FMCD", "FACD"))
    management_rows = _safe_rows(lines, ("@N", "MANAGEMENT", "PLANT", "IRRIG"))
    automatic_irrigation_rows = _safe_rows(
        lines, ("@N", "IRRIGATION", "IMDEP", "ITHRL")
    )
    automatic_nitrogen_rows = _safe_rows(
        lines, ("@N", "NITROGEN", "NMDEP", "NMTHR")
    )
    management_switches = _management_switches(management_rows)

    marker_count = text.count(IRRIGATION_MARKER)
    structural_errors: list[str] = []
    if marker_count != 1:
        structural_errors.append(
            f"AWM irrigation marker count must be exactly one, got {marker_count}"
        )
    if irrigation_rows:
        structural_errors.append(
            "template contains explicit irrigation rows in addition to the AWM marker"
        )
    if not cultivar_rows:
        structural_errors.append("cultivar row not found")
    if not field_rows:
        structural_errors.append("field/weather/soil row not found")
    if not planting_rows:
        structural_errors.append("planting row not found")

    experiment_details = lines[0].strip() if lines else ""
    address = _value_after_label(lines, "@ADDRESS")
    site = _value_after_label(lines, "@SITE")
    note = _value_after_label(lines, "@NOTE")
    fertilizer_n_total = _fertilizer_n_total(fertilizer_rows)

    irrigation_mode = management_switches.get("irrigation")
    fertilization_mode = management_switches.get("fertilization")
    automatic_irrigation_active = irrigation_mode in {"A", "F", "P", "W"}
    automatic_nitrogen_active = fertilization_mode in {"A", "F"}

    review_flags: list[str] = []
    descriptive = " ".join(
        value for value in (experiment_details, address, site) if value
    ).upper()
    field_text = " ".join(field_rows).upper()
    if "JIANGDU" in descriptive and "XJHX" in field_text:
        review_flags.append(
            "descriptive_site_metadata_mentions_JIANGDU_while_field_weather_soil_use_XJHX"
        )
    if fertilizer_n_total is None:
        review_flags.append("explicit_fertilizer_n_schedule_not_parseable")
    elif fertilizer_n_total <= 0.0:
        review_flags.append("explicit_fertilizer_n_total_is_zero")

    if not management_switches:
        review_flags.append("management_switch_row_not_parseable")
    else:
        if irrigation_mode != "R":
            review_flags.append(
                "irrigation_management_switch_must_be_reported_R_for_AWM_policy_rows"
            )
        if fertilization_mode != "R":
            review_flags.append(
                "fertilization_management_switch_must_be_reported_R_for_fixed_N"
            )

    if automatic_irrigation_rows and automatic_irrigation_active:
        review_flags.append("automatic_irrigation_is_active")
    if automatic_nitrogen_rows and automatic_nitrogen_active:
        review_flags.append("automatic_nitrogen_is_active")

    return {
        "path": str(source),
        "sha256": sha256_file(source),
        "marker": IRRIGATION_MARKER,
        "marker_count": marker_count,
        "structural_status": "passed" if not structural_errors else "failed",
        "structural_errors": structural_errors,
        "protocol_ready": not structural_errors and not review_flags,
        "review_flags": review_flags,
        "experiment_details": experiment_details,
        "address": address,
        "site": site,
        "note": note,
        "cultivar_rows": cultivar_rows,
        "field_rows": field_rows,
        "mulch_rows": mulch_rows,
        "planting_rows": planting_rows,
        "explicit_irrigation_rows": irrigation_rows,
        "fertilizer_rows": fertilizer_rows,
        "explicit_fertilizer_n_total_kg_ha": fertilizer_n_total,
        "management_rows": management_rows,
        "management_switches": management_switches,
        "automatic_irrigation_rows": automatic_irrigation_rows,
        "automatic_nitrogen_rows": automatic_nitrogen_rows,
        "automatic_irrigation_active": automatic_irrigation_active,
        "automatic_nitrogen_active": automatic_nitrogen_active,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit a DSSAT COX template without declaring it protocol-ready"
    )
    parser.add_argument("cox")
    parser.add_argument("--output")
    args = parser.parse_args()

    report = audit_cox_template(args.cox)
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()


__all__ = ["audit_cox_template"]
