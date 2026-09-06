from pathlib import Path
import re

import pytest

from awm.dssat.formal_year import (
    DEVELOPMENT_YEARS,
    materialize_formal_cox_year,
    weather_filename,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "run" / "formal" / "awm_protocol_v1_2000.COX.in"


def source_text():
    return SOURCE.read_text(encoding="utf-8")


def test_year_2000_materialization_is_exact_identity():
    source = source_text()
    result = materialize_formal_cox_year(source, target_year=2000)
    assert result.text == source
    assert result.station_id == "XJHX0001"
    assert result.weather_filename == "XJHX0001.WTH"
    assert result.plant_yrdoy == "00119"
    assert result.date_token_count == 18
    assert result.field_wsta_replacement_count == 1
    assert result.note_replacement_count == 1
    assert result.label_year_replacement_count == 2


def test_year_2001_changes_weather_and_calendar_but_not_soil_profile_id():
    result = materialize_formal_cox_year(source_text(), target_year=2001)
    text = result.text

    assert result.station_id == "XJHX0101"
    assert result.weather_filename == "XJHX0101.WTH"
    assert result.plant_yrdoy == "01119"
    assert "*EXP.DETAILS: HUAXING FARM AWM PROTOCOL V1 2001" in text
    assert "XJHX0101 AWM_PROTOCOL_V1" in text

    field_row = next(
        line
        for line in text.splitlines()
        if line.strip().startswith("1 XJHX0001 XJHX0101")
    )
    fields = field_row.split()
    assert fields[1] == "XJHX0001"  # fixed field identifier
    assert fields[2] == "XJHX0101"  # year-specific weather station/file id
    assert fields[11] == "XJHX0001"  # fixed SOIL.SOL profile identifier

    for doy in (103, 116, 119, 133, 161, 173, 201, 231, 288):
        assert f"01{doy:03d}" in text
    assert "HUAXING AWM V1 2001" in text
    assert text.count("{{AWM_IRRIGATION_EVENTS}}") == 1


def test_materialization_preserves_all_source_management_doys():
    source = source_text()
    target = materialize_formal_cox_year(source, target_year=2022).text

    source_dates = re.findall(r"(?<!\d)00(\d{3})(?!\d)", source)
    target_dates = re.findall(r"(?<!\d)22(\d{3})(?!\d)", target)
    assert target_dates == source_dates
    assert len(target_dates) == 18


def test_development_years_exclude_locked_2023_2025_final_test():
    assert DEVELOPMENT_YEARS == tuple(range(2000, 2023))
    assert 2023 not in DEVELOPMENT_YEARS
    assert 2024 not in DEVELOPMENT_YEARS
    assert 2025 not in DEVELOPMENT_YEARS
    assert weather_filename(2022) == "XJHX2201.WTH"


def test_materializer_rejects_missing_or_duplicate_irrigation_marker():
    source = source_text()
    with pytest.raises(ValueError, match="exactly one irrigation marker"):
        materialize_formal_cox_year(
            source.replace("{{AWM_IRRIGATION_EVENTS}}", ""),
            target_year=2001,
        )
    with pytest.raises(ValueError, match="exactly one irrigation marker"):
        materialize_formal_cox_year(
            source + "\n{{AWM_IRRIGATION_EVENTS}}\n",
            target_year=2001,
        )


def test_materializer_rejects_year_without_committed_era5_asset():
    with pytest.raises(ValueError, match="2000..2025"):
        materialize_formal_cox_year(source_text(), target_year=1999)
    with pytest.raises(ValueError, match="2000..2025"):
        materialize_formal_cox_year(source_text(), target_year=2026)
