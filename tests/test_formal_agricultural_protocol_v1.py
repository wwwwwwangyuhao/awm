from __future__ import annotations

import json
from pathlib import Path

from awm.dssat.cox_audit import audit_cox_template
from awm.dssat.management import IRRIGATION_MARKER
from awm.dssat.smoke_runtime import CANONICAL_DAILY_OUT_NAMES


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "agricultural_protocol_v1.json"
COX = ROOT / "run" / "formal" / "awm_protocol_v1_2000.COX.in"
RESET_CONFIG = ROOT / "configs" / "formal_reset_smoke_protocol_v1_2000.json"


def _protocol() -> dict:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def test_protocol_v1_site_calendar_mulch_and_water_are_locked():
    protocol = _protocol()

    assert protocol["protocol_id"] == "awm-agricultural-v1"
    assert protocol["protocol_status"] == "locked_v1_with_provisional_historical_n"

    site = protocol["site"]
    assert site["name"] == "Huaxing Farm"
    assert site["city"] == "Changji"
    assert site["region"] == "Xinjiang"
    assert site["latitude_deg"] == 44.223
    assert site["longitude_deg"] == 87.305
    assert site["soil_id"] == "XJHX0001"

    assert protocol["calendar"]["planting_doy"] == 119
    assert protocol["calendar"]["emergence_doy"] == 133
    assert protocol["calendar"]["decision_horizon_days"] == 125

    assert protocol["mulch"]["PMALB"] == 0.12
    assert protocol["mulch"]["PMWD"] == 22.5
    assert protocol["mulch"]["all_treatments_mulched"] is True

    water = protocol["irrigation"]
    assert water["automatic_irrigation_enabled"] is False
    assert water["management_switch"] == "R"
    assert water["B_ref_total_mm"] == 540.0
    assert water["fixed_preplant_irrigation"] == {
        "timing": "3 days before planting",
        "formal_doy": 116,
        "m3_mu": 30.0,
        "mm": 45.0,
        "policy_controllable": False,
        "source_basis": "present at 30 m3/mu three days before planting in the 2023, 2024 and 2025 Huaxing field COX records",
    }
    assert water["total_water_budget_treatments_mm"] == {
        "W100": 540.0,
        "W80": 432.0,
        "W60": 324.0,
    }
    assert water["policy_water_budget_treatments_mm"] == {
        "W100": 495.0,
        "W80": 387.0,
        "W60": 279.0,
    }
    assert water["max_event_mm"] == 45.0
    assert water["min_positive_execution_mm"] == 0.1
    assert water["execution_resolution_mm"] == 0.1
    assert water["min_interval_days"] == 0
    assert water["nonpolicy_irrigation_mm"] == 45.0
    assert water["ircm_tolerance_mm"] == 0.1


def test_protocol_v1_fixed_n_is_exact_2025_t1_schedule():
    nitrogen = _protocol()["nitrogen"]

    assert nitrogen["rl_action_enabled"] is False
    assert nitrogen["automatic_n_enabled"] is False
    assert nitrogen["management_switch"] == "R"
    assert nitrogen["source_file"] == "data/COX/XJHX2501.COX"
    assert nitrogen["source_treatment"] == "2025_T1"
    assert nitrogen["fertilizer_code"] == "FE010"
    assert nitrogen["application_code"] == "AP005"
    assert nitrogen["FDEP"] == 2

    expected = [
        (161, 8.0),
        (173, 16.0),
        (182, 7.0),
        (190, 21.0),
        (195, 31.0),
        (201, 52.0),
        (212, 20.0),
        (218, 30.0),
        (224, 25.0),
        (231, 25.0),
    ]
    observed = [(event["doy"], event["n_kg_ha"]) for event in nitrogen["events"]]
    assert observed == expected
    assert sum(value for _, value in observed) == 235.0
    assert nitrogen["total_n_kg_ha"] == 235.0


def test_weather_split_is_disjoint_and_station_only_for_final_test():
    weather = _protocol()["weather"]
    training = set(weather["training_years"])
    validation = set(weather["validation_years"])
    final_test = set(weather["final_test_station_years"])

    assert training == set(range(2000, 2018))
    assert validation == set(range(2018, 2023))
    assert final_test == {2023, 2024, 2025}
    assert training.isdisjoint(validation)
    assert training.isdisjoint(final_test)
    assert validation.isdisjoint(final_test)
    assert "models=era5" in weather["historical_source"]
    assert weather["era5_2023_2025_policy"].startswith("noncanonical")


def test_formal_cox_is_protocol_ready_and_has_fixed_preplant_water():
    report = audit_cox_template(
        COX,
        expected_nonpolicy_irrigation_mm=45.0,
        irrigation_tolerance_mm=0.1,
    )
    text = COX.read_text(encoding="utf-8")

    assert report["structural_status"] == "passed"
    assert report["protocol_ready"] is True
    assert report["review_flags"] == []
    assert report["marker_count"] == 1
    assert report["explicit_irrigation_rows"] == [" 1 00116 IR005 45.00"]
    assert report["explicit_nonpolicy_irrigation_total_mm"] == 45.0
    assert report["expected_nonpolicy_irrigation_mm"] == 45.0
    assert report["explicit_fertilizer_n_total_kg_ha"] == 235.0
    assert report["management_switches"]["irrigation"] == "R"
    assert report["management_switches"]["fertilization"] == "R"
    assert report["automatic_irrigation_active"] is False
    assert report["automatic_nitrogen_active"] is False

    assert text.count(IRRIGATION_MARKER) == 1
    assert "HUAXING FARM, CHANGJI, XINJIANG, CHINA" in text
    assert "44.223;87.305" in text
    assert "JIANGDU" not in text.upper()
    assert "JIANGSU" not in text.upper()
    assert " 1 CO IB0007 IB0007_CALIBRATED" in text
    assert " 1  0.12  22.5" in text
    assert " 1 00119 00133" in text
    assert text.count("IR005") == 1


def test_formal_cox_preserves_initial_water_and_n_profile():
    text = COX.read_text(encoding="utf-8")
    expected_rows = [
        " 1     5  0.37    16     8",
        " 1    10  0.36    16     8",
        " 1    20  0.35     8     4",
        " 1    30  0.38     4     2",
        " 1    50  0.38   0.8   0.4",
        " 1    70  0.38  0.16  0.08",
        " 1   100  0.38  0.03 0.016",
        " 1   130  0.38 0.006 0.001",
        " 1   160  0.38 0.001 0.001",
        " 1   190  0.38     0     0",
    ]
    for row in expected_rows:
        assert row in text


def test_formal_reset_config_is_portable_locked_and_accounts_for_preplant_water():
    config = json.loads(RESET_CONFIG.read_text(encoding="utf-8"))

    assert config["status"] == "formal_agricultural_protocol_v1_reset_smoke"
    assert config["formal_protocol_locked"] is True
    assert config["protocol_id"] == "awm-agricultural-v1"
    assert config["cox_template"] == "run/formal/awm_protocol_v1_2000.COX.in"
    assert config["weather_source"] == "era5"
    assert config["weather_filename"] == "XJHX0001.WTH"
    assert config["plant_yrdoy"] == "00119"
    assert config["nonpolicy_irrigation_mm"] == 45.0
    assert config["ircm_tolerance_mm"] == 0.1
    assert tuple(config["daily_out_names"]) == CANONICAL_DAILY_OUT_NAMES

    forbidden = {
        "workspace",
        "dssat_exec",
        "output_dir",
        "rendered_cox",
        "summary_out",
        "daily_out_files",
    }
    assert forbidden.isdisjoint(config)
