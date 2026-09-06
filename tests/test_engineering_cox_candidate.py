from __future__ import annotations

import json
from pathlib import Path

from awm.dssat.cox_audit import audit_cox_template
from awm.dssat.management import IRRIGATION_MARKER
from awm.dssat.smoke_runtime import CANONICAL_DAILY_OUT_NAMES


ROOT = Path(__file__).resolve().parents[1]
COX = ROOT / "run" / "real_smoke" / "v3_2000_engineering.COX.in"
CONFIG = ROOT / "configs" / "engineering_reset_smoke_v3_2000.json"
PROVENANCE = ROOT / "run" / "real_smoke" / "v3_2000_engineering.provenance.json"


def test_engineering_cox_is_structurally_valid_but_not_protocol_ready():
    report = audit_cox_template(COX)

    assert report["structural_status"] == "passed"
    assert report["marker_count"] == 1
    assert report["explicit_irrigation_rows"] == []
    assert report["protocol_ready"] is False

    flags = set(report["review_flags"])
    assert (
        "descriptive_site_metadata_mentions_JIANGDU_while_field_weather_soil_use_XJHX"
        in flags
    )
    assert "explicit_fertilizer_n_total_is_zero" in flags
    assert (
        "automatic_irrigation_settings_present_verify_management_control_semantics"
        in flags
    )
    assert (
        "automatic_nitrogen_settings_present_verify_management_control_semantics"
        in flags
    )


def test_engineering_cox_preserves_v3_candidate_parameters():
    text = COX.read_text(encoding="utf-8")

    assert text.count(IRRIGATION_MARKER) == 1
    assert " 1 CO IB0007 IRRIGATION_ONLY" in text
    assert " 1 XJHX0001 XJHX0001" in text
    assert " 1  0.12  22.5" in text
    assert " 1 00119 00133" in text
    assert " 1 00119 FE010 AP005     2  0.00" in text
    assert "IR005" not in text


def test_engineering_reset_config_is_portable_and_uses_canonical_inventory():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))

    assert config["status"] == "engineering_smoke_only_not_formal_protocol"
    assert config["cox_template"] == "run/real_smoke/v3_2000_engineering.COX.in"
    assert config["weather_source"] == "era5"
    assert config["weather_filename"] == "XJHX0001.WTH"
    assert config["plant_yrdoy"] == "00119"
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


def test_engineering_candidate_references_assets_vendored_in_awm():
    assert (ROOT / "dssat_workspace_template" / "data" / "wth" / "era5" / "XJHX0001.WTH").is_file()

    soil = (
        ROOT
        / "dssat_workspace_template"
        / "data"
        / "soil"
        / "SOIL.SOL"
    ).read_text(encoding="utf-8")
    assert "*XJHX0001" in soil

    cultivar = (
        ROOT
        / "dssat_workspace_template"
        / "Genotype"
        / "COGRO048.CUL"
    ).read_text(encoding="utf-8")
    assert "IB0007" in cultivar


def test_provenance_explicitly_blocks_formal_use():
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))

    assert provenance["formal_protocol_locked"] is False
    assert provenance["source"]["commit"] == "d56336e09fdb9a9aea60ae61eaa892833314ab33"
    assert provenance["engineering_instantiation"]["weather_year"] == 2000
    assert len(provenance["known_review_blockers"]) >= 4
