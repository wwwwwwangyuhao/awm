from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from awm.dssat.runtime_assets import (
    CUSTOM_DSSAT_BUILD_LABEL,
    ERA5_WEATHER_FILENAMES,
    STATION_WEATHER_FILENAMES,
    prepare_worker_from_template,
    render_dssatpro,
    validate_versioned_template,
)


def _fake_template(root: Path) -> Path:
    template = root / "t"
    files = {
        "dscsm048": "#!/bin/sh\nexit 0\n",
        "DATA.CDE": "DATA\n",
        "DETAIL.CDE": "DETAIL\n",
        "SIMULATION.CDE": "SIM\n",
        "Genotype/COGRO048.CUL": "CUL\n",
        "Genotype/COGRO048.ECO": "ECO\n",
        "Genotype/COGRO048.SPE": "SPE\n",
        "StandardData/CO2048.WDA": "WDA\n",
        "StandardData/FERCH048.SDA": "SDA\n",
        "data/soil/SOIL.SOL": "SOIL\n",
    }
    for relative, content in files.items():
        path = template / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    for name in ERA5_WEATHER_FILENAMES:
        path = template / "data" / "wth" / "era5" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{name}\n", encoding="utf-8")
    for name in STATION_WEATHER_FILENAMES:
        path = template / "data" / "wth" / "station" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{name}\n", encoding="utf-8")

    (template / "ASSET_MANIFEST.json").write_text(
        json.dumps(
            {
                "simulator": {
                    "base_version": "4.8.5",
                    "build_label": CUSTOM_DSSAT_BUILD_LABEL,
                }
            }
        ),
        encoding="utf-8",
    )
    return template


def test_render_dssatpro_contains_only_worker_local_paths():
    workspace = Path("/tmp/awmrt/w0")
    profile = render_dssatpro(workspace)

    assert "/tmp/awmrt/w0" in profile
    assert "lrmb" not in profile.lower()
    assert "ppo" not in profile.lower()
    assert "dscsm048 CRGRO048" in profile


def test_validate_versioned_template_requires_complete_weather_inventory(tmp_path):
    template = _fake_template(tmp_path)
    assert validate_versioned_template(template)["status"] == "passed"

    (template / "data" / "wth" / "era5" / "XJHX0001.WTH").unlink()
    with pytest.raises(ValueError, match="ERA5 weather inventory mismatch"):
        validate_versioned_template(template)


def test_prepare_worker_isolated_and_profile_passes_a80():
    with tempfile.TemporaryDirectory(prefix="aw_", dir="/tmp") as root_name:
        root = Path(root_name)
        template = _fake_template(root)
        worker = root / "w"

        report = prepare_worker_from_template(template, worker)

        assert report["status"] == "passed"
        assert (worker / "dscsm048").is_file()
        assert (worker / "DSSATPRO.L48").is_file()
        assert report["fixed_width_preflight"]["status"] == "passed"
        assert (worker / "data" / "output").is_dir()

        profile = (worker / "DSSATPRO.L48").read_text(encoding="utf-8")
        assert str(worker.resolve()) in profile
        assert str(template.resolve()) not in profile


def test_prepare_worker_refuses_implicit_overwrite():
    with tempfile.TemporaryDirectory(prefix="aw_", dir="/tmp") as root_name:
        root = Path(root_name)
        template = _fake_template(root)
        worker = root / "w"
        worker.mkdir()

        with pytest.raises(FileExistsError):
            prepare_worker_from_template(template, worker)
