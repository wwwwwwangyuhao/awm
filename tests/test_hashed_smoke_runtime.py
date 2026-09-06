from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from awm.dssat.runtime_assets import (
    CUSTOM_DSSAT_BUILD_LABEL,
    ERA5_WEATHER_FILENAMES,
    STATION_WEATHER_FILENAMES,
)
from awm.dssat.runtime_paths import runtime_namespace_for_project
from awm.dssat.smoke_runtime import (
    CANONICAL_DAILY_OUT_NAMES,
    discover_project_root,
    prepare_hashed_smoke_worker,
)


def _fake_project(root: Path) -> tuple[Path, Path]:
    project = root / (
        "an_intentionally_long_awm_checkout_name_that_must_not_enter_dssatpro"
    )
    (project / "src" / "awm").mkdir(parents=True)
    (project / "pyproject.toml").write_text("[project]\nname='awm'\n", encoding="utf-8")

    template = project / "dssat_workspace_template"
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
        path.write_text(name + "\n", encoding="utf-8")
    for name in STATION_WEATHER_FILENAMES:
        path = template / "data" / "wth" / "station" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name + "\n", encoding="utf-8")

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

    cox = project / "run" / "real_smoke" / "awm_base.COX.in"
    cox.parent.mkdir(parents=True)
    cox.write_text("{{AWM_IRRIGATION_EVENTS}}\n", encoding="utf-8")

    config = project / "run" / "real_smoke" / "smoke.json"
    return project, config


def _config() -> dict[str, object]:
    return {
        "runtime": {
            "policy_idx": 2,
            "env_idx": 4,
            "replace_worker": True,
        },
        "cox_template": "run/real_smoke/awm_base.COX.in",
        "daily_out_names": list(CANONICAL_DAILY_OUT_NAMES),
        "weather_source": "era5",
        "weather_filename": "XJHX0001.WTH",
        "plant_yrdoy": "00119",
    }


def test_discover_project_root_from_nested_smoke_config():
    with tempfile.TemporaryDirectory(prefix="awsm_", dir="/tmp") as name:
        project, config_path = _fake_project(Path(name))
        config_path.write_text("{}\n", encoding="utf-8")
        assert discover_project_root(config_path) == project.resolve()


def test_prepare_hashed_smoke_worker_uses_short_runtime_not_checkout_path():
    with tempfile.TemporaryDirectory(prefix="awsm_", dir="/tmp") as name:
        root = Path(name)
        project, config_path = _fake_project(root)
        config = _config()
        config_path.write_text(json.dumps(config), encoding="utf-8")
        runtime_base = root / "r"

        worker = prepare_hashed_smoke_worker(
            config_path,
            config,
            runtime_base=runtime_base,
        )

        expected_id = runtime_namespace_for_project(project)
        assert worker.runtime_id == expected_id
        assert worker.workspace == (runtime_base / expected_id / "w" / "p2e4").resolve()
        assert project.name not in str(worker.workspace)
        assert worker.weather_file.name == "XJHX0001.WTH"
        assert worker.soil_file.name == "SOIL.SOL"
        assert tuple(path.name for path in worker.daily_out_files) == CANONICAL_DAILY_OUT_NAMES
        assert len(worker.daily_out_files) == 13
        assert worker.prepare_report["fixed_width_preflight"]["status"] == "passed"

        profile = (worker.workspace / "DSSATPRO.L48").read_text(encoding="utf-8")
        assert project.name not in profile
        assert str(worker.workspace) in profile


def test_canonical_smoke_config_rejects_legacy_explicit_worker_paths():
    with tempfile.TemporaryDirectory(prefix="awsm_", dir="/tmp") as name:
        root = Path(name)
        project, config_path = _fake_project(root)
        config = _config()
        config["workspace"] = str(project / "runtime" / "w0")
        config_path.write_text(json.dumps(config), encoding="utf-8")

        with pytest.raises(KeyError, match="must not hard-code mutable DSSAT runtime"):
            prepare_hashed_smoke_worker(
                config_path,
                config,
                runtime_base=root / "r",
            )


def test_canonical_smoke_requires_exact_13_file_daily_inventory():
    with tempfile.TemporaryDirectory(prefix="awsm_", dir="/tmp") as name:
        root = Path(name)
        _, config_path = _fake_project(root)
        config = _config()
        config["daily_out_names"] = ["PlantGro.OUT", "SoilWat.OUT"]
        config_path.write_text(json.dumps(config), encoding="utf-8")

        with pytest.raises(ValueError, match="canonical 13-file"):
            prepare_hashed_smoke_worker(
                config_path,
                config,
                runtime_base=root / "r",
            )


def test_canonical_smoke_rejects_absolute_cox_template_path():
    with tempfile.TemporaryDirectory(prefix="awsm_", dir="/tmp") as name:
        root = Path(name)
        _, config_path = _fake_project(root)
        config = _config()
        config["cox_template"] = "/tmp/not-portable.COX.in"
        config_path.write_text(json.dumps(config), encoding="utf-8")

        with pytest.raises(ValueError, match="must be project-relative"):
            prepare_hashed_smoke_worker(
                config_path,
                config,
                runtime_base=root / "r",
            )
