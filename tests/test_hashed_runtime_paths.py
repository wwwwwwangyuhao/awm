from __future__ import annotations

import json
from pathlib import Path

import pytest

from awm.dssat.runtime_assets import (
    CUSTOM_DSSAT_BUILD_LABEL,
    ERA5_WEATHER_FILENAMES,
    STATION_WEATHER_FILENAMES,
    prepare_project_worker,
)
from awm.dssat.runtime_paths import (
    DEFAULT_AWM_RUNTIME_BASE,
    WorkspaceRootLock,
    register_dssat_runtime,
    runtime_namespace_for_project,
    runtime_root_for_project,
    worker_workspace_for_project,
)


def _fake_template(root: Path) -> Path:
    template = root / "template"
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
    return template


def test_default_runtime_is_isolated_below_dot_dssat_rt_awm():
    assert DEFAULT_AWM_RUNTIME_BASE.name == "awm"
    assert DEFAULT_AWM_RUNTIME_BASE.parent.name == ".dssat_rt"


def test_project_namespace_is_stable_and_checkout_specific(tmp_path):
    a = tmp_path / "checkout_a"
    b = tmp_path / "checkout_b"
    assert runtime_namespace_for_project(a) == runtime_namespace_for_project(a)
    assert len(runtime_namespace_for_project(a)) == 10
    assert runtime_namespace_for_project(a) != runtime_namespace_for_project(b)


def test_registry_and_runtime_are_confined_to_awm_base(tmp_path):
    runtime_base = tmp_path / ".dssat_rt" / "awm"
    project = tmp_path / "awm_checkout"
    record = register_dssat_runtime(
        project_root=project,
        runtime_base=runtime_base,
    )

    expected_root = runtime_base / runtime_namespace_for_project(project)
    assert Path(record["runtime_root"]) == expected_root.resolve()
    assert (runtime_base / "registry.json").is_file()
    assert not (runtime_base.parent / "registry.json").exists()
    assert (expected_root / "project.json").is_file()


def test_prepare_project_worker_uses_short_hashed_workspace(tmp_path):
    template = _fake_template(tmp_path)
    project = tmp_path / (
        "an_intentionally_very_long_awm_checkout_name_that_must_not_be_written_"
        "into_dssatpro_l48"
    )
    runtime_base = tmp_path / "rt" / "awm"

    report = prepare_project_worker(
        template,
        project_root=project,
        policy_idx=3,
        env_idx=7,
        replace=True,
        runtime_base=runtime_base,
    )

    expected = worker_workspace_for_project(
        project,
        policy_idx=3,
        env_idx=7,
        runtime_base=runtime_base,
    )
    assert Path(str(report["workspace"])) == expected.resolve()
    assert expected.name == "p3e7"
    assert project.name not in str(expected)
    assert report["fixed_width_preflight"]["status"] == "passed"

    profile = (expected / "DSSATPRO.L48").read_text(encoding="utf-8")
    assert str(expected.resolve()) in profile
    assert project.name not in profile


def test_workspace_root_lock_rejects_second_owner(tmp_path):
    project = tmp_path / "awm_checkout"
    runtime_base = tmp_path / "rt" / "awm"
    expected_root = runtime_root_for_project(project, runtime_base=runtime_base)

    with WorkspaceRootLock(project_root=project, runtime_base=runtime_base):
        assert (expected_root / ".workspace.lock").is_file()
        with pytest.raises(RuntimeError, match="already using DSSAT runtime_root"):
            WorkspaceRootLock(
                project_root=project,
                runtime_base=runtime_base,
            ).acquire()
