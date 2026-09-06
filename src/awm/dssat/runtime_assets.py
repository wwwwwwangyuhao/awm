"""Self-contained DSSAT runtime asset management for AWM.

The versioned template lives in ``dssat_workspace_template/``. Mutable workers
are copied from that template into an ignored runtime directory. The template
must never be modified during an episode.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from .workspace import validate_dssatpro_record_width

CUSTOM_DSSAT_BASE_VERSION = "4.8.5"
CUSTOM_DSSAT_BUILD_LABEL = "lab-dssat-4.8.5-mulch"
CUSTOM_DSSAT_EXECUTABLE = "dscsm048"

ERA5_WEATHER_FILENAMES = tuple(
    f"XJHX{year:02d}01.WTH" for year in range(2000 - 2000, 2025 - 2000 + 1)
)
STATION_WEATHER_FILENAMES = (
    "XJHX2301.WTH",
    "XJHX2401.WTH",
    "XJHX2501.WTH",
)

_REQUIRED_TEMPLATE_FILES = (
    "ASSET_MANIFEST.json",
    CUSTOM_DSSAT_EXECUTABLE,
    "DATA.CDE",
    "DETAIL.CDE",
    "SIMULATION.CDE",
    "Genotype/COGRO048.CUL",
    "Genotype/COGRO048.ECO",
    "Genotype/COGRO048.SPE",
    "StandardData/CO2048.WDA",
    "StandardData/FERCH048.SDA",
    "data/soil/SOIL.SOL",
)


def render_dssatpro(workspace: str | Path) -> str:
    """Render a worker-local DSSATPRO.L48 for the custom cotton build."""
    root = Path(workspace).resolve()
    return (
        "*** *  DSSAT PROFILE LINUX* ***\n\n"
        f"MCO // {root} dscsm048 CRGRO048\n\n"
        f"STD // {root / 'StandardData'}\n"
        f"CRD // {root / 'Genotype'}\n\n"
        f"ASD // {root / 'Seasonal'}\n"
        f"AQD // {root / 'Sequence'}\n"
        f"APD // {root / 'Spatial'}\n"
        f"PSD // {root / 'Pest'}\n"
        f"ECD // {root / 'Economic'}\n"
    )


def validate_versioned_template(template_dir: str | Path) -> dict[str, object]:
    """Validate the immutable AWM DSSAT template inventory."""
    root = Path(template_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"DSSAT template directory not found: {root}")

    missing = [name for name in _REQUIRED_TEMPLATE_FILES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "AWM DSSAT template is incomplete; missing: " + ", ".join(missing)
        )

    era5_dir = root / "data" / "wth" / "era5"
    station_dir = root / "data" / "wth" / "station"
    era5 = tuple(sorted(path.name for path in era5_dir.glob("*.WTH")))
    station = tuple(sorted(path.name for path in station_dir.glob("*.WTH")))
    expected_era5 = tuple(sorted(ERA5_WEATHER_FILENAMES))
    expected_station = tuple(sorted(STATION_WEATHER_FILENAMES))
    if era5 != expected_era5:
        raise ValueError(
            "ERA5 weather inventory mismatch: "
            f"expected={expected_era5}, actual={era5}"
        )
    if station != expected_station:
        raise ValueError(
            "Station weather inventory mismatch: "
            f"expected={expected_station}, actual={station}"
        )

    manifest = json.loads((root / "ASSET_MANIFEST.json").read_text(encoding="utf-8"))
    simulator = manifest.get("simulator", {})
    if simulator.get("base_version") != CUSTOM_DSSAT_BASE_VERSION:
        raise ValueError(
            "Unexpected DSSAT base version in ASSET_MANIFEST.json: "
            f"{simulator.get('base_version')!r}"
        )
    if simulator.get("build_label") != CUSTOM_DSSAT_BUILD_LABEL:
        raise ValueError(
            "Unexpected DSSAT build label in ASSET_MANIFEST.json: "
            f"{simulator.get('build_label')!r}"
        )

    return {
        "template": str(root),
        "dssat_exec": str(root / CUSTOM_DSSAT_EXECUTABLE),
        "era5_weather_count": len(era5),
        "station_weather_count": len(station),
        "status": "passed",
    }


def prepare_worker_from_template(
    template_dir: str | Path,
    workspace: str | Path,
    *,
    replace: bool = False,
) -> dict[str, object]:
    """Create one mutable DSSAT worker entirely from the AWM-owned template."""
    template = Path(template_dir).resolve()
    worker = Path(workspace).resolve()
    validate_versioned_template(template)

    if worker == template or template in worker.parents:
        raise ValueError("Worker workspace must not be inside the immutable template")

    if worker.exists():
        if not replace:
            raise FileExistsError(
                f"Worker already exists: {worker}. Pass replace=True explicitly."
            )
        shutil.rmtree(worker)

    worker.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(template, worker)

    for relative in (
        "Seasonal",
        "Sequence",
        "Spatial",
        "Pest",
        "Economic",
        "data/COX",
        "data/irrigation",
        "data/fertilizer",
        "data/output",
    ):
        (worker / relative).mkdir(parents=True, exist_ok=True)

    executable = worker / CUSTOM_DSSAT_EXECUTABLE
    executable.chmod(executable.stat().st_mode | 0o111)

    profile = worker / "DSSATPRO.L48"
    profile.write_text(render_dssatpro(worker), encoding="utf-8", newline="\n")
    width_report = validate_dssatpro_record_width(worker)

    return {
        "template": str(template),
        "workspace": str(worker),
        "dssat_exec": str(executable),
        "dssatpro": str(profile),
        "fixed_width_preflight": width_report,
        "status": "passed",
    }


__all__ = [
    "CUSTOM_DSSAT_BASE_VERSION",
    "CUSTOM_DSSAT_BUILD_LABEL",
    "CUSTOM_DSSAT_EXECUTABLE",
    "ERA5_WEATHER_FILENAMES",
    "STATION_WEATHER_FILENAMES",
    "prepare_worker_from_template",
    "render_dssatpro",
    "validate_versioned_template",
]
