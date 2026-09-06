"""Canonical hashed-runtime preparation for real AWM DSSAT smoke runs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .runtime_assets import prepare_project_worker

CANONICAL_DAILY_OUT_NAMES = (
    "PlantGro.OUT",
    "SoilWat.OUT",
    "Weather.OUT",
    "ET.OUT",
    "GHG.OUT",
    "MgmtOps.OUT",
    "Mulch.OUT",
    "N2O.OUT",
    "PlantC.OUT",
    "PlantN.OUT",
    "SoilTemp.OUT",
    "SoilWater.OUT",
    "SoilNi.OUT",
)

LEGACY_EXPLICIT_RUNTIME_KEYS = (
    "workspace",
    "dssat_exec",
    "output_dir",
    "rendered_cox",
    "summary_out",
    "daily_out_files",
    "episode_artifacts",
)

_SUPPORTED_WEATHER_SOURCES = {"era5", "station"}


@dataclass(frozen=True)
class HashedSmokeWorker:
    project_root: Path
    runtime_id: str
    runtime_root: Path
    workspace: Path
    dssat_exec: Path
    cox_template: Path
    rendered_cox: Path
    summary_out: Path
    daily_out_files: tuple[Path, ...]
    episode_artifacts: tuple[Path, ...]
    weather_file: Path
    soil_file: Path
    prepare_report: Mapping[str, object]


def discover_project_root(config_path: str | Path) -> Path:
    """Find the enclosing AWM checkout from a config stored inside the repo."""
    config = Path(config_path).expanduser().resolve()
    start = config.parent if config.is_file() else config
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "src" / "awm"
        ).is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not discover the AWM project root from config path: "
        f"{config}. Pass project_root explicitly."
    )


def resolve_project_root(
    config_path: str | Path,
    project_root: str | Path | None = None,
) -> Path:
    if project_root is not None:
        root = Path(project_root).expanduser().resolve()
        if not (root / "pyproject.toml").is_file() or not (
            root / "src" / "awm"
        ).is_dir():
            raise FileNotFoundError(f"Not an AWM project root: {root}")
        return root
    return discover_project_root(config_path)


def _project_relative_path(root: Path, raw: Any, *, key: str) -> Path:
    text = str(raw).strip()
    if not text:
        raise ValueError(f"{key} must not be empty")
    path = Path(text)
    if path.is_absolute():
        raise ValueError(
            f"{key} must be project-relative in canonical smoke configs, got: {path}"
        )
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{key} escapes the AWM project root: {path}") from exc
    return resolved


def _worker_leaf_name(raw: Any, *, key: str) -> str:
    text = str(raw).strip()
    path = Path(text)
    if not text or path.name != text or text in {".", ".."}:
        raise ValueError(f"{key} must be a single worker-local filename")
    return text


def reject_legacy_explicit_runtime_keys(config: Mapping[str, Any]) -> None:
    present = [key for key in LEGACY_EXPLICIT_RUNTIME_KEYS if key in config]
    if present:
        raise KeyError(
            "Canonical real-smoke configs must not hard-code mutable DSSAT runtime "
            "paths. Remove legacy keys: " + ", ".join(present)
        )


def prepare_hashed_smoke_worker(
    config_path: str | Path,
    config: Mapping[str, Any],
    *,
    project_root: str | Path | None = None,
    runtime_base: str | Path | None = None,
) -> HashedSmokeWorker:
    """Create a fresh real-smoke worker in AWM's short hashed runtime namespace."""
    reject_legacy_explicit_runtime_keys(config)
    root = resolve_project_root(config_path, project_root)

    if "cox_template" not in config:
        raise KeyError("smoke config missing key: cox_template")
    if "weather_source" not in config:
        raise KeyError("smoke config missing key: weather_source")
    if "weather_filename" not in config:
        raise KeyError("smoke config missing key: weather_filename")

    runtime_spec = config.get("runtime", {})
    if not isinstance(runtime_spec, Mapping):
        raise TypeError("runtime must be a JSON object")
    policy_idx = int(runtime_spec.get("policy_idx", 0))
    env_idx = int(runtime_spec.get("env_idx", 0))
    replace_worker = bool(runtime_spec.get("replace_worker", True))

    template = root / "dssat_workspace_template"
    report = prepare_project_worker(
        template,
        project_root=root,
        policy_idx=policy_idx,
        env_idx=env_idx,
        replace=replace_worker,
        runtime_base=runtime_base,
    )
    workspace = Path(str(report["workspace"])).resolve()

    cox_template = _project_relative_path(
        root,
        config["cox_template"],
        key="cox_template",
    )
    if not cox_template.is_file():
        raise FileNotFoundError(f"Frozen AWM COX template not found: {cox_template}")

    rendered_name = _worker_leaf_name(
        config.get("rendered_cox_name", "AWM_SMOKE.COX"),
        key="rendered_cox_name",
    )
    summary_name = _worker_leaf_name(
        config.get("summary_out_name", "Summary.OUT"),
        key="summary_out_name",
    )

    raw_daily = config.get("daily_out_names", CANONICAL_DAILY_OUT_NAMES)
    if not isinstance(raw_daily, (list, tuple)):
        raise TypeError("daily_out_names must be an array")
    daily_names = tuple(
        _worker_leaf_name(name, key="daily_out_names") for name in raw_daily
    )
    if daily_names != CANONICAL_DAILY_OUT_NAMES:
        raise ValueError(
            "daily_out_names must exactly match the canonical 13-file 74-D state "
            f"inventory: {CANONICAL_DAILY_OUT_NAMES}"
        )

    weather_source = str(config["weather_source"]).strip().lower()
    if weather_source not in _SUPPORTED_WEATHER_SOURCES:
        raise ValueError(
            f"weather_source must be one of {sorted(_SUPPORTED_WEATHER_SOURCES)}"
        )
    weather_name = _worker_leaf_name(
        config["weather_filename"],
        key="weather_filename",
    )
    weather_file = workspace / "data" / "wth" / weather_source / weather_name
    if not weather_file.is_file():
        raise FileNotFoundError(f"Worker weather file not found: {weather_file}")

    soil_relative = str(config.get("soil_relative_path", "data/soil/SOIL.SOL"))
    soil_path = Path(soil_relative)
    if soil_path.is_absolute() or ".." in soil_path.parts:
        raise ValueError("soil_relative_path must stay inside the worker")
    soil_file = (workspace / soil_path).resolve()
    try:
        soil_file.relative_to(workspace)
    except ValueError as exc:
        raise ValueError("soil_relative_path escapes the worker") from exc
    if not soil_file.is_file():
        raise FileNotFoundError(f"Worker soil file not found: {soil_file}")

    summary_out = workspace / summary_name
    daily_out_files = tuple(workspace / name for name in daily_names)
    episode_artifacts = (summary_out, *daily_out_files)

    return HashedSmokeWorker(
        project_root=root,
        runtime_id=str(report["runtime_id"]),
        runtime_root=Path(str(report["runtime_root"])).resolve(),
        workspace=workspace,
        dssat_exec=Path(str(report["dssat_exec"])).resolve(),
        cox_template=cox_template,
        rendered_cox=workspace / rendered_name,
        summary_out=summary_out,
        daily_out_files=daily_out_files,
        episode_artifacts=episode_artifacts,
        weather_file=weather_file,
        soil_file=soil_file,
        prepare_report=dict(report),
    )


__all__ = [
    "CANONICAL_DAILY_OUT_NAMES",
    "HashedSmokeWorker",
    "LEGACY_EXPLICIT_RUNTIME_KEYS",
    "discover_project_root",
    "prepare_hashed_smoke_worker",
    "reject_legacy_explicit_runtime_keys",
    "resolve_project_root",
]
