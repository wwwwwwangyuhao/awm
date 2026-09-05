"""Command-line smoke test for one real worker-local DSSAT season."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .backend import DSSATWorkerBackend, DSSATWorkerPaths
from .management import DSSATExperimentRenderer
from .output_reader import CachedDSSATOutputReader
from .runner import DSSATRunner
from .workspace import validate_dssatpro_record_width


def run_smoke(config_path: str) -> dict[str, object]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    required = (
        "workspace",
        "dssat_exec",
        "output_dir",
        "cox_template",
        "rendered_cox",
        "summary_out",
        "daily_out_files",
        "plant_yrdoy",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise KeyError("smoke config missing keys: " + ", ".join(missing))

    workspace_report = validate_dssatpro_record_width(config["workspace"])
    renderer = DSSATExperimentRenderer(
        template_path=config["cox_template"],
        output_cox_path=config["rendered_cox"],
    )
    runner = DSSATRunner(
        dssat_exec=config["dssat_exec"],
        output_dir=config["output_dir"],
        cox_path=config["rendered_cox"],
        weather_file=config.get("weather_file"),
        soil_file=config.get("soil_file"),
        timeout_seconds=float(config.get("timeout_seconds", 1800.0)),
    )
    reader = CachedDSSATOutputReader(
        summary_out=config["summary_out"],
        out_files=list(config["daily_out_files"]),
        str_fields=config.get("str_fields"),
        date_fields=config.get("date_fields"),
    )
    paths = DSSATWorkerPaths(
        workspace=config["workspace"],
        summary_out=config["summary_out"],
        daily_out_files=tuple(config["daily_out_files"]),
        episode_artifacts=tuple(config.get("episode_artifacts", ())),
    )
    backend = DSSATWorkerBackend(
        renderer=renderer,
        runner=runner,
        reader=reader,
        paths=paths,
    )
    backend.reset_episode()
    planting_state = backend.daily_state(config["plant_yrdoy"])
    summary = backend.season_summary()
    if not planting_state:
        raise RuntimeError("real DSSAT smoke parsed an empty planting-day state")
    if "HWAM" not in summary or "IRCM" not in summary:
        raise KeyError("real DSSAT smoke requires Summary.OUT fields HWAM and IRCM")
    return {
        "status": "passed",
        "workspace_preflight": workspace_report,
        "plant_yrdoy": config["plant_yrdoy"],
        "planting_state_field_count": len(planting_state),
        "summary_has_hwam": True,
        "summary_has_ircm": True,
        "output_reader_metrics": reader.metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test one AWM DSSAT worker")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(json.dumps(run_smoke(args.config), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
