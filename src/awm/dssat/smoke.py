"""Command-line smoke test for one real hashed AWM DSSAT worker."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .backend import DSSATWorkerBackend, DSSATWorkerPaths
from .management import DSSATExperimentRenderer
from .output_reader import CachedDSSATOutputReader
from .runner import DSSATRunner
from .runtime_paths import WorkspaceRootLock
from .smoke_runtime import prepare_hashed_smoke_worker, resolve_project_root
from .workspace import validate_dssatpro_record_width


def run_smoke(
    config_path: str,
    *,
    project_root: str | None = None,
    runtime_base: str | None = None,
) -> dict[str, object]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    required = (
        "cox_template",
        "weather_source",
        "weather_filename",
        "plant_yrdoy",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise KeyError("smoke config missing keys: " + ", ".join(missing))

    root = resolve_project_root(config_path, project_root)
    with WorkspaceRootLock(project_root=root, runtime_base=runtime_base):
        worker = prepare_hashed_smoke_worker(
            config_path,
            config,
            project_root=root,
            runtime_base=runtime_base,
        )
        workspace_report = validate_dssatpro_record_width(worker.workspace)
        renderer = DSSATExperimentRenderer(
            template_path=str(worker.cox_template),
            output_cox_path=str(worker.rendered_cox),
        )
        runner = DSSATRunner(
            dssat_exec=str(worker.dssat_exec),
            output_dir=str(worker.workspace),
            cox_path=str(worker.rendered_cox),
            weather_file=str(worker.weather_file),
            soil_file=str(worker.soil_file),
            timeout_seconds=float(config.get("timeout_seconds", 1800.0)),
        )
        reader = CachedDSSATOutputReader(
            summary_out=str(worker.summary_out),
            out_files=[str(path) for path in worker.daily_out_files],
            str_fields=config.get("str_fields"),
            date_fields=config.get("date_fields"),
        )
        paths = DSSATWorkerPaths(
            workspace=str(worker.workspace),
            summary_out=str(worker.summary_out),
            daily_out_files=tuple(str(path) for path in worker.daily_out_files),
            episode_artifacts=tuple(str(path) for path in worker.episode_artifacts),
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
            "runtime_family": "awm",
            "runtime_id": worker.runtime_id,
            "runtime_root": str(worker.runtime_root),
            "workspace": str(worker.workspace),
            "workspace_preflight": workspace_report,
            "plant_yrdoy": config["plant_yrdoy"],
            "planting_state_field_count": len(planting_state),
            "summary_has_hwam": True,
            "summary_has_ircm": True,
            "output_reader_metrics": reader.metrics,
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-test one real AWM DSSAT worker in the hashed runtime"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--project-root",
        help="AWM checkout root; normally discovered from the config path",
    )
    parser.add_argument(
        "--runtime-base",
        help="Optional override; default is ~/.dssat_rt/awm",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run_smoke(
                args.config,
                project_root=args.project_root,
                runtime_base=args.runtime_base,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
