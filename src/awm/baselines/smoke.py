"""Run one full agricultural baseline episode on a real hashed DSSAT worker."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

from awm.dssat.backend import DSSATWorkerBackend, DSSATWorkerPaths
from awm.dssat.management import DSSATExperimentRenderer
from awm.dssat.output_reader import CachedDSSATOutputReader
from awm.dssat.runner import DSSATRunner
from awm.dssat.runtime_paths import WorkspaceRootLock
from awm.dssat.smoke_runtime import prepare_hashed_smoke_worker, resolve_project_root
from awm.dssat.workspace import validate_dssatpro_record_width
from awm.envs.cotton_water_env import CottonWaterEnv
from awm.envs.dssat_irrigation import DSSATDecisionCalendar, DSSATIrrigationAdapter
from awm.envs.water_budget import IrrigationSystemSpec, WaterBudgetController

from .agricultural import (
    AgriculturalBaseline,
    ConventionalScheduleBaseline,
    PotentialETWaterBalanceBaseline,
    RootZoneREWThresholdBaseline,
)
from .runner import run_baseline_episode


def _require(config: Mapping[str, Any], *keys: str) -> None:
    missing = [key for key in keys if key not in config]
    if missing:
        raise KeyError("baseline smoke config missing keys: " + ", ".join(missing))


def _validate_budget_accounting(config: Mapping[str, Any]) -> None:
    """Keep total seasonal water distinct from the postplant policy quota."""
    if "total_seasonal_budget_mm" not in config:
        return
    total = float(config["total_seasonal_budget_mm"])
    policy = float(config["seasonal_budget_mm"])
    fixed = float(config["nonpolicy_irrigation_mm"])
    for label, value in (("total", total), ("policy", policy), ("nonpolicy", fixed)):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{label} irrigation budget must be finite and >= 0")
    if abs(total - (policy + fixed)) > 1e-9:
        raise ValueError(
            "total_seasonal_budget_mm must equal seasonal_budget_mm + "
            "nonpolicy_irrigation_mm; got "
            f"{total} != {policy} + {fixed}"
        )


def _executed_event_summary(
    step_audits: tuple[Mapping[str, object], ...],
) -> list[dict[str, object]]:
    """Compact event trace preserving desired, constrained and executed water."""
    events: list[dict[str, object]] = []
    for step in step_audits:
        if not bool(step.get("irrigation_event_applied")):
            continue
        events.append(
            {
                "policy_day": int(step["policy_day"]),
                "action_dap": int(step["decision_action_day"]),
                "action_yrdoy": str(step["decision_action_yrdoy"]),
                "baseline_desired_mm": float(step["baseline_desired_mm"]),
                "baseline_constrained_request_mm": float(
                    step["baseline_constrained_request_mm"]
                ),
                "baseline_constraint_adjusted": bool(
                    step["baseline_constraint_adjusted"]
                ),
                "baseline_constraint_reasons": list(
                    step.get("baseline_constraint_reasons", ())
                ),
                "applied_mm": float(step["applied_irrigation_mm"]),
                "adapter_projected": bool(
                    step.get("irrigation_action_projected", False)
                ),
                "adapter_quantized": bool(
                    step.get("irrigation_execution_quantized", False)
                ),
                "adapter_projection_reasons": list(
                    step.get("irrigation_projection_reasons", ())
                ),
            }
        )
    return events


def build_baseline(spec: Mapping[str, Any]) -> AgriculturalBaseline:
    _require(spec, "type")
    kind = str(spec["type"]).strip().lower()
    if kind == "local_conventional":
        _require(spec, "schedule_mm_by_action_dap")
        raw = spec["schedule_mm_by_action_dap"]
        if not isinstance(raw, Mapping):
            raise TypeError("schedule_mm_by_action_dap must be an object")
        return ConventionalScheduleBaseline(
            {int(day): float(mm) for day, mm in raw.items()}
        )
    if kind == "potential_et_water_balance":
        _require(
            spec,
            "trigger_deficit_mm",
            "irrigation_efficiency",
            "effective_rain_fraction",
            "refill_fraction",
        )
        return PotentialETWaterBalanceBaseline(
            trigger_deficit_mm=float(spec["trigger_deficit_mm"]),
            irrigation_efficiency=float(spec["irrigation_efficiency"]),
            effective_rain_fraction=float(spec["effective_rain_fraction"]),
            refill_fraction=float(spec["refill_fraction"]),
        )
    if kind == "root_zone_rew_threshold":
        _require(
            spec,
            "trigger_rew",
            "event_depth_mm",
            "fallback_rew_layers_1_based",
        )
        return RootZoneREWThresholdBaseline(
            trigger_rew=float(spec["trigger_rew"]),
            event_depth_mm=float(spec["event_depth_mm"]),
            fallback_rew_layers=tuple(
                int(x) for x in spec["fallback_rew_layers_1_based"]
            ),
        )
    raise ValueError(f"unsupported agricultural baseline type: {kind!r}")


def run_real_baseline(
    config_path: str,
    *,
    audit_output: str | None = None,
    project_root: str | None = None,
    runtime_base: str | None = None,
) -> dict[str, object]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    _require(
        config,
        "cox_template",
        "weather_source",
        "weather_filename",
        "plant_yrdoy",
        "seasonal_budget_mm",
        "min_event_mm",
        "max_event_mm",
        "min_interval_days",
        "horizon_days",
        "execution_resolution_mm",
        "nonpolicy_irrigation_mm",
        "ircm_tolerance_mm",
        "yield_target_fraction",
        "baseline",
    )
    _validate_budget_accounting(config)

    root = resolve_project_root(config_path, project_root)
    with WorkspaceRootLock(project_root=root, runtime_base=runtime_base):
        worker = prepare_hashed_smoke_worker(
            config_path,
            config,
            project_root=root,
            runtime_base=runtime_base,
        )
        validate_dssatpro_record_width(worker.workspace)
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
        backend = DSSATWorkerBackend(
            renderer=renderer,
            runner=runner,
            reader=reader,
            paths=DSSATWorkerPaths(
                workspace=str(worker.workspace),
                summary_out=str(worker.summary_out),
                daily_out_files=tuple(str(path) for path in worker.daily_out_files),
                episode_artifacts=tuple(str(path) for path in worker.episode_artifacts),
            ),
        )

        water_spec = IrrigationSystemSpec(
            seasonal_budget_mm=float(config["seasonal_budget_mm"]),
            min_event_mm=float(config["min_event_mm"]),
            max_event_mm=float(config["max_event_mm"]),
            min_interval_days=int(config["min_interval_days"]),
            horizon_days=int(config["horizon_days"]),
        )
        controller = WaterBudgetController(water_spec)
        calendar = DSSATDecisionCalendar.from_yrdoy(
            str(config["plant_yrdoy"]),
            horizon_days=int(config["horizon_days"]),
        )
        adapter = DSSATIrrigationAdapter(
            controller=controller,
            backend=backend,
            calendar=calendar,
            execution_resolution_mm=float(config["execution_resolution_mm"]),
            nonpolicy_irrigation_mm=float(config["nonpolicy_irrigation_mm"]),
            summary_tolerance_mm=float(config["ircm_tolerance_mm"]),
        )
        env = CottonWaterEnv(
            backend=backend,
            adapter=adapter,
            plant_yrdoy=str(config["plant_yrdoy"]),
            yield_target_fraction=float(config["yield_target_fraction"]),
        )
        baseline_spec = config["baseline"]
        if not isinstance(baseline_spec, Mapping):
            raise TypeError("baseline must be a JSON object")
        policy = build_baseline(baseline_spec)
        result = run_baseline_episode(env, policy)
        executed_events = _executed_event_summary(result.step_audits)

        total_budget = config.get("total_seasonal_budget_mm")
        audit_payload = {
            "baseline_name": result.baseline_name,
            "config_path": str(Path(config_path).resolve()),
            "runtime_family": "awm",
            "runtime_id": worker.runtime_id,
            "runtime_root": str(worker.runtime_root),
            "workspace": str(worker.workspace),
            "total_seasonal_budget_mm": float(total_budget) if total_budget is not None else None,
            "policy_budget_mm": float(config["seasonal_budget_mm"]),
            "nonpolicy_irrigation_mm": float(config["nonpolicy_irrigation_mm"]),
            "baseline_desired_event_count": result.desired_event_count,
            "hierarchical_requested_event_count": result.requested_event_count,
            "baseline_constraint_adjusted_event_count": result.baseline_constraint_adjusted_event_count,
            "adapter_projected_event_count": result.projected_event_count,
            "executed_events": executed_events,
            "step_audits": list(result.step_audits),
            "terminal_info": dict(result.terminal_info),
        }
        if audit_output:
            destination = Path(audit_output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                json.dumps(audit_payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

        return {
            "status": "passed",
            "runtime_family": "awm",
            "runtime_id": worker.runtime_id,
            "runtime_root": str(worker.runtime_root),
            "workspace": str(worker.workspace),
            "baseline_name": result.baseline_name,
            "water_treatment": config.get("water_treatment"),
            "total_seasonal_budget_mm": float(total_budget) if total_budget is not None else None,
            "policy_budget_mm": float(config["seasonal_budget_mm"]),
            "fixed_nonpolicy_irrigation_mm": float(config["nonpolicy_irrigation_mm"]),
            "HWAM_kg_ha": result.yield_hwam_kg_ha,
            "IRCM_mm": result.dssat_ircm_mm,
            "policy_irrigation_mm": result.policy_irrigation_mm,
            "IWP_kg_m3": result.irrigation_water_productivity_kg_m3,
            "baseline_desired_event_count": result.desired_event_count,
            "requested_event_count": result.requested_event_count,
            "baseline_constraint_adjusted_event_count": result.baseline_constraint_adjusted_event_count,
            "projected_event_count": result.projected_event_count,
            "irrigation_event_count": result.irrigation_event_count,
            "executed_events": executed_events,
            "irrigation_accounting_passed": result.irrigation_accounting_passed,
            "audit_output": str(Path(audit_output).resolve()) if audit_output else None,
            "output_reader_metrics": reader.metrics,
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one full AWM agricultural baseline in the hashed DSSAT runtime"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--audit-output")
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
            run_real_baseline(
                args.config,
                audit_output=args.audit_output,
                project_root=args.project_root,
                runtime_base=args.runtime_base,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
