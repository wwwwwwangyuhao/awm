"""One untrained PPO episode through the formal real-DSSAT AWM stack."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch

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

from .agent import PPOAgent, PPOHyperparameters
from .reward import PPOEpisodeReward


def _require(config: Mapping[str, Any], *keys: str) -> None:
    missing = [key for key in keys if key not in config]
    if missing:
        raise KeyError("PPO smoke config missing keys: " + ", ".join(missing))


def _reference_table(project_root: Path) -> dict[int, float]:
    payload = json.loads(
        (project_root / "configs" / "yield_reference_v1_development.json").read_text(
            encoding="utf-8"
        )
    )
    values = {**payload["training_years"], **payload["validation_years"]}
    return {int(year): float(value) for year, value in values.items()}


def _build_env(config_path: str, config: Mapping[str, Any], *, root: Path, runtime_base: str | None):
    worker = prepare_hashed_smoke_worker(
        config_path, config, project_root=root, runtime_base=runtime_base
    )
    validate_dssatpro_record_width(worker.workspace)
    renderer = DSSATExperimentRenderer(
        template_path=str(worker.cox_template), output_cox_path=str(worker.rendered_cox)
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
    spec = IrrigationSystemSpec(
        seasonal_budget_mm=float(config["seasonal_budget_mm"]),
        min_event_mm=float(config["min_event_mm"]),
        max_event_mm=float(config["max_event_mm"]),
        min_interval_days=int(config["min_interval_days"]),
        horizon_days=int(config["horizon_days"]),
    )
    controller = WaterBudgetController(spec)
    calendar = DSSATDecisionCalendar.from_yrdoy(
        str(config["plant_yrdoy"]), horizon_days=int(config["horizon_days"])
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
        yield_target_fraction=float(config["eta"]),
    )
    return worker, env


def run_real_ppo_smoke(
    config_path: str,
    *,
    project_root: str | None = None,
    runtime_base: str | None = None,
    audit_output: str | None = None,
) -> dict[str, object]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    _require(
        config,
        "cox_template",
        "weather_year",
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
        "eta",
        "seed",
    )
    if int(config["weather_year"]) != 2000 or str(config.get("weather_split")) != "train":
        raise ValueError("formal PPO integration smoke is locked to training year 2000")
    if bool(config.get("state_normalization", False)):
        raise ValueError("untrained integration smoke must not invent normalizer statistics")
    total = float(config.get("total_seasonal_budget_mm", 540.0))
    policy_budget = float(config["seasonal_budget_mm"])
    fixed = float(config["nonpolicy_irrigation_mm"])
    if not math.isclose(total, policy_budget + fixed, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("total seasonal budget must equal policy plus fixed irrigation")

    root = resolve_project_root(config_path, project_root)
    references = _reference_table(root)
    reward = PPOEpisodeReward(
        weather_year=int(config["weather_year"]),
        eta=float(config["eta"]),
        reference_yield_by_year=references,
        policy_budget_mm=policy_budget,
    )
    hparams = PPOHyperparameters(state_dim=79)
    agent = PPOAgent(
        hyperparameters=hparams,
        seed=int(config["seed"]),
        device=str(config.get("device", "cpu")),
    )

    with WorkspaceRootLock(project_root=root, runtime_base=runtime_base):
        worker, env = _build_env(
            config_path, config, root=root, runtime_base=runtime_base
        )
        observation, _ = env.reset()
        step_audits: list[dict[str, object]] = []
        sampled_events = 0
        executed_events = 0
        projected_events = 0
        reruns = 0
        max_logprob_error = 0.0
        water_reward_sum = 0.0
        terminal_info: dict[str, object] | None = None

        while True:
            state = torch.tensor(observation.flat(), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action, value = agent.act(state)
                recomputed, _ = agent.actor.evaluate_behavior(
                    state,
                    action.irrigate,
                    action.raw_amount,
                )
            error = abs(float(action.log_prob.item()) - float(recomputed.item()))
            max_logprob_error = max(max_logprob_error, error)
            irrigate = bool(action.irrigate.item())
            amount_fraction = float(action.amount_fraction.item())
            sampled_events += int(irrigate)
            step = env.step(irrigate=irrigate, amount_fraction=amount_fraction)
            audit = step.irrigation_audit
            executed_events += int(audit.event_applied)
            projected_events += int(audit.water_budget_projected)
            reruns += int(audit.dssat_rerun)
            water_reward_sum += reward.step_reward(float(audit.applied_irrigation_mm))
            audit_row = dict(audit.as_info_dict())
            audit_row.update(
                {
                    "ppo_sampled_gate": irrigate,
                    "ppo_sampled_amount_fraction": amount_fraction,
                    "ppo_raw_amount_latent": float(action.raw_amount.item()),
                    "ppo_behavior_log_prob": float(action.log_prob.item()),
                    "ppo_value": float(value.item()),
                }
            )
            step_audits.append(audit_row)
            observation = step.observation
            if step.terminated:
                terminal_info = dict(step.info)
                break

    if terminal_info is None:
        raise AssertionError("PPO smoke ended without terminal info")
    breakdown = reward.finish(
        yield_kg_ha=float(terminal_info["HWAM"]),
        irrigation_accounting_passed=bool(terminal_info["irrigation_accounting_passed"]),
    )
    if abs(water_reward_sum - breakdown.water_return) > 1e-9:
        raise RuntimeError("step water rewards do not telescope to terminal water return")
    if len(step_audits) != int(config["horizon_days"]):
        raise RuntimeError("PPO smoke did not execute exactly the configured horizon")
    if max_logprob_error > 1e-6:
        raise RuntimeError("sampled PPO log probability did not recompute exactly")

    result = {
        "status": "passed",
        "ppo_protocol_id": "awm-ppo-baseline-v1",
        "integration_smoke_only": True,
        "untrained_policy": True,
        "weather_year": int(config["weather_year"]),
        "eta": float(config["eta"]),
        "seed": int(config["seed"]),
        "runtime_id": worker.runtime_id,
        "workspace": str(worker.workspace),
        "step_count": len(step_audits),
        "sampled_event_count": sampled_events,
        "executed_event_count": executed_events,
        "projected_event_count": projected_events,
        "dssat_rerun_count": reruns,
        "max_behavior_logprob_recompute_error": max_logprob_error,
        "policy_irrigation_mm": float(terminal_info["policy_irrigation_mm"]),
        "IRCM_mm": float(terminal_info["IRCM"]),
        "HWAM_kg_ha": float(terminal_info["HWAM"]),
        "Y_ref_kg_ha": breakdown.reference_yield_kg_ha,
        "yield_retention": breakdown.yield_retention,
        "target_feasible": breakdown.target_feasible,
        "water_return": breakdown.water_return,
        "violation_penalty": breakdown.violation_penalty,
        "episode_return": breakdown.episode_return,
        "irrigation_accounting_passed": True,
        "step_audits": step_audits,
    }
    if audit_output:
        path = Path(audit_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one untrained PPO episode on real DSSAT")
    parser.add_argument("config")
    parser.add_argument("--project-root")
    parser.add_argument("--runtime-base")
    parser.add_argument("--audit-output")
    args = parser.parse_args()
    result = run_real_ppo_smoke(
        args.config,
        project_root=args.project_root,
        runtime_base=args.runtime_base,
        audit_output=args.audit_output,
    )
    compact = {key: value for key, value in result.items() if key != "step_audits"}
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


__all__ = ["run_real_ppo_smoke"]
