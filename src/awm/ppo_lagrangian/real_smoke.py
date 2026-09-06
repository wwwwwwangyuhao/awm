"""One untrained PPO-Lagrangian episode through the formal real-DSSAT stack."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from awm.dssat.runtime_paths import WorkspaceRootLock
from awm.dssat.smoke_runtime import resolve_project_root
from awm.ppo.real_smoke import _build_env, _reference_table

from .agent import PPOLagrangianAgent, PPOLagrangianHyperparameters
from .signals import LagrangianEpisodeSignals


def run_real_smoke(
    config_path: str,
    *,
    project_root: str | None = None,
    runtime_base: str | None = None,
    audit_output: str | None = None,
) -> dict[str, object]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if str(config.get("protocol_id")) != "awm-ppo-lagrangian-baseline-v1":
        raise ValueError("smoke config protocol_id mismatch")
    if int(config.get("weather_year", -1)) != 2000 or str(config.get("weather_split")) != "train":
        raise ValueError("PPO-Lagrangian integration smoke is locked to training year 2000")
    if bool(config.get("state_normalization", False)):
        raise ValueError("untrained smoke must not invent normalizer statistics")
    total = float(config.get("total_seasonal_budget_mm", 540.0))
    policy_budget = float(config["seasonal_budget_mm"])
    fixed = float(config["nonpolicy_irrigation_mm"])
    if not math.isclose(total, policy_budget + fixed, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("total seasonal budget must equal policy plus fixed irrigation")

    root = resolve_project_root(config_path, project_root)
    references = _reference_table(root)
    tracker = LagrangianEpisodeSignals(
        weather_year=int(config["weather_year"]),
        eta=float(config["eta"]),
        reference_yield_by_year=references,
        policy_budget_mm=policy_budget,
    )
    agent = PPOLagrangianAgent(
        hyperparameters=PPOLagrangianHyperparameters(state_dim=79),
        seed=int(config["seed"]),
        device=str(config.get("device", "cpu")),
    )

    with WorkspaceRootLock(project_root=root, runtime_base=runtime_base):
        worker, env = _build_env(config_path, config, root=root, runtime_base=runtime_base)
        observation, _ = env.reset()
        step_audits: list[dict[str, object]] = []
        sampled_events = executed_events = projected_events = reruns = 0
        max_logprob_error = 0.0
        water_reward_sum = 0.0
        terminal_info = None
        while True:
            state = torch.tensor(observation.flat(), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action, reward_value, cost_value = agent.act(state)
                recomputed, _ = agent.actor.evaluate_behavior(
                    state, action.irrigate, action.raw_amount
                )
            max_logprob_error = max(
                max_logprob_error,
                abs(float(action.log_prob.item()) - float(recomputed.item())),
            )
            irrigate = bool(action.irrigate.item())
            sampled_events += int(irrigate)
            step = env.step(
                irrigate=irrigate,
                amount_fraction=float(action.amount_fraction.item()),
            )
            audit = step.irrigation_audit
            executed_events += int(audit.event_applied)
            projected_events += int(audit.water_budget_projected)
            reruns += int(audit.dssat_rerun)
            water_reward_sum += tracker.step_reward(float(audit.applied_irrigation_mm))
            row = dict(audit.as_info_dict())
            row.update(
                {
                    "sampled_gate": irrigate,
                    "sampled_amount_fraction": float(action.amount_fraction.item()),
                    "raw_amount_latent": float(action.raw_amount.item()),
                    "behavior_log_prob": float(action.log_prob.item()),
                    "reward_value": float(reward_value.item()),
                    "cost_value": float(cost_value.item()),
                }
            )
            step_audits.append(row)
            observation = step.observation
            if step.terminated:
                terminal_info = dict(step.info)
                break

    if terminal_info is None:
        raise AssertionError("smoke ended without terminal info")
    breakdown = tracker.finish(
        yield_kg_ha=float(terminal_info["HWAM"]),
        irrigation_accounting_passed=bool(terminal_info["irrigation_accounting_passed"]),
    )
    if abs(water_reward_sum - breakdown.water_return) > 1e-9:
        raise RuntimeError("step water rewards do not telescope to water return")
    if len(step_audits) != int(config["horizon_days"]):
        raise RuntimeError("smoke did not execute exactly the configured horizon")
    if max_logprob_error > 1e-6:
        raise RuntimeError("sampled hierarchical log probability did not recompute")

    result = {
        "status": "passed",
        "protocol_id": "awm-ppo-lagrangian-baseline-v1",
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
        "water_return": breakdown.water_return,
        "signed_constraint_cost": breakdown.signed_constraint_cost,
        "per_year_expected_constraint_satisfied": breakdown.per_year_expected_constraint_satisfied,
        "initial_dual_by_eta": {f"{eta:.2f}": value for eta, value in agent.dual_by_eta.items()},
        "irrigation_accounting_passed": True,
        "step_audits": step_audits,
    }
    if audit_output:
        path = Path(audit_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one untrained PPO-Lagrangian episode on real DSSAT")
    parser.add_argument("config")
    parser.add_argument("--project-root")
    parser.add_argument("--runtime-base")
    parser.add_argument("--audit-output")
    args = parser.parse_args()
    result = run_real_smoke(
        args.config,
        project_root=args.project_root,
        runtime_base=args.runtime_base,
        audit_output=args.audit_output,
    )
    compact = {key: value for key, value in result.items() if key != "step_audits"}
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


__all__ = ["run_real_smoke"]
