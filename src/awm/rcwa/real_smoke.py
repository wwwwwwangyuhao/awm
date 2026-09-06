"""One untrained RCWA-RL episode through the formal real-DSSAT stack."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from awm.ppo.normalization import RunningObservationNormalizer
from awm.ppo.real_env import PPORealEnvFactory
from awm.ppo.scheduler import WeatherEtaCell

from .agent import RCWAAgent
from .signals import RCWAEpisodeSignals


def _references(root: Path) -> dict[int, float]:
    payload = json.loads(
        (root / "configs" / "yield_reference_v1_development.json").read_text(encoding="utf-8")
    )
    values = {**payload["training_years"], **payload["validation_years"]}
    return {int(year): float(value) for year, value in values.items()}


def run_real_rcwa_smoke(
    config_path: str | Path,
    *,
    project_root: str | Path,
    runtime_base: str | Path | None = None,
    audit_output: str | Path | None = None,
) -> dict[str, object]:
    root = Path(project_root).expanduser().resolve()
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if config.get("rcwa_protocol_id") != "awm-rcwa-rl-v1":
        raise ValueError("RCWA smoke protocol id mismatch")
    if int(config["weather_year"]) != 2000 or config.get("weather_split") != "train":
        raise ValueError("formal RCWA smoke is locked to training year 2000")
    if bool(config.get("state_normalization", False)):
        raise ValueError("untrained RCWA smoke must not invent normalizer statistics")

    seed = int(config["seed"])
    eta = float(config["eta"])
    cell = WeatherEtaCell(2000, eta)
    agent = RCWAAgent(seed=seed, device=str(config.get("device", "cpu")))
    normalizer = RunningObservationNormalizer(state_dim=agent.hparams.state_dim)
    tracker = RCWAEpisodeSignals(
        weather_year=2000,
        eta=eta,
        reference_yield_by_year=_references(root),
    )
    factory = PPORealEnvFactory(
        project_root=root,
        work_dir=root / "runtime" / "rcwa_rl_v1" / "single_episode_smoke",
        runtime_base=runtime_base,
        env_idx=0,
    )
    env = factory(cell)
    sampled_events = 0
    executed_events = 0
    projected_events = 0
    reruns = 0
    max_logprob_error = 0.0
    step_count = 0
    terminal_info = None
    water_reward_sum = 0.0
    try:
        observation, _ = env.reset()
        while True:
            raw = np.asarray(observation.flat(), dtype=np.float32)
            normalized = normalizer.normalize(raw)
            state = torch.from_numpy(normalized).unsqueeze(0)
            with torch.no_grad():
                action, reward_value, risk_value = agent.act(state)
                recomputed, _ = agent.actor.evaluate_behavior(
                    state.to(agent.device), action.irrigate, action.raw_amount
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
            observation = step.observation
            step_count += 1
            if step.terminated:
                terminal_info = dict(step.info)
                break
    finally:
        env.close()

    if terminal_info is None:
        raise AssertionError("RCWA smoke ended without terminal info")
    breakdown = tracker.finish(
        yield_kg_ha=float(terminal_info["HWAM"]),
        irrigation_accounting_passed=bool(terminal_info["irrigation_accounting_passed"]),
    )
    if step_count != 125:
        raise RuntimeError("RCWA smoke must contain 125 steps")
    if max_logprob_error > 1e-6:
        raise RuntimeError("sampled RCWA log probability did not recompute exactly")
    if abs(water_reward_sum - breakdown.water_return) > 1e-9:
        raise RuntimeError("RCWA water reward ledger mismatch")

    result = {
        "status": "passed",
        "rcwa_protocol_id": "awm-rcwa-rl-v1",
        "integration_smoke_only": True,
        "untrained_policy": True,
        "weather_year": 2000,
        "eta": eta,
        "seed": seed,
        "step_count": step_count,
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
        "initial_dual_by_eta": {f"{eta:.2f}": value for eta, value in agent.dual_by_eta.items()},
        "irrigation_accounting_passed": True,
        "lower_cvar_computed": False,
        "lower_cvar_note": "A single episode cannot estimate the registered 18-weather-year training LCVaR; use the balanced update gate.",
    }
    if audit_output is not None:
        destination = Path(audit_output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one untrained RCWA-RL episode on real DSSAT")
    parser.add_argument("config")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--runtime-base")
    parser.add_argument("--audit-output")
    args = parser.parse_args()
    result = run_real_rcwa_smoke(
        args.config,
        project_root=args.project_root,
        runtime_base=args.runtime_base,
        audit_output=args.audit_output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


__all__ = ["run_real_rcwa_smoke"]
