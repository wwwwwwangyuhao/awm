"""Diagnostic-only credit audit for frozen RCWA-v10 checkpoints.

This module NEVER updates actor, critics, dual variables, or observation
normalizer statistics. It collects one balanced on-policy DSSAT rollout from a
frozen v10 checkpoint and decomposes the first-order policy-gradient estimate
by weather year and actor parameter block.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

import numpy as np
import torch

from awm.ppo.normalization import NormalizerState, RunningObservationNormalizer
from awm.ppo.real_env import PPORealEnvFactory
from awm.ppo.scheduler import WeatherEtaCell, training_cells
from awm.risk import REGISTERED_ETA_LEVELS

from .agent_v10 import RCWAV10Agent
from .buffer import RCWARolloutBatch, RCWARolloutBuffer
from .rollout import RCWAEpisodeOutcome, collect_episode
from .signals import RCWAEpisodeSignals
from .trainer_v10 import _load_protocol, _load_references, hyperparameters_from_protocol

DIAGNOSTIC_PROTOCOL_ID = "awm-rcwa-v10-credit-diagnostic-v1"
SOURCE_PROTOCOL_ID = "awm-rcwa-rl-v10"
SOURCE_TRAINER_ID = "awm-rcwa-trainer-v10"
SOURCE_COMMIT = "e3f7b8ebc64d6f6337ae74ed44c2ef9d7c9263e5"
DIAGNOSTIC_ACTION_SEED = 20260913


def _load_frozen_checkpoint(*, project_root: Path, checkpoint: Path, device: torch.device):
    protocol = _load_protocol(project_root)
    if protocol.get("rcwa_protocol_id") != SOURCE_PROTOCOL_ID:
        raise RuntimeError("diagnostic requires the frozen RCWA-v10 protocol")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("trainer_protocol_id") != SOURCE_TRAINER_ID:
        raise ValueError("checkpoint trainer protocol mismatch")
    manifest = payload.get("run_manifest") or {}
    if manifest.get("rcwa_protocol_id") != SOURCE_PROTOCOL_ID:
        raise ValueError("checkpoint RCWA protocol mismatch")
    if manifest.get("git_commit") != SOURCE_COMMIT:
        raise ValueError("checkpoint source commit is not the frozen v10 commit")
    seed = int(manifest.get("training_seed", -1))
    hparams = hyperparameters_from_protocol(protocol)
    agent = RCWAV10Agent(hyperparameters=hparams, seed=seed, device=device)
    agent.load_checkpoint_payload(payload["agent"])
    raw = payload["normalizer"]
    normalizer = RunningObservationNormalizer(state_dim=hparams.state_dim)
    normalizer.load_state(NormalizerState(
        count=float(raw["count"]),
        mean=tuple(float(x) for x in raw["mean"]),
        variance=tuple(float(x) for x in raw["variance"]),
    ))
    if int(payload["transition_count"]) != agent.update_index * 6750:
        raise ValueError("checkpoint transition count is inconsistent")
    return payload, protocol, seed, agent, normalizer


def _collect_frozen_rollout(*, agent: RCWAV10Agent, normalizer: RunningObservationNormalizer,
                            env_factory: PPORealEnvFactory, references: dict[int, float]):
    cells = training_cells()  # canonical order, identical for every compared checkpoint
    expected_size = len(cells) * 125
    buffer = RCWARolloutBuffer(
        state_dim=agent.hparams.state_dim,
        expected_size=expected_size,
        gamma=agent.hparams.gamma,
        gae_lambda=agent.hparams.gae_lambda,
        risk_gamma=agent.hparams.risk_gamma,
        risk_gae_lambda=agent.hparams.risk_gae_lambda,
        alpha=agent.hparams.alpha,
        policy_version=agent.policy_version,
    )
    # Common diagnostic RNG removes checkpoint-history RNG as a cross-seed nuisance.
    agent.generator.manual_seed(DIAGNOSTIC_ACTION_SEED)
    raw_sink: list[np.ndarray] = []
    outcomes: list[RCWAEpisodeOutcome] = []
    for cell in cells:
        env = env_factory(cell)
        try:
            tracker = RCWAEpisodeSignals(
                weather_year=cell.weather_year,
                eta=cell.eta,
                reference_yield_by_year=references,
            )
            outcomes.append(collect_episode(
                env,
                cell=cell,
                agent=agent,
                normalizer=normalizer,
                signals=tracker,
                buffer=buffer,
                raw_observation_sink=raw_sink,
            ))
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()
    batch = buffer.finalize()
    # Deliberately DO NOT normalizer.update(...).
    return cells, tuple(outcomes), batch


def _lambda_tensor(agent: RCWAV10Agent, etas: torch.Tensor) -> torch.Tensor:
    return agent._lambda_tensor(etas.to(agent.device))


def _parameter_blocks(agent: RCWAV10Agent) -> dict[str, list[tuple[str, torch.nn.Parameter]]]:
    blocks = {"trunk": [], "gate": [], "amount": []}
    for name, param in agent.actor.named_parameters():
        if name.startswith("trunk."):
            blocks["trunk"].append((name, param))
        elif name.startswith("gate_head."):
            blocks["gate"].append((name, param))
        elif name.startswith("amount_mean_head.") or name.startswith("amount_log_std_head."):
            blocks["amount"].append((name, param))
        else:
            raise RuntimeError(f"unclassified actor parameter: {name}")
    return blocks


def _flat_grad(loss: torch.Tensor, params: list[torch.nn.Parameter], *, retain_graph: bool) -> torch.Tensor:
    grads = torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=False)
    return torch.cat([g.reshape(-1) for g in grads])


def _score_losses(agent: RCWAV10Agent, batch: RCWARolloutBatch, indices: torch.Tensor):
    states = batch.states.to(agent.device)[indices]
    irrigate = batch.irrigate.to(agent.device)[indices]
    raw_amount = batch.raw_amount.to(agent.device)[indices]
    etas = batch.etas.to(agent.device)[indices]
    reward_adv = batch.reward_advantages.to(agent.device)[indices].detach()
    risk_adv = batch.risk_advantages.to(agent.device)[indices].detach()
    log_prob, _ = agent.actor.evaluate_behavior(states, irrigate, raw_amount)
    lam = _lambda_tensor(agent, etas).detach()
    reward_loss = -(log_prob * reward_adv).mean()
    risk_loss = (log_prob * lam * risk_adv).mean()
    total_loss = reward_loss + risk_loss
    return reward_loss, risk_loss, total_loss


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = float(torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b))
    if denom <= 1e-20:
        return float("nan")
    return float(torch.dot(a, b) / denom)


def _block_vector(agent: RCWAV10Agent, loss: torch.Tensor, block: str, *, retain_graph: bool) -> torch.Tensor:
    params = [p for _, p in _parameter_blocks(agent)[block]]
    return _flat_grad(loss, params, retain_graph=retain_graph).detach().cpu()


def gradient_diagnostics(*, agent: RCWAV10Agent, batch: RCWARolloutBatch,
                         weather_by_transition: torch.Tensor) -> tuple[dict[str, Any], dict[str, Any]]:
    if weather_by_transition.shape != (batch.size,):
        raise ValueError("weather_by_transition shape mismatch")
    all_idx = torch.arange(batch.size, device=agent.device)
    reward_loss, risk_loss, total_loss = _score_losses(agent, batch, all_idx)
    blocks = ("trunk", "gate", "amount")
    full: dict[str, dict[str, torch.Tensor]] = {x: {} for x in ("reward", "risk", "total")}
    for block in blocks:
        full["reward"][block] = _block_vector(agent, reward_loss, block, retain_graph=True)
        full["risk"][block] = _block_vector(agent, risk_loss, block, retain_graph=True)
        full["total"][block] = _block_vector(agent, total_loss, block, retain_graph=(block != blocks[-1]))

    years = sorted(int(x) for x in torch.unique(weather_by_transition).tolist())
    weather_vectors: dict[int, dict[str, torch.Tensor]] = {}
    for year in years:
        idx_cpu = torch.nonzero(weather_by_transition == year, as_tuple=False).squeeze(-1)
        idx = idx_cpu.to(agent.device)
        _, _, loss = _score_losses(agent, batch, idx)
        weather_vectors[year] = {}
        for j, block in enumerate(blocks):
            weather_vectors[year][block] = _block_vector(
                agent, loss, block, retain_graph=(j != len(blocks) - 1)
            )

    summary: dict[str, Any] = {
        "gradient_definition": "score-function gradient at frozen on-policy checkpoint; PPO ratio=1, no clipping active",
        "weather_block_snr_definition": "||mean(g_w)|| / sqrt(mean(||g_w-mean(g_w)||^2))",
        "full": {},
        "weather": {},
    }
    for block in blocks:
        g_reward = full["reward"][block]
        g_risk = full["risk"][block]
        g_total = full["total"][block]
        rows = torch.stack([weather_vectors[y][block] for y in years])
        g_mean = rows.mean(dim=0)
        noise = torch.sqrt(torch.mean(torch.sum((rows - g_mean) ** 2, dim=1)))
        signal = torch.linalg.vector_norm(g_mean)
        snr = float(signal / noise) if float(noise) > 0 else float("inf")
        loo_cos: list[float] = []
        loo_norm_ratio: list[float] = []
        for i in range(len(years)):
            loo = torch.cat((rows[:i], rows[i + 1:]), dim=0).mean(dim=0)
            loo_cos.append(_cosine(loo, g_mean))
            denom = float(torch.linalg.vector_norm(g_mean))
            loo_norm_ratio.append(float(torch.linalg.vector_norm(loo)) / denom if denom > 0 else float("nan"))
        summary["full"][block] = {
            "reward_norm": float(torch.linalg.vector_norm(g_reward)),
            "risk_norm": float(torch.linalg.vector_norm(g_risk)),
            "total_norm": float(torch.linalg.vector_norm(g_total)),
            "reward_risk_cosine": _cosine(g_reward, g_risk),
            "weather_signal_norm": float(signal),
            "weather_noise_rms": float(noise),
            "weather_block_snr": snr,
            "loo_cosine_min": min(loo_cos),
            "loo_cosine_median": median(loo_cos),
            "loo_norm_ratio_min": min(loo_norm_ratio),
            "loo_norm_ratio_max": max(loo_norm_ratio),
        }
        for year, vecs in weather_vectors.items():
            summary["weather"].setdefault(str(year), {})[block] = {
                "total_norm": float(torch.linalg.vector_norm(vecs[block])),
                "cosine_to_full_weather_mean": _cosine(vecs[block], g_mean),
            }
    tensors = {
        "years": torch.as_tensor(years, dtype=torch.int64),
        "weather_total_gradients": {
            block: torch.stack([weather_vectors[y][block] for y in years]) for block in blocks
        },
        "full_reward_gradients": {block: full["reward"][block] for block in blocks},
        "full_risk_gradients": {block: full["risk"][block] for block in blocks},
        "full_total_gradients": {block: full["total"][block] for block in blocks},
    }
    return summary, tensors


def _transition_metadata(cells: Iterable[WeatherEtaCell], outcomes: tuple[RCWAEpisodeOutcome, ...]):
    weather: list[int] = []
    episode_index: list[int] = []
    h_i: list[float] = []
    for i, (cell, outcome) in enumerate(zip(cells, outcomes, strict=True)):
        weather.extend([int(cell.weather_year)] * int(outcome.step_count))
        episode_index.extend([i] * int(outcome.step_count))
    return torch.as_tensor(weather, dtype=torch.int64), torch.as_tensor(episode_index, dtype=torch.int64)


def run_diagnostic(*, project_root: Path, checkpoint: Path, output_dir: Path,
                   runtime_base: Path, device: torch.device) -> dict[str, Any]:
    payload, protocol, seed, agent, normalizer = _load_frozen_checkpoint(
        project_root=project_root, checkpoint=checkpoint, device=device
    )
    before_norm = normalizer.state()
    before_dual = dict(agent.dual_by_eta)
    before_actor = {k: v.detach().cpu().clone() for k, v in agent.actor.state_dict().items()}
    references = _load_references(project_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    env_factory = PPORealEnvFactory(
        project_root=project_root,
        work_dir=output_dir / "env",
        runtime_base=runtime_base,
        env_idx=0,
    )
    cells, outcomes, batch = _collect_frozen_rollout(
        agent=agent, normalizer=normalizer, env_factory=env_factory, references=references
    )
    weather, episode_index = _transition_metadata(cells, outcomes)
    grad_summary, grad_tensors = gradient_diagnostics(
        agent=agent, batch=batch, weather_by_transition=weather
    )
    # Frozen-state assertions after all autograd diagnostics.
    if normalizer.state() != before_norm:
        raise RuntimeError("diagnostic mutated observation normalizer")
    if dict(agent.dual_by_eta) != before_dual:
        raise RuntimeError("diagnostic mutated dual variables")
    for name, tensor in agent.actor.state_dict().items():
        if not torch.equal(tensor.detach().cpu(), before_actor[name]):
            raise RuntimeError(f"diagnostic mutated actor parameter {name}")

    episode_hinge = batch.episode_risk_costs.detach().cpu()
    h_i = episode_hinge[episode_index]
    lam = torch.empty_like(batch.etas)
    for eta, value in before_dual.items():
        mask = torch.isclose(batch.etas, torch.tensor(float(eta)), atol=1e-6, rtol=0.0)
        lam[mask] = float(value)
    actor_weight = batch.reward_advantages - lam * batch.risk_advantages
    trace_path = output_dir / "captured_rollout.pt"
    torch.save({
        "diagnostic_protocol_id": DIAGNOSTIC_PROTOCOL_ID,
        "source_commit": SOURCE_COMMIT,
        "checkpoint": str(checkpoint),
        "seed": seed,
        "checkpoint_update": int(payload["agent"]["update_index"]),
        "diagnostic_action_seed": DIAGNOSTIC_ACTION_SEED,
        "states": batch.states.cpu(),
        "irrigate": batch.irrigate.cpu(),
        "raw_amount": batch.raw_amount.cpu(),
        "old_log_probs": batch.old_log_probs.cpu(),
        "etas": batch.etas.cpu(),
        "weather_year": weather,
        "episode_index": episode_index,
        "reward_advantages": batch.reward_advantages.cpu(),
        "risk_advantages": batch.risk_advantages.cpu(),
        "risk_returns_h_i": h_i,
        "actor_weight": actor_weight.cpu(),
        "episode_etas": batch.episode_etas.cpu(),
        "episode_retentions": batch.episode_retentions.cpu(),
        "episode_hinge_costs": episode_hinge,
        "gradient_tensors": grad_tensors,
    }, trace_path)
    summary = {
        "status": "passed",
        "diagnostic_protocol_id": DIAGNOSTIC_PROTOCOL_ID,
        "source_protocol_id": SOURCE_PROTOCOL_ID,
        "source_commit": SOURCE_COMMIT,
        "checkpoint": str(checkpoint),
        "seed": seed,
        "checkpoint_update": int(payload["agent"]["update_index"]),
        "diagnostic_action_seed": DIAGNOSTIC_ACTION_SEED,
        "episode_count": len(outcomes),
        "transition_count": batch.size,
        "normalizer_frozen": True,
        "dual_frozen": True,
        "actor_frozen": True,
        "optimizer_step_performed": False,
        "mean_retention": mean(float(x.signals.yield_retention) for x in outcomes),
        "mean_policy_irrigation_mm": mean(float(x.policy_irrigation_mm) for x in outcomes),
        "tail_metrics": {k: asdict(v) for k, v in batch.tail_metrics.items()},
        "gradient_diagnostics": grad_summary,
        "trace_path": str(trace_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen RCWA-v10 credit diagnostic; no optimizer updates")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--runtime-base", required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    result = run_diagnostic(
        project_root=Path(args.project_root).expanduser().resolve(),
        checkpoint=Path(args.checkpoint).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        runtime_base=Path(args.runtime_base).expanduser().resolve(),
        device=torch.device(args.device),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
