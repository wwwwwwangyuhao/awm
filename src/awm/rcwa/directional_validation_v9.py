"""Real-DSSAT local directional validation for frozen RCWA-v9 policy gradients.

Diagnostic only: no training protocol is modified.  A formal update-1 rollout is
collected from the frozen seed-21 initial policy.  Three natural-gradient
directions are computed at the same policy and KL metric: water-only,
Tail-Q-risk-only, and the v9 combined primal-dual direction.  For each direction
we construct symmetric policies at exact KL delta/16 and evaluate them on all
54 training weather/eta cells with common random numbers.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from awm.ppo.normalization import NormalizerState, RunningObservationNormalizer
from awm.ppo.real_env import PPORealEnvFactory
from awm.ppo.scheduler import balanced_training_cycle
from awm.rcwa.agent_v9 import RCWAV9Agent, clip_envelope_kl_radius
from awm.rcwa.risk_batch import evaluate_eta_tail
from awm.rcwa.rollout import collect_balanced_training_rollout
from awm.rcwa.trainer import _load_references
from awm.rcwa.trainer_v9 import RCWAV9Trainer

ETA_LEVELS = (0.90, 0.95, 0.98)
DIAG_RADIUS_FRACTION = 1.0 / 16.0
DIAG_CRN_SEED = 917_431


def _flat_actor(agent: RCWAV9Agent) -> tuple[list[torch.nn.Parameter], torch.Tensor]:
    params = list(agent.actor.parameters())
    return params, agent._flat_parameters(params).to(agent.device)


def _risk_z(agent: RCWAV9Agent, risk_adv: torch.Tensor, etas: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(risk_adv)
    for key, mask in agent._eta_masks(etas).items():
        z, _, _ = agent._standardize(risk_adv[mask])
        out[mask] = z
    return out


def _natural_direction(
    agent: RCWAV9Agent,
    *,
    states: torch.Tensor,
    irrigate: torch.Tensor,
    raw_amount: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantage: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    params, base = _flat_actor(agent)
    with torch.no_grad():
        old_dist, old_gate = agent.actor.components(states)
        old_mean = old_dist.loc.detach().clone()
        old_scale = old_dist.scale.detach().clone()
        old_gate = old_gate.detach().clone()
        recomputed, _ = agent.actor.evaluate_behavior(states, irrigate, raw_amount)
        lp_error = float((recomputed - old_log_probs).abs().max().item())
    if lp_error > 2e-5:
        raise RuntimeError(f"rollout logprob mismatch: {lp_error}")

    new_lp, _ = agent.actor.evaluate_behavior(states, irrigate, raw_amount)
    surrogate = (torch.exp(new_lp - old_log_probs) * advantage).mean()
    grads = torch.autograd.grad(surrogate, params, allow_unused=True)
    g = agent._flatten_tensors(grads, params).detach()

    def hvp(vector: torch.Tensor) -> torch.Tensor:
        kl = agent._mean_exact_kl(
            states,
            old_gate_logits=old_gate,
            old_mean=old_mean,
            old_scale=old_scale,
        )
        first = torch.autograd.grad(kl, params, create_graph=True, allow_unused=True)
        flat_first = agent._flatten_tensors(first, params)
        second = torch.autograd.grad(torch.dot(flat_first, vector), params, allow_unused=True)
        return agent._flatten_tensors(second, params).detach()

    direction, iters, rel = agent._conjugate_gradient(hvp, g)
    q = float(torch.dot(g, direction).item())
    if not math.isfinite(q) or q <= 0.0:
        raise RuntimeError("non-positive natural-gradient quadratic")
    agent._set_flat_parameters(params, base)
    return direction, {
        "gradient_norm": float(torch.linalg.vector_norm(g).item()),
        "natural_quadratic": q,
        "cg_iterations": float(iters),
        "cg_relative_residual": float(rel),
        "surrogate_at_base": float(surrogate.detach().item()),
    }


def _signed_alpha_for_kl(
    agent: RCWAV9Agent,
    *,
    states: torch.Tensor,
    base: torch.Tensor,
    direction: torch.Tensor,
    target_kl: float,
    sign: float,
) -> tuple[float, float]:
    params = list(agent.actor.parameters())
    # Every signed solve must use the same frozen base policy as its KL reference.
    # The caller may have just materialized the opposite-sign variant.
    agent._set_flat_parameters(params, base)
    with torch.no_grad():
        old_dist, old_gate = agent.actor.components(states)
        old_mean = old_dist.loc.detach().clone()
        old_scale = old_dist.scale.detach().clone()
        old_gate = old_gate.detach().clone()

    @torch.no_grad()
    def kl_at(magnitude: float) -> float:
        agent._set_flat_parameters(params, base + float(sign * magnitude) * direction)
        return float(agent._mean_exact_kl(
            states,
            old_gate_logits=old_gate,
            old_mean=old_mean,
            old_scale=old_scale,
        ).item())

    lo, hi = 0.0, 1.0
    while kl_at(hi) < target_kl:
        hi *= 2.0
        if hi > 1e6:
            agent._set_flat_parameters(params, base)
            raise RuntimeError("could not bracket target KL")
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        if kl_at(mid) < target_kl:
            lo = mid
        else:
            hi = mid
    mag = 0.5 * (lo + hi)
    achieved = kl_at(mag)
    agent._set_flat_parameters(params, base)
    return float(sign * mag), achieved


def _cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def prepare(root: Path, output: Path, runtime_base: Path, device: str) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"diagnostic output already exists and is non-empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    trainer = RCWAV9Trainer(
        project_root=root,
        seed=21,
        device=device,
        output_dir=output / "rollout_source",
        runtime_base=runtime_base / "rollout",
    )
    pre_norm = trainer.normalizer.state()
    rollout = collect_balanced_training_rollout(
        agent=trainer.agent,
        normalizer=trainer.normalizer,
        env_factory=trainer.env_factory,
        reference_yield_by_year=trainer.references,
        training_seed=21,
        update_index=1,
    )
    batch = rollout.batch
    agent = trainer.agent
    states = batch.states.to(agent.device)
    irrigate = batch.irrigate.to(agent.device)
    raw_amount = batch.raw_amount.to(agent.device)
    old_log_probs = batch.old_log_probs.to(agent.device)
    etas = batch.etas.to(agent.device)
    reward_adv = batch.reward_advantages.to(agent.device)

    risk_credit, credit_diag = agent._actor_risk_credit(batch=batch, states=states, etas=etas)
    reward_z, _, _ = agent._standardize(reward_adv)
    risk_z = _risk_z(agent, risk_credit, etas)
    lambdas = agent._lambda_tensor(etas)
    advantages = {
        "water": reward_z.detach(),
        "risk": (-lambdas * risk_z).detach(),
        "combined": (reward_z - lambdas * risk_z).detach(),
    }

    params, base = _flat_actor(agent)
    actor_base = _cpu_state_dict(agent.actor)
    radius = clip_envelope_kl_radius(agent.hparams.clip_epsilon)
    target = radius * DIAG_RADIUS_FRACTION
    variants = output / "variants"
    variants.mkdir(parents=True, exist_ok=True)
    torch.save(actor_base, variants / "base.pt")
    direction_meta: dict[str, Any] = {}
    for name, advantage in advantages.items():
        agent._set_flat_parameters(params, base)
        direction, meta = _natural_direction(
            agent,
            states=states,
            irrigate=irrigate,
            raw_amount=raw_amount,
            old_log_probs=old_log_probs,
            advantage=advantage,
        )
        entry: dict[str, Any] = dict(meta)
        for label, sign in (("plus", 1.0), ("minus", -1.0)):
            alpha, achieved = _signed_alpha_for_kl(
                agent,
                states=states,
                base=base,
                direction=direction,
                target_kl=target,
                sign=sign,
            )
            agent._set_flat_parameters(params, base + alpha * direction)
            torch.save(_cpu_state_dict(agent.actor), variants / f"{name}_{label}.pt")
            entry[f"{label}_alpha"] = alpha
            entry[f"{label}_exact_kl"] = achieved
            agent._set_flat_parameters(params, base)
        agent._set_flat_parameters(params, base)
        direction_meta[name] = entry

    normalizer_payload = asdict(pre_norm)
    (output / "normalizer_pre_rollout.json").write_text(
        json.dumps(normalizer_payload, indent=2) + "\n", encoding="utf-8"
    )
    source_summary = {
        "episode_count": len(rollout.outcomes),
        "mean_policy_irrigation_mm": float(np.mean([x.policy_irrigation_mm for x in rollout.outcomes])),
        "mean_retention": float(np.mean([x.signals.yield_retention for x in rollout.outcomes])),
    }
    manifest = {
        "diagnostic_id": "awm-rcwa-v9-directional-validation-v1",
        "source_protocol": "awm-rcwa-rl-v9",
        "source_commit": "108d7c573fd49f81cb7ebb529df7534db334f773",
        "training_seed": 21,
        "rollout_update_index": 1,
        "trust_region_radius": radius,
        "diagnostic_radius_fraction": DIAG_RADIUS_FRACTION,
        "target_exact_kl": target,
        "common_random_seed": DIAG_CRN_SEED,
        "normalizer_semantics": "pre-rollout frozen normalizer used by the gradient-estimation rollout",
        "source_rollout": source_summary,
        "directions": direction_meta,
        "tail_q_diagnostics": {
            "credit_std_by_eta": credit_diag["tail_q_credit_std_by_eta"],
            "early_amount_correlation_by_eta": credit_diag["tail_q_credit_early_window_amount_correlation_by_eta"],
        },
    }
    (output / "diagnostic_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def _fixed_draws(year: int, eta: float) -> tuple[np.ndarray, np.ndarray]:
    eta_code = {0.90: 90, 0.95: 95, 0.98: 98}[round(float(eta), 2)]
    seed = DIAG_CRN_SEED + int(year) * 1000 + eta_code
    rng = np.random.default_rng(seed)
    return rng.random(125), rng.standard_normal(125)


def evaluate_variant(
    root: Path,
    output: Path,
    runtime_base: Path,
    variant: str,
    device: str,
) -> None:
    destination = output / "evaluations" / f"{variant}.json"
    if destination.exists():
        raise FileExistsError(destination)
    actor_path = output / "variants" / f"{variant}.pt"
    if not actor_path.exists():
        raise FileNotFoundError(actor_path)
    trainer = RCWAV9Trainer(
        project_root=root,
        seed=21,
        device=device,
        output_dir=output / "evaluation_work" / variant,
        runtime_base=runtime_base / variant,
    )
    trainer.agent.actor.load_state_dict(torch.load(actor_path, map_location=trainer.agent.device, weights_only=True))
    raw_norm = json.loads((output / "normalizer_pre_rollout.json").read_text())
    trainer.normalizer.load_state(NormalizerState(
        count=float(raw_norm["count"]),
        mean=tuple(float(x) for x in raw_norm["mean"]),
        variance=tuple(float(x) for x in raw_norm["variance"]),
    ))
    cells = balanced_training_cycle(seed=21, update_index=1)
    rows: list[dict[str, Any]] = []
    for cell in cells:
        env = trainer.env_factory(cell)
        uniforms, normals = _fixed_draws(cell.weather_year, cell.eta)
        try:
            obs, _ = env.reset()
            terminal = None
            for step_index in range(125):
                state_np = trainer.normalizer.normalize(np.asarray(obs.flat(), dtype=np.float32))
                state = torch.from_numpy(state_np).unsqueeze(0).to(trainer.agent.device)
                with torch.no_grad():
                    dist, logits = trainer.agent.actor.components(state)
                    p = float(torch.sigmoid(logits).item())
                    active = bool(uniforms[step_index] < p)
                    raw = float(dist.loc.item() + dist.scale.item() * normals[step_index])
                    amount = (math.tanh(raw) + 1.0) * 0.5 if active else 0.0
                result = env.step(irrigate=active, amount_fraction=amount)
                obs = result.observation
                if result.terminated:
                    terminal = dict(result.info)
                    break
            if terminal is None:
                raise RuntimeError("directional evaluation episode did not terminate")
            retention = float(terminal["HWAM"]) / float(trainer.references[int(cell.weather_year)])
            rows.append({
                "year": int(cell.weather_year),
                "eta": float(cell.eta),
                "retention": retention,
                "policy_irrigation_mm": float(terminal["policy_irrigation_mm"]),
                "total_irrigation_mm": float(terminal["IRCM"]),
                "yield_kg_ha": float(terminal["HWAM"]),
            })
        finally:
            env.close()

    eta_summary: dict[str, Any] = {}
    for eta in ETA_LEVELS:
        group = [r for r in rows if abs(r["eta"] - eta) <= 1e-12]
        metric = evaluate_eta_tail([r["retention"] for r in group], eta=eta)
        eta_summary[f"{eta:.2f}"] = {
            "mean_retention": float(np.mean([r["retention"] for r in group])),
            "lcvar": float(metric.empirical_lcvar),
            "violation": float(metric.violation),
            "mean_policy_irrigation_mm": float(np.mean([r["policy_irrigation_mm"] for r in group])),
        }
    mean_water_return = -float(np.mean([r["policy_irrigation_mm"] for r in rows])) / 495.0
    mean_lcvar = float(np.mean([eta_summary[f"{e:.2f}"]["lcvar"] for e in ETA_LEVELS]))
    mean_violation = float(np.mean([eta_summary[f"{e:.2f}"]["violation"] for e in ETA_LEVELS]))
    true_lagrangian = mean_water_return - mean_violation
    payload = {
        "variant": variant,
        "episode_count": len(rows),
        "mean_retention": float(np.mean([r["retention"] for r in rows])),
        "mean_policy_irrigation_mm": float(np.mean([r["policy_irrigation_mm"] for r in rows])),
        "mean_water_return": mean_water_return,
        "mean_lcvar": mean_lcvar,
        "mean_violation": mean_violation,
        "true_lagrangian": true_lagrangian,
        "eta_summary": eta_summary,
        "cells": rows,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("variant","mean_retention","mean_policy_irrigation_mm","mean_lcvar","true_lagrangian")}, indent=2))


def aggregate(output: Path) -> None:
    variants = ["base", "water_plus", "water_minus", "risk_plus", "risk_minus", "combined_plus", "combined_minus"]
    data = {v: json.loads((output / "evaluations" / f"{v}.json").read_text()) for v in variants}
    base = data["base"]
    rows = []
    for v in variants:
        d = data[v]
        rows.append({
            "variant": v,
            "mean_policy_irrigation_mm": d["mean_policy_irrigation_mm"],
            "delta_policy_irrigation_mm": d["mean_policy_irrigation_mm"] - base["mean_policy_irrigation_mm"],
            "mean_retention": d["mean_retention"],
            "delta_mean_retention": d["mean_retention"] - base["mean_retention"],
            "mean_lcvar": d["mean_lcvar"],
            "delta_mean_lcvar": d["mean_lcvar"] - base["mean_lcvar"],
            "true_lagrangian": d["true_lagrangian"],
            "delta_true_lagrangian": d["true_lagrangian"] - base["true_lagrangian"],
        })
    checks = {
        "water_direction_correct": data["water_plus"]["mean_policy_irrigation_mm"] < data["water_minus"]["mean_policy_irrigation_mm"],
        "risk_direction_correct": data["risk_plus"]["mean_lcvar"] > data["risk_minus"]["mean_lcvar"],
        "combined_direction_correct": data["combined_plus"]["true_lagrangian"] > data["combined_minus"]["true_lagrangian"],
        "risk_plus_improves_over_base": data["risk_plus"]["mean_lcvar"] > base["mean_lcvar"],
        "water_plus_improves_over_base": data["water_plus"]["mean_policy_irrigation_mm"] < base["mean_policy_irrigation_mm"],
        "combined_plus_improves_over_base": data["combined_plus"]["true_lagrangian"] > base["true_lagrangian"],
    }
    payload = {"rows": rows, "directional_checks": checks}
    (output / "directional_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("prepare", "evaluate", "aggregate"))
    p.add_argument("--project-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--runtime-base", default="/home/wangyh24/dr/v9dir1")
    p.add_argument("--device", default="cpu")
    p.add_argument("--variant")
    a = p.parse_args()
    root = Path(a.project_root).expanduser().resolve()
    output = Path(a.output).expanduser().resolve()
    runtime = Path(a.runtime_base).expanduser().resolve()
    if a.mode == "prepare":
        prepare(root, output, runtime, a.device)
    elif a.mode == "evaluate":
        if not a.variant:
            p.error("--variant required for evaluate")
        evaluate_variant(root, output, runtime, a.variant, a.device)
    else:
        aggregate(output)


if __name__ == "__main__":
    main()
