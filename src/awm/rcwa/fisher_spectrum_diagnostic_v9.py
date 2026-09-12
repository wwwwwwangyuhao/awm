"""Lanczos spectrum diagnostic for frozen RCWA-v9 Fisher systems."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from awm.rcwa.fisher_cg_diagnostic_v9 import _flat_tensors
from awm.rcwa.rollout import collect_balanced_training_rollout
from awm.rcwa.trajectory_directional_validation import (
    _replay_identity,
    _restore_source_checkpoint,
    _training_row,
)
from awm.rcwa.trainer_v9 import RCWAV9Trainer

LANCZOS_STEPS = 80
RANDOM_SEED = 20260912


def _lanczos(
    hvp: Callable[[torch.Tensor], torch.Tensor],
    start: torch.Tensor,
    *,
    steps: int = LANCZOS_STEPS,
) -> dict[str, Any]:
    q = start / torch.linalg.vector_norm(start)
    q_prev = torch.zeros_like(q)
    beta_prev = torch.zeros((), device=q.device, dtype=q.dtype)
    basis: list[torch.Tensor] = []
    alphas: list[float] = []
    betas: list[float] = []
    stop_reason = "max_steps"
    for k in range(steps):
        z = hvp(q)
        alpha = torch.dot(q, z)
        z = z - alpha * q - beta_prev * q_prev
        # Full reorthogonalization, two passes for float32 stability.
        for _ in range(2):
            for qi in basis:
                z = z - torch.dot(qi, z) * qi
            z = z - torch.dot(q, z) * q
        beta = torch.linalg.vector_norm(z)
        basis.append(q.detach().clone())
        alphas.append(float(alpha.detach().item()))
        if k == steps - 1:
            break
        b = float(beta.detach().item())
        if (not math.isfinite(b)) or b <= 1e-10:
            stop_reason = f"breakdown:{b}"
            break
        betas.append(b)
        q_prev, q = q, z / beta
        beta_prev = beta

    n = len(alphas)
    t = torch.zeros((n, n), dtype=torch.float64)
    for i, a in enumerate(alphas):
        t[i, i] = a
    for i, b in enumerate(betas[: max(0, n - 1)]):
        t[i, i + 1] = b
        t[i + 1, i] = b
    eig = torch.linalg.eigvalsh(t).cpu().numpy()
    max_abs = float(np.max(np.abs(eig))) if eig.size else 0.0
    positive = eig[eig > max(1e-12, max_abs * 1e-8)]
    negative = eig[eig < -max(1e-12, max_abs * 1e-8)]
    near_zero = eig[np.abs(eig) <= max(1e-12, max_abs * 1e-8)]
    return {
        "steps_completed": n,
        "stop_reason": stop_reason,
        "ritz_min": float(eig[0]),
        "ritz_max": float(eig[-1]),
        "negative_ritz_count": int(negative.size),
        "near_zero_ritz_count": int(near_zero.size),
        "min_positive_ritz": float(positive[0]) if positive.size else None,
        "condition_estimate": (
            float(eig[-1] / positive[0]) if positive.size and eig[-1] > 0 else None
        ),
        "ritz_values": [float(x) for x in eig],
    }


def _symmetry_probe(hvp, dimension: int, device: torch.device) -> dict[str, float]:
    gen = torch.Generator(device="cpu").manual_seed(RANDOM_SEED)
    x = torch.randn(dimension, generator=gen).to(device)
    y = torch.randn(dimension, generator=gen).to(device)
    x = x / torch.linalg.vector_norm(x)
    y = y / torch.linalg.vector_norm(y)
    fx, fy = hvp(x), hvp(y)
    x_fy = float(torch.dot(x, fy).item())
    y_fx = float(torch.dot(y, fx).item())
    denom = max(abs(x_fy), abs(y_fx), 1e-30)
    return {
        "x_F_y": x_fy,
        "y_F_x": y_fx,
        "relative_symmetry_error": abs(x_fy - y_fx) / denom,
        "x_F_x": float(torch.dot(x, fx).item()),
        "y_F_y": float(torch.dot(y, fy).item()),
    }


def diagnose(
    *, root: Path, checkpoint: Path, metrics: Path,
    rollout_update: int, output: Path, runtime_base: Path, device: str,
) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    trainer = RCWAV9Trainer(
        project_root=root, seed=21, device=device,
        output_dir=output / "rollout_source", runtime_base=runtime_base / "r",
    )
    payload = _restore_source_checkpoint(trainer, checkpoint, "v9")
    if trainer.agent.update_index + 1 != rollout_update:
        raise ValueError("checkpoint/update mismatch")
    rollout = collect_balanced_training_rollout(
        agent=trainer.agent, normalizer=trainer.normalizer,
        env_factory=trainer.env_factory, reference_yield_by_year=trainer.references,
        training_seed=21, update_index=rollout_update,
    )
    replay = _replay_identity(rollout, _training_row(metrics, rollout_update))
    agent, batch = trainer.agent, rollout.batch
    states = batch.states.to(agent.device)
    etas = batch.etas.to(agent.device)
    irrigate = batch.irrigate.to(agent.device)
    raw_amount = batch.raw_amount.to(agent.device)
    old_log_probs = batch.old_log_probs.to(agent.device)
    reward_adv = batch.reward_advantages.to(agent.device)
    risk_credit, _ = agent._actor_risk_credit(batch=batch, states=states, etas=etas)
    combined_adv, _ = agent._condition_actor_advantages(
        reward_adv=reward_adv, risk_adv=risk_credit, etas=etas,
    )
    params = list(agent.actor.parameters())
    with torch.no_grad():
        old_dist, old_gate = agent.actor.components(states)
        old_mean = old_dist.loc.detach().clone()
        old_scale = old_dist.scale.detach().clone()
        old_gate = old_gate.detach().clone()
    new_lp, _ = agent.actor.evaluate_behavior(states, irrigate, raw_amount)
    surrogate = (torch.exp(new_lp - old_log_probs) * combined_adv.detach()).mean()
    grads = torch.autograd.grad(surrogate, params, allow_unused=True)
    g = _flat_tensors(list(grads), params).detach()

    def hvp(vector: torch.Tensor) -> torch.Tensor:
        kl = agent._mean_exact_kl(
            states,
            old_gate_logits=old_gate,
            old_mean=old_mean,
            old_scale=old_scale,
        )
        first = torch.autograd.grad(kl, params, create_graph=True, allow_unused=True)
        flat_first = _flat_tensors(list(first), params)
        second = torch.autograd.grad(torch.dot(flat_first, vector), params, allow_unused=True)
        return _flat_tensors(list(second), params).detach()
    grad_start = g / torch.linalg.vector_norm(g)
    gen = torch.Generator(device="cpu").manual_seed(RANDOM_SEED + rollout_update)
    random_start = torch.randn(g.numel(), generator=gen).to(g.device)
    random_start = random_start / torch.linalg.vector_norm(random_start)
    spectrum_grad = _lanczos(hvp, grad_start)
    spectrum_random = _lanczos(hvp, random_start)
    symmetry = _symmetry_probe(hvp, g.numel(), g.device)

    out = {
        "diagnostic_id": "awm-rcwa-v9-fisher-spectrum-v1",
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_update": int(payload["agent"]["update_index"]),
        "rollout_update_index": int(rollout_update),
        "replay_identity": replay,
        "actor_parameter_count": int(g.numel()),
        "policy_gradient_norm": float(torch.linalg.vector_norm(g).item()),
        "surrogate_at_base": float(surrogate.detach().item()),
        "gradient_start_lanczos": spectrum_grad,
        "random_start_lanczos": spectrum_random,
        "symmetry_probe": symmetry,
    }
    destination = output / "fisher_spectrum_diagnostic.json"
    destination.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--source-metrics", required=True)
    p.add_argument("--rollout-update", required=True, type=int)
    p.add_argument("--output", required=True)
    p.add_argument("--runtime-base", required=True)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    diagnose(
        root=Path(a.project_root).resolve(),
        checkpoint=Path(a.checkpoint).resolve(),
        metrics=Path(a.source_metrics).resolve(),
        rollout_update=int(a.rollout_update),
        output=Path(a.output).resolve(),
        runtime_base=Path(a.runtime_base).resolve(),
        device=a.device,
    )


if __name__ == "__main__":
    main()
