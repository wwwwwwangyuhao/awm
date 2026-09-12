"""Actual damped-Fisher CG path on frozen RCWA-v9 batches.

Diagnostic only. Replays a formal rollout exactly, reconstructs the same
combined policy gradient and Fisher HVP, then solves (F + xi I)d = g for the
numerically-derived damping path. No policy update is written back.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable

import torch

from awm.rcwa.agent_v9 import clip_envelope_kl_radius
from awm.rcwa.fisher_cg_diagnostic_v9 import (
    _cg_trace,
    _cosine,
    _exact_kl_for_scaled_direction,
    _flat_tensors,
)
from awm.rcwa.rollout import collect_balanced_training_rollout
from awm.rcwa.trajectory_directional_validation import (
    _replay_identity,
    _restore_source_checkpoint,
    _training_row,
)
from awm.rcwa.trainer_v9 import RCWAV9Trainer
MARKS = (50, 100)


def _path_rows(path: Path) -> list[dict[str, float]]:
    payload = json.loads(path.read_text())
    rows: list[dict[str, float]] = []
    for row in payload["rows"]:
        rows.append({
            "tolerance": float(row["tolerance"]),
            "xi": float(row["global_minimum_damping"]),
            "condition_limit": float(row["condition_limit"]),
        })
    if not rows:
        raise ValueError("regularization path contains no rows")
    return rows


def _true_snapshot(
    *, f_hvp: Callable[[torch.Tensor], torch.Tensor],
    damped_hvp: Callable[[torch.Tensor], torch.Tensor],
    g: torch.Tensor, d: torch.Tensor, xi: float,
) -> dict[str, float]:
    fd = f_hvp(d)
    damped_fd = fd + xi * d
    denom = max(float(torch.linalg.vector_norm(g).item()), 1e-30)
    residual = float(torch.linalg.vector_norm(g - damped_fd).item()) / denom
    q_f = float(torch.dot(d, fd).item())
    q_damped = float(torch.dot(d, damped_fd).item())
    return {
        "true_damped_relative_residual": residual,
        "direction_norm": float(torch.linalg.vector_norm(d).item()),
        "cosine_to_gradient": _cosine(d, g),
        "g_dot_d": float(torch.dot(g, d).item()),
        "d_dot_Fd": q_f,
        "d_dot_damped_Fd": q_damped,
    }

def diagnose_node(
    *, root: Path, checkpoint: Path, metrics: Path, rollout_update: int,
    output: Path, runtime_base: Path, device: str, regularization_path: Path,
) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    trainer = RCWAV9Trainer(
        project_root=root, seed=21, device=device,
        output_dir=output / "rollout_source", runtime_base=runtime_base / "r",
    )
    payload = _restore_source_checkpoint(trainer, checkpoint, "v9")
    if trainer.agent.update_index + 1 != int(rollout_update):
        raise ValueError("checkpoint/update mismatch")
    rollout = collect_balanced_training_rollout(
        agent=trainer.agent, normalizer=trainer.normalizer,
        env_factory=trainer.env_factory,
        reference_yield_by_year=trainer.references,
        training_seed=21, update_index=rollout_update,
    )
    replay = _replay_identity(rollout, _training_row(metrics, rollout_update))
    agent = trainer.agent
    batch = rollout.batch
    states = batch.states.to(agent.device)
    etas = batch.etas.to(agent.device)
    irrigate = batch.irrigate.to(agent.device)
    raw_amount = batch.raw_amount.to(agent.device)
    old_log_probs = batch.old_log_probs.to(agent.device)
    reward_adv = batch.reward_advantages.to(agent.device)
    risk_credit, credit_diag = agent._actor_risk_credit(
        batch=batch, states=states, etas=etas,
    )
    combined_adv, adv_diag = agent._condition_actor_advantages(
        reward_adv=reward_adv, risk_adv=risk_credit, etas=etas,
    )
    params = list(agent.actor.parameters())
    base = agent._flat_parameters(params).to(agent.device)
    with torch.no_grad():
        old_dist, old_gate = agent.actor.components(states)
        old_mean = old_dist.loc.detach().clone()
        old_scale = old_dist.scale.detach().clone()
        old_gate = old_gate.detach().clone()
        recomputed, _ = agent.actor.evaluate_behavior(states, irrigate, raw_amount)
        lp_error = float((recomputed - old_log_probs).abs().max().item())
    if lp_error > 2e-5:
        raise RuntimeError(f"old logprob mismatch {lp_error}")
    new_lp, _ = agent.actor.evaluate_behavior(states, irrigate, raw_amount)
    surrogate = (torch.exp(new_lp - old_log_probs) * combined_adv.detach()).mean()
    grads = torch.autograd.grad(surrogate, params, allow_unused=True)
    g = _flat_tensors(list(grads), params).detach()

    def f_hvp(vector: torch.Tensor) -> torch.Tensor:
        kl = agent._mean_exact_kl(
            states,
            old_gate_logits=old_gate,
            old_mean=old_mean,
            old_scale=old_scale,
        )
        first = torch.autograd.grad(kl, params, create_graph=True, allow_unused=True)
        flat_first = _flat_tensors(list(first), params)
        second = torch.autograd.grad(
            torch.dot(flat_first, vector), params, allow_unused=True,
        )
        out = _flat_tensors(list(second), params).detach()
        if not torch.isfinite(out).all():
            raise FloatingPointError("Fisher HVP became NaN/Inf")
        return out

    radius = clip_envelope_kl_radius(agent.hparams.clip_epsilon)
    path_rows = _path_rows(regularization_path)
    results: list[dict[str, Any]] = []
    for row in path_rows:
        xi = float(row["xi"])

        def damped_hvp(vector: torch.Tensor) -> torch.Tensor:
            return f_hvp(vector) + xi * vector

        snapshots, stop_reason, last_iter = _cg_trace(
            damped_hvp, g, marks=MARKS,
        )
        if not snapshots:
            raise RuntimeError(f"damped CG produced no snapshots for xi={xi}")
        solutions: dict[int, torch.Tensor] = {}
        for k, snap in snapshots.items():
            solutions[k] = snap.pop("solution").to(agent.device)
        final_k = max(solutions)
        final = solutions[final_k]
        per_iter: dict[str, Any] = {}
        for k in sorted(solutions):
            d = solutions[k]
            detail = _true_snapshot(
                f_hvp=f_hvp, damped_hvp=damped_hvp,
                g=g, d=d, xi=xi,
            )
            detail["recursive_relative_residual"] = float(
                snapshots[k]["recursive_relative_residual"]
            )
            detail["cosine_to_100"] = _cosine(d, final)
            q_f = float(detail["d_dot_Fd"])
            if q_f > 0.0 and math.isfinite(q_f):
                alpha = math.sqrt(2.0 * radius / q_f)
                exact_kl = _exact_kl_for_scaled_direction(
                    agent, states=states, base=base, direction=d,
                    alpha=alpha, old_gate=old_gate,
                    old_mean=old_mean, old_scale=old_scale,
                )
                detail["quadratic_alpha_original_F"] = alpha
                detail["exact_kl_at_quadratic_alpha"] = exact_kl
                detail["exact_kl_over_radius"] = exact_kl / radius
            else:
                detail["quadratic_alpha_original_F"] = None
                detail["exact_kl_at_quadratic_alpha"] = None
                detail["exact_kl_over_radius"] = None
            per_iter[str(k)] = detail
        results.append({
            "target_tolerance": float(row["tolerance"]),
            "xi": xi,
            "predicted_condition_limit": float(row["condition_limit"]),
            "cg_stop_reason": stop_reason,
            "cg_last_iteration": int(last_iter),
            "iterations": per_iter,
        })

    output_payload = {
        "diagnostic_id": "awm-rcwa-v9-fisher-damped-cg-v1",
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_update": int(payload["agent"]["update_index"]),
        "rollout_update_index": int(rollout_update),
        "replay_identity": replay,
        "policy_gradient_norm": float(torch.linalg.vector_norm(g).item()),
        "surrogate_at_base": float(surrogate.detach().item()),
        "old_logprob_max_abs_error": lp_error,
        "trust_region_radius": radius,
        "regularization_path": str(regularization_path),
        "results": results,
        "tail_q_credit_std_by_eta": credit_diag["tail_q_credit_std_by_eta"],
        "tail_q_early_amount_correlation_by_eta": credit_diag[
            "tail_q_credit_early_window_amount_correlation_by_eta"
        ],
        "advantage_diagnostics": adv_diag,
    }
    destination = output / "fisher_damped_cg_diagnostic.json"
    destination.write_text(json.dumps(output_payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output_payload, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--source-metrics", required=True)
    p.add_argument("--rollout-update", required=True, type=int)
    p.add_argument("--output", required=True)
    p.add_argument("--runtime-base", required=True)
    p.add_argument("--regularization-path", required=True)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    diagnose_node(
        root=Path(a.project_root).resolve(),
        checkpoint=Path(a.checkpoint).resolve(),
        metrics=Path(a.source_metrics).resolve(),
        rollout_update=a.rollout_update,
        output=Path(a.output).resolve(),
        runtime_base=Path(a.runtime_base).resolve(),
        device=a.device,
        regularization_path=Path(a.regularization_path).resolve(),
    )


if __name__ == "__main__":
    main()
