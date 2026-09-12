"""Prepare damped-Fisher combined-direction variants on frozen v9 nodes.

Diagnostic only. Replays the formal batch exactly, solves the numerically
regularized Fisher system, and materializes symmetric exact-KL variants.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import torch

from awm.rcwa.agent_v9 import clip_envelope_kl_radius
from awm.rcwa.directional_validation_v9 import (
    DIAG_RADIUS_FRACTION,
    _cpu_state_dict,
    _signed_alpha_for_kl,
)
from awm.rcwa.fisher_cg_diagnostic_v9 import _cg_trace, _flat_tensors
from awm.rcwa.fisher_damped_cg_diagnostic_v9 import _path_rows, _true_snapshot
from awm.rcwa.rollout import collect_balanced_training_rollout
from awm.rcwa.trajectory_directional_validation import (
    _replay_identity,
    _restore_source_checkpoint,
    _training_row,
)
from awm.rcwa.trainer_v9 import RCWAV9Trainer

def _state_dict_identical(current: dict[str, torch.Tensor], reference_path: Path) -> bool:
    reference = torch.load(reference_path, map_location="cpu", weights_only=True)
    if current.keys() != reference.keys():
        return False
    return all(torch.equal(current[k].cpu(), reference[k].cpu()) for k in current)


def prepare(
    *, root: Path, checkpoint: Path, metrics: Path, rollout_update: int,
    output: Path, runtime_base: Path, device: str,
    regularization_path: Path, reference_base: Path,
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
    pre_norm = trainer.normalizer.state()
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
    base_state = _cpu_state_dict(agent.actor)
    base_identical = _state_dict_identical(base_state, reference_base)
    if not base_identical:
        raise RuntimeError("replayed base actor does not match reference trajectory base")
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
        return _flat_tensors(list(second), params).detach()

    path_rows = _path_rows(regularization_path)
    row = path_rows[0]
    xi = float(row["xi"])
    target_tolerance = float(row["tolerance"])

    def damped_hvp(vector: torch.Tensor) -> torch.Tensor:
        return f_hvp(vector) + xi * vector

    snapshots, stop_reason, last_iter = _cg_trace(damped_hvp, g, marks=(50,))
    if 50 not in snapshots:
        raise RuntimeError(f"50-step damped CG missing: {stop_reason}")
    direction = snapshots[50].pop("solution").to(agent.device)
    solve_diag = _true_snapshot(
        f_hvp=f_hvp, damped_hvp=damped_hvp,
        g=g, d=direction, xi=xi,
    )
    if solve_diag["true_damped_relative_residual"] > target_tolerance:
        raise RuntimeError(
            f"damped solve missed target tolerance: {solve_diag['true_damped_relative_residual']} > {target_tolerance}"
        )

    radius = clip_envelope_kl_radius(agent.hparams.clip_epsilon)
    target_kl = radius * DIAG_RADIUS_FRACTION
    variants = output / "variants"
    variants.mkdir(parents=True, exist_ok=True)
    torch.save(base_state, variants / "base.pt")
    signed: dict[str, Any] = {}
    for label, sign in (("plus", 1.0), ("minus", -1.0)):
        alpha, achieved = _signed_alpha_for_kl(
            agent, states=states, base=base, direction=direction,
            target_kl=target_kl, sign=sign,
        )
        agent._set_flat_parameters(params, base + alpha * direction)
        torch.save(_cpu_state_dict(agent.actor), variants / f"combined_{label}.pt")
        signed[label] = {"alpha": alpha, "exact_kl": achieved}
        agent._set_flat_parameters(params, base)

    (output / "normalizer_pre_rollout.json").write_text(
        json.dumps(asdict(pre_norm), indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "diagnostic_id": "awm-rcwa-v9-damped-directional-prepare-v1",
        "source_protocol": "awm-rcwa-rl-v9",
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_update": int(payload["agent"]["update_index"]),
        "rollout_update_index": int(rollout_update),
        "replay_identity": replay,
        "reference_base_actor": str(reference_base),
        "base_actor_identical": bool(base_identical),
        "regularization_path": str(regularization_path),
        "damping_xi": xi,
        "target_cg_tolerance": target_tolerance,
        "damped_solve": solve_diag,
        "cg_stop_reason": stop_reason,
        "cg_last_iteration": int(last_iter),
        "target_exact_kl": target_kl,
        "signed_variants": signed,
        "old_logprob_max_abs_error": lp_error,
        "policy_gradient_norm": float(torch.linalg.vector_norm(g).item()),
        "tail_q_credit_std_by_eta": credit_diag["tail_q_credit_std_by_eta"],
        "tail_q_early_amount_correlation_by_eta": credit_diag[
            "tail_q_credit_early_window_amount_correlation_by_eta"
        ],
        "advantage_diagnostics": adv_diag,
    }
    (output / "diagnostic_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))

def main() -> None:
    p = argparse.ArgumentParser(description="Prepare damped-Fisher directional variants")
    p.add_argument("--project-root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--source-metrics", required=True)
    p.add_argument("--rollout-update", type=int, required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--runtime-base", required=True)
    p.add_argument("--regularization-path", required=True)
    p.add_argument("--reference-base", required=True)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    prepare(
        root=Path(a.project_root).resolve(),
        checkpoint=Path(a.checkpoint).resolve(),
        metrics=Path(a.source_metrics).resolve(),
        rollout_update=int(a.rollout_update),
        output=Path(a.output).resolve(),
        runtime_base=Path(a.runtime_base).resolve(),
        device=a.device,
        regularization_path=Path(a.regularization_path).resolve(),
        reference_base=Path(a.reference_base).resolve(),
    )


if __name__ == "__main__":
    main()
