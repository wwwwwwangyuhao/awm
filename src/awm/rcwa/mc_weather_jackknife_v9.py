"""Weather-block jackknife for direct-MC RCWA-v9 policy gradient.

Diagnostic only. Replays one formal batch exactly and compares the full
combined gradient against leave-one-weather-out estimates. No Tail-Q fit,
Fisher solve, actor update, validation, or extra policy evaluation is used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from awm.ppo.scheduler import balanced_training_cycle
from awm.rcwa.fisher_cg_diagnostic_v9 import _flat_tensors, _cosine
from awm.rcwa.rollout import collect_balanced_training_rollout
from awm.rcwa.trajectory_directional_validation import (
    _replay_identity, _restore_source_checkpoint, _training_row,
)
from awm.rcwa.trainer_v9 import RCWAV9Trainer
def _gradient(agent, *, states, irrigate, raw_amount, old_log_probs,
              reward_adv, risk_adv, etas):
    combined, _ = agent._condition_actor_advantages(
        reward_adv=reward_adv, risk_adv=risk_adv, etas=etas,
    )
    new_lp, _ = agent.actor.evaluate_behavior(states, irrigate, raw_amount)
    surrogate = (torch.exp(new_lp - old_log_probs) * combined.detach()).mean()
    params = list(agent.actor.parameters())
    grads = torch.autograd.grad(surrogate, params, allow_unused=True)
    return _flat_tensors(list(grads), params).detach(), float(surrogate.item())


def diagnose(*, root: Path, checkpoint: Path, metrics: Path,
             reference_manifest: Path, rollout_update: int,
             output: Path, runtime_base: Path, device: str) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    trainer = RCWAV9Trainer(
        project_root=root, seed=21, device=device,
        output_dir=output / "rollout_source", runtime_base=runtime_base / "r",
    )
    _restore_source_checkpoint(trainer, checkpoint, "v9")
    if trainer.agent.update_index + 1 != int(rollout_update):
        raise ValueError("checkpoint/update mismatch")
    rollout = collect_balanced_training_rollout(
        agent=trainer.agent, normalizer=trainer.normalizer,
        env_factory=trainer.env_factory, reference_yield_by_year=trainer.references,
        training_seed=21, update_index=rollout_update,
    )
    replay = _replay_identity(rollout, _training_row(metrics, rollout_update))
    batch = rollout.batch
    cells = balanced_training_cycle(seed=21, update_index=rollout_update)
    years = torch.as_tensor(
        [int(cell.weather_year) for cell in cells for _ in range(125)],
        dtype=torch.int64, device=trainer.agent.device,
    )
    if years.numel() != batch.size:
        raise RuntimeError("weather/transition mapping mismatch")
    agent = trainer.agent
    states = batch.states.to(agent.device)
    irrigate = batch.irrigate.to(agent.device)
    raw_amount = batch.raw_amount.to(agent.device)
    old_log_probs = batch.old_log_probs.to(agent.device)
    reward_adv = batch.reward_advantages.to(agent.device)
    risk_adv = batch.risk_advantages.to(agent.device)
    etas = batch.etas.to(agent.device)
    full_grad, full_surrogate = _gradient(
        agent, states=states, irrigate=irrigate, raw_amount=raw_amount,
        old_log_probs=old_log_probs, reward_adv=reward_adv,
        risk_adv=risk_adv, etas=etas,
    )
    full_norm = float(torch.linalg.vector_norm(full_grad).item())
    if full_norm <= 0.0:
        raise RuntimeError("full direct-MC gradient norm is zero")
    ref = json.loads(reference_manifest.read_text(encoding="utf-8"))
    ref_norm = float(ref["policy_gradient_norm"])
    norm_diff = abs(full_norm - ref_norm)
    if norm_diff > 1e-7:
        raise RuntimeError(
            f"full direct-MC gradient norm mismatch: {full_norm} vs {ref_norm}"
        )

    replicates = []
    for year in sorted({int(x) for x in years.detach().cpu().tolist()}):
        mask = years != int(year)
        if int(mask.sum().item()) != 6375:
            raise RuntimeError(f"unexpected LOO transition count for {year}")
        grad, surrogate = _gradient(
            agent, states=states[mask], irrigate=irrigate[mask],
            raw_amount=raw_amount[mask], old_log_probs=old_log_probs[mask],
            reward_adv=reward_adv[mask], risk_adv=risk_adv[mask], etas=etas[mask],
        )
        replicates.append({
            "weather_year_left_out": year,
            "gradient_cosine_to_full": _cosine(grad, full_grad),
            "gradient_norm_ratio": float(torch.linalg.vector_norm(grad).item()) / full_norm,
            "surrogate_at_base": surrogate,
        })

    cosines = [float(r["gradient_cosine_to_full"]) for r in replicates]
    payload = {
        "diagnostic_id": "awm-rcwa-v9-mc-weather-jackknife-v1",
        "source_checkpoint": str(checkpoint.resolve()),
        "rollout_update_index": int(rollout_update),
        "replay_identity": replay,
        "reference_manifest": str(reference_manifest.resolve()),
        "full_gradient_norm": full_norm,
        "reference_gradient_norm": ref_norm,
        "gradient_norm_abs_diff": norm_diff,
        "full_surrogate_at_base": full_surrogate,
        "independent_unit": "weather_year_block_with_three_eta_episodes",
        "weather_block_count": 18,
        "summary": {
            "gradient_cosine_min": min(cosines),
            "gradient_cosine_mean": sum(cosines) / len(cosines),
            "gradient_cosine_below_0p9": sum(x < 0.9 for x in cosines),
            "gradient_cosine_below_0p5": sum(x < 0.5 for x in cosines),
            "gradient_cosine_negative": sum(x < 0.0 for x in cosines),
        },
        "replicates": replicates,
    }
    (output / "mc_weather_jackknife.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload["summary"], indent=2))


def main() -> None:
    p = argparse.ArgumentParser(
        description="Weather-block jackknife for direct-MC v9 policy gradient"
    )
    p.add_argument("--project-root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--source-metrics", required=True)
    p.add_argument("--reference-manifest", required=True)
    p.add_argument("--rollout-update", type=int, required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--runtime-base", required=True)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    diagnose(
        root=Path(a.project_root).resolve(),
        checkpoint=Path(a.checkpoint).resolve(),
        metrics=Path(a.source_metrics).resolve(),
        reference_manifest=Path(a.reference_manifest).resolve(),
        rollout_update=int(a.rollout_update),
        output=Path(a.output).resolve(),
        runtime_base=Path(a.runtime_base).resolve(),
        device=a.device,
    )


if __name__ == "__main__":
    main()
