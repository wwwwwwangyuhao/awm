"""Weather-block jackknife for RCWA-v9 Tail-Q action credit.

Diagnostic only. Replays one formal batch exactly, then refits Tail-Q from the
same checkpoint while leaving out one weather year's three eta episodes.
All credits/actor gradients are evaluated on the same full 6750-transition bank.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch

from awm.ppo.scheduler import balanced_training_cycle
from awm.rcwa.fisher_cg_diagnostic_v9 import _flat_tensors, _cosine
from awm.rcwa.rollout import collect_balanced_training_rollout
from awm.rcwa.trajectory_directional_validation import (
    _replay_identity, _restore_source_checkpoint, _training_row,
)
from awm.rcwa.trainer_v9 import RCWAV9Trainer

def _fit_credit(agent, *, states, action_features, risk_returns, fit_mask, etas):
    agent._prefit_tail_q_critic(
        states=states[fit_mask],
        action_features=action_features[fit_mask],
        risk_returns=risk_returns[fit_mask],
    )
    with torch.no_grad():
        q = agent.tail_q_critic(states, action_features).detach()
        baseline, *_ = agent._policy_q_baseline(states)
        credit = (q - baseline).detach()
    return credit


def _combined_gradient(agent, *, states, irrigate, raw_amount, old_log_probs,
                       reward_adv, risk_credit, etas):
    combined, _ = agent._condition_actor_advantages(
        reward_adv=reward_adv, risk_adv=risk_credit, etas=etas,
    )
    new_lp, _ = agent.actor.evaluate_behavior(states, irrigate, raw_amount)
    surrogate = (torch.exp(new_lp - old_log_probs) * combined.detach()).mean()
    params = list(agent.actor.parameters())
    grads = torch.autograd.grad(surrogate, params, allow_unused=True)
    return _flat_tensors(list(grads), params).detach(), float(surrogate.item())


def _safe_corr(agent, x, y):
    return agent._safe_correlation(x, y)
def diagnose(*, root: Path, checkpoint: Path, metrics: Path, rollout_update: int,
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
    formal_row = _training_row(metrics, rollout_update)
    replay = _replay_identity(rollout, formal_row)
    batch = rollout.batch
    cells = balanced_training_cycle(seed=21, update_index=rollout_update)
    if len(cells) != len(rollout.outcomes) or any(o.step_count != 125 for o in rollout.outcomes):
        raise RuntimeError("unexpected episode layout")
    years = torch.as_tensor(
        [int(cell.weather_year) for cell in cells for _ in range(125)],
        dtype=torch.int64, device=trainer.agent.device,
    )
    if years.numel() != batch.size:
        raise RuntimeError("weather/transition mapping mismatch")
    template = copy.deepcopy(trainer.agent)
    states = batch.states.to(template.device)
    etas = batch.etas.to(template.device)
    irrigate = batch.irrigate.to(template.device)
    raw_amount = batch.raw_amount.to(template.device)
    old_log_probs = batch.old_log_probs.to(template.device)
    reward_adv = batch.reward_advantages.to(template.device)
    risk_returns = batch.risk_returns.to(template.device)
    action_features = template._action_features_from_batch(batch, device=template.device)
    full_mask = torch.ones(batch.size, dtype=torch.bool, device=template.device)

    reference = copy.deepcopy(template)
    full_credit = _fit_credit(
        reference, states=states, action_features=action_features,
        risk_returns=risk_returns, fit_mask=full_mask, etas=etas,
    )
    full_grad, full_surrogate = _combined_gradient(
        reference, states=states, irrigate=irrigate, raw_amount=raw_amount,
        old_log_probs=old_log_probs, reward_adv=reward_adv,
        risk_credit=full_credit, etas=etas,
    )
    full_credit_norm = float(torch.linalg.vector_norm(full_credit).item())
    full_grad_norm = float(torch.linalg.vector_norm(full_grad).item())
    full_tail_diag = reference._tail_q_diagnostics(
        q_values=reference.tail_q_critic(states, action_features).detach(),
        credit=full_credit, etas=etas,
    )
    full_early_diag = reference._tail_q_early_diagnostics(
        batch=batch, action_features=action_features, credit=full_credit, etas=etas,
    )
    formal_opt = formal_row["optimizer"]
    reproduced = True
    reproduction_diffs: dict[str, dict[str, float]] = {}
    for diag_key, formal_key in ((
        "tail_q_credit_std_by_eta", "tail_q_credit_std_by_eta"
    ), (
        "tail_q_credit_early_window_amount_correlation_by_eta",
        "tail_q_credit_early_window_amount_correlation_by_eta"
    )):
        source = full_tail_diag if diag_key in full_tail_diag else full_early_diag
        reproduction_diffs[formal_key] = {}
        for eta_key, formal_value in formal_opt[formal_key].items():
            diff = abs(float(source[diag_key][eta_key]) - float(formal_value))
            reproduction_diffs[formal_key][eta_key] = diff
            reproduced = reproduced and diff <= 1e-8
    if not reproduced:
        raise RuntimeError(f"full-data Tail-Q refit did not reproduce formal diagnostics: {reproduction_diffs}")
    if full_credit_norm <= 0.0 or full_grad_norm <= 0.0:
        raise RuntimeError("reference credit/gradient norm is zero")

    replicates: list[dict[str, Any]] = []
    for year in sorted({int(x) for x in years.detach().cpu().tolist()}):
        replicate = copy.deepcopy(template)
        fit_mask = years != int(year)
        if int(fit_mask.sum().item()) != 6375:
            raise RuntimeError(f"unexpected LOO transition count for {year}")
        credit = _fit_credit(
            replicate, states=states, action_features=action_features,
            risk_returns=risk_returns, fit_mask=fit_mask, etas=etas,
        )
        grad, surrogate = _combined_gradient(
            replicate, states=states, irrigate=irrigate, raw_amount=raw_amount,
            old_log_probs=old_log_probs, reward_adv=reward_adv,
            risk_credit=credit, etas=etas,
        )
        row: dict[str, Any] = {
            "weather_year_left_out": year,
            "credit_cosine_to_full": _cosine(credit, full_credit),
            "credit_norm_ratio": float(torch.linalg.vector_norm(credit).item()) / full_credit_norm,
            "gradient_cosine_to_full": _cosine(grad, full_grad),
            "gradient_norm_ratio": float(torch.linalg.vector_norm(grad).item()) / full_grad_norm,
            "surrogate_at_base": surrogate,
        }
        credit_cos_by_eta: dict[str, float] = {}
        for key, mask in replicate._eta_masks(etas).items():
            credit_cos_by_eta[key] = _cosine(credit[mask], full_credit[mask])
        row["credit_cosine_to_full_by_eta"] = credit_cos_by_eta
        early_diag = replicate._tail_q_early_diagnostics(
            batch=batch, action_features=action_features,
            credit=credit, etas=etas,
        )
        row["early_amount_correlation_by_eta"] = early_diag[
            "tail_q_credit_early_window_amount_correlation_by_eta"
        ]
        replicates.append(row)

    grad_cos = [float(r["gradient_cosine_to_full"]) for r in replicates]
    credit_cos = [float(r["credit_cosine_to_full"]) for r in replicates]
    payload = {
        "diagnostic_id": "awm-rcwa-v9-tail-q-weather-jackknife-v1",
        "source_checkpoint": str(checkpoint.resolve()),
        "source_checkpoint_update": int(trainer.agent.update_index),
        "rollout_update_index": int(rollout_update),
        "replay_identity": replay,
        "independent_unit": "weather_year_block_with_three_eta_episodes",
        "full_transition_count": int(batch.size),
        "loo_transition_count": 6375,
        "weather_block_count": 18,
        "full_refit_reproduced_formal_diagnostics": reproduced,
        "full_refit_reproduction_abs_diffs": reproduction_diffs,
        "reference": {
            "credit_norm": full_credit_norm,
            "gradient_norm": full_grad_norm,
            "surrogate_at_base": full_surrogate,
        },
        "summary": {
            "gradient_cosine_min": min(grad_cos),
            "gradient_cosine_mean": sum(grad_cos) / len(grad_cos),
            "gradient_cosine_below_0p9": sum(x < 0.9 for x in grad_cos),
            "gradient_cosine_below_0p5": sum(x < 0.5 for x in grad_cos),
            "gradient_cosine_negative": sum(x < 0.0 for x in grad_cos),
            "credit_cosine_min": min(credit_cos),
            "credit_cosine_mean": sum(credit_cos) / len(credit_cos),
        },
        "replicates": replicates,
    }
    (output / "tail_q_weather_jackknife.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload["summary"], indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--project-root', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--source-metrics', required=True)
    p.add_argument('--rollout-update', type=int, required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--runtime-base', required=True)
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    diagnose(
        root=Path(a.project_root).resolve(), checkpoint=Path(a.checkpoint).resolve(),
        metrics=Path(a.source_metrics).resolve(), rollout_update=a.rollout_update,
        output=Path(a.output).resolve(), runtime_base=Path(a.runtime_base).resolve(),
        device=a.device,
    )


if __name__ == '__main__':
    main()
