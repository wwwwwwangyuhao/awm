"""RCWA v5: cross-weather, coverage-balanced action-conditioned tail credit.

V5 keeps the RCWA v4 objective, rollout and policy unchanged.  Two persistent
Tail-Q critics are trained on disjoint 2000-2017 weather-year folds.  Every
transition receives action-dependent risk credit from the critic that has never
trained on that transition's weather fold.  Q regression is additionally
balanced over the observed hierarchical-action strata within each eta group.
No validation data, extra environment interaction or fabricated counterfactual
return is used.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from statistics import mean
from typing import Mapping

import torch
import torch.nn.functional as F

from awm.ppo.scheduler import balanced_training_cycle
from awm.risk import REGISTERED_ETA_LEVELS, TRAIN_YEARS

from .agent_v4 import (
    RCWAV4Agent,
    RCWAV4Hyperparameters,
    RCWAV4UpdateStats,
    TailActionValueNetwork,
)
from .buffer import RCWARolloutBatch

V5_PROTOCOL_ID = "awm-rcwa-rl-v5"
ACTION_STRATUM_NAMES = (
    "noop",
    "active_0_025",
    "active_025_050",
    "active_050_075",
    "active_075_100",
)
YEAR_FOLD = {int(year): idx % 2 for idx, year in enumerate(TRAIN_YEARS)}
if sum(v == 0 for v in YEAR_FOLD.values()) != 9 or sum(v == 1 for v in YEAR_FOLD.values()) != 9:
    raise AssertionError("RCWA v5 requires two balanced 9-year training folds")


@dataclass(frozen=True)
class RCWAV5Hyperparameters(RCWAV4Hyperparameters):
    """V5 introduces no new tuned scalar hyperparameters relative to v4."""


@dataclass(frozen=True, slots=True)
class RCWAV5UpdateStats(RCWAV4UpdateStats):
    tail_q_fold_prefit_loss_before: dict[str, float] = field(default_factory=dict)
    tail_q_fold_prefit_loss_after: dict[str, float] = field(default_factory=dict)
    tail_q_fold_prefit_steps: dict[str, int] = field(default_factory=dict)
    tail_q_cross_weather_transition_count: dict[str, int] = field(default_factory=dict)
    tail_q_coverage_stratum_counts: dict[str, dict[str, dict[str, int]]] = field(default_factory=dict)
    tail_q_coverage_max_weight: dict[str, dict[str, float]] = field(default_factory=dict)


class RCWAV5Agent(RCWAV4Agent):
    PROTOCOL_ID = V5_PROTOCOL_ID

    def __init__(
        self,
        *,
        hyperparameters: RCWAV5Hyperparameters | None = None,
        seed: int = 21,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__(
            hyperparameters=hyperparameters or RCWAV5Hyperparameters(),
            seed=seed,
            device=device,
        )
        # The inherited v4 critic is fold 0.  Fold 1 begins bit-identical so
        # cross-fold differences are caused by disjoint weather supervision.
        self.tail_q_critic_fold1: TailActionValueNetwork = copy.deepcopy(
            self.tail_q_critic
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters())
            + list(self.reward_critic.parameters())
            + list(self.risk_critic.parameters())
            + list(self.tail_q_critic.parameters())
            + list(self.tail_q_critic_fold1.parameters()),
            lr=self.hparams.learning_rate,
            betas=(self.hparams.adam_beta1, self.hparams.adam_beta2),
            eps=self.hparams.adam_eps,
        )

    def _transition_weather_folds(
        self, batch: RCWARolloutBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cells = balanced_training_cycle(
            seed=self.seed, update_index=self.update_index + 1
        )
        dones = batch.dones.detach().cpu().to(dtype=torch.bool)
        terminals = torch.nonzero(dones, as_tuple=False).flatten().tolist()
        if len(terminals) != len(cells):
            raise RuntimeError(
                f"RCWA v5 expected {len(cells)} complete episodes, got {len(terminals)}"
            )
        etas = batch.etas.detach().cpu()
        years = torch.empty(batch.size, dtype=torch.int64)
        folds = torch.empty(batch.size, dtype=torch.int64)
        start = 0
        for cell, end in zip(cells, terminals):
            end = int(end)
            if end < start:
                raise RuntimeError("invalid episode boundaries in v5 rollout")
            segment = slice(start, end + 1)
            expected_eta = torch.full_like(etas[segment], float(cell.eta))
            if not torch.allclose(etas[segment], expected_eta, atol=1e-6, rtol=0.0):
                raise RuntimeError("rollout episode order disagrees with balanced training cycle")
            years[segment] = int(cell.weather_year)
            folds[segment] = int(YEAR_FOLD[int(cell.weather_year)])
            start = end + 1
        if start != batch.size:
            raise RuntimeError("episode boundaries did not cover complete v5 rollout")
        return years.to(self.device), folds.to(self.device)

    @staticmethod
    def _coverage_strata(action_features: torch.Tensor) -> torch.Tensor:
        gate = action_features[:, 0] > 0.5
        amount = action_features[:, 1].clamp(0.0, 1.0)
        amount_bin = torch.floor(amount * 4.0).to(torch.int64).clamp(max=3)
        return torch.where(gate, amount_bin + 1, torch.zeros_like(amount_bin))

    def _coverage_weights(
        self,
        *,
        train_mask: torch.Tensor,
        etas: torch.Tensor,
        strata: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, dict[str, int]], dict[str, float]]:
        weights = torch.zeros_like(etas, dtype=torch.float32)
        count_diag: dict[str, dict[str, int]] = {}
        max_weight_diag: dict[str, float] = {}
        for eta in REGISTERED_ETA_LEVELS:
            key = f"{float(eta):.2f}"
            group = train_mask & torch.isclose(
                etas, torch.as_tensor(float(eta), device=etas.device), atol=1e-6, rtol=0.0
            )
            group_n = int(group.sum().item())
            if group_n <= 0:
                raise RuntimeError(f"v5 fold has no samples for eta={key}")
            counts = [int((group & (strata == s)).sum().item()) for s in range(5)]
            present = [s for s, n in enumerate(counts) if n > 0]
            if not present:
                raise RuntimeError(f"v5 fold has no observed action strata for eta={key}")
            k = len(present)
            count_diag[key] = {
                ACTION_STRATUM_NAMES[s]: counts[s] for s in range(5)
            }
            local_max = 0.0
            for s in present:
                sample_weight = float(group_n) / float(k * counts[s])
                mask = group & (strata == s)
                weights[mask] = sample_weight
                local_max = max(local_max, sample_weight)
            # Parameter-free normalization property of N/(K*n_k).
            if abs(float(weights[group].mean().item()) - 1.0) > 1e-5:
                raise AssertionError("v5 coverage weights must have eta-wise mean one")
            max_weight_diag[key] = local_max
        if torch.any(weights[train_mask] <= 0):
            raise RuntimeError("v5 found an unweighted training-fold transition")
        return weights, count_diag, max_weight_diag

    @torch.no_grad()
    def _weighted_q_mse(
        self,
        critic: TailActionValueNetwork,
        states: torch.Tensor,
        action_features: torch.Tensor,
        risk_returns: torch.Tensor,
        mask: torch.Tensor,
        weights: torch.Tensor,
    ) -> float:
        error2 = (critic(states[mask], action_features[mask]) - risk_returns[mask]).pow(2)
        w = weights[mask]
        return float((w * error2).sum().div(w.sum()).item())

    def _prefit_fold_tail_q(
        self,
        *,
        fold: int,
        critic: TailActionValueNetwork,
        states: torch.Tensor,
        action_features: torch.Tensor,
        risk_returns: torch.Tensor,
        etas: torch.Tensor,
        fold_ids: torch.Tensor,
        strata: torch.Tensor,
    ) -> dict[str, object]:
        train_mask = fold_ids == int(fold)
        weights, counts, max_weights = self._coverage_weights(
            train_mask=train_mask, etas=etas, strata=strata
        )
        selected = torch.nonzero(train_mask, as_tuple=False).flatten()
        entry_generator_state = self.generator.get_state()
        q_parameters = list(critic.parameters())
        other_q = self.tail_q_critic_fold1 if fold == 0 else self.tail_q_critic
        frozen_parameters = (
            list(self.actor.named_parameters())
            + list(self.reward_critic.named_parameters())
            + list(self.risk_critic.named_parameters())
            + [(f"other_q.{n}", p) for n, p in other_q.named_parameters()]
        )
        try:
            before = self._weighted_q_mse(
                critic, states, action_features, risk_returns, train_mask, weights
            )
            grad_norms: list[float] = []
            for _epoch in range(int(self.hparams.tail_q_prefit_epochs)):
                order = torch.randperm(
                    int(selected.numel()), generator=self.generator, device=self.device
                )
                shuffled = selected[order]
                for start in range(0, int(shuffled.numel()), self.hparams.minibatch_size):
                    idx = shuffled[start : start + self.hparams.minibatch_size]
                    pred = critic(states[idx], action_features[idx])
                    w = weights[idx]
                    loss = (w * (pred - risk_returns[idx]).pow(2)).sum() / w.sum()
                    if not torch.isfinite(loss):
                        raise FloatingPointError("RCWA v5 fold Tail-Q prefit loss became NaN/Inf")
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    contaminated = [name for name, p in frozen_parameters if p.grad is not None]
                    if contaminated:
                        raise RuntimeError(
                            "RCWA v5 fold Tail-Q prefit touched frozen parameters: "
                            + ", ".join(contaminated)
                        )
                    grad_norm = torch.nn.utils.clip_grad_norm_(q_parameters, self.hparams.max_grad_norm)
                    grad_norms.append(float(torch.as_tensor(grad_norm).detach().item()))
                    self.optimizer.step()
            after = self._weighted_q_mse(
                critic, states, action_features, risk_returns, train_mask, weights
            )
            return {
                "loss_before": before,
                "loss_after": after,
                "grad_mean": mean(grad_norms),
                "grad_max": max(grad_norms),
                "steps": len(grad_norms),
                "counts": counts,
                "max_weights": max_weights,
                "transition_count": int(train_mask.sum().item()),
            }
        finally:
            self.generator.set_state(entry_generator_state)

    def _actor_risk_credit(
        self,
        *,
        batch: RCWARolloutBatch,
        states: torch.Tensor,
        etas: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        risk_returns = batch.risk_returns.to(self.device)
        diagnostics = self._prefit_risk_critic(states=states, risk_returns=risk_returns)
        action_features = self._action_features_from_batch(batch, device=self.device)
        _years, fold_ids = self._transition_weather_folds(batch)
        strata = self._coverage_strata(action_features)
        results: dict[int, dict[str, object]] = {}
        for fold, critic in ((0, self.tail_q_critic), (1, self.tail_q_critic_fold1)):
            results[fold] = self._prefit_fold_tail_q(
                fold=fold,
                critic=critic,
                states=states,
                action_features=action_features,
                risk_returns=risk_returns,
                etas=etas,
                fold_ids=fold_ids,
                strata=strata,
            )
        with torch.no_grad():
            q_values = torch.empty_like(risk_returns)
            fold0 = fold_ids == 0
            fold1 = ~fold0
            # Cross-fit: evaluate each fold with the critic trained exclusively
            # on the opposite weather fold.
            q_values[fold0] = self.tail_q_critic_fold1(
                states[fold0], action_features[fold0]
            )
            q_values[fold1] = self.tail_q_critic(states[fold1], action_features[fold1])
            v_values = self.risk_critic(states).detach()
            credit = (q_values - v_values).detach()
        diagnostics.update({
            "tail_q_prefit_loss_before": mean(float(results[f]["loss_before"]) for f in (0, 1)),
            "tail_q_prefit_loss_after": mean(float(results[f]["loss_after"]) for f in (0, 1)),
            "tail_q_prefit_grad_norm_mean": mean(float(results[f]["grad_mean"]) for f in (0, 1)),
            "tail_q_prefit_grad_norm_max": max(float(results[f]["grad_max"]) for f in (0, 1)),
            "tail_q_prefit_steps": sum(int(results[f]["steps"]) for f in (0, 1)),
            "tail_q_fold_prefit_loss_before": {f"fold{f}": float(results[f]["loss_before"]) for f in (0, 1)},
            "tail_q_fold_prefit_loss_after": {f"fold{f}": float(results[f]["loss_after"]) for f in (0, 1)},
            "tail_q_fold_prefit_steps": {f"fold{f}": int(results[f]["steps"]) for f in (0, 1)},
            "tail_q_cross_weather_transition_count": {
                "fold0_credit_from_fold1": int(fold0.sum().item()),
                "fold1_credit_from_fold0": int(fold1.sum().item()),
            },
            "tail_q_coverage_stratum_counts": {f"fold{f}": results[f]["counts"] for f in (0, 1)},
            "tail_q_coverage_max_weight": {f"fold{f}": results[f]["max_weights"] for f in (0, 1)},
        })
        diagnostics.update(self._tail_q_diagnostics(q_values=q_values, credit=credit, etas=etas))
        diagnostics.update(self._tail_q_early_diagnostics(
            batch=batch, action_features=action_features, credit=credit, etas=etas
        ))
        return credit, diagnostics

    def _build_update_stats(self, fields: Mapping[str, object]):
        return RCWAV5UpdateStats(**fields)

    def checkpoint_payload(self) -> dict[str, object]:
        payload = super().checkpoint_payload()
        payload["tail_q_critic_fold1_state_dict"] = self.tail_q_critic_fold1.state_dict()
        return payload

    def load_checkpoint_payload(self, payload: Mapping[str, object]) -> None:
        if payload.get("protocol_id") != self.PROTOCOL_ID:
            raise ValueError("checkpoint protocol_id mismatch")
        if "tail_q_critic_fold1_state_dict" not in payload:
            raise ValueError("RCWA v5 checkpoint missing fold-1 Tail-Q critic")
        self.tail_q_critic_fold1.load_state_dict(
            payload["tail_q_critic_fold1_state_dict"], strict=True
        )
        super().load_checkpoint_payload(payload)


__all__ = [
    "ACTION_STRATUM_NAMES",
    "RCWAV5Agent",
    "RCWAV5Hyperparameters",
    "RCWAV5UpdateStats",
    "V5_PROTOCOL_ID",
    "YEAR_FOLD",
]
