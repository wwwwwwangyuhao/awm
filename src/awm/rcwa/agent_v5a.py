"""RCWA-v5a: v4 Tail-Q with eta-wise observed-action coverage balancing only."""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Mapping

import torch
import torch.nn.functional as F

from awm.risk import REGISTERED_ETA_LEVELS
from .agent_v4 import RCWAV4Agent, RCWAV4Hyperparameters, RCWAV4UpdateStats
from .buffer import RCWARolloutBatch

V5A_PROTOCOL_ID = "awm-rcwa-rl-v5a"
ACTION_STRATUM_NAMES = (
    "noop", "active_0_025", "active_025_050", "active_050_075", "active_075_100"
)


@dataclass(frozen=True)
class RCWAV5AHyperparameters(RCWAV4Hyperparameters):
    """No new tuned scalar hyperparameters relative to RCWA-v4."""


@dataclass(frozen=True, slots=True)
class RCWAV5AUpdateStats(RCWAV4UpdateStats):
    tail_q_coverage_stratum_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    tail_q_coverage_max_weight_by_eta: dict[str, float] = field(default_factory=dict)
    tail_q_coverage_present_strata_by_eta: dict[str, int] = field(default_factory=dict)


class RCWAV5AAgent(RCWAV4Agent):
    PROTOCOL_ID = V5A_PROTOCOL_ID

    @staticmethod
    def _coverage_strata(action_features: torch.Tensor) -> torch.Tensor:
        gate = action_features[:, 0] > 0.5
        amount = action_features[:, 1].clamp(0.0, 1.0)
        amount_bin = torch.floor(amount * 4.0).to(torch.int64).clamp(max=3)
        return torch.where(gate, amount_bin + 1, torch.zeros_like(amount_bin))

    def _coverage_weights(self, *, etas: torch.Tensor, strata: torch.Tensor):
        weights = torch.zeros_like(etas, dtype=torch.float32)
        counts_diag: dict[str, dict[str, int]] = {}
        max_diag: dict[str, float] = {}
        present_diag: dict[str, int] = {}
        for eta in REGISTERED_ETA_LEVELS:
            key = f"{float(eta):.2f}"
            group = torch.isclose(
                etas, torch.as_tensor(float(eta), device=etas.device), atol=1e-6, rtol=0.0
            )
            n = int(group.sum().item())
            if n <= 0:
                raise RuntimeError(f"v5a has no samples for eta={key}")
            counts = [int((group & (strata == s)).sum().item()) for s in range(5)]
            present = [s for s, count in enumerate(counts) if count > 0]
            if not present:
                raise RuntimeError(f"v5a has no observed action strata for eta={key}")
            k = len(present)
            local_max = 0.0
            for s in present:
                w = float(n) / float(k * counts[s])
                weights[group & (strata == s)] = w
                local_max = max(local_max, w)
            if abs(float(weights[group].mean().item()) - 1.0) > 1e-5:
                raise AssertionError("v5a coverage weights must have eta-wise mean one")
            counts_diag[key] = {ACTION_STRATUM_NAMES[s]: counts[s] for s in range(5)}
            max_diag[key] = local_max
            present_diag[key] = k
        if torch.any(weights <= 0):
            raise RuntimeError("v5a found an unweighted transition")
        return weights, counts_diag, max_diag, present_diag

    @torch.no_grad()
    def _weighted_q_mse(self, states, action_features, risk_returns, weights) -> float:
        err2 = (self.tail_q_critic(states, action_features) - risk_returns).pow(2)
        return float((weights * err2).sum().div(weights.sum()).item())

    def _prefit_coverage_tail_q(self, *, states, action_features, risk_returns, etas):
        entry_generator_state = self.generator.get_state()
        strata = self._coverage_strata(action_features)
        weights, counts, max_weights, present = self._coverage_weights(etas=etas, strata=strata)
        q_parameters = list(self.tail_q_critic.parameters())
        frozen_parameters = (
            list(self.actor.named_parameters())
            + list(self.reward_critic.named_parameters())
            + list(self.risk_critic.named_parameters())
        )
        try:
            n = int(states.shape[0])
            before = self._weighted_q_mse(states, action_features, risk_returns, weights)
            grad_norms: list[float] = []
            for _epoch in range(int(self.hparams.tail_q_prefit_epochs)):
                permutation = torch.randperm(n, generator=self.generator, device=self.device)
                for start in range(0, n, self.hparams.minibatch_size):
                    idx = permutation[start:start+self.hparams.minibatch_size]
                    pred = self.tail_q_critic(states[idx], action_features[idx])
                    w = weights[idx]
                    loss = (w * (pred - risk_returns[idx]).pow(2)).sum() / w.sum()
                    if not torch.isfinite(loss):
                        raise FloatingPointError("RCWA v5a Tail-Q prefit loss became NaN/Inf")
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    contaminated = [name for name, p in frozen_parameters if p.grad is not None]
                    if contaminated:
                        raise RuntimeError(
                            "RCWA v5a Tail-Q prefit touched frozen parameters: " + ", ".join(contaminated)
                        )
                    grad_norm = torch.nn.utils.clip_grad_norm_(q_parameters, self.hparams.max_grad_norm)
                    grad_norms.append(float(torch.as_tensor(grad_norm).detach().item()))
                    self.optimizer.step()
            after = self._weighted_q_mse(states, action_features, risk_returns, weights)
            return {
                "tail_q_prefit_loss_before": before,
                "tail_q_prefit_loss_after": after,
                "tail_q_prefit_grad_norm_mean": mean(grad_norms),
                "tail_q_prefit_grad_norm_max": max(grad_norms),
                "tail_q_prefit_steps": len(grad_norms),
                "tail_q_coverage_stratum_counts": counts,
                "tail_q_coverage_max_weight_by_eta": max_weights,
                "tail_q_coverage_present_strata_by_eta": present,
            }
        finally:
            self.generator.set_state(entry_generator_state)

    def _actor_risk_credit(self, *, batch: RCWARolloutBatch, states: torch.Tensor, etas: torch.Tensor):
        risk_returns = batch.risk_returns.to(self.device)
        diagnostics = self._prefit_risk_critic(states=states, risk_returns=risk_returns)
        action_features = self._action_features_from_batch(batch, device=self.device)
        diagnostics.update(self._prefit_coverage_tail_q(
            states=states, action_features=action_features, risk_returns=risk_returns, etas=etas
        ))
        with torch.no_grad():
            q_values = self.tail_q_critic(states, action_features).detach()
            v_values = self.risk_critic(states).detach()
            credit = (q_values - v_values).detach()
        diagnostics.update(self._tail_q_diagnostics(q_values=q_values, credit=credit, etas=etas))
        diagnostics.update(self._tail_q_early_diagnostics(
            batch=batch, action_features=action_features, credit=credit, etas=etas
        ))
        return credit, diagnostics

    def _build_update_stats(self, fields: Mapping[str, object]):
        return RCWAV5AUpdateStats(**fields)


__all__ = [
    "ACTION_STRATUM_NAMES", "RCWAV5AAgent", "RCWAV5AHyperparameters",
    "RCWAV5AUpdateStats", "V5A_PROTOCOL_ID"
]
