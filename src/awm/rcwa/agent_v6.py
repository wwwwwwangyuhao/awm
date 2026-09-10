"""RCWA-v6: v4 Tail-Q credit with a policy-consistent baseline from the same Q.

The lower-CVaR objective, v4 Tail-Q target/prefit, actor, rollout, dual update,
and interaction budget are unchanged.  Only the actor risk baseline changes:

    A_tail(s,a) = Q_tail(s,a) - E_{a'~pi(.|s)} Q_tail(s,a').

The Bernoulli gate expectation is exact.  The one-dimensional active amount
expectation is evaluated deterministically with fixed Gauss-Hermite quadrature,
so no policy RNG is consumed and no extra environment interaction occurs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import math
from typing import Mapping

import numpy as np
import torch

from .agent_v4 import RCWAV4Agent, RCWAV4Hyperparameters, RCWAV4UpdateStats
from .buffer import RCWARolloutBatch

V6_PROTOCOL_ID = "awm-rcwa-rl-v6"
POLICY_BASELINE_GH_ORDER = 128


@lru_cache(maxsize=8)
def _gauss_hermite_rule(order: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if int(order) <= 0:
        raise ValueError("Gauss-Hermite order must be positive")
    nodes, weights = np.polynomial.hermite.hermgauss(int(order))
    return tuple(float(x) for x in nodes), tuple(float(x) for x in weights)


@dataclass(frozen=True)
class RCWAV6Hyperparameters(RCWAV4Hyperparameters):
    """No new tuned learning hyperparameters relative to RCWA-v4."""


@dataclass(frozen=True, slots=True)
class RCWAV6UpdateStats(RCWAV4UpdateStats):
    tail_q_policy_baseline_mean_by_eta: dict[str, float] = field(default_factory=dict)
    tail_q_policy_baseline_std_by_eta: dict[str, float] = field(default_factory=dict)
    tail_q_policy_noop_q_mean_by_eta: dict[str, float] = field(default_factory=dict)
    tail_q_policy_active_q_mean_by_eta: dict[str, float] = field(default_factory=dict)
    tail_q_policy_gate_probability_mean_by_eta: dict[str, float] = field(default_factory=dict)
    tail_q_policy_expected_advantage_max_abs_error_by_eta: dict[str, float] = field(default_factory=dict)
    tail_q_policy_quadrature_order: int = POLICY_BASELINE_GH_ORDER


class RCWAV6Agent(RCWAV4Agent):
    PROTOCOL_ID = V6_PROTOCOL_ID

    @torch.no_grad()
    def _policy_q_baseline(
        self,
        states: torch.Tensor,
        *,
        quadrature_order: int = POLICY_BASELINE_GH_ORDER,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return V_Q, Q(noop), E[Q(active amount)], and p(irrigate)."""
        order = int(quadrature_order)
        nodes_raw, weights_raw = _gauss_hermite_rule(order)
        nodes = torch.as_tensor(nodes_raw, dtype=states.dtype, device=self.device)
        weights = torch.as_tensor(weights_raw, dtype=states.dtype, device=self.device)
        norm = math.sqrt(math.pi)
        chunk_size = int(self.hparams.minibatch_size)
        baselines: list[torch.Tensor] = []
        noop_values: list[torch.Tensor] = []
        active_values: list[torch.Tensor] = []
        gate_probabilities: list[torch.Tensor] = []
        for start in range(0, int(states.shape[0]), chunk_size):
            s = states[start : start + chunk_size]
            dist, gate_logits = self.actor.components(s)
            gate_p = torch.sigmoid(gate_logits)
            noop_action = torch.zeros((s.shape[0], 2), dtype=s.dtype, device=self.device)
            q_noop = self.tail_q_critic(s, noop_action)

            raw = dist.loc[:, None] + math.sqrt(2.0) * dist.scale[:, None] * nodes[None, :]
            amount = (torch.tanh(raw) + 1.0) * 0.5
            repeated_states = s[:, None, :].expand(-1, order, -1).reshape(-1, s.shape[1])
            active_action = torch.stack(
                (torch.ones_like(amount), amount), dim=-1
            ).reshape(-1, 2)
            q_active_nodes = self.tail_q_critic(repeated_states, active_action).reshape(-1, order)
            q_active = (q_active_nodes * weights[None, :]).sum(dim=1) / norm
            baseline = (1.0 - gate_p) * q_noop + gate_p * q_active

            baselines.append(baseline)
            noop_values.append(q_noop)
            active_values.append(q_active)
            gate_probabilities.append(gate_p)
        return (
            torch.cat(baselines).detach(),
            torch.cat(noop_values).detach(),
            torch.cat(active_values).detach(),
            torch.cat(gate_probabilities).detach(),
        )

    def _policy_baseline_diagnostics(
        self,
        *,
        baseline: torch.Tensor,
        q_noop: torch.Tensor,
        q_active: torch.Tensor,
        gate_p: torch.Tensor,
        etas: torch.Tensor,
    ) -> dict[str, object]:
        mean_b: dict[str, float] = {}
        std_b: dict[str, float] = {}
        mean_noop: dict[str, float] = {}
        mean_active: dict[str, float] = {}
        mean_gate: dict[str, float] = {}
        max_err: dict[str, float] = {}
        expected_advantage = (1.0 - gate_p) * (q_noop - baseline) + gate_p * (q_active - baseline)
        for key, mask in self._eta_masks(etas).items():
            group = baseline[mask]
            mean_b[key] = float(group.mean().item())
            std_b[key] = float(group.std(unbiased=False).item())
            mean_noop[key] = float(q_noop[mask].mean().item())
            mean_active[key] = float(q_active[mask].mean().item())
            mean_gate[key] = float(gate_p[mask].mean().item())
            max_err[key] = float(expected_advantage[mask].abs().max().item())
        return {
            "tail_q_policy_baseline_mean_by_eta": mean_b,
            "tail_q_policy_baseline_std_by_eta": std_b,
            "tail_q_policy_noop_q_mean_by_eta": mean_noop,
            "tail_q_policy_active_q_mean_by_eta": mean_active,
            "tail_q_policy_gate_probability_mean_by_eta": mean_gate,
            "tail_q_policy_expected_advantage_max_abs_error_by_eta": max_err,
            "tail_q_policy_quadrature_order": POLICY_BASELINE_GH_ORDER,
        }

    def _actor_risk_credit(
        self,
        *,
        batch: RCWARolloutBatch,
        states: torch.Tensor,
        etas: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        risk_returns = batch.risk_returns.to(self.device)
        # Preserve v4 risk-value fitting/value-loss behavior for strict isolation.
        diagnostics = self._prefit_risk_critic(states=states, risk_returns=risk_returns)
        action_features = self._action_features_from_batch(batch, device=self.device)
        diagnostics.update(
            self._prefit_tail_q_critic(
                states=states,
                action_features=action_features,
                risk_returns=risk_returns,
            )
        )
        with torch.no_grad():
            q_values = self.tail_q_critic(states, action_features).detach()
            baseline, q_noop, q_active, gate_p = self._policy_q_baseline(states)
            credit = (q_values - baseline).detach()
        diagnostics.update(self._tail_q_diagnostics(q_values=q_values, credit=credit, etas=etas))
        diagnostics.update(
            self._tail_q_early_diagnostics(
                batch=batch,
                action_features=action_features,
                credit=credit,
                etas=etas,
            )
        )
        diagnostics.update(
            self._policy_baseline_diagnostics(
                baseline=baseline,
                q_noop=q_noop,
                q_active=q_active,
                gate_p=gate_p,
                etas=etas,
            )
        )
        return credit, diagnostics

    def _build_update_stats(self, fields: Mapping[str, object]):
        return RCWAV6UpdateStats(**fields)


__all__ = [
    "POLICY_BASELINE_GH_ORDER",
    "RCWAV6Agent",
    "RCWAV6Hyperparameters",
    "RCWAV6UpdateStats",
    "V6_PROTOCOL_ID",
]
