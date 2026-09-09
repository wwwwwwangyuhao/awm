"""RCWA-RL v4: action-conditioned Monte-Carlo tail-Q credit.

RCWA v4 keeps the lower-CVaR objective, empirical tail assignment, dual update,
water objective, actor, state representation, rollout, interaction budget and
evaluation protocol unchanged.  Relative to v3, the actor's tail-risk credit
becomes action-conditioned and long-horizon:

    Q_tail(s_t, a_t) ~= E[h_i | s_t, a_t]
    A_tail(s_t, a_t) = Q_tail(s_t, a_t) - V_tail(s_t)

where h_i=(tau-R_i)_+/alpha is the same frozen episode tail cost used by all
RCWA versions.  Q_tail is fitted only from the already-collected on-policy
batch; no extra DSSAT interaction, schedule prior, DAP penalty or new risk
objective is introduced.

The action input is the behavior action before the WaterBudgetController:
Bernoulli gate plus active-only amount_fraction=(tanh(raw_amount)+1)/2.  This
keeps the critic aligned with the policy's sampled action while the state
already contains remaining/cumulative budget, so controller projection remains
part of the environment dynamics represented in Q_tail.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from awm.risk import REGISTERED_ETA_LEVELS

from .agent import RCWAUpdateStats
from .agent_v3 import (
    EARLY_WINDOW_STEPS,
    POLICY_WATER_BUDGET_MM,
    RCWAV3Agent,
    RCWAV3Hyperparameters,
    RCWAV3UpdateStats,
)
from .buffer import RCWARolloutBatch
from .tail_credit import episode_step_indices


V4_PROTOCOL_ID = "awm-rcwa-rl-v4"


class TailActionValueNetwork(nn.Module):
    """Scalar tail-cost Q(s,a) for hierarchical irrigation actions."""

    def __init__(
        self,
        *,
        state_dim: int = 79,
        hidden_dims: tuple[int, int] = (256, 128),
    ) -> None:
        super().__init__()
        if state_dim <= 0 or len(hidden_dims) != 2 or any(int(x) <= 0 for x in hidden_dims):
            raise ValueError("invalid TailActionValueNetwork dimensions")
        self.state_dim = int(state_dim)
        self.hidden_dims = tuple(int(x) for x in hidden_dims)
        h1, h2 = self.hidden_dims
        self.net = nn.Sequential(
            nn.Linear(self.state_dim + 2, h1),
            nn.LayerNorm(h1),
            nn.Tanh(),
            nn.Linear(h1, h2),
            nn.LayerNorm(h2),
            nn.Tanh(),
        )
        self.value_head = nn.Linear(h2, 1)
        # Match the v2/v3 risk-value stabilization: no random initial tail
        # action signal may leak into the first actor update.
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, state: torch.Tensor, action_features: torch.Tensor) -> torch.Tensor:
        if state.ndim != 2 or state.shape[1] != self.state_dim:
            raise ValueError(f"state must have shape [batch,{self.state_dim}]")
        if action_features.ndim != 2 or action_features.shape != (state.shape[0], 2):
            raise ValueError("action_features must have shape [batch,2]")
        if not torch.isfinite(state).all() or not torch.isfinite(action_features).all():
            raise FloatingPointError("tail-Q input contains NaN/Inf")
        x = torch.cat((state, action_features), dim=-1)
        return self.value_head(self.net(x)).squeeze(-1)


@dataclass(frozen=True)
class RCWAV4Hyperparameters(RCWAV3Hyperparameters):
    """V3 hyperparameters plus a matched tail-Q prefit budget."""

    tail_q_prefit_epochs: int = 10

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.tail_q_prefit_epochs <= 0:
            raise ValueError("tail_q_prefit_epochs must be positive")


@dataclass(frozen=True, slots=True)
class RCWAV4UpdateStats(RCWAV3UpdateStats):
    tail_q_prefit_loss_before: float = 0.0
    tail_q_prefit_loss_after: float = 0.0
    tail_q_prefit_grad_norm_mean: float = 0.0
    tail_q_prefit_grad_norm_max: float = 0.0
    tail_q_prefit_steps: int = 0
    tail_q_value_mean_by_eta: dict[str, float] = field(default_factory=dict)
    tail_q_value_std_by_eta: dict[str, float] = field(default_factory=dict)
    tail_q_credit_mean_by_eta: dict[str, float] = field(default_factory=dict)
    tail_q_credit_std_by_eta: dict[str, float] = field(default_factory=dict)
    tail_q_credit_positive_fraction_by_eta: dict[str, float] = field(default_factory=dict)
    tail_q_credit_early_window_mean_irrigate_by_eta: dict[str, float] = field(default_factory=dict)
    tail_q_credit_early_window_mean_noop_by_eta: dict[str, float] = field(default_factory=dict)
    tail_q_credit_early_window_irrigate_count_by_eta: dict[str, int] = field(default_factory=dict)
    tail_q_credit_early_window_noop_count_by_eta: dict[str, int] = field(default_factory=dict)
    tail_q_credit_early_window_amount_correlation_by_eta: dict[str, float] = field(default_factory=dict)


class RCWAV4Agent(RCWAV3Agent):
    """RCWA v4 agent with an action-conditioned long-horizon tail critic."""

    PROTOCOL_ID = V4_PROTOCOL_ID

    def __init__(
        self,
        *,
        hyperparameters: RCWAV4Hyperparameters | None = None,
        seed: int = 21,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__(
            hyperparameters=hyperparameters or RCWAV4Hyperparameters(),
            seed=seed,
            device=device,
        )
        self.tail_q_critic = TailActionValueNetwork(
            state_dim=self.hparams.state_dim,
            hidden_dims=self.hparams.risk_critic_hidden_dims,
        ).to(self.device)
        # Rebuild before any optimizer state exists.  The original v3
        # actor/reward/risk parameter order is preserved exactly; Q is appended.
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters())
            + list(self.reward_critic.parameters())
            + list(self.risk_critic.parameters())
            + list(self.tail_q_critic.parameters()),
            lr=self.hparams.learning_rate,
            betas=(self.hparams.adam_beta1, self.hparams.adam_beta2),
            eps=self.hparams.adam_eps,
        )

    @staticmethod
    def _action_features_from_batch(batch: RCWARolloutBatch, *, device: torch.device) -> torch.Tensor:
        gate_bool = batch.irrigate.to(device=device, dtype=torch.bool)
        gate = gate_bool.to(dtype=torch.float32)
        raw = batch.raw_amount.to(device=device, dtype=torch.float32)
        amount = torch.where(
            gate_bool,
            (torch.tanh(raw) + 1.0) * 0.5,
            torch.zeros_like(raw),
        )
        return torch.stack((gate, amount), dim=-1)

    @torch.no_grad()
    def _tail_q_mse(
        self,
        states: torch.Tensor,
        action_features: torch.Tensor,
        risk_returns: torch.Tensor,
    ) -> float:
        return float(F.mse_loss(self.tail_q_critic(states, action_features), risk_returns).item())

    def _prefit_tail_q_critic(
        self,
        *,
        states: torch.Tensor,
        action_features: torch.Tensor,
        risk_returns: torch.Tensor,
    ) -> dict[str, object]:
        """Fit only Q_tail to the frozen MC episode tail return.

        The shared actor/PPO generator is restored on every exit path, so the
        extra critic cannot change later actor sampling or minibatch order.
        """
        entry_generator_state = self.generator.get_state()
        q_parameters = list(self.tail_q_critic.parameters())
        frozen_parameters = (
            list(self.actor.named_parameters())
            + list(self.reward_critic.named_parameters())
            + list(self.risk_critic.named_parameters())
        )
        try:
            transition_count = int(states.shape[0])
            loss_before = self._tail_q_mse(states, action_features, risk_returns)
            grad_norms: list[float] = []
            for _epoch in range(int(self.hparams.tail_q_prefit_epochs)):
                permutation = torch.randperm(
                    transition_count, generator=self.generator, device=self.device
                )
                for start in range(0, transition_count, self.hparams.minibatch_size):
                    idx = permutation[start : start + self.hparams.minibatch_size]
                    prediction = self.tail_q_critic(states[idx], action_features[idx])
                    loss = F.mse_loss(prediction, risk_returns[idx])
                    if not torch.isfinite(loss):
                        raise FloatingPointError("RCWA v4 tail-Q prefit loss became NaN/Inf")
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    contaminated = [name for name, p in frozen_parameters if p.grad is not None]
                    if contaminated:
                        raise RuntimeError(
                            "RCWA v4 tail-Q prefit touched frozen parameters: "
                            + ", ".join(contaminated)
                        )
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        q_parameters, self.hparams.max_grad_norm
                    )
                    grad_norms.append(float(torch.as_tensor(grad_norm).detach().item()))
                    self.optimizer.step()
            loss_after = self._tail_q_mse(states, action_features, risk_returns)
            return {
                "tail_q_prefit_loss_before": loss_before,
                "tail_q_prefit_loss_after": loss_after,
                "tail_q_prefit_grad_norm_mean": mean(grad_norms),
                "tail_q_prefit_grad_norm_max": max(grad_norms),
                "tail_q_prefit_steps": len(grad_norms),
            }
        finally:
            self.generator.set_state(entry_generator_state)

    def _tail_q_diagnostics(
        self,
        *,
        q_values: torch.Tensor,
        credit: torch.Tensor,
        etas: torch.Tensor,
    ) -> dict[str, object]:
        q_mean: dict[str, float] = {}
        q_std: dict[str, float] = {}
        c_mean: dict[str, float] = {}
        c_std: dict[str, float] = {}
        positive: dict[str, float] = {}
        for key, mask in self._eta_masks(etas).items():
            q_group = q_values[mask]
            c_group = credit[mask]
            q_mean[key] = float(q_group.mean().item())
            q_std[key] = float(q_group.std(unbiased=False).item())
            c_mean[key] = float(c_group.mean().item())
            c_std[key] = float(c_group.std(unbiased=False).item())
            positive[key] = float((c_group > 0).to(c_group.dtype).mean().item())
        return {
            "tail_q_value_mean_by_eta": q_mean,
            "tail_q_value_std_by_eta": q_std,
            "tail_q_credit_mean_by_eta": c_mean,
            "tail_q_credit_std_by_eta": c_std,
            "tail_q_credit_positive_fraction_by_eta": positive,
        }

    @staticmethod
    def _safe_correlation(x: torch.Tensor, y: torch.Tensor) -> float:
        if x.numel() < 2:
            return 0.0
        x0 = x - x.mean()
        y0 = y - y.mean()
        denom = torch.sqrt(torch.sum(x0 * x0) * torch.sum(y0 * y0))
        if float(denom.item()) <= 1e-12:
            return 0.0
        return float((torch.sum(x0 * y0) / denom).item())

    def _tail_q_early_diagnostics(
        self,
        *,
        batch: RCWARolloutBatch,
        action_features: torch.Tensor,
        credit: torch.Tensor,
        etas: torch.Tensor,
    ) -> dict[str, object]:
        early = episode_step_indices(batch.dones).to(credit.device) < EARLY_WINDOW_STEPS
        irrigate = batch.irrigate.to(credit.device)
        amount = action_features[:, 1]
        mean_i: dict[str, float] = {}
        mean_0: dict[str, float] = {}
        count_i: dict[str, int] = {}
        count_0: dict[str, int] = {}
        amount_corr: dict[str, float] = {}
        for key, eta_mask in self._eta_masks(etas).items():
            window = early & eta_mask
            active = window & irrigate
            noop = window & ~irrigate
            count_i[key] = int(active.sum().item())
            count_0[key] = int(noop.sum().item())
            mean_i[key] = float(credit[active].mean().item()) if count_i[key] else 0.0
            mean_0[key] = float(credit[noop].mean().item()) if count_0[key] else 0.0
            amount_corr[key] = self._safe_correlation(amount[active], credit[active])
        return {
            "tail_q_credit_early_window_mean_irrigate_by_eta": mean_i,
            "tail_q_credit_early_window_mean_noop_by_eta": mean_0,
            "tail_q_credit_early_window_irrigate_count_by_eta": count_i,
            "tail_q_credit_early_window_noop_count_by_eta": count_0,
            "tail_q_credit_early_window_amount_correlation_by_eta": amount_corr,
        }

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
        diagnostics.update(
            self._prefit_tail_q_critic(
                states=states,
                action_features=action_features,
                risk_returns=risk_returns,
            )
        )
        with torch.no_grad():
            q_values = self.tail_q_critic(states, action_features).detach()
            v_values = self.risk_critic(states).detach()
            credit = (q_values - v_values).detach()
        diagnostics.update(self._tail_q_diagnostics(q_values=q_values, credit=credit, etas=etas))
        diagnostics.update(
            self._tail_q_early_diagnostics(
                batch=batch,
                action_features=action_features,
                credit=credit,
                etas=etas,
            )
        )
        return credit, diagnostics

    def _build_update_stats(self, fields: Mapping[str, object]) -> RCWAUpdateStats:
        return RCWAV4UpdateStats(**fields)  # type: ignore[arg-type]

    def checkpoint_payload(self) -> dict[str, object]:
        payload = super().checkpoint_payload()
        payload["tail_q_critic_state_dict"] = self.tail_q_critic.state_dict()
        return payload

    def load_checkpoint_payload(self, payload: Mapping[str, object]) -> None:
        if payload.get("protocol_id") != self.PROTOCOL_ID:
            raise ValueError("checkpoint protocol_id mismatch")
        if "tail_q_critic_state_dict" not in payload:
            raise ValueError("RCWA v4 checkpoint missing tail-Q critic")
        self.tail_q_critic.load_state_dict(payload["tail_q_critic_state_dict"], strict=True)
        super().load_checkpoint_payload(payload)


__all__ = [
    "RCWAV4Agent",
    "RCWAV4Hyperparameters",
    "RCWAV4UpdateStats",
    "TailActionValueNetwork",
    "V4_PROTOCOL_ID",
]
