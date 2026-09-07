"""Balanced complete-episode rollout storage for RCWA-RL v1."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from .risk_batch import EtaTailRiskMetrics, evaluate_registered_eta_groups


@dataclass(frozen=True, slots=True)
class RCWARolloutBatch:
    states: torch.Tensor
    irrigate: torch.Tensor
    raw_amount: torch.Tensor
    old_log_probs: torch.Tensor
    etas: torch.Tensor
    reward_values: torch.Tensor
    risk_values: torch.Tensor
    reward_returns: torch.Tensor
    risk_returns: torch.Tensor
    reward_advantages: torch.Tensor
    risk_advantages: torch.Tensor
    rewards: torch.Tensor
    risk_costs: torch.Tensor
    dones: torch.Tensor
    episode_etas: torch.Tensor
    episode_retentions: torch.Tensor
    episode_risk_costs: torch.Tensor
    tail_metrics: dict[str, EtaTailRiskMetrics]
    policy_version: int

    @property
    def size(self) -> int:
        return int(self.states.shape[0])


class RCWARolloutBuffer:
    def __init__(
        self,
        *,
        state_dim: int = 79,
        expected_size: int = 6750,
        gamma: float = 1.0,
        gae_lambda: float = 1.0,
        risk_gamma: float = 1.0,
        risk_gae_lambda: float = 1.0,
        alpha: float = 0.20,
        policy_version: int = 0,
    ) -> None:
        if state_dim <= 0 or expected_size <= 0:
            raise ValueError("state_dim and expected_size must be positive")
        for name, value in (
            ("gamma", gamma),
            ("gae_lambda", gae_lambda),
            ("risk_gamma", risk_gamma),
            ("risk_gae_lambda", risk_gae_lambda),
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must lie in [0,1]")
        if not 0.0 < float(alpha) <= 1.0:
            raise ValueError("alpha must lie in (0,1]")
        self.state_dim = int(state_dim)
        self.expected_size = int(expected_size)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.risk_gamma = float(risk_gamma)
        self.risk_gae_lambda = float(risk_gae_lambda)
        self.alpha = float(alpha)
        self.policy_version = int(policy_version)
        self._finalized = False
        self._states: list[np.ndarray] = []
        self._irrigate: list[bool] = []
        self._raw_amount: list[float] = []
        self._log_probs: list[float] = []
        self._etas: list[float] = []
        self._reward_values: list[float] = []
        self._risk_values: list[float] = []
        self._rewards: list[float] = []
        self._risk_costs: list[float] = []
        self._dones: list[bool] = []
        self._episode_etas: list[float] = []
        self._episode_retentions: list[float] = []
        self._episode_terminal_indices: list[int] = []

    def __len__(self) -> int:
        return len(self._states)

    def append(
        self,
        *,
        state,
        irrigate: bool,
        raw_amount: float,
        log_prob: float,
        eta: float,
        reward_value: float,
        risk_value: float,
        reward: float,
        done: bool,
    ) -> None:
        if self._finalized:
            raise RuntimeError("cannot append to finalized rollout")
        if len(self) >= self.expected_size:
            raise RuntimeError("rollout exceeded expected_size")
        array = state.detach().cpu().numpy() if isinstance(state, torch.Tensor) else np.asarray(state)
        array = np.asarray(array, dtype=np.float32)
        if array.shape != (self.state_dim,):
            raise ValueError(f"state must have shape ({self.state_dim},), got {array.shape}")
        if not np.isfinite(array).all():
            raise FloatingPointError("state contains NaN/Inf")
        for name, value in {
            "raw_amount": raw_amount,
            "log_prob": log_prob,
            "eta": eta,
            "reward_value": reward_value,
            "risk_value": risk_value,
            "reward": reward,
        }.items():
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite numeric")
        self._states.append(array.copy())
        self._irrigate.append(bool(irrigate))
        self._raw_amount.append(float(raw_amount))
        self._log_probs.append(float(log_prob))
        self._etas.append(float(eta))
        self._reward_values.append(float(reward_value))
        self._risk_values.append(float(risk_value))
        self._rewards.append(float(reward))
        self._risk_costs.append(0.0)
        self._dones.append(bool(done))

    def register_terminal_retention(self, *, eta: float, retention: float) -> None:
        if self._finalized:
            raise RuntimeError("rollout already finalized")
        if not self._dones or self._dones[-1] is not True:
            raise RuntimeError("terminal retention may be registered only after a done transition")
        if not math.isfinite(float(eta)) or not math.isfinite(float(retention)) or float(retention) < 0.0:
            raise ValueError("eta/retention must be finite and retention >= 0")
        if abs(self._etas[-1] - float(eta)) > 1e-9:
            raise ValueError("terminal eta disagrees with last transition eta")
        if self._episode_terminal_indices and self._episode_terminal_indices[-1] == len(self) - 1:
            raise RuntimeError("terminal retention already registered for this episode")
        self._episode_etas.append(float(eta))
        self._episode_retentions.append(float(retention))
        self._episode_terminal_indices.append(len(self) - 1)

    @staticmethod
    def _gae(*, values, signals, dones, gamma: float, lam: float):
        advantages = np.zeros(len(values), dtype=np.float64)
        gae = 0.0
        for idx in range(len(values) - 1, -1, -1):
            next_value = 0.0 if idx == len(values) - 1 else values[idx + 1]
            nonterminal = 1.0 - dones[idx]
            delta = signals[idx] + gamma * next_value * nonterminal - values[idx]
            gae = delta + gamma * lam * nonterminal * gae
            advantages[idx] = gae
        return advantages, advantages + values

    def finalize(self) -> RCWARolloutBatch:
        if self._finalized:
            raise RuntimeError("rollout may be finalized only once")
        if len(self) != self.expected_size:
            raise RuntimeError(f"rollout size mismatch: {len(self)} != {self.expected_size}")
        done_count = sum(bool(x) for x in self._dones)
        if done_count != len(self._episode_retentions):
            raise RuntimeError(
                f"every episode must register exactly one retention: {done_count} != {len(self._episode_retentions)}"
            )
        episode_risk_costs, tail_metrics = evaluate_registered_eta_groups(
            episode_etas=self._episode_etas,
            retentions=self._episode_retentions,
            alpha=self.alpha,
            expected_per_eta=18,
        )
        for terminal_index, risk_cost in zip(self._episode_terminal_indices, episode_risk_costs):
            self._risk_costs[terminal_index] = float(risk_cost)

        reward_values = np.asarray(self._reward_values, dtype=np.float64)
        risk_values = np.asarray(self._risk_values, dtype=np.float64)
        rewards = np.asarray(self._rewards, dtype=np.float64)
        risk_costs = np.asarray(self._risk_costs, dtype=np.float64)
        dones = np.asarray(self._dones, dtype=np.float64)
        reward_adv, reward_returns = self._gae(
            values=reward_values,
            signals=rewards,
            dones=dones,
            gamma=self.gamma,
            lam=self.gae_lambda,
        )
        risk_adv, risk_returns = self._gae(
            values=risk_values,
            signals=risk_costs,
            dones=dones,
            gamma=self.risk_gamma,
            lam=self.risk_gae_lambda,
        )
        self._finalized = True
        return RCWARolloutBatch(
            states=torch.from_numpy(np.stack(self._states).astype(np.float32)),
            irrigate=torch.as_tensor(self._irrigate, dtype=torch.bool),
            raw_amount=torch.as_tensor(self._raw_amount, dtype=torch.float32),
            old_log_probs=torch.as_tensor(self._log_probs, dtype=torch.float32),
            etas=torch.as_tensor(self._etas, dtype=torch.float32),
            reward_values=torch.as_tensor(reward_values, dtype=torch.float32),
            risk_values=torch.as_tensor(risk_values, dtype=torch.float32),
            reward_returns=torch.as_tensor(reward_returns, dtype=torch.float32),
            risk_returns=torch.as_tensor(risk_returns, dtype=torch.float32),
            reward_advantages=torch.as_tensor(reward_adv, dtype=torch.float32),
            risk_advantages=torch.as_tensor(risk_adv, dtype=torch.float32),
            rewards=torch.as_tensor(rewards, dtype=torch.float32),
            risk_costs=torch.as_tensor(risk_costs, dtype=torch.float32),
            dones=torch.as_tensor(self._dones, dtype=torch.bool),
            episode_etas=torch.as_tensor(self._episode_etas, dtype=torch.float32),
            episode_retentions=torch.as_tensor(self._episode_retentions, dtype=torch.float32),
            episode_risk_costs=torch.as_tensor(episode_risk_costs, dtype=torch.float32),
            tail_metrics=tail_metrics,
            policy_version=self.policy_version,
        )


__all__ = ["RCWARolloutBatch", "RCWARolloutBuffer"]
