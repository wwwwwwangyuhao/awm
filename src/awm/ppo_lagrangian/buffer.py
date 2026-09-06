"""Balanced complete-episode rollout storage for PPO-Lagrangian v1."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch


@dataclass(frozen=True, slots=True)
class LagrangianRolloutBatch:
    states: torch.Tensor
    irrigate: torch.Tensor
    raw_amount: torch.Tensor
    old_log_probs: torch.Tensor
    etas: torch.Tensor
    reward_values: torch.Tensor
    cost_values: torch.Tensor
    reward_returns: torch.Tensor
    cost_returns: torch.Tensor
    reward_advantages: torch.Tensor
    cost_advantages: torch.Tensor
    rewards: torch.Tensor
    costs: torch.Tensor
    dones: torch.Tensor
    episode_etas: torch.Tensor
    episode_costs: torch.Tensor
    policy_version: int

    @property
    def size(self) -> int:
        return int(self.states.shape[0])


class LagrangianRolloutBuffer:
    """Store one 54-episode strictly on-policy reward/cost rollout."""

    def __init__(
        self,
        *,
        state_dim: int = 79,
        expected_size: int = 6750,
        gamma: float = 1.0,
        gae_lambda: float = 1.0,
        cost_gamma: float = 1.0,
        cost_gae_lambda: float = 1.0,
        policy_version: int = 0,
    ) -> None:
        if state_dim <= 0 or expected_size <= 0:
            raise ValueError("state_dim and expected_size must be positive")
        for name, value in (
            ("gamma", gamma),
            ("gae_lambda", gae_lambda),
            ("cost_gamma", cost_gamma),
            ("cost_gae_lambda", cost_gae_lambda),
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must lie in [0,1]")
        if policy_version < 0:
            raise ValueError("policy_version must be >= 0")
        self.state_dim = int(state_dim)
        self.expected_size = int(expected_size)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.cost_gamma = float(cost_gamma)
        self.cost_gae_lambda = float(cost_gae_lambda)
        self.policy_version = int(policy_version)
        self._finalized = False
        self._states: list[np.ndarray] = []
        self._irrigate: list[bool] = []
        self._raw_amount: list[float] = []
        self._log_probs: list[float] = []
        self._etas: list[float] = []
        self._reward_values: list[float] = []
        self._cost_values: list[float] = []
        self._rewards: list[float] = []
        self._costs: list[float] = []
        self._dones: list[bool] = []
        self._episode_etas: list[float] = []
        self._episode_costs: list[float] = []

    def __len__(self) -> int:
        return len(self._states)

    def append(
        self,
        *,
        state: np.ndarray | torch.Tensor,
        irrigate: bool,
        raw_amount: float,
        log_prob: float,
        eta: float,
        reward_value: float,
        cost_value: float,
        reward: float,
        cost: float,
        done: bool,
    ) -> None:
        if self._finalized:
            raise RuntimeError("cannot append to finalized rollout")
        if len(self) >= self.expected_size:
            raise RuntimeError("rollout exceeded expected_size")
        array = (
            state.detach().cpu().numpy()
            if isinstance(state, torch.Tensor)
            else np.asarray(state)
        )
        array = np.asarray(array, dtype=np.float32)
        if array.shape != (self.state_dim,):
            raise ValueError(f"state must have shape ({self.state_dim},), got {array.shape}")
        if not np.isfinite(array).all():
            raise FloatingPointError("state contains NaN/Inf")
        scalars = {
            "raw_amount": raw_amount,
            "log_prob": log_prob,
            "eta": eta,
            "reward_value": reward_value,
            "cost_value": cost_value,
            "reward": reward,
            "cost": cost,
        }
        for name, item in scalars.items():
            if not isinstance(item, (int, float)) or not math.isfinite(float(item)):
                raise ValueError(f"{name} must be finite numeric")
        self._states.append(array.copy())
        self._irrigate.append(bool(irrigate))
        self._raw_amount.append(float(raw_amount))
        self._log_probs.append(float(log_prob))
        self._etas.append(float(eta))
        self._reward_values.append(float(reward_value))
        self._cost_values.append(float(cost_value))
        self._rewards.append(float(reward))
        self._costs.append(float(cost))
        self._dones.append(bool(done))

    def set_terminal_cost(self, *, eta: float, cost: float) -> None:
        if self._finalized:
            raise RuntimeError("rollout already finalized")
        if not self._costs:
            raise RuntimeError("cannot set terminal cost on empty rollout")
        if self._dones[-1] is not True:
            raise RuntimeError("terminal cost may be set only after a done transition")
        if not math.isfinite(float(cost)) or not math.isfinite(float(eta)):
            raise ValueError("eta and terminal cost must be finite")
        if abs(self._etas[-1] - float(eta)) > 1e-12:
            raise ValueError("terminal eta disagrees with last transition eta")
        if abs(self._costs[-1]) > 1e-12:
            raise RuntimeError("terminal cost already populated")
        self._costs[-1] = float(cost)
        self._episode_etas.append(float(eta))
        self._episode_costs.append(float(cost))

    @staticmethod
    def _gae(
        *,
        values: np.ndarray,
        signals: np.ndarray,
        dones: np.ndarray,
        gamma: float,
        lam: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        advantages = np.zeros(len(values), dtype=np.float64)
        gae = 0.0
        for idx in range(len(values) - 1, -1, -1):
            next_value = 0.0 if idx == len(values) - 1 else values[idx + 1]
            nonterminal = 1.0 - dones[idx]
            delta = signals[idx] + gamma * next_value * nonterminal - values[idx]
            gae = delta + gamma * lam * nonterminal * gae
            advantages[idx] = gae
        return advantages, advantages + values

    def finalize(self) -> LagrangianRolloutBatch:
        if self._finalized:
            raise RuntimeError("rollout may be finalized only once")
        if len(self) != self.expected_size:
            raise RuntimeError(f"rollout size mismatch: {len(self)} != {self.expected_size}")
        if not self._dones or not self._dones[-1]:
            raise RuntimeError("rollout must end at a complete episode boundary")
        done_count = sum(bool(x) for x in self._dones)
        if done_count != len(self._episode_costs):
            raise RuntimeError(
                f"every episode must have exactly one registered terminal cost: {done_count} != {len(self._episode_costs)}"
            )

        reward_values = np.asarray(self._reward_values, dtype=np.float64)
        cost_values = np.asarray(self._cost_values, dtype=np.float64)
        rewards = np.asarray(self._rewards, dtype=np.float64)
        costs = np.asarray(self._costs, dtype=np.float64)
        dones = np.asarray(self._dones, dtype=np.float64)
        reward_adv, reward_returns = self._gae(
            values=reward_values,
            signals=rewards,
            dones=dones,
            gamma=self.gamma,
            lam=self.gae_lambda,
        )
        cost_adv, cost_returns = self._gae(
            values=cost_values,
            signals=costs,
            dones=dones,
            gamma=self.cost_gamma,
            lam=self.cost_gae_lambda,
        )
        self._finalized = True
        return LagrangianRolloutBatch(
            states=torch.from_numpy(np.stack(self._states).astype(np.float32)),
            irrigate=torch.as_tensor(self._irrigate, dtype=torch.bool),
            raw_amount=torch.as_tensor(self._raw_amount, dtype=torch.float32),
            old_log_probs=torch.as_tensor(self._log_probs, dtype=torch.float32),
            etas=torch.as_tensor(self._etas, dtype=torch.float32),
            reward_values=torch.as_tensor(reward_values, dtype=torch.float32),
            cost_values=torch.as_tensor(cost_values, dtype=torch.float32),
            reward_returns=torch.as_tensor(reward_returns, dtype=torch.float32),
            cost_returns=torch.as_tensor(cost_returns, dtype=torch.float32),
            reward_advantages=torch.as_tensor(reward_adv, dtype=torch.float32),
            cost_advantages=torch.as_tensor(cost_adv, dtype=torch.float32),
            rewards=torch.as_tensor(rewards, dtype=torch.float32),
            costs=torch.as_tensor(costs, dtype=torch.float32),
            dones=torch.as_tensor(self._dones, dtype=torch.bool),
            episode_etas=torch.as_tensor(self._episode_etas, dtype=torch.float32),
            episode_costs=torch.as_tensor(self._episode_costs, dtype=torch.float32),
            policy_version=self.policy_version,
        )


__all__ = ["LagrangianRolloutBatch", "LagrangianRolloutBuffer"]
