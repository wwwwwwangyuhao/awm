"""Complete-episode on-policy rollout storage for AWM PPO v1."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True, slots=True)
class PPORolloutBatch:
    states: torch.Tensor
    irrigate: torch.Tensor
    raw_amount: torch.Tensor
    old_log_probs: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    policy_version: int

    @property
    def size(self) -> int:
        return int(self.states.shape[0])


class PPORolloutBuffer:
    """Store exactly one balanced, complete, strictly on-policy PPO rollout."""

    def __init__(
        self,
        *,
        state_dim: int = 79,
        expected_size: int = 6750,
        gamma: float = 1.0,
        gae_lambda: float = 0.95,
        policy_version: int = 0,
    ) -> None:
        if state_dim <= 0 or expected_size <= 0:
            raise ValueError("state_dim and expected_size must be positive")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must lie in [0,1]")
        if not 0.0 <= gae_lambda <= 1.0:
            raise ValueError("gae_lambda must lie in [0,1]")
        if policy_version < 0:
            raise ValueError("policy_version must be >= 0")
        self.state_dim = int(state_dim)
        self.expected_size = int(expected_size)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.policy_version = int(policy_version)
        self._finalized = False
        self._states: list[np.ndarray] = []
        self._irrigate: list[bool] = []
        self._raw_amount: list[float] = []
        self._log_probs: list[float] = []
        self._values: list[float] = []
        self._rewards: list[float] = []
        self._dones: list[bool] = []

    def __len__(self) -> int:
        return len(self._states)

    def append(
        self,
        *,
        state: np.ndarray | torch.Tensor,
        irrigate: bool,
        raw_amount: float,
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
    ) -> None:
        if self._finalized:
            raise RuntimeError("cannot append to a finalized PPO rollout")
        if len(self) >= self.expected_size:
            raise RuntimeError("PPO rollout exceeded expected_size")
        array = (
            state.detach().cpu().numpy()
            if isinstance(state, torch.Tensor)
            else np.asarray(state)
        )
        array = np.asarray(array, dtype=np.float32)
        if array.shape != (self.state_dim,):
            raise ValueError(
                f"state must have shape ({self.state_dim},), got {array.shape}"
            )
        if not np.isfinite(array).all():
            raise FloatingPointError("state contains NaN/Inf")
        scalars = {
            "raw_amount": raw_amount,
            "log_prob": log_prob,
            "value": value,
            "reward": reward,
        }
        for name, item in scalars.items():
            if not isinstance(item, (int, float)) or not math.isfinite(float(item)):
                raise ValueError(f"{name} must be finite numeric")
        self._states.append(array.copy())
        self._irrigate.append(bool(irrigate))
        self._raw_amount.append(float(raw_amount))
        self._log_probs.append(float(log_prob))
        self._values.append(float(value))
        self._rewards.append(float(reward))
        self._dones.append(bool(done))

    def add_terminal_reward(self, amount: float) -> None:
        """Add the terminal yield-target penalty to the last transition."""
        if self._finalized:
            raise RuntimeError("rollout is already finalized")
        if not self._rewards:
            raise RuntimeError("cannot add terminal reward to an empty rollout")
        if self._dones[-1] is not True:
            raise RuntimeError("terminal reward may be added only after a done transition")
        if not isinstance(amount, (int, float)) or not math.isfinite(float(amount)):
            raise ValueError("terminal reward must be finite numeric")
        self._rewards[-1] += float(amount)

    def finalize(self) -> PPORolloutBatch:
        if self._finalized:
            raise RuntimeError("PPO rollout may be finalized only once")
        if len(self) != self.expected_size:
            raise RuntimeError(
                f"PPO rollout size mismatch: {len(self)} != {self.expected_size}"
            )
        if not self._dones[-1]:
            raise RuntimeError("PPO v1 rollout must end at a complete episode boundary")
        if not any(self._dones):
            raise RuntimeError("PPO rollout contains no complete episode")

        values = np.asarray(self._values, dtype=np.float64)
        rewards = np.asarray(self._rewards, dtype=np.float64)
        dones = np.asarray(self._dones, dtype=np.float64)
        advantages = np.zeros(len(self), dtype=np.float64)
        gae = 0.0
        for idx in range(len(self) - 1, -1, -1):
            if idx == len(self) - 1:
                next_value = 0.0
            else:
                next_value = values[idx + 1]
            nonterminal = 1.0 - dones[idx]
            delta = rewards[idx] + self.gamma * next_value * nonterminal - values[idx]
            gae = delta + self.gamma * self.gae_lambda * nonterminal * gae
            advantages[idx] = gae
        returns = advantages + values
        self._finalized = True
        return PPORolloutBatch(
            states=torch.from_numpy(np.stack(self._states).astype(np.float32)),
            irrigate=torch.as_tensor(self._irrigate, dtype=torch.bool),
            raw_amount=torch.as_tensor(self._raw_amount, dtype=torch.float32),
            old_log_probs=torch.as_tensor(self._log_probs, dtype=torch.float32),
            values=torch.as_tensor(values, dtype=torch.float32),
            returns=torch.as_tensor(returns, dtype=torch.float32),
            advantages=torch.as_tensor(advantages, dtype=torch.float32),
            rewards=torch.as_tensor(rewards, dtype=torch.float32),
            dones=torch.as_tensor(self._dones, dtype=torch.bool),
            policy_version=self.policy_version,
        )


__all__ = ["PPORolloutBatch", "PPORolloutBuffer"]
