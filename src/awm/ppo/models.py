"""Neural models for the AWM standard PPO baseline.

The actor represents the mixed action distribution

    pi(g, z | s) = Bernoulli(g | s) * Normal(z | s)^g

where ``g=0`` is exact no irrigation and, only when ``g=1``, the Gaussian
latent is transformed to ``amount_fraction=(tanh(z)+1)/2``.  PPO stores and
re-evaluates the sampled gate and pre-tanh latent; DSSAT execution projection
never substitutes for this behavior probability contract.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


@dataclass(frozen=True, slots=True)
class HierarchicalActionBatch:
    irrigate: torch.Tensor
    amount_fraction: torch.Tensor
    raw_amount: torch.Tensor
    log_prob: torch.Tensor


class _MLPTrunk(nn.Module):
    def __init__(self, state_dim: int, hidden_dims: tuple[int, int]) -> None:
        super().__init__()
        h1, h2 = hidden_dims
        self.net = nn.Sequential(
            nn.Linear(state_dim, h1),
            nn.LayerNorm(h1),
            nn.Tanh(),
            nn.Linear(h1, h2),
            nn.LayerNorm(h2),
            nn.Tanh(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class HierarchicalIrrigationActor(nn.Module):
    """Single-policy exact-no-op irrigation actor for 79D AWM observations."""

    def __init__(
        self,
        *,
        state_dim: int = 79,
        hidden_dims: tuple[int, int] = (256, 128),
        log_std_min: float = -5.0,
        log_std_max: float = 1.0,
    ) -> None:
        super().__init__()
        if state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if len(hidden_dims) != 2 or any(int(x) <= 0 for x in hidden_dims):
            raise ValueError("hidden_dims must contain two positive integers")
        if not log_std_min < log_std_max:
            raise ValueError("log_std_min must be < log_std_max")
        self.state_dim = int(state_dim)
        self.hidden_dims = tuple(int(x) for x in hidden_dims)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.trunk = _MLPTrunk(self.state_dim, self.hidden_dims)
        h = self.hidden_dims[-1]
        self.gate_head = nn.Linear(h, 1)
        self.amount_mean_head = nn.Linear(h, 1)
        self.amount_log_std_head = nn.Linear(h, 1)
        # No hand-coded irrigation sparsity prior: p(irrigate)=0.5 initially.
        nn.init.zeros_(self.gate_head.weight)
        nn.init.zeros_(self.gate_head.bias)
        self.register_buffer("_log_two", torch.tensor(math.log(2.0)))

    def _validate_state(self, state: torch.Tensor) -> None:
        if state.ndim != 2 or state.shape[1] != self.state_dim:
            raise ValueError(
                f"state must have shape [batch,{self.state_dim}], got {tuple(state.shape)}"
            )
        if not torch.isfinite(state).all():
            raise FloatingPointError("state contains NaN/Inf")

    def components(self, state: torch.Tensor) -> tuple[Normal, torch.Tensor]:
        self._validate_state(state)
        hidden = self.trunk(state)
        gate_logits = self.gate_head(hidden).squeeze(-1)
        mean = self.amount_mean_head(hidden).squeeze(-1)
        log_std = self.amount_log_std_head(hidden).squeeze(-1).clamp(
            self.log_std_min, self.log_std_max
        )
        return Normal(mean, torch.exp(log_std)), gate_logits

    @staticmethod
    def _bernoulli_log_prob(logits: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        target = active.to(dtype=logits.dtype)
        return -F.binary_cross_entropy_with_logits(logits, target, reduction="none")

    def _amount_log_prob(self, dist: Normal, raw_amount: torch.Tensor) -> torch.Tensor:
        """Density after z -> (tanh(z)+1)/2 for active irrigation only."""
        base = dist.log_prob(raw_amount)
        # log|d tanh(z)/dz|, written stably.
        tanh_log_det = 2.0 * (
            self._log_two.to(dtype=raw_amount.dtype)
            - raw_amount
            - F.softplus(-2.0 * raw_amount)
        )
        # amount_fraction=(tanh(z)+1)/2 adds scale 1/2.
        return base - tanh_log_det + self._log_two.to(dtype=raw_amount.dtype)

    def evaluate_behavior(
        self,
        state: torch.Tensor,
        irrigate: torch.Tensor,
        raw_amount: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if irrigate.ndim != 1 or raw_amount.ndim != 1:
            raise ValueError("irrigate and raw_amount must be 1D tensors")
        if irrigate.shape[0] != state.shape[0] or raw_amount.shape[0] != state.shape[0]:
            raise ValueError("behavior tensors must match state batch size")
        if not torch.isfinite(raw_amount).all():
            raise FloatingPointError("raw_amount contains NaN/Inf")
        dist, gate_logits = self.components(state)
        active = irrigate.to(dtype=torch.bool)
        gate_lp = self._bernoulli_log_prob(gate_logits, active)
        amount_lp = self._amount_log_prob(dist, raw_amount)
        log_prob = gate_lp + torch.where(active, amount_lp, torch.zeros_like(amount_lp))
        if not torch.isfinite(log_prob).all():
            raise FloatingPointError("hierarchical log probability contains NaN/Inf")
        # Diagnostic entropy surrogate. Formal entropy coefficient is zero in v1.
        gate_p = torch.sigmoid(gate_logits)
        gate_entropy = -(
            gate_p * F.logsigmoid(gate_logits)
            + (1.0 - gate_p) * F.logsigmoid(-gate_logits)
        )
        entropy = gate_entropy + gate_p * dist.entropy()
        return log_prob, entropy

    def sample(
        self,
        state: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> HierarchicalActionBatch:
        dist, gate_logits = self.components(state)
        gate_draw = torch.rand(
            gate_logits.shape,
            generator=generator,
            device=gate_logits.device,
            dtype=gate_logits.dtype,
        )
        active = gate_draw < torch.sigmoid(gate_logits)
        eps = torch.randn(
            dist.loc.shape,
            generator=generator,
            device=dist.loc.device,
            dtype=dist.loc.dtype,
        )
        raw = dist.loc + dist.scale * eps
        amount_fraction = torch.where(
            active,
            (torch.tanh(raw) + 1.0) * 0.5,
            torch.zeros_like(raw),
        )
        log_prob, _ = self.evaluate_behavior(state, active, raw)
        return HierarchicalActionBatch(
            irrigate=active,
            amount_fraction=amount_fraction,
            raw_amount=raw,
            log_prob=log_prob,
        )

    def deterministic(self, state: torch.Tensor) -> HierarchicalActionBatch:
        dist, gate_logits = self.components(state)
        active = torch.sigmoid(gate_logits) >= 0.5
        raw = dist.loc
        amount_fraction = torch.where(
            active,
            (torch.tanh(raw) + 1.0) * 0.5,
            torch.zeros_like(raw),
        )
        log_prob, _ = self.evaluate_behavior(state, active, raw)
        return HierarchicalActionBatch(active, amount_fraction, raw, log_prob)


class IrrigationValueNetwork(nn.Module):
    def __init__(
        self,
        *,
        state_dim: int = 79,
        hidden_dims: tuple[int, int] = (256, 128),
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.trunk = _MLPTrunk(self.state_dim, hidden_dims)
        self.value_head = nn.Linear(int(hidden_dims[-1]), 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim != 2 or state.shape[1] != self.state_dim:
            raise ValueError(
                f"state must have shape [batch,{self.state_dim}], got {tuple(state.shape)}"
            )
        return self.value_head(self.trunk(state)).squeeze(-1)


__all__ = [
    "HierarchicalActionBatch",
    "HierarchicalIrrigationActor",
    "IrrigationValueNetwork",
]
