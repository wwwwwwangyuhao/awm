"""Strictly on-policy PPO update core for the AWM irrigation baseline."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from statistics import mean
from typing import Mapping

import torch
import torch.nn.functional as F

from .buffer import PPORolloutBatch
from .models import HierarchicalActionBatch, HierarchicalIrrigationActor, IrrigationValueNetwork


@dataclass(frozen=True, slots=True)
class PPOHyperparameters:
    state_dim: int = 79
    actor_hidden_dims: tuple[int, int] = (256, 128)
    critic_hidden_dims: tuple[int, int] = (256, 128)
    learning_rate: float = 1e-4
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    update_epochs: int = 10
    minibatch_size: int = 450
    value_loss_coefficient: float = 0.5
    entropy_coefficient: float = 0.0
    max_grad_norm: float = 0.5

    def __post_init__(self) -> None:
        if self.state_dim <= 0:
            raise ValueError("state_dim must be positive")
        for name in ("learning_rate", "clip_epsilon", "max_grad_norm"):
            if not math.isfinite(float(getattr(self, name))) or float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be finite and > 0")
        if self.update_epochs <= 0 or self.minibatch_size <= 0:
            raise ValueError("update_epochs and minibatch_size must be positive")
        if not 0.0 <= self.gamma <= 1.0 or not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gamma and gae_lambda must lie in [0,1]")
        if self.value_loss_coefficient < 0.0 or self.entropy_coefficient < 0.0:
            raise ValueError("loss coefficients must be >= 0")


@dataclass(frozen=True, slots=True)
class PPOUpdateStats:
    update_index: int
    rollout_policy_version: int
    new_policy_version: int
    sample_count: int
    actor_loss: float
    value_loss: float
    total_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    grad_norm: float
    advantage_mean_before_normalization: float
    advantage_std_before_normalization: float


class PPOAgent:
    """Single actor/critic PPO with no off-policy or population data path."""

    def __init__(
        self,
        *,
        hyperparameters: PPOHyperparameters | None = None,
        seed: int = 21,
        device: str | torch.device = "cpu",
    ) -> None:
        self.hparams = hyperparameters or PPOHyperparameters()
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be an integer >= 0")
        self.seed = seed
        self.device = torch.device(device)
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        self.actor = HierarchicalIrrigationActor(
            state_dim=self.hparams.state_dim,
            hidden_dims=self.hparams.actor_hidden_dims,
        ).to(self.device)
        self.critic = IrrigationValueNetwork(
            state_dim=self.hparams.state_dim,
            hidden_dims=self.hparams.critic_hidden_dims,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.hparams.learning_rate,
        )
        self.policy_version = 0
        self.update_index = 0
        generator_device = self.device if self.device.type == "cuda" else torch.device("cpu")
        self.generator = torch.Generator(device=generator_device)
        self.generator.manual_seed(seed)

    @torch.no_grad()
    def act(self, state: torch.Tensor) -> tuple[HierarchicalActionBatch, torch.Tensor]:
        state = state.to(self.device, dtype=torch.float32)
        action = self.actor.sample(state, generator=self.generator)
        value = self.critic(state)
        return action, value

    @torch.no_grad()
    def deterministic_action(self, state: torch.Tensor) -> tuple[HierarchicalActionBatch, torch.Tensor]:
        state = state.to(self.device, dtype=torch.float32)
        return self.actor.deterministic(state), self.critic(state)

    def update(self, batch: PPORolloutBatch) -> PPOUpdateStats:
        if batch.policy_version != self.policy_version:
            raise RuntimeError(
                "strict on-policy violation: rollout policy_version "
                f"{batch.policy_version} != current {self.policy_version}"
            )
        n = batch.size
        if n <= 0:
            raise ValueError("PPO rollout is empty")
        if n % self.hparams.minibatch_size != 0:
            raise ValueError(
                f"rollout size {n} must be divisible by minibatch_size "
                f"{self.hparams.minibatch_size}"
            )

        states = batch.states.to(self.device)
        irrigate = batch.irrigate.to(self.device)
        raw_amount = batch.raw_amount.to(self.device)
        old_log_probs = batch.old_log_probs.to(self.device)
        returns = batch.returns.to(self.device)
        advantages = batch.advantages.to(self.device)
        adv_mean = float(advantages.mean().item())
        adv_std = float(advantages.std(unbiased=False).item())
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )

        actor_losses: list[float] = []
        value_losses: list[float] = []
        total_losses: list[float] = []
        entropies: list[float] = []
        approx_kls: list[float] = []
        clip_fractions: list[float] = []
        grad_norms: list[float] = []

        for _epoch in range(self.hparams.update_epochs):
            permutation = torch.randperm(
                n,
                generator=self.generator,
                device=self.device,
            )
            for start in range(0, n, self.hparams.minibatch_size):
                idx = permutation[start : start + self.hparams.minibatch_size]
                new_log_prob, entropy = self.actor.evaluate_behavior(
                    states[idx], irrigate[idx], raw_amount[idx]
                )
                log_ratio = new_log_prob - old_log_probs[idx]
                ratio = torch.exp(log_ratio)
                surrogate1 = ratio * advantages[idx]
                surrogate2 = ratio.clamp(
                    1.0 - self.hparams.clip_epsilon,
                    1.0 + self.hparams.clip_epsilon,
                ) * advantages[idx]
                actor_loss = -torch.minimum(surrogate1, surrogate2).mean()
                predicted_value = self.critic(states[idx])
                value_loss = F.mse_loss(predicted_value, returns[idx])
                entropy_mean = entropy.mean()
                total_loss = (
                    actor_loss
                    + self.hparams.value_loss_coefficient * value_loss
                    - self.hparams.entropy_coefficient * entropy_mean
                )
                if not torch.isfinite(total_loss):
                    raise FloatingPointError("PPO loss became NaN/Inf")

                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.hparams.max_grad_norm,
                )
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = (
                        (torch.abs(ratio - 1.0) > self.hparams.clip_epsilon)
                        .to(torch.float32)
                        .mean()
                    )
                actor_losses.append(float(actor_loss.detach().item()))
                value_losses.append(float(value_loss.detach().item()))
                total_losses.append(float(total_loss.detach().item()))
                entropies.append(float(entropy_mean.detach().item()))
                approx_kls.append(float(approx_kl.detach().item()))
                clip_fractions.append(float(clip_fraction.detach().item()))
                grad_norms.append(float(torch.as_tensor(grad_norm).detach().item()))

        old_version = self.policy_version
        self.policy_version += 1
        self.update_index += 1
        return PPOUpdateStats(
            update_index=self.update_index,
            rollout_policy_version=old_version,
            new_policy_version=self.policy_version,
            sample_count=n,
            actor_loss=mean(actor_losses),
            value_loss=mean(value_losses),
            total_loss=mean(total_losses),
            entropy=mean(entropies),
            approx_kl=mean(approx_kls),
            clip_fraction=mean(clip_fractions),
            grad_norm=mean(grad_norms),
            advantage_mean_before_normalization=adv_mean,
            advantage_std_before_normalization=adv_std,
        )

    def checkpoint_payload(self) -> dict[str, object]:
        return {
            "protocol_id": "awm-ppo-baseline-v1",
            "seed": self.seed,
            "policy_version": self.policy_version,
            "update_index": self.update_index,
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "generator_state": self.generator.get_state(),
            "hyperparameters": asdict(self.hparams),
        }

    def load_checkpoint_payload(self, payload: Mapping[str, object]) -> None:
        if payload.get("protocol_id") != "awm-ppo-baseline-v1":
            raise ValueError("checkpoint protocol_id is not awm-ppo-baseline-v1")
        if int(payload.get("seed", -1)) != self.seed:
            raise ValueError("checkpoint seed does not match PPOAgent seed")
        if payload.get("hyperparameters") != asdict(self.hparams):
            raise ValueError("checkpoint hyperparameters do not match PPOAgent configuration")
        self.actor.load_state_dict(payload["actor_state_dict"], strict=True)
        self.critic.load_state_dict(payload["critic_state_dict"], strict=True)
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.policy_version = int(payload["policy_version"])
        self.update_index = int(payload["update_index"])
        if self.policy_version < 0 or self.update_index < 0:
            raise ValueError("checkpoint version/update index must be nonnegative")
        self.generator.set_state(payload["generator_state"])


__all__ = ["PPOAgent", "PPOHyperparameters", "PPOUpdateStats"]
