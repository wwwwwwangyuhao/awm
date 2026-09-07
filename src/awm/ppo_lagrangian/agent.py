"""Strictly on-policy PPO-Lagrangian agent for expected retention constraints."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from statistics import mean
from typing import Mapping

import torch
import torch.nn.functional as F

from awm.ppo.models import HierarchicalActionBatch, HierarchicalIrrigationActor, IrrigationValueNetwork
from awm.risk import REGISTERED_ETA_LEVELS

from .buffer import LagrangianRolloutBatch

_EPS = 1e-12


@dataclass(frozen=True, slots=True)
class PPOLagrangianHyperparameters:
    state_dim: int = 79
    actor_hidden_dims: tuple[int, int] = (256, 128)
    reward_critic_hidden_dims: tuple[int, int] = (256, 128)
    cost_critic_hidden_dims: tuple[int, int] = (256, 128)
    learning_rate: float = 1e-4
    gamma: float = 1.0
    gae_lambda: float = 1.0
    cost_gamma: float = 1.0
    cost_gae_lambda: float = 1.0
    clip_epsilon: float = 0.2
    update_epochs: int = 10
    minibatch_size: int = 450
    reward_value_loss_coefficient: float = 0.5
    cost_value_loss_coefficient: float = 0.5
    entropy_coefficient: float = 0.0
    max_grad_norm: float = 0.5
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    dual_initial_value: float = 1.0
    dual_learning_rate: float = 0.05

    def __post_init__(self) -> None:
        if self.state_dim <= 0:
            raise ValueError("state_dim must be positive")
        for name in (
            "learning_rate",
            "clip_epsilon",
            "max_grad_norm",
            "adam_eps",
            "dual_learning_rate",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0")
        if self.update_epochs <= 0 or self.minibatch_size <= 0:
            raise ValueError("update_epochs and minibatch_size must be positive")
        for name in ("gamma", "gae_lambda", "cost_gamma", "cost_gae_lambda"):
            if not 0.0 <= float(getattr(self, name)) <= 1.0:
                raise ValueError(f"{name} must lie in [0,1]")
        for name in (
            "reward_value_loss_coefficient",
            "cost_value_loss_coefficient",
            "entropy_coefficient",
            "dual_initial_value",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be >= 0")
        if not 0.0 <= self.adam_beta1 < 1.0 or not 0.0 <= self.adam_beta2 < 1.0:
            raise ValueError("Adam beta parameters must lie in [0,1)")


@dataclass(frozen=True, slots=True)
class PPOLagrangianUpdateStats:
    update_index: int
    rollout_policy_version: int
    new_policy_version: int
    sample_count: int
    actor_loss: float
    reward_value_loss: float
    cost_value_loss: float
    total_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    grad_norm: float
    lagrangian_advantage_mean_before_normalization: float
    lagrangian_advantage_std_before_normalization: float
    dual_before: dict[str, float]
    dual_after: dict[str, float]
    mean_constraint_cost_by_eta: dict[str, float]


class PPOLagrangianAgent:
    """Shared actor with reward/cost critics and eta-specific dual variables."""

    def __init__(
        self,
        *,
        hyperparameters: PPOLagrangianHyperparameters | None = None,
        seed: int = 21,
        device: str | torch.device = "cpu",
    ) -> None:
        self.hparams = hyperparameters or PPOLagrangianHyperparameters()
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be an integer >= 0")
        self.seed = int(seed)
        self.device = torch.device(device)
        torch.manual_seed(self.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)
        self.actor = HierarchicalIrrigationActor(
            state_dim=self.hparams.state_dim,
            hidden_dims=self.hparams.actor_hidden_dims,
        ).to(self.device)
        self.reward_critic = IrrigationValueNetwork(
            state_dim=self.hparams.state_dim,
            hidden_dims=self.hparams.reward_critic_hidden_dims,
        ).to(self.device)
        self.cost_critic = IrrigationValueNetwork(
            state_dim=self.hparams.state_dim,
            hidden_dims=self.hparams.cost_critic_hidden_dims,
        ).to(self.device)
        # Compatibility with the frozen deterministic PPO evaluator.
        self.critic = self.reward_critic
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters())
            + list(self.reward_critic.parameters())
            + list(self.cost_critic.parameters()),
            lr=self.hparams.learning_rate,
            betas=(self.hparams.adam_beta1, self.hparams.adam_beta2),
            eps=self.hparams.adam_eps,
        )
        self.dual_by_eta = {
            float(eta): float(self.hparams.dual_initial_value)
            for eta in REGISTERED_ETA_LEVELS
        }
        self.policy_version = 0
        self.update_index = 0
        generator_device = self.device if self.device.type == "cuda" else torch.device("cpu")
        self.generator = torch.Generator(device=generator_device)
        self.generator.manual_seed(self.seed)

    @torch.no_grad()
    def act(
        self, state: torch.Tensor
    ) -> tuple[HierarchicalActionBatch, torch.Tensor, torch.Tensor]:
        state = state.to(self.device, dtype=torch.float32)
        action = self.actor.sample(state, generator=self.generator)
        return action, self.reward_critic(state), self.cost_critic(state)

    @torch.no_grad()
    def deterministic_action(
        self, state: torch.Tensor
    ) -> tuple[HierarchicalActionBatch, torch.Tensor]:
        state = state.to(self.device, dtype=torch.float32)
        return self.actor.deterministic(state), self.reward_critic(state)

    def _lambda_tensor(self, etas: torch.Tensor) -> torch.Tensor:
        result = torch.empty_like(etas, dtype=torch.float32, device=self.device)
        matched = torch.zeros_like(etas, dtype=torch.bool, device=self.device)
        for eta, multiplier in self.dual_by_eta.items():
            mask = torch.isclose(
                etas,
                torch.tensor(float(eta), dtype=etas.dtype, device=etas.device),
                atol=1e-6,
                rtol=0.0,
            )
            result[mask] = float(multiplier)
            matched |= mask
        if not bool(matched.all().item()):
            bad = etas[~matched].detach().cpu().tolist()
            raise ValueError(f"batch contains unregistered eta values: {bad[:5]}")
        return result

    def _dual_update(
        self, *, episode_etas: torch.Tensor, episode_costs: torch.Tensor
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        before = {f"{eta:.2f}": float(value) for eta, value in self.dual_by_eta.items()}
        means: dict[str, float] = {}
        after: dict[str, float] = {}
        for eta in REGISTERED_ETA_LEVELS:
            eta_value = float(eta)
            mask = torch.isclose(
                episode_etas,
                torch.tensor(eta_value, dtype=episode_etas.dtype, device=episode_etas.device),
                atol=1e-6,
                rtol=0.0,
            )
            count = int(mask.sum().item())
            if count != 18:
                raise RuntimeError(
                    f"formal balanced rollout requires 18 episode costs for eta={eta_value:.2f}, got {count}"
                )
            mean_cost = float(episode_costs[mask].mean().item())
            if not math.isfinite(mean_cost):
                raise FloatingPointError("dual mean constraint cost became NaN/Inf")
            updated = max(
                0.0,
                float(self.dual_by_eta[eta_value])
                + self.hparams.dual_learning_rate * mean_cost,
            )
            self.dual_by_eta[eta_value] = updated
            key = f"{eta_value:.2f}"
            means[key] = mean_cost
            after[key] = updated
        return before, after, means

    def update(self, batch: LagrangianRolloutBatch) -> PPOLagrangianUpdateStats:
        if batch.policy_version != self.policy_version:
            raise RuntimeError(
                "strict on-policy violation: rollout policy_version "
                f"{batch.policy_version} != current {self.policy_version}"
            )
        n = batch.size
        if n <= 0 or n % self.hparams.minibatch_size != 0:
            raise ValueError("rollout must be non-empty and divisible by minibatch_size")

        states = batch.states.to(self.device)
        irrigate = batch.irrigate.to(self.device)
        raw_amount = batch.raw_amount.to(self.device)
        old_log_probs = batch.old_log_probs.to(self.device)
        etas = batch.etas.to(self.device)
        reward_returns = batch.reward_returns.to(self.device)
        cost_returns = batch.cost_returns.to(self.device)
        reward_adv = batch.reward_advantages.to(self.device)
        cost_adv = batch.cost_advantages.to(self.device)
        episode_etas = batch.episode_etas.to(self.device)
        episode_costs = batch.episode_costs.to(self.device)

        # Freeze lambda_k for the whole PPO update. The dual update happens only
        # after the actor/critics have consumed this on-policy batch.
        lambda_per_transition = self._lambda_tensor(etas)
        lagrangian_adv = reward_adv - lambda_per_transition * cost_adv
        adv_mean = float(lagrangian_adv.mean().item())
        adv_std = float(lagrangian_adv.std(unbiased=False).item())
        lagrangian_adv = (lagrangian_adv - lagrangian_adv.mean()) / (
            lagrangian_adv.std(unbiased=False) + 1e-8
        )

        actor_losses: list[float] = []
        reward_value_losses: list[float] = []
        cost_value_losses: list[float] = []
        total_losses: list[float] = []
        entropies: list[float] = []
        approx_kls: list[float] = []
        clip_fractions: list[float] = []
        grad_norms: list[float] = []

        parameters = (
            list(self.actor.parameters())
            + list(self.reward_critic.parameters())
            + list(self.cost_critic.parameters())
        )
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
                surrogate1 = ratio * lagrangian_adv[idx]
                surrogate2 = ratio.clamp(
                    1.0 - self.hparams.clip_epsilon,
                    1.0 + self.hparams.clip_epsilon,
                ) * lagrangian_adv[idx]
                actor_loss = -torch.minimum(surrogate1, surrogate2).mean()
                reward_value_loss = F.mse_loss(
                    self.reward_critic(states[idx]), reward_returns[idx]
                )
                cost_value_loss = F.mse_loss(
                    self.cost_critic(states[idx]), cost_returns[idx]
                )
                entropy_mean = entropy.mean()
                total_loss = (
                    actor_loss
                    + self.hparams.reward_value_loss_coefficient * reward_value_loss
                    + self.hparams.cost_value_loss_coefficient * cost_value_loss
                    - self.hparams.entropy_coefficient * entropy_mean
                )
                if not torch.isfinite(total_loss):
                    raise FloatingPointError("PPO-Lagrangian loss became NaN/Inf")
                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    parameters, self.hparams.max_grad_norm
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
                reward_value_losses.append(float(reward_value_loss.detach().item()))
                cost_value_losses.append(float(cost_value_loss.detach().item()))
                total_losses.append(float(total_loss.detach().item()))
                entropies.append(float(entropy_mean.detach().item()))
                approx_kls.append(float(approx_kl.detach().item()))
                clip_fractions.append(float(clip_fraction.detach().item()))
                grad_norms.append(float(torch.as_tensor(grad_norm).detach().item()))

        dual_before, dual_after, mean_costs = self._dual_update(
            episode_etas=episode_etas,
            episode_costs=episode_costs,
        )
        old_version = self.policy_version
        self.policy_version += 1
        self.update_index += 1
        return PPOLagrangianUpdateStats(
            update_index=self.update_index,
            rollout_policy_version=old_version,
            new_policy_version=self.policy_version,
            sample_count=n,
            actor_loss=mean(actor_losses),
            reward_value_loss=mean(reward_value_losses),
            cost_value_loss=mean(cost_value_losses),
            total_loss=mean(total_losses),
            entropy=mean(entropies),
            approx_kl=mean(approx_kls),
            clip_fraction=mean(clip_fractions),
            grad_norm=mean(grad_norms),
            lagrangian_advantage_mean_before_normalization=adv_mean,
            lagrangian_advantage_std_before_normalization=adv_std,
            dual_before=dual_before,
            dual_after=dual_after,
            mean_constraint_cost_by_eta=mean_costs,
        )

    def checkpoint_payload(self) -> dict[str, object]:
        return {
            "protocol_id": "awm-ppo-lagrangian-baseline-v1",
            "seed": self.seed,
            "policy_version": self.policy_version,
            "update_index": self.update_index,
            "actor_state_dict": self.actor.state_dict(),
            "reward_critic_state_dict": self.reward_critic.state_dict(),
            "cost_critic_state_dict": self.cost_critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "generator_state": self.generator.get_state(),
            "dual_by_eta": {f"{eta:.2f}": value for eta, value in self.dual_by_eta.items()},
            "hyperparameters": asdict(self.hparams),
        }

    def load_checkpoint_payload(self, payload: Mapping[str, object]) -> None:
        if payload.get("protocol_id") != "awm-ppo-lagrangian-baseline-v1":
            raise ValueError("checkpoint protocol_id mismatch")
        if int(payload.get("seed", -1)) != self.seed:
            raise ValueError("checkpoint seed does not match agent seed")
        if payload.get("hyperparameters") != asdict(self.hparams):
            raise ValueError("checkpoint hyperparameters do not match agent configuration")
        self.actor.load_state_dict(payload["actor_state_dict"], strict=True)
        self.reward_critic.load_state_dict(payload["reward_critic_state_dict"], strict=True)
        self.cost_critic.load_state_dict(payload["cost_critic_state_dict"], strict=True)
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        raw_duals = payload["dual_by_eta"]
        restored: dict[float, float] = {}
        for eta in REGISTERED_ETA_LEVELS:
            key = f"{float(eta):.2f}"
            if key not in raw_duals:
                raise ValueError(f"checkpoint missing dual for eta={key}")
            value = float(raw_duals[key])
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("checkpoint dual values must be finite and >= 0")
            restored[float(eta)] = value
        self.dual_by_eta = restored
        self.policy_version = int(payload["policy_version"])
        self.update_index = int(payload["update_index"])
        if self.policy_version < 0 or self.update_index < 0:
            raise ValueError("checkpoint version/update index must be nonnegative")
        self.generator.set_state(payload["generator_state"])


__all__ = [
    "PPOLagrangianAgent",
    "PPOLagrangianHyperparameters",
    "PPOLagrangianUpdateStats",
]
