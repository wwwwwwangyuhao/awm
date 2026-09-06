"""Strictly on-policy lower-CVaR primal-dual PPO agent for RCWA-RL v1."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from statistics import mean
from typing import Mapping

import torch
import torch.nn.functional as F

from awm.ppo.models import HierarchicalActionBatch, HierarchicalIrrigationActor, IrrigationValueNetwork
from awm.risk import REGISTERED_ETA_LEVELS

from .buffer import RCWARolloutBatch


@dataclass(frozen=True, slots=True)
class RCWAHyperparameters:
    state_dim: int = 79
    actor_hidden_dims: tuple[int, int] = (256, 128)
    reward_critic_hidden_dims: tuple[int, int] = (256, 128)
    risk_critic_hidden_dims: tuple[int, int] = (256, 128)
    learning_rate: float = 1e-4
    gamma: float = 1.0
    gae_lambda: float = 1.0
    risk_gamma: float = 1.0
    risk_gae_lambda: float = 1.0
    alpha: float = 0.20
    clip_epsilon: float = 0.2
    update_epochs: int = 10
    minibatch_size: int = 450
    reward_value_loss_coefficient: float = 0.5
    risk_value_loss_coefficient: float = 0.5
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
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must lie in (0,1]")
        for name in ("learning_rate", "clip_epsilon", "max_grad_norm", "adam_eps", "dual_learning_rate"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0")
        for name in ("gamma", "gae_lambda", "risk_gamma", "risk_gae_lambda"):
            if not 0.0 <= float(getattr(self, name)) <= 1.0:
                raise ValueError(f"{name} must lie in [0,1]")
        for name in (
            "reward_value_loss_coefficient",
            "risk_value_loss_coefficient",
            "entropy_coefficient",
            "dual_initial_value",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be >= 0")
        if self.update_epochs <= 0 or self.minibatch_size <= 0:
            raise ValueError("update_epochs and minibatch_size must be positive")
        if not 0.0 <= self.adam_beta1 < 1.0 or not 0.0 <= self.adam_beta2 < 1.0:
            raise ValueError("Adam beta parameters must lie in [0,1)")


@dataclass(frozen=True, slots=True)
class RCWAUpdateStats:
    update_index: int
    rollout_policy_version: int
    new_policy_version: int
    sample_count: int
    actor_loss: float
    reward_value_loss: float
    risk_value_loss: float
    total_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    grad_norm: float
    combined_advantage_mean_before_normalization: float
    combined_advantage_std_before_normalization: float
    dual_before: dict[str, float]
    dual_after: dict[str, float]
    tau_by_eta: dict[str, float]
    lcvar_by_eta: dict[str, float]
    violation_by_eta: dict[str, float]


class RCWAAgent:
    def __init__(
        self,
        *,
        hyperparameters: RCWAHyperparameters | None = None,
        seed: int = 21,
        device: str | torch.device = "cpu",
    ) -> None:
        self.hparams = hyperparameters or RCWAHyperparameters()
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
        self.risk_critic = IrrigationValueNetwork(
            state_dim=self.hparams.state_dim,
            hidden_dims=self.hparams.risk_critic_hidden_dims,
        ).to(self.device)
        self.critic = self.reward_critic
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters())
            + list(self.reward_critic.parameters())
            + list(self.risk_critic.parameters()),
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
    def act(self, state: torch.Tensor) -> tuple[HierarchicalActionBatch, torch.Tensor, torch.Tensor]:
        state = state.to(self.device, dtype=torch.float32)
        action = self.actor.sample(state, generator=self.generator)
        return action, self.reward_critic(state), self.risk_critic(state)

    @torch.no_grad()
    def deterministic_action(self, state: torch.Tensor) -> tuple[HierarchicalActionBatch, torch.Tensor]:
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
            raise ValueError("batch contains unregistered eta values")
        return result

    def _dual_update(self, batch: RCWARolloutBatch):
        before = {f"{eta:.2f}": float(value) for eta, value in self.dual_by_eta.items()}
        after: dict[str, float] = {}
        tau: dict[str, float] = {}
        lcvar: dict[str, float] = {}
        violation: dict[str, float] = {}
        for eta in REGISTERED_ETA_LEVELS:
            key = f"{float(eta):.2f}"
            if key not in batch.tail_metrics:
                raise RuntimeError(f"missing tail metrics for eta={key}")
            metric = batch.tail_metrics[key]
            if metric.sample_count != 18:
                raise RuntimeError(f"eta={key} tail group must contain 18 episodes")
            g = float(metric.violation)
            updated = max(0.0, float(self.dual_by_eta[float(eta)]) + self.hparams.dual_learning_rate * g)
            self.dual_by_eta[float(eta)] = updated
            after[key] = updated
            tau[key] = float(metric.tau)
            lcvar[key] = float(metric.empirical_lcvar)
            violation[key] = g
        return before, after, tau, lcvar, violation

    def update(self, batch: RCWARolloutBatch) -> RCWAUpdateStats:
        if batch.policy_version != self.policy_version:
            raise RuntimeError(
                f"strict on-policy violation: rollout policy_version {batch.policy_version} != current {self.policy_version}"
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
        risk_returns = batch.risk_returns.to(self.device)
        reward_adv = batch.reward_advantages.to(self.device)
        risk_adv = batch.risk_advantages.to(self.device)

        lambda_per_transition = self._lambda_tensor(etas)
        combined_adv = reward_adv - lambda_per_transition * risk_adv
        adv_mean = float(combined_adv.mean().item())
        adv_std = float(combined_adv.std(unbiased=False).item())
        combined_adv = (combined_adv - combined_adv.mean()) / (combined_adv.std(unbiased=False) + 1e-8)

        actor_losses: list[float] = []
        reward_value_losses: list[float] = []
        risk_value_losses: list[float] = []
        total_losses: list[float] = []
        entropies: list[float] = []
        approx_kls: list[float] = []
        clip_fractions: list[float] = []
        grad_norms: list[float] = []
        parameters = list(self.actor.parameters()) + list(self.reward_critic.parameters()) + list(self.risk_critic.parameters())

        for _epoch in range(self.hparams.update_epochs):
            permutation = torch.randperm(n, generator=self.generator, device=self.device)
            for start in range(0, n, self.hparams.minibatch_size):
                idx = permutation[start : start + self.hparams.minibatch_size]
                new_log_prob, entropy = self.actor.evaluate_behavior(states[idx], irrigate[idx], raw_amount[idx])
                log_ratio = new_log_prob - old_log_probs[idx]
                ratio = torch.exp(log_ratio)
                surrogate1 = ratio * combined_adv[idx]
                surrogate2 = ratio.clamp(1.0 - self.hparams.clip_epsilon, 1.0 + self.hparams.clip_epsilon) * combined_adv[idx]
                actor_loss = -torch.minimum(surrogate1, surrogate2).mean()
                reward_value_loss = F.mse_loss(self.reward_critic(states[idx]), reward_returns[idx])
                risk_value_loss = F.mse_loss(self.risk_critic(states[idx]), risk_returns[idx])
                entropy_mean = entropy.mean()
                total_loss = (
                    actor_loss
                    + self.hparams.reward_value_loss_coefficient * reward_value_loss
                    + self.hparams.risk_value_loss_coefficient * risk_value_loss
                    - self.hparams.entropy_coefficient * entropy_mean
                )
                if not torch.isfinite(total_loss):
                    raise FloatingPointError("RCWA loss became NaN/Inf")
                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, self.hparams.max_grad_norm)
                self.optimizer.step()
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = (torch.abs(ratio - 1.0) > self.hparams.clip_epsilon).float().mean()
                actor_losses.append(float(actor_loss.detach().item()))
                reward_value_losses.append(float(reward_value_loss.detach().item()))
                risk_value_losses.append(float(risk_value_loss.detach().item()))
                total_losses.append(float(total_loss.detach().item()))
                entropies.append(float(entropy_mean.detach().item()))
                approx_kls.append(float(approx_kl.detach().item()))
                clip_fractions.append(float(clip_fraction.detach().item()))
                grad_norms.append(float(torch.as_tensor(grad_norm).detach().item()))

        dual_before, dual_after, tau, lcvar, violation = self._dual_update(batch)
        old_version = self.policy_version
        self.policy_version += 1
        self.update_index += 1
        return RCWAUpdateStats(
            update_index=self.update_index,
            rollout_policy_version=old_version,
            new_policy_version=self.policy_version,
            sample_count=n,
            actor_loss=mean(actor_losses),
            reward_value_loss=mean(reward_value_losses),
            risk_value_loss=mean(risk_value_losses),
            total_loss=mean(total_losses),
            entropy=mean(entropies),
            approx_kl=mean(approx_kls),
            clip_fraction=mean(clip_fractions),
            grad_norm=mean(grad_norms),
            combined_advantage_mean_before_normalization=adv_mean,
            combined_advantage_std_before_normalization=adv_std,
            dual_before=dual_before,
            dual_after=dual_after,
            tau_by_eta=tau,
            lcvar_by_eta=lcvar,
            violation_by_eta=violation,
        )

    def checkpoint_payload(self) -> dict[str, object]:
        return {
            "protocol_id": "awm-rcwa-rl-v1",
            "seed": self.seed,
            "policy_version": self.policy_version,
            "update_index": self.update_index,
            "actor_state_dict": self.actor.state_dict(),
            "reward_critic_state_dict": self.reward_critic.state_dict(),
            "risk_critic_state_dict": self.risk_critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "generator_state": self.generator.get_state(),
            "dual_by_eta": {f"{eta:.2f}": value for eta, value in self.dual_by_eta.items()},
            "hyperparameters": asdict(self.hparams),
        }

    def load_checkpoint_payload(self, payload: Mapping[str, object]) -> None:
        if payload.get("protocol_id") != "awm-rcwa-rl-v1":
            raise ValueError("checkpoint protocol_id mismatch")
        if int(payload.get("seed", -1)) != self.seed:
            raise ValueError("checkpoint seed does not match agent seed")
        if payload.get("hyperparameters") != asdict(self.hparams):
            raise ValueError("checkpoint hyperparameters do not match agent configuration")
        self.actor.load_state_dict(payload["actor_state_dict"], strict=True)
        self.reward_critic.load_state_dict(payload["reward_critic_state_dict"], strict=True)
        self.risk_critic.load_state_dict(payload["risk_critic_state_dict"], strict=True)
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        raw_duals = payload["dual_by_eta"]
        restored: dict[float, float] = {}
        for eta in REGISTERED_ETA_LEVELS:
            key = f"{float(eta):.2f}"
            restored[float(eta)] = float(raw_duals[key])
        self.dual_by_eta = restored
        self.policy_version = int(payload["policy_version"])
        self.update_index = int(payload["update_index"])
        self.generator.set_state(payload["generator_state"])


__all__ = ["RCWAAgent", "RCWAHyperparameters", "RCWAUpdateStats"]
