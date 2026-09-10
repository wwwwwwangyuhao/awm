"""Strictly on-policy stabilized lower-CVaR primal-dual PPO agent for RCWA-RL v2."""
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
    actor_grad_norm: float
    reward_critic_grad_norm: float
    risk_critic_grad_norm: float
    combined_advantage_mean_after_conditioning: float
    combined_advantage_std_after_conditioning: float
    reward_advantage_mean_before_conditioning: float
    reward_advantage_std_before_conditioning: float
    risk_advantage_mean_before_conditioning_by_eta: dict[str, float]
    risk_advantage_std_before_conditioning_by_eta: dict[str, float]
    dual_before: dict[str, float]
    dual_after: dict[str, float]
    tau_by_eta: dict[str, float]
    lcvar_by_eta: dict[str, float]
    violation_by_eta: dict[str, float]


class RCWAAgent:
    # Checkpoint protocol identity.  RCWA v3 subclasses override this so a v2
    # payload can never be silently accepted as a v3 payload.
    PROTOCOL_ID = "awm-rcwa-rl-v2"

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
        # V2 stabilization: the extra risk baseline must not inject a random
        # action-correlated signal into the very first primal update.  The
        # actor and reward critic still exactly match PPO initialization.
        torch.nn.init.zeros_(self.risk_critic.value_head.weight)
        torch.nn.init.zeros_(self.risk_critic.value_head.bias)
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

    @staticmethod
    def _standardize(values: torch.Tensor) -> tuple[torch.Tensor, float, float]:
        mean_value = values.mean()
        std_value = values.std(unbiased=False)
        normalized = (values - mean_value) / (std_value + 1e-8)
        return normalized, float(mean_value.item()), float(std_value.item())

    def _condition_actor_advantages(
        self,
        *,
        reward_adv: torch.Tensor,
        risk_adv: torch.Tensor,
        etas: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        """Condition reward/risk components without erasing dual scale.

        V1 standardized the already dual-weighted combined advantage.  That
        made a larger lambda change direction but largely removed its intended
        scale effect.  V2 standardizes each component first, with the sparse
        tail-risk component standardized independently inside each eta group,
        and then applies the frozen eta-specific dual multiplier.
        """
        reward_z, reward_mean, reward_std = self._standardize(reward_adv)
        risk_z = torch.empty_like(risk_adv)
        risk_mean_by_eta: dict[str, float] = {}
        risk_std_by_eta: dict[str, float] = {}
        matched = torch.zeros_like(etas, dtype=torch.bool, device=etas.device)
        for eta in REGISTERED_ETA_LEVELS:
            eta_value = float(eta)
            key = f"{eta_value:.2f}"
            mask = torch.isclose(
                etas,
                torch.tensor(eta_value, dtype=etas.dtype, device=etas.device),
                atol=1e-6,
                rtol=0.0,
            )
            if not bool(mask.any().item()):
                raise RuntimeError(f"missing transitions for eta={key}")
            group_z, group_mean, group_std = self._standardize(risk_adv[mask])
            risk_z[mask] = group_z
            risk_mean_by_eta[key] = group_mean
            risk_std_by_eta[key] = group_std
            matched |= mask
        if not bool(matched.all().item()):
            raise ValueError("batch contains unregistered eta values")

        lambda_per_transition = self._lambda_tensor(etas)
        combined = reward_z - lambda_per_transition * risk_z
        diagnostics: dict[str, object] = {
            "reward_advantage_mean_before_conditioning": reward_mean,
            "reward_advantage_std_before_conditioning": reward_std,
            "risk_advantage_mean_before_conditioning_by_eta": risk_mean_by_eta,
            "risk_advantage_std_before_conditioning_by_eta": risk_std_by_eta,
            "combined_advantage_mean_after_conditioning": float(combined.mean().item()),
            "combined_advantage_std_after_conditioning": float(combined.std(unbiased=False).item()),
        }
        return combined, diagnostics

    def _actor_risk_credit(
        self,
        *,
        batch: RCWARolloutBatch,
        states: torch.Tensor,
        etas: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        """Return the actor's tail-risk credit plus extra update diagnostics.

        V2 uses the Monte-Carlo complete-episode risk advantage; with gamma=1
        it is the same number ``h_i`` on every transition of an episode, which
        is exactly the delayed-credit failure v3 addresses.  RCWA v3 overrides
        this hook with a frozen one-step TD credit instead.
        """
        del states, etas
        return batch.risk_advantages.to(self.device), {}

    def _build_update_stats(self, fields: Mapping[str, object]) -> RCWAUpdateStats:
        return RCWAUpdateStats(**fields)  # type: ignore[arg-type]

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
        actor_risk_credit, credit_diagnostics = self._actor_risk_credit(
            batch=batch,
            states=states,
            etas=etas,
        )

        combined_adv, advantage_diagnostics = self._condition_actor_advantages(
            reward_adv=reward_adv,
            risk_adv=actor_risk_credit,
            etas=etas,
        )
        adv_mean = float(combined_adv.mean().item())
        adv_std = float(combined_adv.std(unbiased=False).item())

        actor_losses: list[float] = []
        reward_value_losses: list[float] = []
        risk_value_losses: list[float] = []
        total_losses: list[float] = []
        entropies: list[float] = []
        approx_kls: list[float] = []
        clip_fractions: list[float] = []
        actor_grad_norms: list[float] = []
        reward_critic_grad_norms: list[float] = []
        risk_critic_grad_norms: list[float] = []
        actor_parameters = list(self.actor.parameters())
        reward_critic_parameters = list(self.reward_critic.parameters())
        risk_critic_parameters = list(self.risk_critic.parameters())

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
                # V2 clips the actor and the two critics independently so a
                # transient critic spike cannot globally shrink the actor step.
                actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                    actor_parameters, self.hparams.max_grad_norm
                )
                reward_critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                    reward_critic_parameters, self.hparams.max_grad_norm
                )
                risk_critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                    risk_critic_parameters, self.hparams.max_grad_norm
                )
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
                actor_grad_norms.append(float(torch.as_tensor(actor_grad_norm).detach().item()))
                reward_critic_grad_norms.append(
                    float(torch.as_tensor(reward_critic_grad_norm).detach().item())
                )
                risk_critic_grad_norms.append(
                    float(torch.as_tensor(risk_critic_grad_norm).detach().item())
                )

        dual_before, dual_after, tau, lcvar, violation = self._dual_update(batch)
        old_version = self.policy_version
        self.policy_version += 1
        self.update_index += 1
        fields: dict[str, object] = {
            "update_index": self.update_index,
            "rollout_policy_version": old_version,
            "new_policy_version": self.policy_version,
            "sample_count": n,
            "actor_loss": mean(actor_losses),
            "reward_value_loss": mean(reward_value_losses),
            "risk_value_loss": mean(risk_value_losses),
            "total_loss": mean(total_losses),
            "entropy": mean(entropies),
            "approx_kl": mean(approx_kls),
            "clip_fraction": mean(clip_fractions),
            "grad_norm": mean(actor_grad_norms),
            "actor_grad_norm": mean(actor_grad_norms),
            "reward_critic_grad_norm": mean(reward_critic_grad_norms),
            "risk_critic_grad_norm": mean(risk_critic_grad_norms),
            "combined_advantage_mean_after_conditioning": adv_mean,
            "combined_advantage_std_after_conditioning": adv_std,
            "reward_advantage_mean_before_conditioning": float(
                advantage_diagnostics["reward_advantage_mean_before_conditioning"]
            ),
            "reward_advantage_std_before_conditioning": float(
                advantage_diagnostics["reward_advantage_std_before_conditioning"]
            ),
            "risk_advantage_mean_before_conditioning_by_eta": dict(
                advantage_diagnostics["risk_advantage_mean_before_conditioning_by_eta"]
            ),
            "risk_advantage_std_before_conditioning_by_eta": dict(
                advantage_diagnostics["risk_advantage_std_before_conditioning_by_eta"]
            ),
            "dual_before": dual_before,
            "dual_after": dual_after,
            "tau_by_eta": tau,
            "lcvar_by_eta": lcvar,
            "violation_by_eta": violation,
        }
        fields.update(credit_diagnostics)
        return self._build_update_stats(fields)

    def checkpoint_payload(self) -> dict[str, object]:
        return {
            "protocol_id": self.PROTOCOL_ID,
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
        if payload.get("protocol_id") != self.PROTOCOL_ID:
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
        generator_state = payload["generator_state"]
        if not isinstance(generator_state, torch.Tensor):
            raise TypeError("checkpoint generator_state must be a torch.Tensor")
        self.generator.set_state(generator_state.detach().cpu())


__all__ = ["RCWAAgent", "RCWAHyperparameters", "RCWAUpdateStats"]
