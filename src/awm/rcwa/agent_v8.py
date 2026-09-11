"""RCWA-v8: v6 policy-consistent Tail-Q with twin-critic reliability shrinkage.

V8 keeps the v6 primary Tail-Q, policy-consistent baseline, lower-CVaR objective,
rollout, dual update and PPO optimization unchanged.  A second Tail-Q estimator
is trained on the same complete on-policy batch with the same target and fit
budget, but with independent initialization, optimizer and minibatch RNG.  The
auxiliary critic never replaces the primary actor credit.  Its sole purpose is
to estimate function-approximation uncertainty from disagreement.

For each eta, center the two policy-consistent credits A1 and A2 and define
M=(A1+A2)/2 and D=A1-A2.  Under the working two-estimator error model,
nu=Var(D)/2 estimates per-estimator noise variance and
s2=max(0, Var(M)-nu/2) estimates shared action-risk signal variance.  The
parameter-free reliability rho=s2/(s2+nu) multiplies the usual within-eta
standardized primary v6 credit.  Small but critic-consistent action separation
can therefore retain full strength, while disagreement-dominated credit is
attenuated without an outcome-dispersion threshold.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from statistics import mean
from typing import Mapping

import torch
import torch.nn.functional as F

from awm.risk import REGISTERED_ETA_LEVELS
from .agent_v4 import TailActionValueNetwork
from .agent_v6 import (
    POLICY_BASELINE_GH_ORDER,
    RCWAV6Agent,
    RCWAV6Hyperparameters,
    RCWAV6UpdateStats,
    _gauss_hermite_rule,
)
from .buffer import RCWARolloutBatch

V8_PROTOCOL_ID = "awm-rcwa-rl-v8"


@dataclass(frozen=True)
class RCWAV8Hyperparameters(RCWAV6Hyperparameters):
    """No tuned learning hyperparameters are added relative to RCWA-v6."""


@dataclass(frozen=True, slots=True)
class RCWAV8UpdateStats(RCWAV6UpdateStats):
    aux_tail_q_prefit_loss_before: float = 0.0
    aux_tail_q_prefit_loss_after: float = 0.0
    aux_tail_q_prefit_grad_norm_mean: float = 0.0
    aux_tail_q_prefit_grad_norm_max: float = 0.0
    aux_tail_q_prefit_steps: int = 0
    twin_q_primary_credit_std_by_eta: dict[str, float] = field(default_factory=dict)
    twin_q_aux_credit_std_by_eta: dict[str, float] = field(default_factory=dict)
    twin_q_disagreement_variance_by_eta: dict[str, float] = field(default_factory=dict)
    twin_q_noise_variance_by_eta: dict[str, float] = field(default_factory=dict)
    twin_q_consensus_signal_variance_by_eta: dict[str, float] = field(default_factory=dict)
    twin_q_reliability_by_eta: dict[str, float] = field(default_factory=dict)
    twin_q_credit_correlation_by_eta: dict[str, float] = field(default_factory=dict)
    twin_q_aux_expected_advantage_max_abs_error_by_eta: dict[str, float] = field(default_factory=dict)
    twin_q_auxiliary_seed: int = 0


class RCWAV8Agent(RCWAV6Agent):
    PROTOCOL_ID = V8_PROTOCOL_ID

    def __init__(
        self,
        *,
        hyperparameters: RCWAV8Hyperparameters | None = None,
        seed: int = 21,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__(
            hyperparameters=hyperparameters or RCWAV8Hyperparameters(),
            seed=seed,
            device=device,
        )
        # +1 is a fixed estimator identity, not a tuned learning parameter.
        self.auxiliary_seed = int(self.seed + 1)
        # Construct on CPU under a forked RNG so v6 primary initialization and
        # process-global RNG state are unchanged. Moving to device consumes no RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.auxiliary_seed)
            auxiliary = TailActionValueNetwork(
                state_dim=self.hparams.state_dim,
                hidden_dims=self.hparams.risk_critic_hidden_dims,
            )
        self.aux_tail_q_critic = auxiliary.to(self.device)
        self.aux_tail_q_optimizer = torch.optim.Adam(
            self.aux_tail_q_critic.parameters(),
            lr=self.hparams.learning_rate,
            betas=(self.hparams.adam_beta1, self.hparams.adam_beta2),
            eps=self.hparams.adam_eps,
        )
        generator_device = self.device if self.device.type == "cuda" else torch.device("cpu")
        self.aux_generator = torch.Generator(device=generator_device)
        self.aux_generator.manual_seed(self.auxiliary_seed)

    @torch.no_grad()
    def _aux_tail_q_mse(
        self,
        states: torch.Tensor,
        action_features: torch.Tensor,
        risk_returns: torch.Tensor,
    ) -> float:
        return float(F.mse_loss(self.aux_tail_q_critic(states, action_features), risk_returns).item())

    def _prefit_aux_tail_q_critic(
        self,
        *,
        states: torch.Tensor,
        action_features: torch.Tensor,
        risk_returns: torch.Tensor,
    ) -> dict[str, object]:
        """Fit only the auxiliary Q on the same full frozen batch as primary Q."""
        # Primary prefit leaves its last gradients allocated; clear them before
        # auxiliary fitting so isolation checks remain meaningful.
        self.optimizer.zero_grad(set_to_none=True)
        params = list(self.aux_tail_q_critic.parameters())
        transition_count = int(states.shape[0])
        loss_before = self._aux_tail_q_mse(states, action_features, risk_returns)
        grad_norms: list[float] = []
        for _epoch in range(int(self.hparams.tail_q_prefit_epochs)):
            permutation = torch.randperm(
                transition_count, generator=self.aux_generator, device=self.device
            )
            for start in range(0, transition_count, self.hparams.minibatch_size):
                idx = permutation[start : start + self.hparams.minibatch_size]
                prediction = self.aux_tail_q_critic(states[idx], action_features[idx])
                loss = F.mse_loss(prediction, risk_returns[idx])
                if not torch.isfinite(loss):
                    raise FloatingPointError("RCWA v8 auxiliary tail-Q prefit loss became NaN/Inf")
                self.aux_tail_q_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(params, self.hparams.max_grad_norm)
                grad_norms.append(float(torch.as_tensor(grad_norm).detach().item()))
                self.aux_tail_q_optimizer.step()
        self.aux_tail_q_optimizer.zero_grad(set_to_none=True)
        loss_after = self._aux_tail_q_mse(states, action_features, risk_returns)
        return {
            "aux_tail_q_prefit_loss_before": loss_before,
            "aux_tail_q_prefit_loss_after": loss_after,
            "aux_tail_q_prefit_grad_norm_mean": mean(grad_norms),
            "aux_tail_q_prefit_grad_norm_max": max(grad_norms),
            "aux_tail_q_prefit_steps": len(grad_norms),
            "twin_q_auxiliary_seed": self.auxiliary_seed,
        }

    @torch.no_grad()
    def _policy_q_baseline_for_critic(
        self,
        states: torch.Tensor,
        critic: TailActionValueNetwork,
        *,
        quadrature_order: int = POLICY_BASELINE_GH_ORDER,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
            q_noop = critic(s, noop_action)
            raw = dist.loc[:, None] + math.sqrt(2.0) * dist.scale[:, None] * nodes[None, :]
            amount = (torch.tanh(raw) + 1.0) * 0.5
            repeated_states = s[:, None, :].expand(-1, order, -1).reshape(-1, s.shape[1])
            active_action = torch.stack((torch.ones_like(amount), amount), dim=-1).reshape(-1, 2)
            q_nodes = critic(repeated_states, active_action).reshape(-1, order)
            q_active = (q_nodes * weights[None, :]).sum(dim=1) / norm
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

    def _twin_reliability_diagnostics(
        self,
        *,
        primary_credit: torch.Tensor,
        aux_credit: torch.Tensor,
        aux_baseline: torch.Tensor,
        aux_q_noop: torch.Tensor,
        aux_q_active: torch.Tensor,
        gate_p: torch.Tensor,
        etas: torch.Tensor,
    ) -> dict[str, object]:
        primary_std: dict[str, float] = {}
        aux_std: dict[str, float] = {}
        disagreement_var: dict[str, float] = {}
        noise_var: dict[str, float] = {}
        signal_var: dict[str, float] = {}
        reliability: dict[str, float] = {}
        correlation: dict[str, float] = {}
        identity_error: dict[str, float] = {}
        rho_by_eta: dict[str, float] = {}
        aux_expected = (1.0 - gate_p) * (aux_q_noop - aux_baseline) + gate_p * (
            aux_q_active - aux_baseline
        )
        for key, mask in self._eta_masks(etas).items():
            a1 = primary_credit[mask]
            a2 = aux_credit[mask]
            c1 = a1 - a1.mean()
            c2 = a2 - a2.mean()
            m = 0.5 * (c1 + c2)
            d = c1 - c2
            v1 = c1.pow(2).mean()
            v2 = c2.pow(2).mean()
            vd = d.pow(2).mean()
            vn = 0.5 * vd
            vm = m.pow(2).mean()
            vs = torch.clamp(vm - 0.5 * vn, min=0.0)
            denom = vs + vn
            rho = torch.where(denom > 1e-16, vs / denom, torch.zeros_like(denom))
            corr_denom = torch.sqrt(v1 * v2)
            corr = torch.where(
                corr_denom > 1e-16,
                (c1 * c2).mean() / corr_denom,
                torch.zeros_like(corr_denom),
            )
            primary_std[key] = float(torch.sqrt(v1).item())
            aux_std[key] = float(torch.sqrt(v2).item())
            disagreement_var[key] = float(vd.item())
            noise_var[key] = float(vn.item())
            signal_var[key] = float(vs.item())
            rho_value = float(torch.clamp(rho, 0.0, 1.0).item())
            reliability[key] = rho_value
            correlation[key] = float(torch.clamp(corr, -1.0, 1.0).item())
            identity_error[key] = float(aux_expected[mask].abs().max().item())
            rho_by_eta[key] = rho_value
        self._v8_reliability_by_eta = rho_by_eta
        return {
            "twin_q_primary_credit_std_by_eta": primary_std,
            "twin_q_aux_credit_std_by_eta": aux_std,
            "twin_q_disagreement_variance_by_eta": disagreement_var,
            "twin_q_noise_variance_by_eta": noise_var,
            "twin_q_consensus_signal_variance_by_eta": signal_var,
            "twin_q_reliability_by_eta": reliability,
            "twin_q_credit_correlation_by_eta": correlation,
            "twin_q_aux_expected_advantage_max_abs_error_by_eta": identity_error,
        }

    def _actor_risk_credit(
        self,
        *,
        batch: RCWARolloutBatch,
        states: torch.Tensor,
        etas: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        primary_credit, diagnostics = super()._actor_risk_credit(
            batch=batch, states=states, etas=etas
        )
        action_features = self._action_features_from_batch(batch, device=self.device)
        risk_returns = batch.risk_returns.to(self.device)
        diagnostics.update(
            self._prefit_aux_tail_q_critic(
                states=states,
                action_features=action_features,
                risk_returns=risk_returns,
            )
        )
        with torch.no_grad():
            aux_q = self.aux_tail_q_critic(states, action_features).detach()
            aux_baseline, aux_q_noop, aux_q_active, gate_p = self._policy_q_baseline_for_critic(
                states, self.aux_tail_q_critic
            )
            aux_credit = (aux_q - aux_baseline).detach()
        diagnostics.update(
            self._twin_reliability_diagnostics(
                primary_credit=primary_credit,
                aux_credit=aux_credit,
                aux_baseline=aux_baseline,
                aux_q_noop=aux_q_noop,
                aux_q_active=aux_q_active,
                gate_p=gate_p,
                etas=etas,
            )
        )
        return primary_credit, diagnostics

    def _condition_actor_advantages(
        self,
        *,
        reward_adv: torch.Tensor,
        risk_adv: torch.Tensor,
        etas: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        if not hasattr(self, "_v8_reliability_by_eta"):
            raise RuntimeError("RCWA v8 reliability must be computed before advantage conditioning")
        reward_z, reward_mean, reward_std = self._standardize(reward_adv)
        risk_conditioned = torch.empty_like(risk_adv)
        risk_mean_by_eta: dict[str, float] = {}
        risk_std_by_eta: dict[str, float] = {}
        matched = torch.zeros_like(etas, dtype=torch.bool, device=etas.device)
        for eta in REGISTERED_ETA_LEVELS:
            eta_value = float(eta)
            key = f"{eta_value:.2f}"
            mask = torch.isclose(
                etas,
                torch.as_tensor(eta_value, dtype=etas.dtype, device=etas.device),
                atol=1e-6,
                rtol=0.0,
            )
            if not bool(mask.any().item()):
                raise RuntimeError(f"missing transitions for eta={key}")
            group_z, group_mean, group_std = self._standardize(risk_adv[mask])
            rho = float(self._v8_reliability_by_eta[key])
            risk_conditioned[mask] = group_z * rho
            risk_mean_by_eta[key] = group_mean
            risk_std_by_eta[key] = group_std
            matched |= mask
        if not bool(matched.all().item()):
            raise ValueError("batch contains unregistered eta values")
        lambda_per_transition = self._lambda_tensor(etas)
        combined = reward_z - lambda_per_transition * risk_conditioned
        return combined, {
            "reward_advantage_mean_before_conditioning": reward_mean,
            "reward_advantage_std_before_conditioning": reward_std,
            "risk_advantage_mean_before_conditioning_by_eta": risk_mean_by_eta,
            "risk_advantage_std_before_conditioning_by_eta": risk_std_by_eta,
            "combined_advantage_mean_after_conditioning": float(combined.mean().item()),
            "combined_advantage_std_after_conditioning": float(combined.std(unbiased=False).item()),
        }

    def _build_update_stats(self, fields: Mapping[str, object]):
        return RCWAV8UpdateStats(**fields)

    def checkpoint_payload(self) -> dict[str, object]:
        payload = super().checkpoint_payload()
        payload["aux_tail_q_critic_state_dict"] = self.aux_tail_q_critic.state_dict()
        payload["aux_tail_q_optimizer_state_dict"] = self.aux_tail_q_optimizer.state_dict()
        payload["aux_generator_state"] = self.aux_generator.get_state()
        payload["twin_q_auxiliary_seed"] = self.auxiliary_seed
        return payload

    def load_checkpoint_payload(self, payload: Mapping[str, object]) -> None:
        if payload.get("protocol_id") != self.PROTOCOL_ID:
            raise ValueError("checkpoint protocol_id mismatch")
        if int(payload.get("twin_q_auxiliary_seed", -1)) != self.auxiliary_seed:
            raise ValueError("checkpoint auxiliary seed does not match v8 estimator identity")
        if "aux_tail_q_critic_state_dict" not in payload or "aux_tail_q_optimizer_state_dict" not in payload:
            raise ValueError("RCWA v8 checkpoint missing auxiliary Tail-Q state")
        self.aux_tail_q_critic.load_state_dict(payload["aux_tail_q_critic_state_dict"], strict=True)
        self.aux_tail_q_optimizer.load_state_dict(payload["aux_tail_q_optimizer_state_dict"])
        aux_state = payload.get("aux_generator_state")
        if not isinstance(aux_state, torch.Tensor):
            raise TypeError("checkpoint aux_generator_state must be a torch.Tensor")
        self.aux_generator.set_state(aux_state.detach().cpu())
        super().load_checkpoint_payload(payload)


__all__ = [
    "RCWAV8Agent",
    "RCWAV8Hyperparameters",
    "RCWAV8UpdateStats",
    "V8_PROTOCOL_ID",
]
