"""RCWA-v7: v6 policy-consistent Tail-Q with episode-aware noise-floor conditioning.

V7 changes only the conditioning of the already-frozen v6 actor risk credit.
For each eta, the 18 complete episodes are the independent tail-outcome units;
the 125 transitions inside an episode share the same Monte-Carlo tail label.
The empirical standard error of those episode tail targets defines a parameter-
free noise floor.  A tiny action-risk separation is therefore not automatically
inflated to unit variance by within-eta standardization.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping

import torch

from awm.risk import REGISTERED_ETA_LEVELS
from .agent_v6 import RCWAV6Agent, RCWAV6Hyperparameters, RCWAV6UpdateStats
from .buffer import RCWARolloutBatch

V7_PROTOCOL_ID = "awm-rcwa-rl-v7"


@dataclass(frozen=True)
class RCWAV7Hyperparameters(RCWAV6Hyperparameters):
    """No new tuned learning hyperparameters relative to RCWA-v6."""


@dataclass(frozen=True, slots=True)
class RCWAV7UpdateStats(RCWAV6UpdateStats):
    risk_conditioning_episode_count_by_eta: dict[str, int] = field(default_factory=dict)
    risk_conditioning_episode_tail_std_by_eta: dict[str, float] = field(default_factory=dict)
    risk_conditioning_noise_floor_by_eta: dict[str, float] = field(default_factory=dict)
    risk_conditioning_reliability_by_eta: dict[str, float] = field(default_factory=dict)


class RCWAV7Agent(RCWAV6Agent):
    PROTOCOL_ID = V7_PROTOCOL_ID

    def _episode_noise_floor_diagnostics(
        self,
        *,
        batch: RCWARolloutBatch,
        credit: torch.Tensor,
        etas: torch.Tensor,
    ) -> dict[str, object]:
        dones = batch.dones.to(self.device, dtype=torch.bool)
        risk_returns = batch.risk_returns.to(self.device)
        if int(dones.sum().item()) <= 0:
            raise RuntimeError("RCWA v7 requires complete episodes for risk conditioning")
        terminal_etas = etas[dones]
        terminal_targets = risk_returns[dones]
        counts: dict[str, int] = {}
        target_std: dict[str, float] = {}
        noise_floor: dict[str, float] = {}
        reliability: dict[str, float] = {}
        floors: dict[str, float] = {}
        for eta in REGISTERED_ETA_LEVELS:
            eta_value = float(eta)
            key = f"{eta_value:.2f}"
            transition_mask = torch.isclose(
                etas,
                torch.as_tensor(eta_value, dtype=etas.dtype, device=etas.device),
                atol=1e-6,
                rtol=0.0,
            )
            episode_mask = torch.isclose(
                terminal_etas,
                torch.as_tensor(eta_value, dtype=terminal_etas.dtype, device=terminal_etas.device),
                atol=1e-6,
                rtol=0.0,
            )
            n = int(episode_mask.sum().item())
            if n <= 1:
                raise RuntimeError(f"RCWA v7 needs at least two complete episodes for eta={key}")
            h = terminal_targets[episode_mask]
            h_std = float(h.std(unbiased=False).item())
            floor = h_std / math.sqrt(float(n))
            a_std = float(credit[transition_mask].std(unbiased=False).item())
            if h_std <= 1e-12:
                rho = 0.0
            elif floor <= 1e-12:
                rho = 1.0
            else:
                rho = min(1.0, max(0.0, a_std / floor))
            counts[key] = n
            target_std[key] = h_std
            noise_floor[key] = floor
            reliability[key] = rho
            floors[key] = floor
        self._v7_noise_floor_by_eta = floors
        return {
            "risk_conditioning_episode_count_by_eta": counts,
            "risk_conditioning_episode_tail_std_by_eta": target_std,
            "risk_conditioning_noise_floor_by_eta": noise_floor,
            "risk_conditioning_reliability_by_eta": reliability,
        }

    def _actor_risk_credit(
        self,
        *,
        batch: RCWARolloutBatch,
        states: torch.Tensor,
        etas: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        credit, diagnostics = super()._actor_risk_credit(batch=batch, states=states, etas=etas)
        diagnostics.update(
            self._episode_noise_floor_diagnostics(batch=batch, credit=credit, etas=etas)
        )
        return credit, diagnostics

    def _condition_actor_advantages(
        self,
        *,
        reward_adv: torch.Tensor,
        risk_adv: torch.Tensor,
        etas: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        if not hasattr(self, "_v7_noise_floor_by_eta"):
            raise RuntimeError("RCWA v7 noise floors must be computed before advantage conditioning")
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
            floor = float(self._v7_noise_floor_by_eta[key])
            if floor <= 1e-12:
                rho = 0.0
            else:
                rho = min(1.0, max(0.0, group_std / floor))
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
        return RCWAV7UpdateStats(**fields)


__all__ = [
    "RCWAV7Agent", "RCWAV7Hyperparameters", "RCWAV7UpdateStats", "V7_PROTOCOL_ID"
]
