"""RCWA-v10: raw episodic-MC lower-CVaR primal-dual PPO.

V10 removes component standardization from RCWA-v2. The actor uses the
unmodified primal-dual advantage

    A_actor = A_water - lambda_eta * A_tail_MC,

where ``A_tail_MC`` is the complete-episode Monte-Carlo tail-risk advantage
already produced by the formal rollout buffer. No Tail-Q, reliability model,
Fisher solve, damping, or validation feedback is used.
"""
from __future__ import annotations

from typing import Mapping

import torch

from awm.risk import REGISTERED_ETA_LEVELS

from .agent import RCWAAgent, RCWAHyperparameters

V10_PROTOCOL_ID = "awm-rcwa-rl-v10"
_BASE_PROTOCOL_ID = "awm-rcwa-rl-v2"


class RCWAV10Agent(RCWAAgent):
    """RCWA-v2 architecture/optimizer with the raw primal-dual actor signal."""

    PROTOCOL_ID = V10_PROTOCOL_ID
    def _condition_actor_advantages(
        self,
        *,
        reward_adv: torch.Tensor,
        risk_adv: torch.Tensor,
        etas: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        """Return the unstandardized primal-dual actor advantage."""
        lambda_per_transition = self._lambda_tensor(etas)
        combined = reward_adv - lambda_per_transition * risk_adv

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
            group = risk_adv[mask]
            risk_mean_by_eta[key] = float(group.mean().item())
            risk_std_by_eta[key] = float(group.std(unbiased=False).item())
            matched |= mask
        if not bool(matched.all().item()):
            raise ValueError("batch contains unregistered eta values")
        diagnostics: dict[str, object] = {
            "reward_advantage_mean_before_conditioning": float(reward_adv.mean().item()),
            "reward_advantage_std_before_conditioning": float(
                reward_adv.std(unbiased=False).item()
            ),
            "risk_advantage_mean_before_conditioning_by_eta": risk_mean_by_eta,
            "risk_advantage_std_before_conditioning_by_eta": risk_std_by_eta,
            "combined_advantage_mean_after_conditioning": float(combined.mean().item()),
            "combined_advantage_std_after_conditioning": float(
                combined.std(unbiased=False).item()
            ),
        }
        return combined, diagnostics

    def checkpoint_payload(self) -> dict[str, object]:
        payload = super().checkpoint_payload()
        payload["protocol_id"] = V10_PROTOCOL_ID
        return payload

    def load_checkpoint_payload(self, payload: Mapping[str, object]) -> None:
        if payload.get("protocol_id") != V10_PROTOCOL_ID:
            raise ValueError("checkpoint protocol_id mismatch")
        shadow = dict(payload)
        shadow["protocol_id"] = _BASE_PROTOCOL_ID
        super().load_checkpoint_payload(shadow)


__all__ = [
    "RCWAV10Agent",
    "RCWAHyperparameters",
    "V10_PROTOCOL_ID",
]
