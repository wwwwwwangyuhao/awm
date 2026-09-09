"""RCWA-RL v3: lower-CVaR primal-dual PPO with frozen one-step tail TD credit.

Only the *temporal* credit assigned to the actor changes relative to RCWA v2.
Every frozen scientific invariant is inherited unchanged: retention, alpha,
empirical LCVaR semantics, episode tail cost, dual update, water objective,
actor architecture, budgets, interaction budget, seeds and cadence.

What v3 changes, and only this:

1. The risk critic is first **prefit** to the frozen Monte-Carlo tail-risk
   return ``h_i`` on the same on-policy batch (no extra environment
   interaction, only risk-critic gradients, same optimizer and learning rate).
2. The **post-prefit risk critic is frozen** and used to compute a one-step
   tail TD residual

       delta_t^risk = c_t + gamma_risk * V_r(s_{t+1}) * (1 - d_t) - V_r(s_t)

   under ``no_grad``.  This fixed tensor, not the MC return, becomes the
   actor's tail-risk credit for the whole PPO update.  It is standardized
   within each eta group and combined with the reward advantage exactly as in
   v2: ``A_actor = A_reward_z - lambda_eta * z_eta(delta_risk)``.
3. The TD credit telescopes to ``h_i - V_r(s_0)`` for ``gamma_risk=1``, so the
   actor still optimizes the same terminal tail information, merely
   redistributed over time by a learned Markov risk value.  This is verified
   at runtime on every update.

Critic vs actor targets (must not be confused):
* risk critic  -> Monte-Carlo ``risk_returns`` (unchanged from v2), also
  during the main PPO optimization.
* actor        -> frozen one-step TD residual of the prefit risk critic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Mapping

import torch
import torch.nn.functional as F

from awm.risk import REGISTERED_ETA_LEVELS

from .agent import RCWAAgent, RCWAHyperparameters, RCWAUpdateStats
from .buffer import RCWARolloutBatch
from .tail_credit import (
    episode_step_indices,
    next_risk_values,
    tail_td_credit,
    verify_tail_credit_telescoping,
)


V3_PROTOCOL_ID = "awm-rcwa-rl-v3"

# Diagnostics-only constants.  ``EARLY_WINDOW_STEPS`` is the DAP 0-30 window
# inferred from the fixed 125-step episode indexing; ``POLICY_WATER_BUDGET_MM``
# converts the frozen step reward (-applied_mm/495) back to applied millimetres.
# Neither value is used as a training input.
EARLY_WINDOW_STEPS = 30
POLICY_WATER_BUDGET_MM = 495.0


# ``slots=True`` is intentionally not used here: dataclass recreates slotted
# classes, which would break the zero-argument ``super()`` call in
# ``__post_init__`` that chains the inherited v2 validation.
@dataclass(frozen=True)
class RCWAV3Hyperparameters(RCWAHyperparameters):
    """V2 hyperparameters plus the risk-critic prefit budget.

    ``risk_credit_prefit_epochs`` is deliberately matched to
    ``update_epochs`` (10) so it is not an independently tuned scientific
    hyperparameter.
    """

    risk_credit_prefit_epochs: int = 10

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.risk_credit_prefit_epochs <= 0:
            raise ValueError("risk_credit_prefit_epochs must be positive")


@dataclass(frozen=True, slots=True)
class RCWAV3UpdateStats(RCWAUpdateStats):
    """V2 update statistics plus the v3 tail-credit diagnostics."""

    risk_credit_prefit_loss_before: float = 0.0
    risk_credit_prefit_loss_after: float = 0.0
    risk_credit_prefit_grad_norm_mean: float = 0.0
    risk_credit_prefit_grad_norm_max: float = 0.0
    risk_credit_prefit_steps: int = 0
    risk_td_credit_mean_by_eta: dict[str, float] = field(default_factory=dict)
    risk_td_credit_std_by_eta: dict[str, float] = field(default_factory=dict)
    risk_td_credit_positive_fraction_by_eta: dict[str, float] = field(default_factory=dict)
    risk_td_telescoping_max_abs_error: float = 0.0
    risk_td_credit_early_window_mean_irrigate_by_eta: dict[str, float] = field(
        default_factory=dict
    )
    risk_td_credit_early_window_mean_noop_by_eta: dict[str, float] = field(default_factory=dict)
    risk_td_credit_early_window_irrigate_count_by_eta: dict[str, int] = field(
        default_factory=dict
    )
    risk_td_credit_early_window_noop_count_by_eta: dict[str, int] = field(default_factory=dict)
    risk_td_credit_early_window_mean_applied_irrigation_mm_by_eta: dict[str, float] = field(
        default_factory=dict
    )


class RCWAV3Agent(RCWAAgent):
    """RCWA v3 agent: identical to v2 except for the actor's tail-risk credit."""

    PROTOCOL_ID = V3_PROTOCOL_ID

    def __init__(
        self,
        *,
        hyperparameters: RCWAV3Hyperparameters | None = None,
        seed: int = 21,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__(
            hyperparameters=hyperparameters or RCWAV3Hyperparameters(),
            seed=seed,
            device=device,
        )

    # ------------------------------------------------------------------
    # Phase B: risk-critic prefit on the frozen Monte-Carlo risk return.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _risk_value_mse(self, states: torch.Tensor, risk_returns: torch.Tensor) -> float:
        return float(F.mse_loss(self.risk_critic(states), risk_returns).item())

    def _prefit_risk_critic(
        self,
        *,
        states: torch.Tensor,
        risk_returns: torch.Tensor,
    ) -> dict[str, object]:
        """Fit only the risk critic to the frozen MC tail-risk return.

        Uses the existing shared Adam optimizer at the existing learning rate
        so no second optimizer semantics are introduced.  Only risk-critic
        parameters receive gradients; Adam skips parameters whose ``grad`` is
        ``None``, so the actor and the reward critic are left bit-identical.

        The prefit shuffles its own minibatches with ``self.generator``, which
        is also the actor/behavior and PPO-minibatch generator.  Its entry
        state is therefore restored here so the prefit cannot advance the
        actor/optimization RNG timeline: with v2-identical risk credit, a v3
        update stays bit-identical to v2 and no attribution confound is
        introduced by the prefit.  The prefit's own draws are unchanged; only
        the state observed after it returns is.
        """
        entry_generator_state = self.generator.get_state()
        risk_parameters = list(self.risk_critic.parameters())
        try:
            frozen_parameters = list(self.actor.named_parameters()) + list(
                self.reward_critic.named_parameters()
            )
            transition_count = int(states.shape[0])
            loss_before = self._risk_value_mse(states, risk_returns)
            grad_norms: list[float] = []
            for _epoch in range(int(self.hparams.risk_credit_prefit_epochs)):
                permutation = torch.randperm(
                    transition_count, generator=self.generator, device=self.device
                )
                for start in range(0, transition_count, self.hparams.minibatch_size):
                    idx = permutation[start : start + self.hparams.minibatch_size]
                    loss = F.mse_loss(self.risk_critic(states[idx]), risk_returns[idx])
                    if not torch.isfinite(loss):
                        raise FloatingPointError("RCWA v3 risk-critic prefit loss became NaN/Inf")
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    contaminated = [name for name, p in frozen_parameters if p.grad is not None]
                    if contaminated:
                        raise RuntimeError(
                            "RCWA v3 prefit must not touch actor/reward-critic parameters: "
                            + ", ".join(contaminated)
                        )
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        risk_parameters, self.hparams.max_grad_norm
                    )
                    grad_norms.append(float(torch.as_tensor(grad_norm).detach().item()))
                    self.optimizer.step()
            loss_after = self._risk_value_mse(states, risk_returns)
            return {
                "risk_credit_prefit_loss_before": loss_before,
                "risk_credit_prefit_loss_after": loss_after,
                "risk_credit_prefit_grad_norm_mean": mean(grad_norms),
                "risk_credit_prefit_grad_norm_max": max(grad_norms),
                "risk_credit_prefit_steps": len(grad_norms),
            }
        finally:
            # Restore the entry state even when the prefit raises so the
            # actor/behavior RNG timeline is never perturbed by the prefit.
            self.generator.set_state(entry_generator_state)

    # ------------------------------------------------------------------
    # Phase C: frozen one-step tail TD credit for the actor.
    # ------------------------------------------------------------------
    def _eta_masks(self, etas: torch.Tensor) -> dict[str, torch.Tensor]:
        masks: dict[str, torch.Tensor] = {}
        matched = torch.zeros_like(etas, dtype=torch.bool, device=etas.device)
        for eta in REGISTERED_ETA_LEVELS:
            key = f"{float(eta):.2f}"
            mask = torch.isclose(
                etas,
                torch.tensor(float(eta), dtype=etas.dtype, device=etas.device),
                atol=1e-6,
                rtol=0.0,
            )
            if not bool(mask.any().item()):
                raise RuntimeError(f"missing transitions for eta={key}")
            masks[key] = mask
            matched |= mask
        if not bool(matched.all().item()):
            raise ValueError("batch contains unregistered eta values")
        return masks

    def _frozen_tail_td_credit(
        self,
        *,
        states: torch.Tensor,
        risk_costs: torch.Tensor,
        dones: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Return ``(credit, values, telescoping_max_abs_error)`` after prefit.

        The credit is computed under ``no_grad`` and detached: it is a fixed
        tensor for the whole PPO update and is never recomputed when the risk
        critic keeps training on its Monte-Carlo value loss afterwards.

        The residual is accumulated in float64 so that the telescoping
        identity is not masked by float32 summation noise over a 125-step
        episode; the tensor handed to the actor is float32 again.
        """
        gamma = float(self.hparams.risk_gamma)
        with torch.no_grad():
            values = self.risk_critic(states).detach()
            double_values = values.to(torch.float64)
            double_costs = risk_costs.to(torch.float64)
            credit = tail_td_credit(
                costs=double_costs,
                values=double_values,
                next_values=next_risk_values(double_values, dones),
                dones=dones,
                gamma=gamma,
            )
            max_error = verify_tail_credit_telescoping(
                credits=credit,
                costs=double_costs,
                values=double_values,
                dones=dones,
                gamma=gamma,
            )
        return credit.to(values.dtype).detach(), values, max_error

    def _td_credit_diagnostics(
        self,
        *,
        credit: torch.Tensor,
        etas: torch.Tensor,
    ) -> dict[str, object]:
        mean_by_eta: dict[str, float] = {}
        std_by_eta: dict[str, float] = {}
        positive_by_eta: dict[str, float] = {}
        for key, mask in self._eta_masks(etas).items():
            group = credit[mask]
            mean_by_eta[key] = float(group.mean().item())
            std_by_eta[key] = float(group.std(unbiased=False).item())
            positive_by_eta[key] = float((group > 0.0).to(group.dtype).mean().item())
        return {
            "risk_td_credit_mean_by_eta": mean_by_eta,
            "risk_td_credit_std_by_eta": std_by_eta,
            "risk_td_credit_positive_fraction_by_eta": positive_by_eta,
        }

    def _early_window_diagnostics(
        self,
        *,
        batch: RCWARolloutBatch,
        credit: torch.Tensor,
        etas: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict[str, object]:
        """DAP 0-30 diagnostic split; diagnostics only, never a training input."""
        early = episode_step_indices(dones).to(credit.device) < EARLY_WINDOW_STEPS
        irrigate = batch.irrigate.to(credit.device)
        applied_mm = -batch.rewards.to(credit.device) * POLICY_WATER_BUDGET_MM
        mean_irrigate: dict[str, float] = {}
        mean_noop: dict[str, float] = {}
        count_irrigate: dict[str, int] = {}
        count_noop: dict[str, int] = {}
        mean_applied: dict[str, float] = {}
        for key, mask in self._eta_masks(etas).items():
            window = early & mask
            did_irrigate = window & irrigate
            noop = window & ~irrigate
            count_irrigate[key] = int(did_irrigate.sum().item())
            count_noop[key] = int(noop.sum().item())
            mean_irrigate[key] = (
                float(credit[did_irrigate].mean().item()) if count_irrigate[key] else 0.0
            )
            mean_noop[key] = float(credit[noop].mean().item()) if count_noop[key] else 0.0
            mean_applied[key] = (
                float(applied_mm[window].mean().item()) if int(window.sum().item()) else 0.0
            )
        return {
            "risk_td_credit_early_window_mean_irrigate_by_eta": mean_irrigate,
            "risk_td_credit_early_window_mean_noop_by_eta": mean_noop,
            "risk_td_credit_early_window_irrigate_count_by_eta": count_irrigate,
            "risk_td_credit_early_window_noop_count_by_eta": count_noop,
            "risk_td_credit_early_window_mean_applied_irrigation_mm_by_eta": mean_applied,
        }

    def _actor_risk_credit(
        self,
        *,
        batch: RCWARolloutBatch,
        states: torch.Tensor,
        etas: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        diagnostics = self._prefit_risk_critic(
            states=states,
            risk_returns=batch.risk_returns.to(self.device),
        )
        risk_costs = batch.risk_costs.to(self.device)
        dones = batch.dones.to(self.device)
        credit, _values, max_error = self._frozen_tail_td_credit(
            states=states,
            risk_costs=risk_costs,
            dones=dones,
        )
        diagnostics["risk_td_telescoping_max_abs_error"] = max_error
        diagnostics.update(self._td_credit_diagnostics(credit=credit, etas=etas))
        diagnostics.update(
            self._early_window_diagnostics(
                batch=batch,
                credit=credit,
                etas=etas,
                dones=dones,
            )
        )
        return credit, diagnostics

    def _build_update_stats(self, fields: Mapping[str, object]) -> RCWAUpdateStats:
        return RCWAV3UpdateStats(**fields)  # type: ignore[arg-type]


__all__ = [
    "EARLY_WINDOW_STEPS",
    "POLICY_WATER_BUDGET_MM",
    "RCWAV3Agent",
    "RCWAV3Hyperparameters",
    "RCWAV3UpdateStats",
    "V3_PROTOCOL_ID",
]
