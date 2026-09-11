"""RCWA-v9: v6 Tail-Q objective with a single KL-constrained natural policy update.

V9 removes PPO clipping from the actor instead of stacking another reliability
heuristic on top of v6.  The actor solves the local primal-dual trust-region
problem

    max_theta E_{pi_old}[ (pi_theta/pi_old) A_combined ]
    s.t. E_s[ KL(pi_old(.|s) || pi_theta(.|s)) ] <= delta.

For the hierarchical irrigation policy, the exact per-state KL is

    KL(Bern_old || Bern_new) + p_old * KL(N_old || N_new),

because the active tanh/affine transform is bijective.  The trust-region radius
is not tuned: delta=-0.5*log(1-epsilon^2), the worst old||new KL compatible
with the original symmetric PPO likelihood-ratio envelope [1-epsilon,1+epsilon]
and probability normalization.  With epsilon=0.2, delta=0.020410997....

The natural-gradient direction uses the exact KL Hessian.  A deterministic
one-dimensional numerical solve selects the best empirical surrogate point
along that direction inside the KL-feasible interval.  Tail-Q, the
policy-consistent Q baseline, lower-CVaR duals, rollout, critics and DSSAT
interaction are inherited unchanged from v6.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from statistics import mean
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

from .agent_v6 import RCWAV6Agent, RCWAV6Hyperparameters, RCWAV6UpdateStats
from .buffer import RCWARolloutBatch

V9_PROTOCOL_ID = "awm-rcwa-rl-v9"
# Purely numerical solver controls; they are not scientific/tuned parameters.
_CG_MAX_ITERS = 50
_CG_REL_TOL = 1e-6
_KL_BISECTION_ITERS = 40
_GOLDEN_SECTION_ITERS = 32


def clip_envelope_kl_radius(epsilon: float) -> float:
    """Map a symmetric ratio envelope to its worst normalized old||new KL."""
    eps = float(epsilon)
    if not 0.0 < eps < 1.0:
        raise ValueError("epsilon must lie in (0,1)")
    return -0.5 * math.log1p(-(eps * eps))


@dataclass(frozen=True)
class RCWAV9Hyperparameters(RCWAV6Hyperparameters):
    """No new tuned learning hyperparameters relative to RCWA-v6."""


@dataclass(frozen=True, slots=True)
class RCWAV9UpdateStats(RCWAV6UpdateStats):
    trust_region_radius: float = 0.0
    trust_region_exact_kl: float = 0.0
    trust_region_predicted_kl: float = 0.0
    trust_region_surrogate_before: float = 0.0
    trust_region_surrogate_after: float = 0.0
    trust_region_surrogate_improvement: float = 0.0
    trust_region_step_alpha: float = 0.0
    trust_region_quadratic_alpha: float = 0.0
    trust_region_natural_quadratic: float = 0.0
    trust_region_policy_gradient_norm: float = 0.0
    trust_region_cg_iterations: int = 0
    trust_region_cg_relative_residual: float = 0.0
    trust_region_actor_step_accepted: bool = False
    trust_region_old_logprob_max_abs_error: float = 0.0
    reference_clip_fraction: float = 0.0


class RCWAV9Agent(RCWAV6Agent):
    PROTOCOL_ID = V9_PROTOCOL_ID

    @staticmethod
    def _flat_parameters(parameters: Sequence[torch.nn.Parameter]) -> torch.Tensor:
        return torch.cat([p.detach().reshape(-1) for p in parameters])

    @staticmethod
    def _flatten_tensors(
        tensors: Sequence[torch.Tensor | None], parameters: Sequence[torch.nn.Parameter]
    ) -> torch.Tensor:
        pieces: list[torch.Tensor] = []
        for tensor, parameter in zip(tensors, parameters, strict=True):
            pieces.append(torch.zeros_like(parameter).reshape(-1) if tensor is None else tensor.reshape(-1))
        return torch.cat(pieces)

    @staticmethod
    @torch.no_grad()
    def _set_flat_parameters(parameters: Sequence[torch.nn.Parameter], flat: torch.Tensor) -> None:
        offset = 0
        for parameter in parameters:
            count = parameter.numel()
            parameter.copy_(flat[offset : offset + count].view_as(parameter))
            offset += count
        if offset != flat.numel():
            raise ValueError("flat parameter vector has incorrect length")

    @staticmethod
    def _hierarchical_kl_from_old(
        *,
        old_gate_logits: torch.Tensor,
        old_mean: torch.Tensor,
        old_scale: torch.Tensor,
        new_gate_logits: torch.Tensor,
        new_mean: torch.Tensor,
        new_scale: torch.Tensor,
    ) -> torch.Tensor:
        p_old = torch.sigmoid(old_gate_logits)
        log_p_old = F.logsigmoid(old_gate_logits)
        log_q_old = F.logsigmoid(-old_gate_logits)
        log_p_new = F.logsigmoid(new_gate_logits)
        log_q_new = F.logsigmoid(-new_gate_logits)
        gate_kl = p_old * (log_p_old - log_p_new) + (1.0 - p_old) * (log_q_old - log_q_new)
        normal_kl = (
            torch.log(new_scale / old_scale)
            + (old_scale.square() + (old_mean - new_mean).square()) / (2.0 * new_scale.square())
            - 0.5
        )
        return gate_kl + p_old * normal_kl

    def _mean_exact_kl(
        self,
        states: torch.Tensor,
        *,
        old_gate_logits: torch.Tensor,
        old_mean: torch.Tensor,
        old_scale: torch.Tensor,
    ) -> torch.Tensor:
        new_dist, new_gate_logits = self.actor.components(states)
        return self._hierarchical_kl_from_old(
            old_gate_logits=old_gate_logits,
            old_mean=old_mean,
            old_scale=old_scale,
            new_gate_logits=new_gate_logits,
            new_mean=new_dist.loc,
            new_scale=new_dist.scale,
        ).mean()

    def _conjugate_gradient(self, hvp, b: torch.Tensor) -> tuple[torch.Tensor, int, float]:
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        rr = torch.dot(r, r)
        rr0 = float(rr.detach().item())
        if rr0 <= 0.0:
            return x, 0, 0.0
        iterations = 0
        for iterations in range(1, _CG_MAX_ITERS + 1):
            hp = hvp(p)
            curvature = torch.dot(p, hp)
            if not torch.isfinite(curvature) or float(curvature.detach().item()) <= 0.0:
                break
            alpha = rr / curvature
            x = x + alpha * p
            r = r - alpha * hp
            rr_new = torch.dot(r, r)
            if not torch.isfinite(rr_new):
                raise FloatingPointError("v9 conjugate-gradient residual became NaN/Inf")
            if float(rr_new.detach().item()) <= rr0 * (_CG_REL_TOL ** 2):
                rr = rr_new
                break
            beta = rr_new / rr
            p = r + beta * p
            rr = rr_new
        relative = math.sqrt(max(0.0, float(rr.detach().item())) / rr0)
        return x.detach(), iterations, relative

    def _trust_region_actor_step(
        self,
        *,
        states: torch.Tensor,
        irrigate: torch.Tensor,
        raw_amount: torch.Tensor,
        old_log_probs: torch.Tensor,
        combined_adv: torch.Tensor,
    ) -> dict[str, object]:
        parameters = list(self.actor.parameters())
        old_flat = self._flat_parameters(parameters).to(self.device)
        with torch.no_grad():
            old_dist, old_gate_logits = self.actor.components(states)
            old_mean = old_dist.loc.detach().clone()
            old_scale = old_dist.scale.detach().clone()
            old_gate_logits = old_gate_logits.detach().clone()
            recomputed_old, _ = self.actor.evaluate_behavior(states, irrigate, raw_amount)
            old_lp_error = float((recomputed_old - old_log_probs).abs().max().item())
        if old_lp_error > 2e-5:
            raise RuntimeError(f"v9 rollout logprob mismatch before trust-region update: {old_lp_error}")

        def surrogate_tensor() -> torch.Tensor:
            new_log_prob, _ = self.actor.evaluate_behavior(states, irrigate, raw_amount)
            ratio = torch.exp(new_log_prob - old_log_probs)
            return (ratio * combined_adv).mean()

        surrogate_before_t = surrogate_tensor()
        grads = torch.autograd.grad(surrogate_before_t, parameters, allow_unused=True)
        g = self._flatten_tensors(grads, parameters).detach()
        g_norm = float(torch.linalg.vector_norm(g).item())
        radius = clip_envelope_kl_radius(self.hparams.clip_epsilon)

        if not math.isfinite(g_norm) or g_norm <= 1e-12:
            return {
                "actor_loss": float(-surrogate_before_t.detach().item()),
                "entropy": 0.0,
                "exact_kl": 0.0,
                "reference_clip_fraction": 0.0,
                "radius": radius,
                "surrogate_before": float(surrogate_before_t.detach().item()),
                "surrogate_after": float(surrogate_before_t.detach().item()),
                "step_alpha": 0.0,
                "quadratic_alpha": 0.0,
                "natural_quadratic": 0.0,
                "policy_gradient_norm": g_norm,
                "cg_iterations": 0,
                "cg_relative_residual": 0.0,
                "accepted": False,
                "old_logprob_max_abs_error": old_lp_error,
                "predicted_kl": 0.0,
            }

        def hvp(vector: torch.Tensor) -> torch.Tensor:
            kl = self._mean_exact_kl(
                states,
                old_gate_logits=old_gate_logits,
                old_mean=old_mean,
                old_scale=old_scale,
            )
            first = torch.autograd.grad(kl, parameters, create_graph=True, allow_unused=True)
            flat_first = self._flatten_tensors(first, parameters)
            directional = torch.dot(flat_first, vector)
            second = torch.autograd.grad(directional, parameters, allow_unused=True)
            out = self._flatten_tensors(second, parameters).detach()
            if not torch.isfinite(out).all():
                raise FloatingPointError("v9 KL Hessian-vector product became NaN/Inf")
            return out

        direction, cg_iterations, cg_relative = self._conjugate_gradient(hvp, g)
        natural_quadratic = float(torch.dot(g, direction).item())
        if not math.isfinite(natural_quadratic) or natural_quadratic <= 0.0:
            self._set_flat_parameters(parameters, old_flat)
            raise RuntimeError("v9 natural-gradient curvature is non-positive")
        quadratic_alpha = math.sqrt(2.0 * radius / natural_quadratic)

        @torch.no_grad()
        def evaluate_alpha(alpha: float) -> tuple[float, float, float, float]:
            self._set_flat_parameters(parameters, old_flat + float(alpha) * direction)
            kl = float(
                self._mean_exact_kl(
                    states,
                    old_gate_logits=old_gate_logits,
                    old_mean=old_mean,
                    old_scale=old_scale,
                ).item()
            )
            new_log_prob, entropy = self.actor.evaluate_behavior(states, irrigate, raw_amount)
            ratio = torch.exp(new_log_prob - old_log_probs)
            surrogate = float((ratio * combined_adv).mean().item())
            reference_clip = float(
                (torch.abs(ratio - 1.0) > self.hparams.clip_epsilon).to(ratio.dtype).mean().item()
            )
            return surrogate, kl, float(entropy.mean().item()), reference_clip

        s0 = float(surrogate_before_t.detach().item())
        s_cap, kl_cap, ent_cap, clip_cap = evaluate_alpha(quadratic_alpha)
        if not all(math.isfinite(x) for x in (s_cap, kl_cap, ent_cap, clip_cap)):
            self._set_flat_parameters(parameters, old_flat)
            raise FloatingPointError("v9 trust-region candidate became NaN/Inf")
        alpha_cap = quadratic_alpha
        if kl_cap > radius:
            lo, hi = 0.0, quadratic_alpha
            for _ in range(_KL_BISECTION_ITERS):
                mid = 0.5 * (lo + hi)
                _, kl_mid, _, _ = evaluate_alpha(mid)
                if kl_mid <= radius:
                    lo = mid
                else:
                    hi = mid
            alpha_cap = lo
            s_cap, kl_cap, ent_cap, clip_cap = evaluate_alpha(alpha_cap)

        # Maximize the *actual empirical surrogate* along the natural direction
        # over the KL-feasible interval. Golden-section iterations are only a
        # deterministic numerical solve, not a tuned learning mechanism.
        candidates: list[tuple[float, float, float, float, float]] = [
            (0.0, s0, 0.0, 0.0, 0.0),
            (alpha_cap, s_cap, kl_cap, ent_cap, clip_cap),
        ]
        if alpha_cap > 0.0:
            phi = (math.sqrt(5.0) - 1.0) / 2.0
            a, b = 0.0, alpha_cap
            x1 = b - phi * (b - a)
            x2 = a + phi * (b - a)
            e1 = evaluate_alpha(x1); e2 = evaluate_alpha(x2)
            candidates.extend([(x1, *e1), (x2, *e2)])
            for _ in range(_GOLDEN_SECTION_ITERS):
                if e1[0] < e2[0]:
                    a = x1
                    x1, e1 = x2, e2
                    x2 = a + phi * (b - a)
                    e2 = evaluate_alpha(x2)
                    candidates.append((x2, *e2))
                else:
                    b = x2
                    x2, e2 = x1, e1
                    x1 = b - phi * (b - a)
                    e1 = evaluate_alpha(x1)
                    candidates.append((x1, *e1))

        feasible = [item for item in candidates if item[2] <= radius + 1e-7 and math.isfinite(item[1])]
        best = max(feasible, key=lambda item: item[1]) if feasible else candidates[0]
        if best[1] <= s0:
            best = candidates[0]
        best_alpha, best_surrogate, best_kl, best_entropy, best_clip = best
        self._set_flat_parameters(parameters, old_flat + best_alpha * direction)
        predicted_kl = 0.5 * best_alpha * best_alpha * natural_quadratic
        if best_kl > radius + 1e-6:
            self._set_flat_parameters(parameters, old_flat)
            raise RuntimeError("v9 exact KL exceeded trust-region radius")
        return {
            "actor_loss": -best_surrogate,
            "entropy": best_entropy,
            "exact_kl": best_kl,
            "reference_clip_fraction": best_clip,
            "radius": radius,
            "surrogate_before": s0,
            "surrogate_after": best_surrogate,
            "step_alpha": best_alpha,
            "quadratic_alpha": quadratic_alpha,
            "natural_quadratic": natural_quadratic,
            "policy_gradient_norm": g_norm,
            "cg_iterations": cg_iterations,
            "cg_relative_residual": cg_relative,
            "accepted": bool(best_alpha > 0.0),
            "old_logprob_max_abs_error": old_lp_error,
            "predicted_kl": predicted_kl,
        }

    def update(self, batch: RCWARolloutBatch) -> RCWAV9UpdateStats:
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
            batch=batch, states=states, etas=etas
        )
        combined_adv, advantage_diagnostics = self._condition_actor_advantages(
            reward_adv=reward_adv, risk_adv=actor_risk_credit, etas=etas
        )
        adv_mean = float(combined_adv.mean().item())
        adv_std = float(combined_adv.std(unbiased=False).item())

        trust = self._trust_region_actor_step(
            states=states,
            irrigate=irrigate,
            raw_amount=raw_amount,
            old_log_probs=old_log_probs,
            combined_adv=combined_adv.detach(),
        )

        # Preserve v6 critic/Tail-Q optimizer semantics exactly.  The main loop
        # now updates only reward/risk state-value critics; actor movement has
        # already been solved by the KL-constrained policy step above.
        reward_value_losses: list[float] = []
        risk_value_losses: list[float] = []
        critic_total_losses: list[float] = []
        reward_grad_norms: list[float] = []
        risk_grad_norms: list[float] = []
        actor_parameters = list(self.actor.parameters())
        reward_parameters = list(self.reward_critic.parameters())
        risk_parameters = list(self.risk_critic.parameters())
        q_parameters = list(self.tail_q_critic.parameters())
        for _epoch in range(self.hparams.update_epochs):
            permutation = torch.randperm(n, generator=self.generator, device=self.device)
            for start in range(0, n, self.hparams.minibatch_size):
                idx = permutation[start : start + self.hparams.minibatch_size]
                reward_value_loss = F.mse_loss(self.reward_critic(states[idx]), reward_returns[idx])
                risk_value_loss = F.mse_loss(self.risk_critic(states[idx]), risk_returns[idx])
                critic_loss = (
                    self.hparams.reward_value_loss_coefficient * reward_value_loss
                    + self.hparams.risk_value_loss_coefficient * risk_value_loss
                )
                if not torch.isfinite(critic_loss):
                    raise FloatingPointError("RCWA v9 critic loss became NaN/Inf")
                self.optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                if any(p.grad is not None for p in actor_parameters):
                    raise RuntimeError("v9 critic update touched actor parameters")
                if any(p.grad is not None for p in q_parameters):
                    raise RuntimeError("v9 main critic update touched Tail-Q parameters")
                reward_norm = torch.nn.utils.clip_grad_norm_(reward_parameters, self.hparams.max_grad_norm)
                risk_norm = torch.nn.utils.clip_grad_norm_(risk_parameters, self.hparams.max_grad_norm)
                self.optimizer.step()
                reward_value_losses.append(float(reward_value_loss.detach().item()))
                risk_value_losses.append(float(risk_value_loss.detach().item()))
                critic_total_losses.append(float(critic_loss.detach().item()))
                reward_grad_norms.append(float(torch.as_tensor(reward_norm).detach().item()))
                risk_grad_norms.append(float(torch.as_tensor(risk_norm).detach().item()))

        dual_before, dual_after, tau, lcvar, violation = self._dual_update(batch)
        old_version = self.policy_version
        self.policy_version += 1
        self.update_index += 1
        actor_loss = float(trust["actor_loss"])
        fields: dict[str, object] = {
            "update_index": self.update_index,
            "rollout_policy_version": old_version,
            "new_policy_version": self.policy_version,
            "sample_count": n,
            "actor_loss": actor_loss,
            "reward_value_loss": mean(reward_value_losses),
            "risk_value_loss": mean(risk_value_losses),
            "total_loss": actor_loss + mean(critic_total_losses),
            "entropy": float(trust["entropy"]),
            "approx_kl": float(trust["exact_kl"]),
            "clip_fraction": float(trust["reference_clip_fraction"]),
            "grad_norm": float(trust["policy_gradient_norm"]),
            "actor_grad_norm": float(trust["policy_gradient_norm"]),
            "reward_critic_grad_norm": mean(reward_grad_norms),
            "risk_critic_grad_norm": mean(risk_grad_norms),
            "combined_advantage_mean_after_conditioning": adv_mean,
            "combined_advantage_std_after_conditioning": adv_std,
            "reward_advantage_mean_before_conditioning": float(advantage_diagnostics["reward_advantage_mean_before_conditioning"]),
            "reward_advantage_std_before_conditioning": float(advantage_diagnostics["reward_advantage_std_before_conditioning"]),
            "risk_advantage_mean_before_conditioning_by_eta": dict(advantage_diagnostics["risk_advantage_mean_before_conditioning_by_eta"]),
            "risk_advantage_std_before_conditioning_by_eta": dict(advantage_diagnostics["risk_advantage_std_before_conditioning_by_eta"]),
            "dual_before": dual_before,
            "dual_after": dual_after,
            "tau_by_eta": tau,
            "lcvar_by_eta": lcvar,
            "violation_by_eta": violation,
            "trust_region_radius": float(trust["radius"]),
            "trust_region_exact_kl": float(trust["exact_kl"]),
            "trust_region_predicted_kl": float(trust["predicted_kl"]),
            "trust_region_surrogate_before": float(trust["surrogate_before"]),
            "trust_region_surrogate_after": float(trust["surrogate_after"]),
            "trust_region_surrogate_improvement": float(trust["surrogate_after"])-float(trust["surrogate_before"]),
            "trust_region_step_alpha": float(trust["step_alpha"]),
            "trust_region_quadratic_alpha": float(trust["quadratic_alpha"]),
            "trust_region_natural_quadratic": float(trust["natural_quadratic"]),
            "trust_region_policy_gradient_norm": float(trust["policy_gradient_norm"]),
            "trust_region_cg_iterations": int(trust["cg_iterations"]),
            "trust_region_cg_relative_residual": float(trust["cg_relative_residual"]),
            "trust_region_actor_step_accepted": bool(trust["accepted"]),
            "trust_region_old_logprob_max_abs_error": float(trust["old_logprob_max_abs_error"]),
            "reference_clip_fraction": float(trust["reference_clip_fraction"]),
        }
        fields.update(credit_diagnostics)
        return self._build_update_stats(fields)

    def _build_update_stats(self, fields: Mapping[str, object]):
        return RCWAV9UpdateStats(**fields)


__all__ = [
    "RCWAV9Agent",
    "RCWAV9Hyperparameters",
    "RCWAV9UpdateStats",
    "V9_PROTOCOL_ID",
    "clip_envelope_kl_radius",
]
