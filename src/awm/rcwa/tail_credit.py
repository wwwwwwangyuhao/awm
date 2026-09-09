"""Pure temporal-credit helpers for RCWA-RL v3 tail risk.

Two different tail quantities coexist in RCWA v3 and must never be confused:

* **Monte-Carlo risk return** (``risk_returns``): the complete-episode target
  used to *fit* the risk critic.  With ``gamma=1`` it is the episode tail cost
  ``h_i`` repeated on every transition of episode ``i``.  It carries no
  temporal information at all.

* **One-step tail TD credit** (``delta_t^risk``): the actor's local tail-risk
  credit, computed from a *frozen* fitted risk value function

      delta_t^risk = c_t + gamma_risk * V_r(s_{t+1}) * (1 - d_t) - V_r(s_t)

  where ``c_t`` is the existing per-transition risk cost (zero except on the
  terminal transition, where it is ``h_i``).  It redistributes the same
  terminal tail information over time; it does not add a new risk objective,
  a new reward term, or any agronomic schedule prior.

This module is deliberately parameter-free, environment-free and side-effect
free so the telescoping identity can be unit-tested directly.
"""
from __future__ import annotations

import torch


# Numerical guard for the telescoping identity.  The identity is verified in
# float64 accumulation, so these tolerances only separate floating-point noise
# (about 1e-14) from a structural violation of the credit definition.  They are
# not scientific hyperparameters and are not exposed through the protocol.
TELESCOPING_ABS_TOL = 1e-6
TELESCOPING_REL_TOL = 1e-6


def _flat(tensor: torch.Tensor, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    result = torch.as_tensor(tensor)
    if dtype is not None:
        result = result.to(dtype)
    return result.reshape(-1)


def next_risk_values(values: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
    """Return ``V_r(s_{t+1})`` with the terminal next value defined as zero.

    Transitions are stored as contiguous complete episodes, so ``s_{t+1}`` is
    simply the next stored state; at a done boundary the next value is forced
    to zero instead of leaking into the following episode.
    """
    flat_values = _flat(values)
    flat_dones = _flat(dones)
    if flat_dones.shape[0] != flat_values.shape[0]:
        raise ValueError("values and dones must contain the same number of transitions")
    next_values = torch.zeros_like(flat_values)
    if next_values.numel() > 1:
        next_values[:-1] = flat_values[1:]
    return next_values * (1.0 - flat_dones.to(flat_values.dtype))


def tail_td_credit(
    *,
    costs: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 1.0,
) -> torch.Tensor:
    """One-step tail TD residual ``c_t + gamma * V_r(s_{t+1}) * (1-d_t) - V_r(s_t)``."""
    flat_costs = _flat(costs)
    flat_values = _flat(values)
    flat_next = _flat(next_values)
    flat_dones = _flat(dones)
    if not (flat_costs.shape == flat_values.shape == flat_next.shape == flat_dones.shape):
        raise ValueError("costs, values, next_values and dones must share one shape")
    nonterminal = 1.0 - flat_dones.to(flat_values.dtype)
    return flat_costs + gamma * flat_next * nonterminal - flat_values


def episode_boundaries(dones: torch.Tensor) -> list[tuple[int, int]]:
    """Return ``(start, end)`` index pairs for contiguous complete episodes."""
    flags = [bool(x) for x in _flat(dones).tolist()]
    boundaries: list[tuple[int, int]] = []
    start = 0
    for index, done in enumerate(flags):
        if done:
            boundaries.append((start, index + 1))
            start = index + 1
    if start != len(flags):
        raise ValueError("rollout does not end on an episode boundary")
    return boundaries


def episode_step_indices(dones: torch.Tensor) -> torch.Tensor:
    """Return the within-episode step index of every transition."""
    flags = [bool(x) for x in _flat(dones).tolist()]
    indices = torch.zeros(len(flags), dtype=torch.long)
    step = 0
    for index, done in enumerate(flags):
        indices[index] = step
        step = 0 if done else step + 1
    return indices


def tail_credit_telescoping_terms(
    *,
    credits: torch.Tensor,
    costs: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-episode ``(achieved, expected)`` discounted credit sums.

    For a complete episode with terminal next value zero,

        sum_t gamma^t delta_t = sum_t gamma^t c_t - V_r(s_0)

    which for ``gamma=1`` and terminal-only cost reduces to the RCWA v3
    contract relation ``sum_t delta_t = h_i - V_r(s_0)``.
    """
    flat_credits = _flat(credits, dtype=torch.float64)
    flat_costs = _flat(costs, dtype=torch.float64)
    flat_values = _flat(values, dtype=torch.float64)
    boundaries = episode_boundaries(dones)
    device = flat_credits.device
    achieved = torch.zeros(len(boundaries), dtype=torch.float64, device=device)
    expected = torch.zeros(len(boundaries), dtype=torch.float64, device=device)
    for position, (start, end) in enumerate(boundaries):
        length = end - start
        weights = torch.pow(
            float(gamma),
            torch.arange(length, dtype=torch.float64, device=device),
        )
        achieved[position] = torch.dot(weights, flat_credits[start:end])
        expected[position] = torch.dot(weights, flat_costs[start:end]) - flat_values[start]
    return achieved, expected


def tail_credit_telescoping_errors(
    *,
    credits: torch.Tensor,
    costs: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 1.0,
) -> list[float]:
    """Return the absolute per-episode telescoping error of the TD credit."""
    achieved, expected = tail_credit_telescoping_terms(
        credits=credits,
        costs=costs,
        values=values,
        dones=dones,
        gamma=gamma,
    )
    return [float(value) for value in torch.abs(achieved - expected).tolist()]


def verify_tail_credit_telescoping(
    *,
    credits: torch.Tensor,
    costs: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 1.0,
    abs_tol: float = TELESCOPING_ABS_TOL,
    rel_tol: float = TELESCOPING_REL_TOL,
) -> float:
    """Check the telescoping identity and return the maximum absolute error.

    Raises ``RuntimeError`` when the frozen actor credit does not telescope.
    The check is never disabled at runtime: a violation means the actor would
    be optimized against a credit that no longer sums to the episode tail cost.
    """
    achieved, expected = tail_credit_telescoping_terms(
        credits=credits,
        costs=costs,
        values=values,
        dones=dones,
        gamma=gamma,
    )
    errors = torch.abs(achieved - expected)
    scale = torch.abs(expected).max() if expected.numel() else torch.zeros((), dtype=torch.float64)
    tolerance = float(abs_tol) + float(rel_tol) * float(scale.item())
    max_error = float(errors.max().item()) if errors.numel() else 0.0
    if not max_error <= tolerance:
        worst = int(torch.argmax(errors).item()) if errors.numel() else -1
        raise RuntimeError(
            "RCWA v3 tail TD credit violates the telescoping identity: "
            f"max_abs_error={max_error!r} tolerance={tolerance!r} episode_index={worst} "
            f"gamma={gamma!r}"
        )
    return max_error


__all__ = [
    "TELESCOPING_ABS_TOL",
    "TELESCOPING_REL_TOL",
    "episode_boundaries",
    "episode_step_indices",
    "next_risk_values",
    "tail_credit_telescoping_errors",
    "tail_credit_telescoping_terms",
    "tail_td_credit",
    "verify_tail_credit_telescoping",
]
