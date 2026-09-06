"""Exact empirical lower-CVaR batch semantics for RCWA-RL v1."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

from awm.risk import REGISTERED_ETA_LEVELS, empirical_lower_cvar


@dataclass(frozen=True, slots=True)
class EtaTailRiskMetrics:
    eta: float
    sample_count: int
    alpha: float
    tau: float
    empirical_lcvar: float
    ru_lcvar: float
    violation: float
    hinge_costs: tuple[float, ...]


def empirical_lower_quantile(values: Sequence[float], *, alpha: float) -> float:
    if not values:
        raise ValueError("at least one value is required")
    if not 0.0 < float(alpha) <= 1.0:
        raise ValueError("alpha must lie in (0,1]")
    ordered = sorted(float(v) for v in values)
    if not all(math.isfinite(v) for v in ordered):
        raise ValueError("values must be finite")
    index = max(0, min(len(ordered) - 1, math.ceil(alpha * len(ordered) - 1e-12) - 1))
    return ordered[index]


def evaluate_eta_tail(
    retentions: Sequence[float],
    *,
    eta: float,
    alpha: float = 0.20,
) -> EtaTailRiskMetrics:
    values = tuple(float(v) for v in retentions)
    if not values:
        raise ValueError("retentions must be non-empty")
    if not all(math.isfinite(v) and v >= 0.0 for v in values):
        raise ValueError("retentions must be finite and >= 0")
    tau = empirical_lower_quantile(values, alpha=alpha)
    hinge = tuple(max(0.0, tau - value) / float(alpha) for value in values)
    ru_lcvar = tau - sum(hinge) / len(values)
    exact = empirical_lower_cvar(values, alpha=alpha)
    if not math.isclose(ru_lcvar, exact, rel_tol=0.0, abs_tol=1e-10):
        raise RuntimeError(
            "Rockafellar-Uryasev batch value disagrees with frozen empirical LCVaR: "
            f"ru={ru_lcvar}, exact={exact}"
        )
    return EtaTailRiskMetrics(
        eta=float(eta),
        sample_count=len(values),
        alpha=float(alpha),
        tau=float(tau),
        empirical_lcvar=float(exact),
        ru_lcvar=float(ru_lcvar),
        violation=float(eta) - float(exact),
        hinge_costs=hinge,
    )


def evaluate_registered_eta_groups(
    *,
    episode_etas: Sequence[float],
    retentions: Sequence[float],
    alpha: float = 0.20,
    expected_per_eta: int = 18,
) -> tuple[tuple[float, ...], dict[str, EtaTailRiskMetrics]]:
    if len(episode_etas) != len(retentions):
        raise ValueError("episode_etas and retentions length mismatch")
    assigned = [0.0] * len(retentions)
    metrics: dict[str, EtaTailRiskMetrics] = {}
    matched = [False] * len(retentions)
    for eta in REGISTERED_ETA_LEVELS:
        eta_value = float(eta)
        indices = [
            i
            for i, raw in enumerate(episode_etas)
            if math.isclose(float(raw), eta_value, rel_tol=0.0, abs_tol=1e-9)
        ]
        if len(indices) != int(expected_per_eta):
            raise RuntimeError(
                f"formal RCWA rollout requires {expected_per_eta} episodes for eta={eta_value:.2f}, got {len(indices)}"
            )
        group = [float(retentions[i]) for i in indices]
        item = evaluate_eta_tail(group, eta=eta_value, alpha=alpha)
        key = f"{eta_value:.2f}"
        metrics[key] = item
        for local_index, episode_index in enumerate(indices):
            assigned[episode_index] = item.hinge_costs[local_index]
            matched[episode_index] = True
    if not all(matched):
        bad = [episode_etas[i] for i, ok in enumerate(matched) if not ok]
        raise ValueError(f"unregistered eta values present: {bad[:5]}")
    return tuple(assigned), metrics


__all__ = [
    "EtaTailRiskMetrics",
    "empirical_lower_quantile",
    "evaluate_eta_tail",
    "evaluate_registered_eta_groups",
]
