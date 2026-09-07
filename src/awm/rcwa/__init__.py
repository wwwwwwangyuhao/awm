"""RCWA-RL v1: conditional lower-CVaR primal-dual irrigation RL."""

from .agent import RCWAAgent, RCWAHyperparameters, RCWAUpdateStats
from .buffer import RCWARolloutBatch, RCWARolloutBuffer
from .risk_batch import (
    EtaTailRiskMetrics,
    empirical_lower_quantile,
    evaluate_eta_tail,
    evaluate_registered_eta_groups,
)
from .signals import RCWAEpisodeBreakdown, RCWAEpisodeSignals

__all__ = [
    "EtaTailRiskMetrics",
    "RCWAAgent",
    "RCWAEpisodeBreakdown",
    "RCWAEpisodeSignals",
    "RCWAHyperparameters",
    "RCWARolloutBatch",
    "RCWARolloutBuffer",
    "RCWAUpdateStats",
    "empirical_lower_quantile",
    "evaluate_eta_tail",
    "evaluate_registered_eta_groups",
]
