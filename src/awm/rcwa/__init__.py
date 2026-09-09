"""RCWA-RL v2 (stabilized conditional lower-CVaR primal-dual PPO) and v3 (frozen one-step tail TD credit)."""

from .agent import RCWAAgent, RCWAHyperparameters, RCWAUpdateStats
from .agent_v3 import (
    RCWAV3Agent,
    RCWAV3Hyperparameters,
    RCWAV3UpdateStats,
    V3_PROTOCOL_ID,
)
from .buffer import RCWARolloutBatch, RCWARolloutBuffer
from .risk_batch import (
    EtaTailRiskMetrics,
    empirical_lower_quantile,
    evaluate_eta_tail,
    evaluate_registered_eta_groups,
)
from .signals import RCWAEpisodeBreakdown, RCWAEpisodeSignals
from .tail_credit import (
    episode_boundaries,
    episode_step_indices,
    next_risk_values,
    tail_credit_telescoping_errors,
    tail_credit_telescoping_terms,
    tail_td_credit,
    verify_tail_credit_telescoping,
)

__all__ = [
    "EtaTailRiskMetrics",
    "RCWAAgent",
    "RCWAEpisodeBreakdown",
    "RCWAEpisodeSignals",
    "RCWAHyperparameters",
    "RCWARolloutBatch",
    "RCWARolloutBuffer",
    "RCWAUpdateStats",
    "RCWAV3Agent",
    "RCWAV3Hyperparameters",
    "RCWAV3UpdateStats",
    "V3_PROTOCOL_ID",
    "empirical_lower_quantile",
    "episode_boundaries",
    "episode_step_indices",
    "evaluate_eta_tail",
    "evaluate_registered_eta_groups",
    "next_risk_values",
    "tail_credit_telescoping_errors",
    "tail_credit_telescoping_terms",
    "tail_td_credit",
    "verify_tail_credit_telescoping",
]
