"""RCWA-RL v2-v4 risk-aware irrigation agents."""

from .agent import RCWAAgent, RCWAHyperparameters, RCWAUpdateStats
from .agent_v3 import (
    RCWAV3Agent,
    RCWAV3Hyperparameters,
    RCWAV3UpdateStats,
    V3_PROTOCOL_ID,
)
from .agent_v4 import (
    RCWAV4Agent,
    RCWAV4Hyperparameters,
    RCWAV4UpdateStats,
    TailActionValueNetwork,
    V4_PROTOCOL_ID,
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
    "RCWAV4Agent",
    "RCWAV4Hyperparameters",
    "RCWAV4UpdateStats",
    "TailActionValueNetwork",
    "V4_PROTOCOL_ID",
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
