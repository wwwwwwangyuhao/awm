"""Canonical ss_refactor reward configuration.

Phase 2 keeps the proven season-level trade-offs from the earlier simple daily
reward, but removes cultivar-target centering and hidden-history settlement.
The old objective (up to the action-independent constant ``-0.12*7000``) was:

    0.12 * HWAM
    - 10 * irrigation / 500
    - 10 * nitrogen / 250
    - 0.1 * irrigation_events
    - 0.1 * fertilizer_events
    + 0.1 * joint_events

The three event terms are exactly ``-0.1`` per management day: irrigation only,
nitrogen only, or both on the same day all cost 0.1. Dividing the complete
objective by 120 is a positive linear rescaling, so it preserves the same
policy preference while giving a numerically convenient yield scale of
1000 kg/ha:

    R = HWAM / 1000
        - irrigation / 6000
        - nitrogen / 3000
        - management_days / 1200

With gamma=1, daily GWAD differences telescope to mature HWAM and all resource
and operation costs are paid immediately from the *applied* action. Therefore
no cumulative irrigation/nitrogen/event history is required in the policy
state.
"""
from __future__ import annotations

YIELD_SCALE_KG_HA = 1000.0
IRRIGATION_SCALE_MM = 6000.0
NITROGEN_SCALE_KG_HA = 3000.0
MANAGEMENT_EVENT_SCALE = 1200.0

DAILY_YIELD_FIELD = "GWAD"
FINAL_YIELD_FIELD = "HWAM"
MANAGEMENT_EVENT_MODE = "any_applied_action"
REQUIRED_GAMMA = 1.0

REWARD_CONFIG = {
    "yield-scale-kg-ha": YIELD_SCALE_KG_HA,
    "irrigation-scale-mm": IRRIGATION_SCALE_MM,
    "nitrogen-scale-kg-ha": NITROGEN_SCALE_KG_HA,
    "management-event-scale": MANAGEMENT_EVENT_SCALE,
    "daily-yield-field": DAILY_YIELD_FIELD,
    "final-yield-field": FINAL_YIELD_FIELD,
    "management-event-mode": MANAGEMENT_EVENT_MODE,
}


def validate_reward_config(config=None) -> dict:
    cfg = dict(REWARD_CONFIG if config is None else config)
    for key in (
        "yield-scale-kg-ha",
        "irrigation-scale-mm",
        "nitrogen-scale-kg-ha",
        "management-event-scale",
    ):
        value = float(cfg[key])
        if value <= 0.0:
            raise ValueError(f"{key} must be positive, got {value}.")
        cfg[key] = value

    if cfg.get("daily-yield-field") != "GWAD":
        raise ValueError("ss_refactor Phase 2 requires daily-yield-field='GWAD'.")
    if cfg.get("final-yield-field") != "HWAM":
        raise ValueError("ss_refactor Phase 2 requires final-yield-field='HWAM'.")
    if cfg.get("management-event-mode") != "any_applied_action":
        raise ValueError(
            "ss_refactor Phase 2 requires management-event-mode="
            "'any_applied_action'."
        )
    return cfg


validate_reward_config()

__all__ = [
    "YIELD_SCALE_KG_HA",
    "IRRIGATION_SCALE_MM",
    "NITROGEN_SCALE_KG_HA",
    "MANAGEMENT_EVENT_SCALE",
    "DAILY_YIELD_FIELD",
    "FINAL_YIELD_FIELD",
    "MANAGEMENT_EVENT_MODE",
    "REQUIRED_GAMMA",
    "REWARD_CONFIG",
    "validate_reward_config",
]
