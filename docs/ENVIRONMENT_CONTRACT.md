# Water-budget environment contract v1

This document defines the integration boundary between an RL policy and DSSAT for the first AWM study.

## Scope

The environment layer is responsible only for operational irrigation feasibility and seasonal water accounting. It must not contain algorithm-specific losses, reward shaping, PPO/SAC logic, or DSSAT crop-model internals.

The first paper optimizes irrigation only. Nitrogen management remains fixed to the validated agronomic schedule defined by the formal experiment protocol.

## Required experiment parameters

The following values are deliberately required and have no code defaults:

- `seasonal_budget_mm` — seasonal irrigation allocation `B`;
- `min_event_mm` — minimum executable positive irrigation event `I_min`;
- `max_event_mm` — maximum irrigation depth allowed per decision `I_max`;
- `min_interval_days` — minimum interval between positive irrigation events `d_min`;
- `horizon_days` — number of management decision days.

Before formal experiments, each numerical value must be frozen in `EXPERIMENT_PROTOCOL.md` with a field-system, agronomic, or treatment-design justification.

## Policy action interface

The policy proposes a hierarchical action:

1. `irrigate` in `{false, true}`;
2. `amount_fraction` in `[0, 1]` when irrigation is requested.

When an irrigation event is feasible, the normalized amount is mapped to

`I_t = I_min + u_t * (I_max,t - I_min)`

where

`I_max,t = min(I_max, remaining seasonal budget)`.

The policy therefore has an exact no-irrigation action and cannot intentionally request arbitrarily small positive irrigation depths.

## Hard feasibility rules

Before DSSAT execution, the controller enforces:

1. cumulative irrigation never exceeds the seasonal allocation;
2. positive events never exceed event capacity;
3. positive events are never smaller than the minimum effective event;
4. the minimum event interval is respected;
5. if remaining seasonal water is below `I_min`, the only feasible irrigation action is exact zero;
6. the same management day cannot be committed twice.

These rules are hard constraints, not reward penalties.

## Auditing

Every decision returns an `IrrigationDecision` record containing:

- policy event request;
- policy normalized amount request;
- applied irrigation depth;
- water remaining before and after the action;
- current feasible maximum event depth;
- whether projection was necessary;
- machine-readable projection reasons.

Formal rollouts must persist these fields so that reported seasonal irrigation can be reconstructed independently of the RL buffer.

## Policy-facing water state

The controller exposes only information available at decision time:

- remaining seasonal water fraction;
- cumulative irrigation fraction;
- remaining management-horizon fraction;
- days since the last positive irrigation event;
- requested yield-protection fraction.

DSSAT crop, soil, and current-weather state is appended by the DSSAT environment adapter, not by this module.

Before the first positive irrigation event, `days_since_last_irrigation = day + 1` for zero-indexed decision days. This convention is fixed for reproducibility.

## DSSAT adapter contract

The future DSSAT adapter must obey this order on each decision day:

1. build the observation from current/past DSSAT information plus water-budget state;
2. obtain hierarchical policy action;
3. pass it through `WaterBudgetController.step`;
4. write only `decision.applied_mm` to the DSSAT management record;
5. execute/refresh DSSAT according to the existing simulator protocol;
6. log the full `IrrigationDecision` audit record;
7. advance to the next management day.

No future DSSAT daily output or seasonal summary variable may enter the policy observation.

## Values intentionally not frozen yet

The v1 code does **not** choose numerical values for `B`, `I_min`, `I_max`, or `d_min`. Those are agricultural experimental parameters and must be supported by the actual irrigation system, local management recommendation, or preregistered treatment design before formal training.
