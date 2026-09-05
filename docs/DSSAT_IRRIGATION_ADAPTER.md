# DSSAT Irrigation Adapter Contract

## Purpose

This layer connects the hard seasonal-water controller to the mutable DSSAT worker without introducing algorithm-specific reward or policy logic.

The canonical timing rule is:

- policy observation at decision day `d`;
- management action applied on biological day `d + 1`;
- exact zero irrigation does not rewrite management files and does not rerun DSSAT;
- a positive executed irrigation event is written to the worker-local irrigation management state and then triggers a full DSSAT rerun and output-cache refresh.

This preserves the DSSAT-RL semantics already used by the project while making the executed irrigation amount auditable.

## Action pipeline

```text
policy request
(event, amount_fraction)
        |
        v
WaterBudgetController
(seasonal budget, interval, min/max event)
        |
        v
execution-resolution canonicalization
        |
        v
applied_irrigation_mm
        |
        +-- 0 mm --> no management write, no DSSAT rerun
        |
        +-- >0 mm -> write irrigation at DAP d+1
                    -> rerun complete DSSAT season
                    -> refresh daily-output cache
```

The rollout must store both the policy request and the executed action. Agricultural metrics must be based on executed irrigation, never the raw actor output.

## Required rollout audit fields

Every step records:

- policy day;
- DSSAT action day/date/YYDDD;
- requested event;
- requested amount fraction;
- canonical amount fraction after execution-resolution mapping;
- applied irrigation depth (mm);
- remaining seasonal water before and after the action;
- whether the action was projected;
- projection reasons;
- whether the DSSAT management file was written;
- whether DSSAT was rerun.

## Execution resolution

The actor may output a continuous amount, but DSSAT and the real irrigation system can only execute finite-precision depths. Therefore the requested amount is canonicalized to an explicitly configured `execution_resolution_mm` before the amount is committed to the seasonal water account.

This prevents a mismatch in which the controller counts one amount while DSSAT executes a rounded amount.

The formal experiment must freeze `execution_resolution_mm` using the coarser of the real irrigation-system resolution and the DSSAT management-file representation used in the experiment.

## Failure semantics

A write or DSSAT-rerun failure faults the current adapter instance. The current trajectory must be discarded and the worker must undergo a full episode reset before any new rollout is accepted.

Continuing after a partial management write is prohibited because policy water accounting and mutable DSSAT state could diverge.

## Terminal seasonal-water reconciliation

At the terminal transition only, DSSAT `Summary.OUT::IRCM` is reconciled against:

```text
expected_IRCM = fixed_nonpolicy_irrigation + sum(executed_policy_irrigation)
```

The comparison must pass a pre-specified tolerance. A mismatch is a hard experiment error, not a warning.

`nonpolicy_irrigation_mm` is explicit because pre-plant or otherwise fixed irrigation may be present in an experimental protocol but is not controlled by the policy.

## Information boundary

The adapter does not expose `Summary.OUT`, final yield, future daily states, or any future-weather information during the season. Terminal `IRCM` reconciliation is strictly an accounting audit after episode termination.
