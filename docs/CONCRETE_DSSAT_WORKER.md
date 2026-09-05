# Concrete DSSAT worker contract

## Purpose

Step 3 converts the water-budget and irrigation-adapter contracts into a real worker-local DSSAT execution path without importing the old study's hard-coded site, cultivar, soil, mulch or nitrogen settings.

## Experiment ownership

The DSSAT experiment file is split into two layers:

1. **Externally validated COX template** — owns cultivar, soil initial conditions, planting, fixed nitrogen management, residue/mulch, simulation controls and any non-policy management.
2. **AWM irrigation renderer** — replaces exactly one `{{AWM_IRRIGATION_EVENTS}}` marker with policy-executed irrigation rows.

The RL environment is not allowed to synthesize or silently modify the first layer.

A source template contains the normal DSSAT irrigation header followed by the marker, for example:

```text
@I IDATE  IROP IRVAL
{{AWM_IRRIGATION_EVENTS}}
```

The marker-bearing file is never passed to DSSAT. The rendered worker-local `.COX` contains only DSSAT rows.

## Worker execution order

Episode reset:

```text
clean declared episode outputs
-> clear policy irrigation ledger
-> render COX with no policy events
-> run complete DSSAT season
-> parse newest Summary.OUT and daily OUT blocks
-> expose planting-day daily state only
```

Positive irrigation step:

```text
policy request
-> hard water-budget projection
-> execution-resolution canonicalization
-> append one YYDDD irrigation row to rendered COX
-> complete DSSAT rerun
-> refresh output cache
-> expose only next-day daily state
```

Exact no-op step:

```text
policy request = no irrigation
-> no COX write
-> no DSSAT rerun
-> O(1) next-day lookup from existing complete-season cache
```

Terminal step:

```text
read Summary.OUT only after termination
-> require HWAM and IRCM
-> reconcile IRCM with fixed non-policy irrigation + executed policy irrigation
-> keep HWAM/IRCM out of policy observation
```

## Observation contract

The leakage-safe cotton state remains 74 dimensions. Step 3 appends five water-allocation features:

- remaining seasonal irrigation fraction;
- cumulative policy irrigation fraction;
- remaining decision-horizon fraction;
- days since last irrigation;
- requested yield-protection fraction.

The current structured observation therefore flattens to **79 numeric values**.

A discrete phenological-stage encoding is intentionally **not invented in Step 3**. Crop-development variables plus `dap_frac` already expose development continuously. If a separate stage variable is later exposed to the policy, its agronomic definition and encoding must first be frozen in `EXPERIMENT_PROTOCOL.md`. Stage labels may be used earlier for post-hoc AWM mechanism analysis.

## Fixed nitrogen

There is no fertilizer action and no fertilizer-write method in the Step-3 backend. Nitrogen must be fixed in the externally validated COX template. This preserves causal attribution of the first paper to irrigation management.

## Real-worker smoke prerequisite

Unit tests can verify the execution contract with fake runners/readers, but a real DSSAT smoke requires external assets deliberately absent from this repository:

- DSSAT executable;
- validated COX template;
- weather file;
- soil file;
- `DSSATPRO.L48` / runtime profile;
- required daily OUT inventory.

Before formal training, one real worker must complete a full episode and pass the terminal IRCM reconciliation audit.
