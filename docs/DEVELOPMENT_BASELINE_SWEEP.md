# Development agricultural-baseline sweep

## Purpose

This stage evaluates the three frozen non-learning agricultural comparators on
historical ERA5 development weather before any PPO or RCWA-RL training.
It is an engineering/agronomic reference sweep, not a final-test evaluation.

The canonical matrix is

```text
23 ERA5 years (2000-2022)
× 3 agricultural baselines
× 3 water treatments (W100/W80/W60)
= 207 episodes
```

The weather split is immutable for this workflow:

- training/development weather: 2000-2017;
- validation/development weather: 2018-2022;
- final station test: 2023-2025 — **forbidden in this sweep**.

`awm.baselines.development_sweep` rejects 2023, 2024 and 2025 rather than
providing an override flag.

## One agricultural protocol, many calendar years

The only versioned formal template is

```text
run/formal/awm_protocol_v1_2000.COX.in
```

Its SHA-256 is locked by the sweep driver.  `awm.dssat.formal_year` creates
runtime-only year instances.  It preserves every management day-of-year while
changing the YY prefix of valid DSSAT `YYDDD` tokens.

The materializer also distinguishes identifiers that must and must not vary:

- `WSTA` follows the year-specific weather file (`XJHXYY01`);
- `@NOTE` and human-readable year labels follow the target year;
- `ID_SOIL` remains `XJHX0001`, because `SOIL.SOL` defines that fixed profile;
- the fixed field/profile identity is not globally renamed;
- `{{AWM_IRRIGATION_EVENTS}}` remains exactly once.

For example, year 2001 uses:

```text
weather: XJHX0101.WTH
planting: 01119
fixed preplant irrigation: 01116, 45 mm
WSTA: XJHX0101
ID_SOIL: XJHX0001
```

All fertilizer and other formal management DOYs are preserved with the same
calendar-year transformation.

## Three-layer irrigation audit

Each agricultural step now records three distinct levels:

1. `baseline_desired_mm` — raw water depth requested by the agricultural rule;
2. `baseline_constrained_request_mm` — request after common quota/event/interval
   feasibility has been encoded into the hierarchical action;
3. `applied_irrigation_mm` — final DSSAT-executed depth after adapter
   canonicalization and execution resolution.

If a W60 rule wants 45 mm but only 9 mm remains, the audit must therefore show
approximately:

```text
baseline_desired_mm = 45
baseline_constrained_request_mm = 9
baseline_constraint_adjusted = true
baseline_constraint_reasons = [remaining_seasonal_budget]
applied_irrigation_mm = 9
```

This is intentionally separate from adapter-level projection/quantization.
A baseline-level quota adjustment must not be misreported as an adapter
projection.

## Water accounting

Every method shares the same fixed establishment irrigation and treatment
budgets:

| Treatment | Total seasonal irrigation cap | Fixed preplant | Postplant policy quota |
| --- | ---: | ---: | ---: |
| W100 | 540 mm | 45 mm | 495 mm |
| W80 | 432 mm | 45 mm | 387 mm |
| W60 | 324 mm | 45 mm | 279 mm |

Every completed episode must pass

```text
IRCM_DSSAT = 45 mm + executed postplant irrigation
```

within the frozen IRCM tolerance.  Any mismatch aborts the episode and must not
enter downstream statistics.

## Preparing inputs without DSSAT execution

From the AWM repository root:

```bash
PYTHONPATH="$PWD/src" \
python -m awm.baselines.development_sweep \
  --project-root . \
  --prepare-only
```

Generated files are written under the ignored runtime tree:

```text
runtime/development_baseline_sweep_v1/
├── manifest.json
├── inputs/
│   ├── 2000/
│   ├── 2001/
│   └── ...
└── audits/
```

No generated year-specific COX is versioned in Git.

## Running the canonical 207-episode sweep

```bash
PYTHONPATH="$PWD/src" \
python -m awm.baselines.development_sweep \
  --project-root . \
  --years 2000-2022 \
  --methods conventional,et,rew \
  --treatments W100,W80,W60
```

The run is deliberately sequential because all episodes use the same canonical
worker namespace and mutable DSSAT workspace.  Parallel execution must use
separate worker identities and is not part of this v1 protocol.

Outputs include:

```text
results.jsonl   # checkpointed after every successful episode
results.csv     # compact analysis table
manifest.json   # selected years/methods/treatments and locked final years
audits/*.json   # full per-step traces
```

Use `--resume` only to continue the same frozen-code/protocol run after an
interruption.  If code or protocol inputs change, start a fresh work directory
instead of mixing rows from different implementations.

## Interpretation boundary

The sweep is intended to establish agricultural baseline distributions and
weather sensitivity.  It does **not** by itself define the paper's formal
`Y_ref`, `eta`, or `alpha`.  Those quantities remain separate protocol choices
and must not be selected by looking at the 2023-2025 station final-test set.
