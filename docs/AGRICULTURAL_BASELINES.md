# Agricultural baselines and real-worker asset freeze

## Scope

Step 4 adds non-learning agricultural comparison policies before any RL
baseline or RCWA-RL implementation. Every baseline uses the same
`WaterBudgetController` and `DSSATIrrigationAdapter` as learned policies, so
seasonal quota, event-depth limits, irrigation interval, execution resolution
and terminal IRCM reconciliation are identical across methods.

## Baseline A — local conventional schedule

`ConventionalScheduleBaseline` accepts an explicit map from **biological action
DAP** (1..T) to event depth in mm.

No schedule is embedded in code. The formal schedule must come from the local
field protocol/recommendation and be frozen in `EXPERIMENT_PROTOCOL.md`.

When a water-scarcity treatment cannot execute the original event exactly, the
common water-budget controller performs the same projection used for every
method. Requested and executed depths are both logged.

## Baseline B — DSSAT potential-ET water balance

`PotentialETWaterBalanceBaseline` is a causal ET-based comparator using only
current/past information. It uses the DSSAT daily variables:

- `EOAA`: potential evapotranspiration, mm d-1;
- `PRED`: observed precipitation for the current simulated day.

The internal deficit ledger is

`D <- max(0, D + EOAA - f_eff * PRED)`.

An event is requested only after `D` reaches a pre-registered trigger. After
execution,

`D <- max(0, D - eta_I * I_executed)`.

`trigger_deficit_mm`, effective-rain fraction, irrigation efficiency and refill
fraction are protocol quantities with no software defaults.

This comparator is deliberately **not called FAO-56**. The canonical DSSAT
weather files currently contain SRAD, TMAX, TMIN, RAIN and WIND but not the
full meteorological input set needed to claim a strict FAO-56
Penman-Monteith ET0 calculation. If a later experiment adds validated humidity
or ET0 data, a separate FAO-56 comparator can be preregistered.

## Baseline C — root-zone REW threshold

`RootZoneREWThresholdBaseline` computes active-root-zone water status from the
same leakage-safe DSSAT state used by RL:

`REW_root = sum(REWi * RLiD) / sum(RLiD), i=1..10`.

Before root-length values are numerically available, the formal experiment must
supply an explicit fallback layer set; no fallback soil depth is hidden in
code.

When `REW_root <= trigger_rew`, the rule requests the pre-registered event
depth. Hard interval/quota/capacity constraints remain external and common.

## Episode outcomes

`run_baseline_episode()` returns:

- terminal HWAM;
- DSSAT IRCM;
- executed policy irrigation;
- irrigation event count;
- requested/projected event counts;
- complete per-step action audit;
- irrigation water productivity `HWAM / (10 * IRCM)`.

Agricultural water use is always based on executed management, never raw rule
or actor requests.

## Freezing a validated COX template

The old LRMB/SAPG workflow generated COX dynamically and therefore does not
contain a static validated COX file in Git. AWM now includes:

```bash
python -m awm.dssat.freeze_template \
  --source-cox /path/to/known_good_reset.COX \
  --output-template /path/to/awm_base.COX.in \
  --report /path/to/awm_base.template.json
```

The preferred source is a **successful no-policy/reset worker COX**. By default
the command refuses a source containing explicit irrigation rows. This forces
the researcher to classify any existing irrigation as policy or fixed
non-policy management before it can be stripped.

The report records SHA-256 for both source and frozen template. The formal
protocol must lock the template hash.

## Legacy asset provenance discovered in `lrmb`

The already-run DSSAT workflow at `lrmb` branch `exp/sapg`, commit
`b6257a29249969ea4b43849debe3e65657902e7d`, contains:

- `dssat_workspace_template/dscsm048`;
- `dssat_workspace_template/DSSATPRO.L48`;
- `dssat_workspace_template/Genotype/*`;
- `dssat_workspace_template/data/soil/SOIL.SOL`;
- ERA5 weather files for 2000–2025;
- station weather files for 2023–2025.

The old canonical environment used site code `XJHX`, planting DOY 119,
emergence DOY 133 and a 125-decision horizon.

These facts establish **candidate asset provenance only**. They do not prove
that the AWM paper should inherit every agronomic setting. Formal use requires
review and protocol locking of the actual COX template, soil/genotype files,
weather split and management constants.

## Real binary step still required

GitHub-side development cannot execute the user's Linux DSSAT binary in its
worker runtime. Before formal training, run on the DSSAT server:

1. freeze a known-good reset COX;
2. populate the real smoke JSON with worker-local paths;
3. run `python -m awm.dssat.smoke --config ...`;
4. run one complete 125-day agricultural-baseline episode;
5. require terminal IRCM reconciliation to pass.

Only after those checks should a learned RL baseline be implemented.
