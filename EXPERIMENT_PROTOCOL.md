# Experimental Protocol — Pre-registration Skeleton

This document records empirical quantities that must be justified and frozen before formal test-set evaluation.

## 1. Field, DSSAT experiment and irrigation-system constants

- Study region/site: **TBD**
- Irrigation system: **TBD**
- Validated DSSAT COX template path/version: **TBD**
- Validated DSSAT COX template SHA-256: **TBD**
- Frozen-template provenance report path/SHA-256: **TBD**
- Reference seasonal irrigation, `B_ref` (mm): **TBD**
- Minimum effective irrigation event, `I_min` (mm): **TBD**
- Maximum event/application capacity, `I_max` (mm day^-1 or event^-1): **TBD**
- Minimum interval between irrigation events, `d_min` (days): **TBD**
- Executable irrigation-depth resolution, `execution_resolution_mm`: **TBD**
- Fixed/non-policy irrigation included in DSSAT `IRCM`, `nonpolicy_irrigation_mm`: **TBD**
- Terminal DSSAT irrigation-accounting tolerance, `IRCM_tolerance_mm`: **TBD**
- Fixed nitrogen schedule: **TBD; owned by the validated COX template**

No value above should be chosen solely to make RL training easier. `execution_resolution_mm` must be justified from the effective execution precision of the real irrigation system and DSSAT management representation used in the experiment. Any fixed irrigation not chosen by the policy must be declared before training rather than absorbed into policy irrigation after the fact.

The first paper must not regenerate cultivar, initial soil conditions, mulch/residue settings, planting management or nitrogen management from hidden Python defaults. Those quantities belong to the externally validated DSSAT experiment template. The RL layer may inject only the policy irrigation rows at the explicit `{{AWM_IRRIGATION_EVENTS}}` marker.

A candidate template generated from an older dynamic DSSAT workflow is not automatically validated. The source rendered COX, source SHA-256, frozen-template SHA-256, and the decision to retain/remove any pre-existing irrigation rows must be archived. Prefer freezing a successful no-policy/reset worker COX. Existing explicit irrigation rows may not be stripped without first classifying them as policy or fixed non-policy management.

## 2. Yield target and risk settings

- Reference-yield definition, `Y_ref`: **TBD**
- Canonical yield-protection target, `eta`: **TBD**
- Lower-tail risk fraction, `alpha`: **TBD**

Candidate starting values such as `eta = 0.95` or `alpha = 0.10` are hypotheses only until justified and frozen.

## 3. Seasonal water-scarcity treatments

The intended main treatments are:

- `W100`: **TBD mm**
- `W80`: **TBD mm**
- `W60`: **TBD mm**

Each treatment must be defined relative to an agronomic/reference irrigation allocation, with exact depths frozen before formal evaluation.

## 4. Weather-data split

- Training weather years: **TBD**
- Validation/model-selection weather years: **TBD**
- Independent final-test weather years: **TBD**

Rules:

1. The three sets must be disjoint.
2. Final-test years cannot influence architecture, reward/constraint design, hyperparameters, checkpoint selection, or baseline tuning.
3. All competing methods are evaluated on exactly the same final-test weather realizations.
4. Weather-source identity and file hashes/version must be archived with the protocol lock.

## 5. Climate and crop-stage definitions

Predefine thresholds for:

- dry years;
- normal years;
- wet years;
- hot–dry compound years.

Classification must depend only on meteorological variables, not on algorithm outcomes.

Crop-stage boundaries used for post-hoc AWM mechanism analysis are also **TBD** and must be fixed from agronomic/DSSAT phenology definitions before formal analysis.

### Policy phenology encoding

The current Step-3 canonical numeric policy observation is **79D = 74D leakage-safe DSSAT state + 5D water-allocation state**. No extra discrete phenological-stage code is exposed to the actor at this stage. Crop development is already represented continuously by DSSAT crop variables plus `dap_frac`.

Accordingly, the earlier problem-formulation symbol `g_t` is interpreted in canonical v1 as developmental information contained within `x_t` and `dap_frac`, not as an additional arbitrarily coded scalar. Adding a separate stage input later is permitted only after its agronomic boundaries and numeric encoding are preregistered here.

## 6. Agricultural management baselines

Three non-learning agricultural baselines are implemented before any learned RL comparator. They all use the same `WaterBudgetController` and `DSSATIrrigationAdapter` as learned policies, so seasonal quota, minimum/maximum event depth, minimum event interval, execution resolution, and terminal IRCM reconciliation are identical across methods.

### 6.1 Local conventional irrigation

Implementation: `ConventionalScheduleBaseline`.

Formal quantities to freeze:

- source of local schedule (field protocol/recommendation): **TBD**
- source version/date/citation: **TBD**
- schedule expressed as biological action DAP -> desired irrigation depth (mm): **TBD**

The schedule uses the canonical temporal contract: policy day `d` selects irrigation applied on biological action DAP `d+1`. No conventional event dates or depths are embedded as software defaults.

Under W100/W80/W60, the conventional rule is not allowed to bypass the common water budget. If an original event cannot be executed because the treatment quota/capacity/interval is active, requested and executed amounts are both logged and the common controller determines the feasible action.

### 6.2 DSSAT potential-ET water-balance irrigation

Implementation: `PotentialETWaterBalanceBaseline`.

The comparator is causal and uses current/past DSSAT state only. The current implementation uses daily `EOAA` as DSSAT potential evapotranspiration and `PRED` as current-day precipitation. Its observed deficit ledger is:

`D <- max(0, D + EOAA - f_eff * PRED)`.

An irrigation request is generated only after the deficit reaches the pre-specified trigger. Executed irrigation is fed back into the ledger:

`D <- max(0, D - irrigation_efficiency * I_executed)`.

Formal quantities to freeze:

- `trigger_deficit_mm`: **TBD**
- `irrigation_efficiency`: **TBD**
- `effective_rain_fraction`: **TBD**
- `refill_fraction`: **TBD**
- agronomic/literature justification for all four quantities: **TBD**

This baseline is deliberately not labelled FAO-56 in canonical v1. The currently inherited weather inventory contains SRAD, TMAX, TMIN, RAIN, and WIND but does not provide the full validated meteorological input set needed to claim a strict FAO-56 Penman–Monteith ET0 implementation. A true FAO-56 baseline may be added only as a separately preregistered comparator when the required meteorological inputs/ET0 source are available and validated.

### 6.3 Root-zone REW threshold irrigation

Implementation: `RootZoneREWThresholdBaseline`.

The rule estimates active root-zone water status from the same decision-time variables exposed to the policy:

`REW_root = sum(REWi * RLiD) / sum(RLiD), i=1..10`.

Before root-length values are numerically available, an explicitly frozen set of 1-based REW layers is averaged. No fallback soil depth is hidden in code.

Formal quantities to freeze:

- `trigger_rew`: **TBD**
- `event_depth_mm`: **TBD**
- `fallback_rew_layers_1_based`: **TBD**
- agronomic/literature justification: **TBD**

### 6.4 Learned methods

4. Standard RL baseline: **TBD algorithm/configuration; implement only after the three agricultural baselines and real DSSAT smoke are accepted**
5. RCWA-RL: **implement after the standard RL baseline contract is frozen**

The three agricultural baselines must be calibrated/tuned, if tuning is necessary, using training/validation conditions only. Final-test weather cannot be used to select conventional schedule variants, ET parameters, or REW thresholds.

## 7. Replication

- Independent RL training seeds per learned method: **TBD**
- DSSAT stochasticity, if any: **TBD**
- Site/soil repetitions: **TBD**
- Field-block repetitions, if field validation is performed: **TBD**

Best-seed-only reporting is prohibited.

Deterministic agricultural baselines are evaluated once per unique site × soil × weather × water-budget condition unless an explicitly stochastic component is introduced. Statistical replication comes from the paired independent environmental units, not repeated identical deterministic simulations.

## 8. Primary outcomes

The following are primary and must be reported for every final management comparison:

- cotton yield (kg ha^-1);
- seasonal irrigation (mm);
- irrigation-water saving relative to baseline (%);
- irrigation water productivity (kg m^-3);
- lower-tail yield reliability;
- probability of meeting the pre-specified yield target.

Seasonal irrigation used in all primary outcomes must be based on executed management. Raw actor/rule requests must never be summed as agricultural water use.

For the current implementation, irrigation water productivity is calculated from terminal DSSAT seasonal irrigation as:

`IWP = HWAM / (10 * IRCM)`

when `IRCM > 0`, with units kg m^-3 if HWAM is kg ha^-1 and IRCM is mm.

## 9. Secondary/mechanistic outcomes

Subject to validated DSSAT outputs:

- ET and crop water productivity;
- stage-wise irrigation allocation;
- root-zone soil-water status;
- crop-water stress indicators;
- deep drainage;
- runoff;
- soil-water-storage change;
- dynamic marginal value of irrigation water.

## 10. DSSAT irrigation accounting audit

For every accepted terminal episode, reconcile DSSAT seasonal irrigation against the externally audited management ledger:

`expected_IRCM = nonpolicy_irrigation_mm + sum(executed_policy_irrigation_mm)`

The episode is valid only if:

`abs(DSSAT_IRCM - expected_IRCM) <= IRCM_tolerance_mm`.

A failed reconciliation is an experiment error and the episode must not enter training summaries, model-selection statistics, or final-test results.

Per-step rollout/baseline records must preserve both requested and executed irrigation, including projection/quantization reasons and whether DSSAT was rerun.

## 11. Real-worker preflight and smoke lock

Before any RL training run is accepted:

1. identify the exact legacy/current DSSAT executable, runtime profile, soil, genotype, weather, and generated experiment-file provenance;
2. freeze a known-good rendered no-policy/reset COX with `python -m awm.dssat.freeze_template ...`;
3. archive the source/template SHA-256 report and manually verify that fixed nitrogen, cultivar, initial soil conditions, planting, mulch/residue, and non-policy management are the intended AWM experiment;
4. the worker-local `DSSATPRO.L48` fixed-width preflight must pass;
5. one real DSSAT worker must complete a baseline reset season from the validated COX template;
6. the planting-day daily state must parse successfully;
7. `Summary.OUT` must contain at least `HWAM` and `IRCM`;
8. one complete 125-day agricultural-baseline episode must finish with a passing terminal IRCM reconciliation audit;
9. run all three agricultural baseline policies on at least one designated smoke weather year and archive their action audits;
10. only then begin implementation/training of the learned standard RL baseline.

The exact smoke configuration, template hashes, DSSAT asset versions, and smoke outputs used for the formal experiment must be archived with the protocol lock.

### Candidate legacy provenance currently identified

The previously operational DSSAT workflow in repository `wwwwwwangyuhao/lrmb`, branch `exp/sapg`, commit `b6257a29249969ea4b43849debe3e65657902e7d` contains candidate runtime assets including `dssat_workspace_template/dscsm048`, `DSSATPRO.L48`, genotype files, `SOIL.SOL`, ERA5 weather for 2000–2025, and station weather for 2023–2025. The old canonical environment used site code XJHX, planting DOY 119, emergence DOY 133, and a 125-decision horizon.

These values establish provenance only. They are not formal AWM agronomic constants until the frozen COX and experiment protocol are reviewed and locked.

## 12. Formal-test lock

Before the independent final-test set is run, create a tagged protocol version containing all previously `TBD` quantities. After this lock:

- do not alter `eta`, `alpha`, operational constraints, execution resolution, fixed irrigation accounting, DSSAT template, conventional schedule, ET baseline parameters, REW baseline parameters, learned-method definitions, climate thresholds, crop-stage definitions, or test years in response to final-test performance;
- any change requires a new protocol version and a new untouched final-test set.
