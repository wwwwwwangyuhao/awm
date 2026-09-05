# Experimental Protocol — Pre-registration Skeleton

This document records empirical quantities that must be justified and frozen before formal test-set evaluation.

## 1. Field, DSSAT experiment and irrigation-system constants

- Study region/site: **TBD**
- Irrigation system: **TBD**
- Validated DSSAT COX template path/version: **TBD**
- Validated DSSAT COX template SHA-256: **TBD**
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

Required baselines:

1. Local conventional irrigation: **TBD protocol**
2. ET-based irrigation: **TBD protocol**
3. Soil-water-threshold irrigation: **TBD protocol**
4. Standard RL baseline: **TBD algorithm/configuration**
5. RCWA-RL

All baselines must share the same operational constraints and information availability unless a specific experiment states otherwise.

## 7. Replication

- Independent RL training seeds per method: **TBD**
- DSSAT stochasticity, if any: **TBD**
- Site/soil repetitions: **TBD**
- Field-block repetitions, if field validation is performed: **TBD**

Best-seed-only reporting is prohibited.

## 8. Primary outcomes

The following are primary and must be reported for every final management comparison:

- cotton yield (kg ha^-1);
- seasonal irrigation (mm);
- irrigation-water saving relative to baseline (%);
- irrigation water productivity (kg m^-3);
- lower-tail yield reliability;
- probability of meeting the pre-specified yield target.

Seasonal irrigation used in all primary outcomes must be based on executed management. Raw actor output must never be summed as agricultural water use.

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

Per-step rollout records must preserve both requested and executed irrigation, including projection/quantization reasons and whether DSSAT was rerun.

## 11. Real-worker preflight and smoke lock

Before any RL training run is accepted:

1. the worker-local `DSSATPRO.L48` fixed-width preflight must pass;
2. one real DSSAT worker must complete a baseline reset season from the validated COX template;
3. the planting-day daily state must parse successfully;
4. `Summary.OUT` must contain at least `HWAM` and `IRCM`;
5. one complete irrigation-policy episode must finish with a passing terminal IRCM reconciliation audit.

The exact smoke configuration used for the formal experiment must be archived with the protocol lock.

## 12. Formal-test lock

Before the independent final-test set is run, create a tagged protocol version containing all previously `TBD` quantities. After this lock:

- do not alter `eta`, `alpha`, operational constraints, execution resolution, fixed irrigation accounting, DSSAT template, baseline definitions, climate thresholds, crop-stage definitions, or test years in response to final-test performance;
- any change requires a new protocol version and a new untouched final-test set.
