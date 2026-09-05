# Experimental Protocol — Pre-registration Skeleton

This document records empirical quantities that must be justified and frozen before formal test-set evaluation.

## 1. Field and irrigation-system constants

- Study region/site: **TBD**
- Irrigation system: **TBD**
- Reference seasonal irrigation, `B_ref` (mm): **TBD**
- Minimum effective irrigation event, `I_min` (mm): **TBD**
- Maximum event/application capacity, `I_max` (mm day^-1 or event^-1): **TBD**
- Minimum interval between irrigation events, `d_min` (days): **TBD**
- Fixed nitrogen schedule: **TBD**

No value above should be chosen solely to make RL training easier.

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

## 5. Climate-class definitions

Predefine thresholds for:

- dry years;
- normal years;
- wet years;
- hot–dry compound years.

Classification must depend only on meteorological variables, not on algorithm outcomes.

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

## 10. Formal-test lock

Before the independent final-test set is run, create a tagged protocol version containing all previously `TBD` quantities. After this lock:

- do not alter `eta`, `alpha`, operational constraints, baseline definitions, climate thresholds, or test years in response to final-test performance;
- any change requires a new protocol version and a new untouched final-test set.
