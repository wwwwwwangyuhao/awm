# Experimental Protocol — Pre-registration Skeleton

This document records empirical quantities that must be justified and frozen before formal test-set evaluation. The detailed agricultural lock is `docs/AGRICULTURAL_PROTOCOL_V1.md`; machine-readable constants are in `configs/agricultural_protocol_v1.json`.

## 1. Field, DSSAT experiment and irrigation-system constants

- Study region/site: **Huaxing Farm, Changji, Xinjiang, China; 44.223 N, 87.305 E**
- Irrigation system: **film-mulched drip irrigation**
- Validated DSSAT COX template path/version: **`run/formal/awm_protocol_v1_2000.COX.in`, agricultural protocol v1**
- DSSAT build: **lab-modified DSSAT 4.8.5 mulch build, `dscsm048`**
- Reference total seasonal irrigation, `B_ref_total`: **540 mm = 360 m3 mu^-1**
- Fixed preplant establishment irrigation: **45 mm = 30 m3 mu^-1, three days before planting; non-policy**
- Policy-controllable postplant quota: **W100 495 mm; W80 387 mm; W60 279 mm**
- Total seasonal water treatments: **W100 540 mm; W80 432 mm; W60 324 mm**
- Minimum positive irrigation execution: **0.1 mm**
- Maximum event/application capacity, `I_max`: **45 mm event^-1**
- Minimum interval between irrigation events, `d_min`: **0 days**
- Executable irrigation-depth resolution: **0.1 mm**
- Fixed/non-policy irrigation included in DSSAT `IRCM`: **45 mm**
- Terminal DSSAT irrigation-accounting tolerance: **0.1 mm**
- Fixed nitrogen schedule: **2025 T1 historical schedule, 235 kg N ha^-1, 10 positive events; provisional historical N locked for protocol v1**
- Planting/emergence: **DOY 119 / DOY 133**
- Cultivar: **calibrated `IB0007`**
- Mulch: **`PMALB=0.12`, `PMWD=22.5`**

The RL layer injects only postplant policy irrigation rows at `{{AWM_IRRIGATION_EVENTS}}`. The formal COX owns the fixed 45-mm preplant establishment irrigation and all fixed non-water management.

The WaterBudgetController quota is the postplant policy-controllable quota. Every accepted episode reconciles

`expected_IRCM = 45 + sum(executed_policy_irrigation_mm)`.

No empirical constant may be changed solely to make RL training easier.

## 2. Yield target and risk settings

- Reference-yield definition, `Y_ref(w)`: **year-specific W100 full-water DSSAT reference under the same weather and fixed non-water management**
- Canonical yield-protection target, `eta`: **TBD before learned-method formal evaluation**
- Lower-tail risk fraction, `alpha`: **TBD before learned-method formal evaluation**

`Y_ref(w)` is offline reference information and is never exposed to the actor.

## 3. Seasonal water-scarcity treatments

Total seasonal limits, including the common 45-mm preplant event:

- `W100`: **540 mm total = 45 fixed + 495 policy-controllable**
- `W80`: **432 mm total = 45 fixed + 387 policy-controllable**
- `W60`: **324 mm total = 45 fixed + 279 policy-controllable**

The quota is an upper bound, not a target use.

## 4. Weather-data split

- Training weather years: **ERA5 2000-2017**
- Validation/model-selection weather years: **ERA5 2018-2022**
- Independent final-test weather years: **Huaxing Farm station 2023-2025**

Historical ERA5 provenance is archived under `provenance/weather/open_meteo_era5_pipeline_legacy/`. The final station set is untouched during architecture, reward/constraint design, hyperparameter selection, checkpoint selection and agricultural-baseline parameter selection.

## 5. Climate and crop-stage definitions

Predefine thresholds for dry, normal, wet and hot-dry compound years before post-hoc mechanism analysis. Classification must depend only on meteorological variables, not algorithm outcomes.

Crop-stage boundaries used for post-hoc AWM mechanism analysis are also **TBD** and must be fixed from agronomic/DSSAT phenology definitions before formal analysis.

### Policy phenology encoding

The canonical numeric policy observation is **79D = 74D leakage-safe DSSAT state + 5D water-allocation state**. No extra discrete phenological-stage code is exposed to the actor. Crop development is represented continuously by DSSAT crop variables plus `dap_frac`.

## 6. Agricultural management baselines

Three non-learning agricultural baselines are implemented before any learned RL comparator. All use the same formal COX, fixed 45-mm preplant irrigation, postplant WaterBudgetController, action quantization and terminal IRCM reconciliation as learned policies.

### 6.1 Local conventional irrigation

Implementation: `ConventionalScheduleBaseline`.

Source timing is the Huaxing 2023 W100 field schedule (`dssat_cultivar_calibration/data/COX/XJHX2301.COX`), because that treatment is the field 360 m3 mu^-1 / 540-mm full-water reference. Its preplant 45 mm is handled by the formal COX; postplant timing is translated to biological action DAP.

The final quota-normalized event depths for W100/W80/W60 are frozen in the agricultural-baseline protocol, not as hidden software defaults. All requested and executed amounts remain audited through the common controller.

### 6.2 DSSAT potential-ET water-balance irrigation

Implementation: `PotentialETWaterBalanceBaseline`.

The comparator is causal and uses current/past DSSAT state only. It uses daily `EOAA` as DSSAT potential evapotranspiration and `PRED` as current-day precipitation:

`D <- max(0, D + EOAA - f_eff * PRED)`.

Executed irrigation is fed back as

`D <- max(0, D - irrigation_efficiency * I_executed)`.

The formal parameter values and their literature/field justifications are frozen in the agricultural-baseline protocol. This comparator is deliberately not labelled a full FAO-56 Penman-Monteith implementation.

### 6.3 Root-zone REW threshold irrigation

Implementation: `RootZoneREWThresholdBaseline`.

`REW_root = sum(REWi * RLiD) / sum(RLiD), i=1..10`.

Before root length is positive, a preregistered shallow-layer fallback is used. The formal threshold, event depth and fallback layers are frozen in the agricultural-baseline protocol.

### 6.4 Learned methods

4. Standard RL baseline: implement only after the three agricultural baseline formal smoke runs are accepted.
5. RCWA-RL: implement after the standard RL baseline contract is frozen.

Agricultural-baseline parameter selection must not inspect 2023-2025 station final-test outcomes.

## 7. Replication

- Independent RL training seeds per learned method: **TBD**
- DSSAT stochasticity: deterministic for fixed input and build unless demonstrated otherwise
- Site/soil repetitions: **one locked site/soil profile in protocol v1**
- Field-block repetitions for later real field validation: **TBD from field design**

Best-seed-only reporting is prohibited.

Deterministic agricultural baselines are evaluated once per unique site × soil × weather × water-budget condition.

## 8. Primary outcomes

Primary outcomes:

- cotton yield (kg ha^-1);
- total seasonal irrigation (mm), including the fixed 45-mm preplant event;
- postplant policy-controlled irrigation (mm), reported separately;
- irrigation-water saving relative to the preregistered baseline (%);
- irrigation water productivity (kg m^-3);
- lower-tail yield reliability;
- probability of meeting the pre-specified yield target.

Seasonal irrigation is based only on executed management. Raw actor/rule requests are never summed as agricultural water use.

`IWP = HWAM / (10 * IRCM)` for `IRCM > 0`.

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

For every accepted terminal episode:

`expected_IRCM = 45 mm + sum(executed_policy_irrigation_mm)`.

The episode is valid only if

`abs(DSSAT_IRCM - expected_IRCM) <= 0.1 mm`.

A failed reconciliation is an experiment error and cannot enter training summaries, model selection or final statistics.

## 11. Real-worker preflight and smoke lock

Before learned-method training is accepted:

1. the exact DSSAT executable/assets and formal COX provenance are versioned;
2. worker-local `DSSATPRO.L48` A80 preflight passes;
3. formal reset runs in the hashed AWM runtime;
4. planting-day state parses;
5. all 13 canonical daily OUT files are present/non-empty;
6. formal reset reconciles `IRCM=45 mm` with zero policy irrigation and `NICM=235 kg ha^-1`;
7. each of the three agricultural baselines completes a 125-day formal smoke with terminal IRCM reconciliation;
8. only then begin the learned standard RL baseline.

## 12. Formal-test lock

Before 2023-2025 station final evaluation, freeze all remaining `TBD` quantities, including `eta`, `alpha`, baseline parameter definitions, climate thresholds, crop-stage analysis definitions and learned-method settings. Do not alter them in response to final-test performance; any scientific change requires a new protocol version and a new untouched final-test set.
