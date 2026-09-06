# Agricultural Baselines v1

Machine-readable source of truth: `configs/agricultural_baselines_v1.json`.

All three non-learning comparators use the same formal agricultural protocol, custom DSSAT 4.8.5 mulch build, fixed 45-mm preplant establishment irrigation, postplant `WaterBudgetController`, 45-mm event cap, 0.1-mm execution resolution and terminal IRCM reconciliation as learned policies.

For every episode:

`IRCM_expected = 45 mm fixed preplant + sum(executed postplant irrigation)`.

The W100/W80/W60 total seasonal limits are 540/432/324 mm; therefore their postplant policy quotas are 495/387/279 mm.

## A. Quota-normalized Huaxing conventional timing

Implementation: `ConventionalScheduleBaseline`.

Timing provenance is the 2023 Huaxing W100 field treatment in `wwwwwwangyuhao/dssat_cultivar_calibration:data/COX/XJHX2301.COX`. That treatment used the local full-water level of 360 m3 mu^-1 = 540 mm and planted on DOY 124. Its positive postplant irrigation DOYs were:

`160, 170, 180, 190, 195, 202, 209, 216, 223, 230, 237`.

Subtracting planting DOY gives canonical biological action DAPs:

`36, 46, 56, 66, 71, 78, 85, 92, 99, 106, 113`.

Historical individual event depths sometimes exceeded the protocol-v1 45-mm operational cap. The comparator therefore preserves the **local field timing** but quota-normalizes depth onto the common feasible action envelope rather than granting the conventional baseline an exception.

- W100 postplant: 11 × 45.0 = 495.0 mm.
- W80 postplant: nine 35.2-mm + two 35.1-mm events = 387.0 mm.
- W60 postplant: seven 25.4-mm + four 25.3-mm events = 279.0 mm.

All amounts lie on the common 0.1-mm execution grid. The fixed preplant 45 mm is not requested by this policy; it is already in the formal COX.

This W100 schedule is also the protocol-v1 timing reference for year-specific `Y_ref(w)` simulations.

## B. Causal DSSAT potential-ET deficit rule

Implementation: `PotentialETWaterBalanceBaseline`.

It uses only current/past DSSAT variables:

- `EOAA`: DSSAT potential evapotranspiration (mm d^-1);
- `PRED`: current-day precipitation (mm).

Ledger:

`D <- max(0, D + EOAA - f_eff * PRED)`.

When `D >= 40.5 mm`, the rule requests:

`I_requested = D / 0.90`,

subject to the common 45-mm event cap and remaining quota. After execution:

`D <- max(0, D - 0.90 * I_executed)`.

Frozen parameters:

- `irrigation_efficiency = 0.90`. FAO gives 90% as an indicative drip-irrigation field application efficiency: https://www.fao.org/4/t7202e/t7202e08.htm
- `trigger_deficit_mm = 40.5 = 0.90 × 45`. A capacity-sized 45-mm gross event therefore corresponds to one threshold-sized net deficit.
- `effective_rain_fraction = 1.0`. All causal DSSAT precipitation is credited; no unvalidated empirical effective-rain/runoff coefficient is introduced.
- `refill_fraction = 1.0`. The observed deficit is fully targeted, while the common executor enforces event and seasonal limits.

This rule is **not** described as a complete FAO-56 Penman-Monteith scheduler. It is a transparent DSSAT-potential-ET water-balance comparator.

## C. Root-zone REW threshold rule

Implementation: `RootZoneREWThresholdBaseline`.

The policy computes:

`REW_root = sum(REWi * RLiD) / sum(RLiD), i=1..10`.

Frozen parameters:

- `trigger_rew = 0.35`;
- `event_depth_mm = 45.0`;
- fallback layers before positive root-length weights: REW layers 1-4 (0-30 cm by the protocol layer boundaries).

The threshold is derived from cotton depletion fraction `p=0.65`: FAO defines `p` as the fraction of total available water that may be depleted before stress begins, so under AWM's `REW=(SW-LL)/(DUL-LL)` normalization the corresponding remaining-water threshold is approximately `1-p=0.35`. FAO56 defines RAW = p·TAW and a later Agricultural Water Management cotton drip-irrigation study independently recommends a baseline p of 0.65.

Sources:

- FAO56 soil-water stress definition: https://www.fao.org/4/X0490E/x0490e0e.htm
- Cotton p=0.65 evidence: DOI `10.1016/j.agwat.2021.106881`.

The top-30-cm fallback is used only before DSSAT provides positive root-length weights; thereafter the rule is dynamically root-weighted.

## No validation-set tuning for v1

The v1 baseline parameters are fixed **before** 2023-2025 station evaluation from field operations, physical constraints and external agronomic references. They are not optimized against final-test performance and, by default, are not hyperparameter-tuned on 2018-2022 either. Validation years are used for sanity and robustness assessment.

Changing these definitions after inspecting station-test outcomes requires a new baseline protocol version and cannot overwrite v1.

## Formal smoke gate

GitHub Actions can execute the repository-vendored `dscsm048` directly on Ubuntu. Before the standard learned RL baseline is implemented, each of the three W100 baseline definitions must complete a full 125-decision real-DSSAT episode with:

- the formal COX;
- 45-mm fixed preplant irrigation;
- fixed 235 kg N ha^-1;
- 13 canonical daily OUT files;
- passing `IRCM = 45 + policy irrigation` reconciliation;
- complete requested/executed action audit.

After W100 acceptance, W80/W60 configs and the 2000-2022 development-weather sweep are generated from the same locked definitions.
