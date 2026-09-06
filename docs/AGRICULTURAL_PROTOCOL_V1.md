# AWM Agricultural Protocol v1

This document freezes the agricultural-management constants used by the first AWM irrigation-control study. The machine-readable source of truth is `configs/agricultural_protocol_v1.json`.

## 1. Protocol status

Protocol identifier: `awm-agricultural-v1`.

This is a versioned lock. The current fixed nitrogen schedule is a real historical 2025 T1 schedule and is intentionally labelled **provisional historical N for protocol v1**. It is nevertheless frozen for v1 reproducibility. Replacing it later with a newly available field schedule requires a new protocol version; v1 must not be silently rewritten after seeing model or final-test performance.

## 2. Study site

- Site: Huaxing Farm, Changji, Xinjiang, China.
- DSSAT site code: `XJHX`.
- Soil profile: `XJHX0001`.
- Coordinates: 44.223 N, 87.305 E.
- Irrigation system: film-mulched drip irrigation.

Historical `JIANGDU`, `JIANGSU`, and 32.585/119.7 descriptive metadata are legacy template residue and are forbidden in the formal AWM template.

## 3. Crop, calendar and initial state

- Crop: cotton.
- Calibrated DSSAT cultivar: `IB0007`.
- Real-world cultivar name: pending final paper metadata; the calibrated DSSAT identifier is already fixed.
- Planting DOY: 119.
- Emergence DOY: 133.
- Decision horizon: 125 policy decisions.
- Temporal contract: policy day `d` controls biological action DAP `d+1`.

The initial 10-layer soil-water, NH4 and NO3 profile is copied from the existing calibrated Huaxing COX profile and is held fixed across weather years so that the weather-year experiment does not confound changing initial conditions with climate variability.

## 4. Film mulch

All treatments use plastic-film mulch.

- `PMALB = 0.12`.
- `PMWD = 22.5`, derived as `30 * 0.75` in the laboratory implementation.
- No film-removal or film-failure process is introduced in protocol v1.

Methodological source: Li et al. (2019), *Simulation of cotton growth and soil water content under film-mulched drip irrigation using modified CSM-CROPGRO-cotton model*, Agricultural Water Management 218:124-138, DOI `10.1016/j.agwat.2019.03.041`.

## 5. Nitrogen: fixed non-policy management

The first AWM paper optimizes irrigation only. Nitrogen is not part of the RL action.

DSSAT fertilization management uses `FERTI=R`, meaning the simulator applies only explicitly reported fertilizer rows. The automatic-nitrogen parameter row may remain in the standard FileX section for compatibility but is inactive under `FERTI=R`.

Protocol-v1 fixed-N source:

- repository: `wwwwwwangyuhao/dssat_cultivar_calibration`;
- file: `data/COX/XJHX2501.COX`;
- treatment: 2025 T1;
- fertilizer code: `FE010`;
- application code: `AP005`;
- `FDEP = 2`;
- seasonal N total: 235 kg N ha^-1.

Positive N applications are fixed by DOY:

| DOY | N (kg ha^-1) |
| ---: | ---: |
| 161 | 8 |
| 173 | 16 |
| 182 | 7 |
| 190 | 21 |
| 195 | 31 |
| 201 | 52 |
| 212 | 20 |
| 218 | 30 |
| 224 | 25 |
| 231 | 25 |
| **Total** | **235** |

The P and K numeric fields from the historical rows are retained in the COX for provenance, but protocol v1 keeps `PHOSP=N` and `POTAS=N`; therefore P and K stress are not active DSSAT model dimensions in this study.

## 6. Irrigation action, establishment water and hard constraints

DSSAT irrigation management uses `IRRIG=R`: only explicit fixed-management rows plus AWM-generated policy rows are active. Automatic irrigation is disabled.

### 6.1 Fixed preplant establishment irrigation

The 2023, 2024 and 2025 Huaxing field COX records all contain the same preplant event: 30 m3 mu^-1 exactly three days before planting. Under the study conversion `1 m3 mu^-1 = 1.5 mm`, this is 45 mm.

Protocol v1 therefore fixes:

- preplant irrigation: 45 mm;
- formal 2000-instance date: DOY 116, three days before planting DOY 119;
- policy controllable: no;
- shared by every agricultural baseline, learned policy and reference simulation;
- included in DSSAT seasonal `IRCM` and all reported seasonal irrigation totals.

The policy never receives an action before planting. Its water-allocation state and `WaterBudgetController` account only for postplant controllable irrigation.

### 6.2 Total and policy-controllable seasonal budgets

The local normal full-irrigation reference is 360 m3 mu^-1 = 540 mm. The canonical **total seasonal** water-availability treatments are:

- `W100_total = 540 mm`;
- `W80_total = 432 mm`;
- `W60_total = 324 mm`.

Subtracting the common fixed 45-mm preplant event gives the postplant policy quotas enforced by `WaterBudgetController`:

- `W100_policy = 495 mm`;
- `W80_policy = 387 mm`;
- `W60_policy = 279 mm`.

Thus for every accepted episode,

`IRCM_expected = 45 mm + sum(executed postplant policy irrigation)`.

The total budget is an upper bound, not a target consumption. Policies may finish below their postplant quota when the yield-protection requirement can be met with less irrigation.

Execution constraints for postplant actions:

- maximum event depth: 45 mm, corresponding to the field statement that a single event does not exceed 30 m3 mu^-1;
- exact no-op: 0 mm;
- minimum positive execution quantum: 0.1 mm;
- execution resolution: 0.1 mm;
- minimum interval: 0 days; no hand-coded spacing rule is imposed;
- fixed/non-policy seasonal irrigation: 45 mm;
- IRCM reconciliation tolerance: 0.1 mm.

Thus the action executor accepts either zero irrigation or a positive depth on the 0.1-mm grid up to 45 mm, subject to the remaining **postplant** quota.

## 7. Weather provenance and split

Historical weather is generated from the Open-Meteo Historical Weather API using `models=era5`, with the recovered downloader and WTH conversion source archived under:

`provenance/weather/open_meteo_era5_pipeline_legacy/`.

The recovered source documents:

- API endpoint `https://archive-api.open-meteo.com/v1/archive`;
- coordinates 44.223 N, 87.305 E;
- timezone `Asia/Urumqi`;
- default model `era5`;
- Open-Meteo wind requested in m s^-1;
- DSSAT WIND conversion `m/s * 86.4 -> km/day`.

The legacy scheduler in that archive is provenance only and is **not** the canonical AWM split.

Protocol-v1 split:

- training: ERA5 2000-2017 (18 years);
- validation/model selection: ERA5 2018-2022 (5 years);
- untouched final test: Huaxing Farm station weather 2023-2025 (3 years).

The final-test station years may not influence reward design, architecture, hyperparameters, checkpoint selection, baseline tuning, risk settings or any other model-selection decision.

The repository also contains legacy `era5`-directory copies for 2023-2025. They are noncanonical for protocol v1 and must not be used for model selection or final evaluation.

## 8. Reference yield

For weather year `w`, define the yield reference as the W100 full-water DSSAT reference under the same weather realization and exactly the same non-water management:

`Y_ref(w) = W100 reference yield for weather year w`.

The concrete W100 postplant schedule is frozen with the agricultural-baseline protocol; its total irrigation must equal 540 mm including the common 45-mm preplant event.

Risk/yield comparisons should use `Y_pi(w) / Y_ref(w)`. `Y_ref(w)` is an offline evaluation/constraint reference and is never exposed to the actor observation.

The numerical yield-protection target `eta` and lower-tail fraction `alpha` remain to be frozen before learned-method formal evaluation.

## 9. Formal COX v1

Canonical template:

`run/formal/awm_protocol_v1_2000.COX.in`

Required properties:

- exactly one `{{AWM_IRRIGATION_EVENTS}}` marker;
- exactly one fixed non-policy irrigation row, `00116 IR005 45.00`;
- no other pre-existing irrigation rows;
- site metadata contains Huaxing/Changji/Xinjiang and no Jiangdu/Jiangsu residue;
- `IB0007` cultivar;
- `PMALB=0.12`, `PMWD=22.5`;
- planting/emergence DOY 119/133;
- the 10 positive fixed-N events above, summing to 235 kg N ha^-1;
- `IRRIG=R` and `FERTI=R`;
- WATER and NITRO simulation options enabled;
- P and K simulation options disabled.

## 10. What remains outside protocol v1

The following remain to be frozen separately before the corresponding formal analyses:

- real-world cultivar name for `IB0007` in manuscript prose;
- `eta` and `alpha` for the RCWA-RL yield-risk constraint;
- climate-category thresholds for dry/normal/wet/hot-dry post-hoc analysis;
- the quota-normalized conventional postplant schedule;
- ET/water-balance baseline parameters;
- REW baseline parameters;
- RL training seeds and learned-method hyperparameters.

A later field-derived nitrogen schedule may motivate protocol v2, but protocol v1 remains an immutable reproducible experiment definition.
