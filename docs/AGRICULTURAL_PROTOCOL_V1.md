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

## 6. Irrigation action and hard constraints

DSSAT irrigation management uses `IRRIG=R`: only explicit AWM-generated irrigation rows are active. Automatic irrigation is therefore disabled.

The local normal full-irrigation reference is 360 m3 mu^-1. Using the exact area conversion used by this study,

`1 m3 mu^-1 = 1.5 mm`,

therefore

`B_ref = 360 * 1.5 = 540 mm`.

Canonical seasonal water-availability treatments are:

- `W100 = 540 mm`;
- `W80 = 432 mm`;
- `W60 = 324 mm`.

The budget is an upper bound, not a target consumption. Policies are allowed and encouraged to finish below the budget when the yield-protection requirement can be met with less irrigation.

Execution constraints:

- maximum event depth: 45 mm, corresponding to the field statement that a single event does not exceed 30 m3 mu^-1;
- exact no-op: 0 mm;
- minimum positive execution quantum: 0.1 mm;
- execution resolution: 0.1 mm;
- minimum interval: 0 days; no hand-coded spacing rule is imposed;
- non-policy seasonal irrigation: 0 mm;
- IRCM reconciliation tolerance: 0.1 mm.

Thus the action executor accepts either zero irrigation or a positive depth on the 0.1-mm grid up to 45 mm, subject to remaining seasonal budget.

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

For weather year `w`, define the yield reference as the full-water DSSAT yield under the same weather realization and exactly the same non-water management:

`Y_ref(w) = full-water reference yield for weather year w`.

Risk/yield comparisons should use the normalized quantity

`Y_pi(w) / Y_ref(w)`.

This avoids treating a single fixed absolute yield as equally attainable under every weather year. `Y_ref(w)` is an offline evaluation/constraint reference and is not exposed to the actor observation.

The numerical yield-protection target `eta` and lower-tail fraction `alpha` remain to be frozen before learned-method formal evaluation.

## 9. Formal COX v1

Canonical template:

`run/formal/awm_protocol_v1_2000.COX.in`

The 2000 instance is an engineering instantiation of the protocol used for real-DSSAT integration tests. Weather-year execution will transplant the year while keeping calendar DOYs and all non-weather management fixed.

Required properties:

- exactly one `{{AWM_IRRIGATION_EVENTS}}` marker;
- no pre-existing explicit irrigation rows;
- site metadata contains Huaxing/Changji/Xinjiang and no Jiangdu/Jiangsu residue;
- `IB0007` cultivar;
- `PMALB=0.12`, `PMWD=22.5`;
- planting/emergence DOY 119/133;
- the 10 positive fixed-N events above, summing to 235 kg N ha^-1;
- `IRRIG=R` and `FERTI=R`;
- WATER and NITRO simulation options enabled;
- P and K simulation options disabled.

## 10. What remains outside protocol v1

The following are intentionally not invented here and must be frozen separately before the corresponding analyses:

- real-world cultivar name for `IB0007` in manuscript prose;
- `eta` and `alpha` for the RCWA-RL yield-risk constraint;
- climate-category thresholds for dry/normal/wet/hot-dry post-hoc analysis;
- conventional baseline schedule;
- ET/water-balance baseline trigger and efficiency parameters;
- REW baseline threshold and event depth;
- RL training seeds and learned-method hyperparameters.

A later field-derived nitrogen schedule may motivate protocol v2, but protocol v1 remains an immutable reproducible experiment definition.
