# AWM

Research repository for **risk-constrained adaptive irrigation-water allocation for cotton using DSSAT and reinforcement learning**, designed around Agricultural Water Management questions rather than a generic RL benchmark.

The frozen version-1 scientific and optimization contract is defined in [`PROBLEM_FORMULATION.md`](PROBLEM_FORMULATION.md).

The versioned agricultural-management lock is defined in [`docs/AGRICULTURAL_PROTOCOL_V1.md`](docs/AGRICULTURAL_PROTOCOL_V1.md), with machine-readable constants in [`configs/agricultural_protocol_v1.json`](configs/agricultural_protocol_v1.json). The canonical formal DSSAT template is [`run/formal/awm_protocol_v1_2000.COX.in`](run/formal/awm_protocol_v1_2000.COX.in).

## Version-1 scope

- DSSAT is the crop–soil–water simulator.
- RL controls irrigation only; nitrogen is fixed to a versioned explicit schedule.
- Seasonal irrigation water is explicitly limited.
- Future weather is hidden from the policy.
- Irrigation actions must satisfy realistic operational constraints.
- Primary outcomes are yield reliability, seasonal irrigation use, irrigation water productivity, growth-stage water allocation, and robustness across independent weather conditions.
- The working algorithm is **RCWA-RL (Risk-Constrained Water Allocation Reinforcement Learning)**.

## Protocol-v1 agricultural lock

- Huaxing Farm, Changji, Xinjiang, China; 44.223 N, 87.305 E.
- Planting/emergence DOY 119/133; calibrated `IB0007`.
- Film mulch: `PMALB=0.12`, `PMWD=22.5`.
- Fixed protocol-v1 N: historical 2025 T1 schedule, 235 kg N ha^-1; automatic N off.
- `B_ref=540 mm`; W100/W80/W60 = 540/432/324 mm.
- Event range: exact no-op or positive 0.1-mm-grid execution up to 45 mm; no hand-coded minimum interval.
- ERA5 train 2000-2017; ERA5 validation 2018-2022; Huaxing station final test 2023-2025.

## Development order

1. Freeze empirical irrigation-system and agronomic constants. **Protocol v1 complete.**
2. Implement and test the water-budget-aware DSSAT environment contract. **Core plumbing and real-DSSAT smoke complete.**
3. Freeze and validate the three agricultural baselines.
4. Implement a standard RL baseline.
5. Implement RCWA-RL components incrementally with ablations.
6. Freeze remaining risk settings (`eta`, `alpha`) and learned-method training protocol before independent final testing.
7. Run final-test weather only after all model-selection decisions are locked.
