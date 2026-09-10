# RCWA-RL Ablation Protocol v1

This protocol is preregistered before completion of the main PPO, PPO-Lagrangian, and RCWA-RL formal runs. Its purpose is to prevent choosing ablations after seeing the main learned-method results.

The machine-readable source of truth is `configs/rcwa_ablation_protocol_v1.json`.

## A0: Full RCWA-RL

The main method uses a single eta-conditional policy with registered targets 0.90, 0.95, and 0.98 and the frozen lower-tail risk level alpha=0.20.

## A1: Mean-constraint component control

No duplicate training run is required. The already registered PPO-Lagrangian baseline serves as the component control that replaces RCWA-RL's lower-CVaR constraint with an expected-retention constraint while preserving the PPO action stack, interaction budget, validation cadence, and seeds.

This answers the primary question:

> Does lower-tail risk control add value beyond using an explicit expected-yield constraint?

## A2: No conditional sharing

Three independent RCWA policies are trained per seed, one for each fixed eta. Each policy receives the same 18 training weather years and 200 updates, for 3600 episodes per eta-specific policy.

Across the three policies:

- total episodes per seed = 10,800;
- total decisions per seed = 1,350,000;
- the interaction budget therefore matches the full conditional RCWA method.

The 79-D observation is retained, but `yield_target_fraction` is constant inside each independent policy and therefore cannot provide cross-target conditioning information.

Checkpoint selection is performed on a **bundle**, not separately per eta. At update 10, for example, the eta=0.90, 0.95, and 0.98 policies form one candidate bundle. The same is true at updates 20,...,200. Therefore the ablation has 20 selection opportunities, not 60. One shared update index is selected using all 15 validation cells and the same lexicographic feasibility/water rule.

## A3/A4: Alpha sensitivity

Two development-only sensitivity runs are preregistered:

- alpha=0.10;
- alpha=0.30.

They use the same seeds and interaction budget as the main method. Their purpose is sensitivity analysis only. They may not be used to replace the registered alpha=0.20 after observing results.

## Data governance

All v1 ablations use only:

- training weather: 2000-2017 ERA5;
- validation weather: 2018-2022 ERA5.

They do not access 2023-2025 station outcomes under this protocol. If station evaluation of an ablation is ever required, that requires a separate ablation final-test protocol frozen before those station outcomes are used.

No hyperparameter may be retuned from the main RCWA result solely to improve an ablation result.
