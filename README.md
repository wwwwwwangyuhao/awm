# AWM

Research repository for **risk-constrained adaptive irrigation-water allocation for cotton using DSSAT and reinforcement learning**, designed around Agricultural Water Management questions rather than a generic RL benchmark.

The frozen version-1 scientific and optimization contract is defined in [`PROBLEM_FORMULATION.md`](PROBLEM_FORMULATION.md).

## Version-1 scope

- DSSAT is the crop–soil–water simulator.
- RL controls irrigation only; nitrogen is fixed to a validated recommended schedule.
- Seasonal irrigation water is explicitly limited.
- Future weather is hidden from the policy.
- Irrigation actions must satisfy realistic operational constraints.
- Primary outcomes are yield reliability, seasonal irrigation use, irrigation water productivity, growth-stage water allocation, and robustness across independent weather conditions.
- The working algorithm is **RCWA-RL (Risk-Constrained Water Allocation Reinforcement Learning)**.

## Development order

1. Freeze empirical irrigation-system and agronomic constants.
2. Implement and test the water-budget-aware DSSAT environment contract.
3. Implement agricultural baselines.
4. Implement a standard RL baseline.
5. Implement RCWA-RL components incrementally with ablations.
6. Freeze train/validation/test weather protocol.
7. Run formal AWM experiments only after the above contracts are fixed.
