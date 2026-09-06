# AWM Learned-Method Evaluation Contract v1

This contract freezes how PPO and RCWA-RL checkpoints are selected from the development validation split. The machine-readable source is `configs/learned_method_evaluation_contract_v1.json`.

## 1. Scope

Checkpoint selection is performed independently **within each preregistered training seed**. A seed is a replication unit, not a model-selection axis: all preregistered seeds remain in the reported experiment and no best seed may be chosen.

For a conditional policy, one selected checkpoint must serve all registered protection targets:

\[
\eta\in\{0.90,0.95,0.98\}.
\]

It is forbidden to choose one checkpoint for `eta=0.90` and another for `eta=0.98` from the same training run.

## 2. Validation data

Every candidate checkpoint is evaluated on exactly:

\[
5\text{ validation weather years}\times3\eta=15
\]

weather-target cells.

The validation weather years are ERA5 2018-2022. They may not produce gradient updates. Station years 2023-2025 are forbidden during checkpoint selection.

The exact candidate checkpoint cadence is algorithm-specific and must be frozen before training. Once training starts, all checkpoints emitted by that registered cadence belong to the candidate set; candidates cannot be selectively omitted because their validation result is inconvenient.

## 3. Per-eta risk metrics

For every candidate and every eta, compute the Risk Contract v1 quantity separately across the five validation years:

\[
L_\eta=\mathrm{LCVaR}_{0.20}(R_\pi\mid\eta),
\qquad
m_\eta=L_\eta-\eta.
\]

The checkpoint is **jointly feasible** iff

\[
m_{0.90}\ge0,\quad m_{0.95}\ge0,\quad m_{0.98}\ge0.
\]

Different eta levels are never pooled into one CVaR sample.

## 4. Irrigation selection metric

For every candidate, compute mean total seasonal irrigation over all 15 validation cells, with equal weight for each year and each eta.

Total seasonal irrigation includes the common 45-mm preplant establishment event. Because this fixed amount is shared by every method and checkpoint, it does not change ranking, but retaining it keeps the reported metric agronomically interpretable.

## 5. Selection when feasible checkpoints exist

If at least one candidate is jointly feasible, choose the candidate by the following deterministic lexicographic order:

1. minimum mean validation total irrigation across all 15 cells;
2. maximum minimum eta-specific risk margin;
3. maximum mean eta-specific risk margin;
4. earlier training step;
5. lexicographically smaller checkpoint identifier.

The first criterion implements the scientific objective: after all requested yield-protection constraints are met, select the checkpoint that uses the least water.

## 6. Selection when training never produces a jointly feasible checkpoint

If no candidate is jointly feasible, the training run is **not reclassified as successful**. A deterministic representative is nevertheless required for diagnosis and method comparison.

Define eta-specific positive shortfall

\[
s_\eta=\max(0,\eta-L_\eta).
\]

Select by:

1. minimum worst shortfall `max_eta s_eta`;
2. minimum mean shortfall across the three eta levels;
3. minimum mean validation total irrigation across all 15 cells;
4. maximum minimum risk margin;
5. earlier training step;
6. lexicographically smaller checkpoint identifier.

Such a checkpoint receives status `selected_infeasible`. The paper and experiment logs must retain that infeasibility; the fallback is only a deterministic selection rule.

## 7. Evaluation action mode

The exact deterministic/stochastic action rule depends on the policy distribution used by each learned algorithm. Therefore the algorithm-specific protocol must freeze, before training:

- checkpoint cadence;
- evaluation action mode;
- if stochastic evaluation is used, the number of rollouts and evaluation seeds.

Once declared, the same evaluation mode must be used for every candidate checkpoint and the final evaluation. Validation outcomes cannot be used to change this mode.

## 8. Final-test lock

Before station weather 2023-2025 is evaluated, all of the following must already be frozen:

- implementation and hyperparameters;
- training seeds;
- checkpoint candidate cadence;
- evaluation action mode;
- the selected checkpoint for every training seed.

Station weather cannot trigger checkpoint reselection, hyperparameter changes, seed selection or retraining.

## 9. Implementation

- machine-readable contract: `configs/learned_method_evaluation_contract_v1.json`;
- selector: `src/awm/evaluation/checkpoint_selection.py`;
- tests: `tests/test_checkpoint_selection.py`.

The selector implements the ordering above directly; later PPO and RCWA-RL training code should emit validation reports into this common selector rather than reimplementing checkpoint ranking independently.
