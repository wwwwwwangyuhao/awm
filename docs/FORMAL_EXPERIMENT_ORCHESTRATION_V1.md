# Formal Experiment Orchestration v1

This document defines the read-only monitoring, checkpoint-selection, and development-export workflow for the three formal learned irrigation methods:

- PPO v1;
- PPO-Lagrangian v1;
- RCWA-RL v1.

The machine-readable source of truth is `configs/formal_experiment_orchestration_v1.json`.

## Training completion versus selection readiness

A seed is not considered ready for scientific selection merely because update 200 was reached. A seed is `selection_ready` only when all of the following are true:

1. `train_updates.jsonl` contains contiguous updates 1-200;
2. candidate checkpoints exist for exactly updates 10,20,...,200;
3. deterministic validation reports exist for exactly updates 10,20,...,200;
4. the run manifest protocol and training seed match the frozen method contract.

This prevents an update-200 trainer crash during final checkpoint/validation serialization from being mistaken for a complete experiment.

## Monitor

`awm.experiments.formal_monitor` is read-only. It reports per method and seed:

- manifest provenance;
- completed update count;
- recovery/candidate/validation artifact counts;
- actual trainer-process presence when run on the training host;
- `running`, `stopped_incomplete`, `artifact_incomplete`, or `selection_ready` state.

It does not kill, resume, modify, or select training runs.

## Formal checkpoint selector

`awm.experiments.formal_selection` does not define a new score. It invokes the already frozen `awm.evaluation.checkpoint_selection` contract.

For each seed it requires all 20 validation reports and all 20 candidate checkpoints. It then records the unique selected checkpoint together with:

- checkpoint SHA-256;
- selected validation-report SHA-256;
- source run-manifest SHA-256;
- source Git commit and protocol ID;
- complete frozen selection metrics.

All five seeds remain independent replicates. The selector never chooses a best seed.

## Development exporter

`awm.experiments.development_export` consumes the selected-checkpoint documents and exports only the deterministic 2018-2022 validation evidence already generated during checkpoint selection.

It writes:

- one row per method × seed × eta × validation year;
- one row per method × seed × eta containing LCVaR and irrigation metrics;
- cross-seed summaries using mean and sample standard deviation.

The exporter rejects any 2023-2025 row. It cannot be used to open or populate the station final test.

## Final-test lock

Nothing in this orchestration layer grants access to 2023-2025 station outcomes. Station reference yields and station learned-policy results remain locked until the main learned methods and selected checkpoints are frozen under the separate final-evaluation protocol.
