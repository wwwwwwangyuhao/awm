# LRMB V3 environment snapshot imported into AWM

## Source lock

This directory is a deliberately isolated source snapshot taken from:

- repository: `wwwwwwangyuhao/lrmb`
- branch: `exp/v3-recoverable-policy-manifold-train10000`
- source commit: `d56336e09fdb9a9aea60ae61eaa892833314ab33`
- source tree: `6010966b70cc2499209b0c95d1bea4d0fd685251`

The source repository was treated as **read only**. No branch, commit, file, worktree, or runtime in `lrmb` was modified to create this snapshot.

## Why this exists

AWM needs an auditable copy of the DSSAT/cotton environment implementation that was already exercised in the LRMB V3 line, without carrying the LRMB/SAPG/RPM learning system into the new water-management project.

This directory is therefore a migration/reference layer. It is not the canonical AWM package and it does not override `src/awm`.

## Imported source components

The following environment-side source is retained:

- `dssat/dates.py`
- `dssat/file_handler.py`
- `dssat/output_reader.py`
- `dssat/runner.py`
- `configs/base_settings.py`
- `configs/environment.py`
- `configs/reward.py`
- `envs/cotton/env.py`
- `envs/cotton/observation.py`
- `envs/cotton/reward.py`
- `envs/cotton/state_schema.py`

These files preserve the historical implementation needed to audit DSSAT date semantics, complete-season reruns, COX generation, output parsing, the 74-D cotton state, and the historical environment objective.

## Intentionally excluded algorithm/training code

The migration deliberately does **not** include:

- `algorithms/` — LRMB, SAPG, RPM, actors, critics, buffers, recovery/manifold logic, losses;
- `models/` — policy/value model implementations;
- `pipeline/` — rollout/training orchestration, LRMB reporting, checkpoint/resume and training entrypoints;
- `launch.py`, `run.py`, `run_lrmb_*` — algorithm launch/training entrypoints;
- `configs/experiments/` — LRMB/SAPG training and smoke configurations;
- `configs/train_settings.py` and follower policy settings;
- `toy_v2/`;
- algorithm-specific tests;
- `dssat/workspace.py` from LRMB V3, because it imports the old `pipeline.workspace_layout_v1` lifecycle stack. AWM already owns a separate worker/workspace contract under `src/awm`;
- large DSSAT binaries, generated runtime directories, OUT files, checkpoints and training results.

## Important semantic warning

The imported historical environment is **reference only**. It is not the first-paper AWM contract.

In particular, LRMB V3 historically contains:

- a two-dimensional continuous action `[irrigation, nitrogen]`;
- dynamically generated COX content with site/cultivar/initial-condition values embedded in Python;
- the old additive water-nitrogen reward;
- legacy maximum irrigation/nitrogen bounds.

Those semantics must not silently leak into the new study.

The canonical AWM implementation remains under `src/awm` and currently requires:

- irrigation-only policy control;
- nitrogen fixed by the validated experiment/protocol;
- a hard seasonal irrigation-water budget;
- hierarchical event/amount irrigation actions;
- requested versus executed irrigation audits;
- terminal DSSAT `IRCM` reconciliation;
- 79-D policy observation = 74-D leakage-safe DSSAT state + 5-D water-allocation state;
- no terminal `HWAM/IRCM` leakage to the actor.

## Agronomic provenance warning

`legacy_v3/dssat/file_handler.py` contains historical Jiangdu/XJHX experiment assumptions such as cultivar, initial soil water/nitrogen, planting settings, mulch and other management details. Their presence in this snapshot means only that they existed in the source implementation.

They are **not** approved AWM experimental constants. Do not copy them into the canonical AWM protocol unless they are independently reviewed, validated and frozen in `EXPERIMENT_PROTOCOL.md`.

## Data/runtime assets

The source LRMB branch also contains DSSAT runtime assets and weather data. They are not duplicated in this Git migration because the purpose of `legacy_v3` is source-code provenance and because large binaries/runtime artifacts should not be casually duplicated into the active AWM code tree.

If AWM later needs those assets, they must be copied into an AWM-owned runtime location without modifying the source LRMB repository, and their hashes/provenance must be recorded.

## Modification rule

Prefer adapting functionality into `src/awm` with tests rather than editing this snapshot. If a source-reference file must be changed, document why it is no longer an exact source copy.
