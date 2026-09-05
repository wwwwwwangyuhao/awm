# Real DSSAT worker and agricultural-baseline smoke runbook

This runbook is the server-side step that cannot be executed by GitHub-side development. It must be completed on the Linux machine that already runs DSSAT successfully.

## 1. Start from an existing successful worker

Use the same DSSAT executable/runtime family that has already produced valid cotton simulations. The legacy asset manifest in `configs/legacy_asset_candidate.json` points to candidate assets in `wwwwwwangyuhao/lrmb@b6257a29249969ea4b43849debe3e65657902e7d`.

Do **not** copy an arbitrary old training COX containing an RL irrigation trajectory and call it the AWM baseline template.

Preferred source:

- reset/no-policy worker;
- successful complete DSSAT season;
- correct site/year/cultivar/soil/fixed-N management;
- no policy-generated irrigation rows.

## 2. Freeze the rendered COX and record hashes

From the AWM checkout:

```bash
python -m awm.dssat.freeze_template \
  --source-cox /absolute/path/to/known_good_reset.COX \
  --output-template /absolute/path/to/awm_base.COX.in \
  --report /absolute/path/to/awm_base.template.json
```

The default command fails if explicit irrigation rows are present. Inspect and classify them first. Only if they are confirmed policy-generated and should be removed may you deliberately use:

```bash
python -m awm.dssat.freeze_template \
  --source-cox /absolute/path/to/source.COX \
  --output-template /absolute/path/to/awm_base.COX.in \
  --report /absolute/path/to/awm_base.template.json \
  --allow-strip-existing-irrigation
```

Before continuing, manually verify the frozen template's cultivar, planting/emergence dates, initial soil conditions, fixed nitrogen, mulch/residue settings, simulation controls and any non-policy management. Record the template/report hashes in `EXPERIMENT_PROTOCOL.md`.

## 3. Baseline reset smoke

Copy `configs/real_baseline_smoke.example.json` to a run-specific JSON and replace every `TBD` value with actual worker-local paths/parameters.

First run the lower-level reset smoke if desired:

```bash
python -m awm.dssat.smoke --config /absolute/path/to/real_worker_smoke.json
```

Required result:

- `DSSATPRO.L48` A80 preflight passes;
- DSSAT complete-season return code is zero;
- planting-day state parses;
- Summary.OUT contains HWAM and IRCM.

## 4. Full 125-day conventional baseline smoke

Create a baseline config with the preregistered local schedule:

```json
"baseline": {
  "type": "local_conventional",
  "schedule_mm_by_action_dap": {
    "TBD": "TBD"
  }
}
```

Then run:

```bash
python -m awm.baselines.smoke \
  --config /absolute/path/to/conventional_smoke.json \
  --audit-output /absolute/path/to/conventional_smoke.audit.json
```

Do not accept the run unless:

- status is `passed`;
- the episode reaches all 125 decisions;
- terminal irrigation accounting passes;
- `IRCM = nonpolicy_irrigation + executed policy irrigation` within the frozen tolerance;
- no future/seasonal field appears in the policy observation;
- the audit shows plausible event dates and executed depths.

## 5. Potential-ET baseline smoke

Use:

```json
"baseline": {
  "type": "potential_et_water_balance",
  "trigger_deficit_mm": "TBD",
  "irrigation_efficiency": "TBD",
  "effective_rain_fraction": "TBD",
  "refill_fraction": "TBD"
}
```

Run the same CLI with a separate audit file. Confirm that only current/past `EOAA` and `PRED` drive the internal deficit; no future weather is supplied.

## 6. Root-zone REW baseline smoke

Use:

```json
"baseline": {
  "type": "root_zone_rew_threshold",
  "trigger_rew": "TBD",
  "event_depth_mm": "TBD",
  "fallback_rew_layers_1_based": ["TBD"]
}
```

Confirm that trigger decisions follow root-length-weighted REW and that early-season fallback layers match the preregistered soil/root-zone interpretation.

## 7. Archive before learned RL work

Archive together:

- exact AWM commit SHA;
- DSSAT executable/version/hash if available;
- DSSATPRO.L48;
- frozen COX template and provenance report;
- soil/genotype/weather files or their hashes/version identifiers;
- exact smoke JSONs;
- all three action-audit JSONs;
- stdout/stderr logs;
- final HWAM, IRCM and IWP summaries.

Only after these three agricultural baselines complete the real-worker smoke should the standard learned RL baseline be implemented.
