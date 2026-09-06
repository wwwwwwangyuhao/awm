# Real DSSAT worker and agricultural-baseline smoke runbook

This runbook assumes the one-time asset migration in
`docs/SELF_CONTAINED_DSSAT_RUNTIME.md` has already been completed and committed
to AWM.

The real smoke must use only AWM-owned assets under:

```text
dssat_workspace_template/
```

LRMB must not be consulted at runtime.

## 1. Verify the versioned template

From the AWM checkout:

```bash
PYTHONPATH="$PWD/src" python - <<'PY'
from awm.dssat.runtime_assets import validate_versioned_template
print(validate_versioned_template("dssat_workspace_template"))
PY
```

The template must contain the laboratory custom DSSAT 4.8.5 mulch build,
canonical Genotype/StandardData, soil, and weather inventories.

## 2. Build an isolated mutable worker

```bash
PYTHONPATH="$PWD/src" python - <<'PY'
from awm.dssat.runtime_assets import prepare_worker_from_template

report = prepare_worker_from_template(
    "dssat_workspace_template",
    "runtime/w0",
    replace=True,
)
print(report)
PY
```

Required:

- worker paths contain no LRMB/PPO references;
- `DSSATPRO.L48` is generated inside `runtime/w0`;
- DSSATPRO A80 preflight passes;
- `runtime/w0/dscsm048` is executable.

## 3. Freeze a validated AWM COX template

Do not invent a new agronomic experiment file just to make smoke pass. Start
from a known-good rendered COX with the intended cultivar, planting/emergence,
soil initial conditions, fixed nitrogen, mulch settings and simulation
controls.

```bash
PYTHONPATH="$PWD/src" python -m awm.dssat.freeze_template \
  --source-cox /absolute/path/to/known_good_reset.COX \
  --output-template /absolute/path/to/awm_base.COX.in \
  --report /absolute/path/to/awm_base.template.json
```

The default fails if explicit irrigation rows are present. Classify them before
using `--allow-strip-existing-irrigation`.

Once the final AWM COX base is reviewed, it should also be versioned inside AWM
rather than left in an LRMB runtime.

## 4. Reset smoke

Create a run-specific JSON using only:

- `runtime/w0/dscsm048`;
- `runtime/w0/Genotype`;
- `runtime/w0/StandardData`;
- `runtime/w0/data/soil/SOIL.SOL`;
- `runtime/w0/data/wth/...`;
- the frozen AWM COX template.

The daily output inventory must cover all 13 files required by the 74-D DSSAT
state:

```text
PlantGro.OUT
SoilWat.OUT
Weather.OUT
ET.OUT
GHG.OUT
MgmtOps.OUT
Mulch.OUT
N2O.OUT
PlantC.OUT
PlantN.OUT
SoilTemp.OUT
SoilWater.OUT
SoilNi.OUT
```

Run:

```bash
PYTHONPATH="$PWD/src" python -m awm.dssat.smoke \
  --config /absolute/path/to/real_worker_smoke.json
```

Required:

- DSSAT complete-season return code zero;
- planting-day daily state parses;
- Summary.OUT contains HWAM and IRCM;
- no nonterminal observation receives Summary.OUT fields.

## 5. Full 125-day engineering smoke

Before formal agricultural parameters are locked, use a clearly labelled
engineering-only one-event config to exercise:

```text
observation
→ requested irrigation
→ water-budget projection
→ COX write
→ custom DSSAT rerun
→ output refresh
→ terminal IRCM reconciliation
```

Run:

```bash
PYTHONPATH="$PWD/src" python -m awm.baselines.smoke \
  --config /absolute/path/to/one_event_smoke.json \
  --audit-output /absolute/path/to/one_event_smoke.audit.json
```

Do not accept the run unless:

- all 125 decisions complete;
- the intended positive event triggers exactly one management write/rerun;
- no-op days do not rerun DSSAT;
- action DAP follows the frozen `policy_day d -> management day d+1` semantics;
- terminal irrigation accounting passes.

## 6. Agricultural baseline smoke

Only after formal baseline parameters are preregistered, run:

1. Local Conventional Irrigation;
2. Potential-ET Water Balance;
3. Root-zone REW Threshold.

All three must use the same AWM-owned worker assets and the same hard
irrigation-system constraints.

## 7. Archive reproducibility evidence

Archive:

- exact AWM commit SHA;
- `dssat_workspace_template/ASSET_MANIFEST.json`;
- custom `dscsm048` SHA256;
- generated worker `DSSATPRO.L48`;
- frozen AWM COX + hash;
- soil/genotype/weather hashes;
- smoke JSONs;
- action audit JSONs;
- stdout/stderr;
- HWAM, IRCM, IWP;
- IRCM reconciliation.

A successful run that depends on any `/home/.../lrmb/...` runtime path does not
count as an AWM isolation pass.
