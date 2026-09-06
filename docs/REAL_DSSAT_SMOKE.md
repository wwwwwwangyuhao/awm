# Real DSSAT worker and agricultural-baseline smoke runbook

This runbook assumes the self-contained AWM DSSAT assets have already been
committed under:

```text
dssat_workspace_template/
```

LRMB is not a runtime dependency.

## 1. Canonical runtime model

Version-controlled assets remain inside the AWM checkout, but DSSAT never runs
directly in the checkout. Mutable workers are created under the short hashed
runtime:

```text
~/.dssat_rt/awm/<sha256(resolved_project_root)[:10]>/w/p0e0/
```

The AWM runtime has its own registry and lock files under
`~/.dssat_rt/awm/`; it does not share LRMB worker directories, registry state,
or locks.

Each real smoke run:

1. discovers the AWM project root from the config path unless `--project-root`
   is supplied;
2. acquires the per-checkout `WorkspaceRootLock`;
3. copies `dssat_workspace_template/` into the hashed worker;
4. generates a worker-local `DSSATPRO.L48`;
5. enforces the DSSAT A80 fixed-width preflight;
6. executes DSSAT only inside the hashed worker;
7. releases the lock when the full smoke run exits.

## 2. Verify the immutable template

From the AWM checkout:

```bash
PYTHONPATH="$PWD/src" python - <<'PY'
from awm.dssat.runtime_assets import validate_versioned_template
print(validate_versioned_template("dssat_workspace_template"))
PY
```

The template must contain the laboratory custom DSSAT 4.8.5 mulch build,
canonical Genotype/StandardData, soil, and weather inventories.

Do not manually create `runtime/w0` in the checkout. `prepare_project_worker()`
is the canonical worker allocator.

## 3. Freeze a validated AWM COX template

Do not invent a new agronomic experiment file just to make smoke pass. Start
from a known-good rendered COX with the intended cultivar, planting/emergence,
soil initial conditions, fixed nitrogen, mulch settings, and simulation
controls.

The frozen template should live inside the AWM checkout, for example:

```text
run/real_smoke/awm_base.COX.in
```

Freeze it with:

```bash
PYTHONPATH="$PWD/src" python -m awm.dssat.freeze_template \
  --source-cox /absolute/path/to/known_good_reset.COX \
  --output-template "$PWD/run/real_smoke/awm_base.COX.in" \
  --report "$PWD/run/real_smoke/awm_base.template.json"
```

The default fails if explicit irrigation rows are present. Classify them before
using `--allow-strip-existing-irrigation`.

## 4. Portable real-smoke config

Canonical smoke JSONs do **not** contain machine-specific mutable runtime paths.
The following legacy keys are rejected:

```text
workspace
dssat_exec
output_dir
rendered_cox
summary_out
daily_out_files
episode_artifacts
```

Instead configure only portable inputs, including:

```json
{
  "runtime": {
    "policy_idx": 0,
    "env_idx": 0,
    "replace_worker": true
  },
  "cox_template": "run/real_smoke/awm_base.COX.in",
  "rendered_cox_name": "AWM_SMOKE.COX",
  "summary_out_name": "Summary.OUT",
  "weather_source": "era5",
  "weather_filename": "XJHX0001.WTH",
  "soil_relative_path": "data/soil/SOIL.SOL",
  "plant_yrdoy": "00119"
}
```

`cox_template` must be project-relative. Weather and soil are resolved from the
copied AWM worker itself, not from LRMB or another checkout.

The canonical daily output inventory is fixed to all 13 files required by the
74-D DSSAT state:

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

If `daily_out_names` is supplied, it must exactly match this inventory.

## 5. Reset smoke

Run:

```bash
PYTHONPATH="$PWD/src" python -m awm.dssat.smoke \
  --config "$PWD/run/real_smoke/reset_smoke.json"
```

Normally no `--project-root` or `--runtime-base` is needed. The CLI discovers
the checkout and uses `~/.dssat_rt/awm` automatically.

Required output includes:

```text
runtime_family = awm
runtime_id = <10 hex>
runtime_root = ~/.dssat_rt/awm/<10 hex>
workspace = ~/.dssat_rt/awm/<10 hex>/w/p0e0
```

Acceptance criteria:

- DSSAT complete-season return code zero;
- A80 preflight passes;
- planting-day daily state parses;
- Summary.OUT contains HWAM and IRCM;
- no nonterminal observation receives Summary.OUT fields;
- no generated `DSSATPRO.L48` contains the checkout name, LRMB, or PPO paths.

## 6. Full 125-day engineering smoke

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
  --config "$PWD/run/real_smoke/one_event_smoke.json" \
  --audit-output "$PWD/run/real_smoke/one_event_smoke.audit.json"
```

The baseline smoke uses the same hashed worker allocation and holds the same
per-checkout workspace lock for the entire 125-day episode.

Do not accept the run unless:

- all 125 decisions complete;
- the intended positive event triggers exactly one management write/rerun;
- no-op days do not rerun DSSAT;
- action DAP follows the frozen `policy_day d -> management day d+1` semantics;
- terminal irrigation accounting passes.

## 7. Agricultural baseline smoke

Only after formal baseline parameters are preregistered, run:

1. Local Conventional Irrigation;
2. Potential-ET Water Balance;
3. Root-zone REW Threshold.

All three use the same AWM-owned template, hashed runtime allocator, and hard
irrigation-system constraints.

## 8. Runtime inspection

List the AWM runtime registry:

```bash
python -m json.tool "$HOME/.dssat_rt/awm/registry.json"
```

For the current checkout, print its namespace and worker path:

```bash
PYTHONPATH="$PWD/src" python - <<'PY'
from pathlib import Path
from awm.dssat import runtime_namespace_for_project, worker_workspace_for_project
root = Path.cwd().resolve()
print(runtime_namespace_for_project(root))
print(worker_workspace_for_project(root, policy_idx=0, env_idx=0))
PY
```

## 9. Archive reproducibility evidence

Archive:

- exact AWM commit SHA;
- runtime ID and generated worker path;
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

A successful run that depends on any `/home/.../lrmb/...` path or on a mutable
worker inside the Git checkout does not count as the canonical AWM isolation
pass.
