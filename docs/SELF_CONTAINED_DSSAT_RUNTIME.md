# Self-contained DSSAT runtime for AWM

## Goal

A fresh clone of `wwwwwwangyuhao/awm` contains every static DSSAT asset required
by the AWM cotton environment. LRMB is not a runtime dependency.

The immutable, version-controlled simulator template lives in:

```text
dssat_workspace_template/
```

DSSAT must never run directly in that directory.

## Simulator identity

The AWM study deliberately keeps the laboratory executable already used by the
cotton experiments:

- DSSAT base: 4.8.5
- executable: `dscsm048`
- AWM build label: `lab-dssat-4.8.5-mulch`
- source LRMB V3 Git blob SHA-1:
  `37c88710d0518d4e20a02881e884652973f559c3`
- source size: 14,603,784 bytes

This executable is not an unmodified official DSSAT 4.8.5 binary. It is the
laboratory build based on DSSAT 4.8.5 with mulch functionality required by the
experiment COX setup. Exact source-patch/compiler provenance remains a separate
reproducibility item until archived.

## Why mutable workers do not live under the checkout

DSSAT v4.8.x reads `DSSATPRO.L48` through legacy fixed-width A80 records. If a
worker inherits an arbitrarily long Git checkout/worktree path, a valid checkout
name can make an L48 record exceed 80 ASCII bytes.

Therefore AWM separates:

```text
Git checkout
  dssat_workspace_template/     immutable, version controlled

~/.dssat_rt/awm/<10hex>/        mutable, generated, short path
  project.json
  .workspace.lock
  w/
    p0e0/
    p0e1/
    ...
  e/
  archives/
```

The 10-hex namespace is:

```text
sha256(resolved_project_root)[:10]
```

Renaming or moving a checkout intentionally creates a new namespace. Reusing
the same resolved checkout path reuses the same namespace.

## LRMB coexistence

LRMB already uses `~/.dssat_rt/<hash>/...`. AWM deliberately uses the child
namespace:

```text
~/.dssat_rt/awm/<hash>/...
```

AWM also keeps its own:

```text
~/.dssat_rt/awm/registry.json
~/.dssat_rt/awm/.registry.lock
```

It does not read or write LRMB's root-level `~/.dssat_rt/registry.json`, worker
roots, archives, or locks. Sharing the top-level `.dssat_rt` directory is only a
filesystem grouping convention; the mutable runtime namespaces are isolated.

The AWM runtime base can be overridden explicitly with:

```bash
export AWM_DSSAT_RUNTIME_BASE=/short/custom/path
```

The A80 preflight remains mandatory even with the short-path layout.

## Canonical worker creation

New AWM code should use the high-level project-aware API:

```python
from pathlib import Path
from awm.dssat import prepare_project_worker

project_root = Path.cwd().resolve()
report = prepare_project_worker(
    project_root / "dssat_workspace_template",
    project_root=project_root,
    policy_idx=0,
    env_idx=0,
    replace=True,
)
print(report["workspace"])
```

For a checkout such as:

```text
/home/wangyh24/awm_some_very_long_experiment_name
```

the actual DSSAT worker is still similar to:

```text
/home/wangyh24/.dssat_rt/awm/12ab34cd56/w/p0e0
```

The checkout name itself is not written into worker `DSSATPRO.L48` records.

`prepare_worker_from_template(template, workspace)` remains available only as a
low-level explicit-path primitive for tests and special tooling. Canonical AWM
training/evaluation code should use `prepare_project_worker`.

## Runtime registry

Calling the project-aware runtime API creates:

```text
~/.dssat_rt/awm/registry.json
```

Each record maps a 10-character runtime ID back to the resolved AWM checkout and
records worker/evaluation/archive roots. Updates are protected by a Linux
`flock` on `.registry.lock`.

Each runtime root also contains:

```text
project.json
```

for direct human inspection.

## Process lock

`WorkspaceRootLock` provides an exclusive lock for one checkout's hashed DSSAT
runtime. A future training launcher must hold this lock for the full training
process, not only during worker creation, so two accidental launches of the same
checkout cannot overwrite the same `p<policy>e<env>` workers.

Example:

```python
from awm.dssat import WorkspaceRootLock

with WorkspaceRootLock(project_root=project_root):
    # create workers and keep the lock while the run owns them
    ...
```

## Worker naming

Training workers use compact algorithm-neutral names:

```text
p<policy_idx>e<env_idx>
```

Examples:

```text
p0e0
p0e1
p1e0
```

This keeps L48 paths short and does not encode PPO, RCWA-RL, or any other
algorithm into simulator infrastructure.

## A80 safety

Short paths are the primary design constraint, but every generated worker still
runs `validate_dssatpro_record_width()` before DSSAT execution. Every meaningful
profile record must be ASCII and no longer than 80 bytes.

Thus the runtime contract is:

```text
short path by construction
+
A80 validation before execution
```

## Asset provenance

`dssat_workspace_template/ASSET_MANIFEST.json` records SHA256 and Git-blob SHA-1
for the vendored simulator assets, the simulator build identity, weather
inventory, and excluded historical files.

The historical absolute-path `DSSATPRO.L48` is kept only as provenance. AWM
workers always generate a fresh worker-local profile.

## Zero-LRMB acceptance test

A complete isolation check is:

1. fresh clone AWM to any checkout name;
2. do not clone LRMB;
3. call `prepare_project_worker`;
4. verify the worker is under `~/.dssat_rt/awm/<hash>/w/...`;
5. verify worker `DSSATPRO.L48` contains no `lrmb` or `ppo` path;
6. pass unit tests;
7. pass real DSSAT reset smoke;
8. pass the complete 125-day one-event irrigation smoke and IRCM reconciliation.
