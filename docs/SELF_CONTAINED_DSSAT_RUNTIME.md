# Self-contained DSSAT runtime for AWM

## Goal

After the one-time asset migration is committed, a fresh clone of
`wwwwwwangyuhao/awm` must contain every DSSAT asset required by the AWM cotton
environment. LRMB is then no longer a runtime dependency.

The versioned assets live in:

```text
dssat_workspace_template/
```

Mutable episode workers live in an ignored runtime directory such as:

```text
runtime/w0/
```

Never run DSSAT directly in the versioned template.

## Simulator identity

The AWM study deliberately keeps the laboratory executable already used by the
cotton experiments:

- DSSAT base: 4.8.5
- executable: `dscsm048`
- AWM build label: `lab-dssat-4.8.5-mulch`
- source LRMB V3 Git blob SHA-1:
  `37c88710d0518d4e20a02881e884652973f559c3`
- source size: 14,603,784 bytes

This executable is **not** an unmodified official DSSAT 4.8.5 binary. It is the
laboratory build based on DSSAT 4.8.5 with the mulch functionality required by
the experiment COX setup. The exact source patch/compiler provenance remains a
separate reproducibility item and is recorded as TBD until archived.

## One-time migration

The migration utility is stored in AWM itself:

```bash
cd /home/wangyh24/awm_step4

PYTHONPATH="$PWD/src" \
python scripts/import_dssat_assets.py \
  --source-repo /home/wangyh24/lrmb
```

The command reads exactly:

```text
wwwwwwangyuhao/lrmb
commit d56336e09fdb9a9aea60ae61eaa892833314ab33
path dssat_workspace_template/
```

by `git archive`.

It does **not** checkout that commit, switch the LRMB branch, reset LRMB, create
an LRMB worktree, modify LRMB files, or create an LRMB commit.

The script validates the custom executable's Git blob identity and size before
copying anything.

## Canonical asset subset

The AWM template includes:

- `dscsm048`;
- root CDE files needed by the executable;
- canonical cotton `COGRO048.CUL`, `COGRO048.ECO`, `COGRO048.SPE`;
- `StandardData/CO2048.WDA` and `FERCH048.SDA`;
- DSSAT CDE data files;
- `data/soil/SOIL.SOL`;
- ERA5 weather `XJHX0001.WTH` through `XJHX2501.WTH`;
- station weather for 2023, 2024 and 2025.

The following historical/reference assets are deliberately not part of the
canonical runtime:

- `Genotype/COGRO048_别人的参数.CUL`;
- `data/wth/legacy_import/`;
- `DSSATPRO_完整版.L48`.

The old absolute-path `DSSATPRO.L48` is kept only under
`provenance/source_DSSATPRO.L48`; it is never used to run AWM.

## Provenance

The migration writes:

```text
dssat_workspace_template/ASSET_MANIFEST.json
```

with SHA256 and Git-blob SHA-1 for every copied file, the fixed LRMB source
commit, the simulator build label, weather inventory, and excluded historical
items.

After migration, review:

```bash
python -m json.tool dssat_workspace_template/ASSET_MANIFEST.json | less
```

Then run:

```bash
PYTHONPATH="$PWD/src" python -m pytest
```

If correct, commit only to the AWM branch:

```bash
git add dssat_workspace_template
git commit -m "assets: vendor custom DSSAT 4.8.5 mulch runtime"
git push
```

From that commit onward, a fresh AWM clone no longer needs LRMB to prepare a
worker.

## Worker creation

AWM creates a writable worker from the immutable template:

```python
from awm.dssat.runtime_assets import prepare_worker_from_template

prepare_worker_from_template(
    "dssat_workspace_template",
    "runtime/w0",
    replace=True,
)
```

This:

1. copies the AWM-owned template;
2. makes the custom executable executable;
3. creates writable DSSAT episode directories;
4. generates a worker-local `DSSATPRO.L48`;
5. runs the A80 fixed-width preflight.

The generated profile contains only AWM worker paths. It must never reference
`lrmb`, `ppo`, or any other historical repository.

## Zero-LRMB acceptance test

After the migration commit is pushed, the strongest isolation check is:

1. fresh-clone AWM to a new directory;
2. do not clone LRMB there;
3. prepare a worker from `dssat_workspace_template`;
4. pass unit tests;
5. pass real DSSAT reset smoke;
6. pass a complete 125-day one-event irrigation smoke and IRCM reconciliation.

Only after that should AWM be considered fully detached from LRMB.
