# AWM versioned DSSAT workspace template

This directory is part of the AWM repository and is the immutable source for
mutable DSSAT workers.

## Simulator identity

- DSSAT base version: 4.8.5
- build label: `lab-dssat-4.8.5-mulch`
- executable: `dscsm048`
- source executable Git blob: `37c88710d0518d4e20a02881e884652973f559c3`
- source executable size: `14603784` bytes

The executable is the laboratory-custom DSSAT 4.8.5 build used with the
experiment's mulch-enabled COX configuration. It must not be described as an
unmodified official DSSAT 4.8.5 binary.

`ASSET_MANIFEST.json` records SHA256 and Git-blob hashes for every migrated
asset.

## Runtime rule

Never run episodes directly in this versioned directory. Build a mutable worker
with `awm.dssat.runtime_assets.prepare_worker_from_template(...)`; mutable
workers belong under ignored `runtime/` or another explicitly chosen short
runtime root.

The historical source `DSSATPRO.L48` is retained only under
`provenance/source_DSSATPRO.L48`. AWM renders a new worker-local profile so no
runtime path depends on LRMB.
