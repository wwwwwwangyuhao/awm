# Engineering real-DSSAT reset smoke

This gate validates the AWM-owned laboratory custom DSSAT 4.8.5 mulch build
before any formal agricultural baseline or learned policy is run.

The candidate in this branch is intentionally **not** a paper protocol. It is a
V3-derived engineering fixture used only to prove that the simulator binary,
hashed runtime, mulch-aware COX syntax, canonical daily outputs, and output
reader work together on a real server.

## Candidate

- weather: ERA5 `XJHX0001.WTH` (year 2000);
- soil identifier: `XJHX0001`;
- cultivar: `IB0007`;
- planting: `00119`;
- emergence: `00133`;
- mulch: `PMALB=0.12`, `PMWD=22.5`;
- explicit policy irrigation: none;
- AWM irrigation marker: exactly one.

Source provenance is recorded in:

```text
run/real_smoke/v3_2000_engineering.provenance.json
```

Known scientific blockers are deliberately retained and documented:

1. historical descriptive site metadata says JIANGDU/Jiangsu while the actual
   field/weather/soil identifiers are XJHX;
2. the no-fertilizer-file fallback in the historical generator writes an
   explicit `FAMN=0` row, so this is not the formal fixed-N AWM protocol;
3. automatic irrigation/nitrogen parameter rows are present and their effective
   behavior must be determined from the real DSSAT result;
4. historical initial soil water/mineral-N values remain candidate provenance.

Do not use this fixture for calibration, baseline tuning, RL training, model
selection, or paper-result generation.

## 1. Audit the COX candidate

From the checkout root:

```bash
PYTHONPATH="$PWD/src" python -m awm.dssat.cox_audit \
  run/real_smoke/v3_2000_engineering.COX.in \
  --output run/real_smoke/v3_2000_engineering.audit.json
```

Expected structural result:

```text
structural_status = passed
marker_count = 1
explicit_irrigation_rows = []
protocol_ready = false
```

`protocol_ready=false` is intentional. The review flags are scientific
information, not a software failure.

## 2. Run the real reset smoke

```bash
PYTHONPATH="$PWD/src" python -m awm.dssat.smoke \
  --config configs/engineering_reset_smoke_v3_2000.json \
  2>&1 | tee run/real_smoke/v3_2000_reset.log
```

The command automatically creates and runs in:

```text
~/.dssat_rt/awm/<sha256(project_root)[:10]>/w/p0e0
```

It must not run DSSAT from the Git checkout and must not reference LRMB.

## 3. Technical acceptance criteria

The reset smoke must report:

```text
status = passed
workspace_preflight.status = passed
daily_output_file_count = 13
summary_has_hwam = true
summary_has_ircm = true
policy_irrigation_event_count = 0
policy_irrigation_mm = 0.0
```

All thirteen daily output files must exist and be non-empty:

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

The returned `workspace` must be under `~/.dssat_rt/awm/`.

## 4. Irrigation gate

Because this reset contains zero AWM policy irrigation, inspect:

```text
IRCM_mm
nonpolicy_irrigation_detected
```

If:

```text
IRCM_mm == 0
nonpolicy_irrigation_detected == false
```

then the candidate has not introduced seasonal irrigation during the zero-policy
reset and it is acceptable to proceed to a one-event engineering smoke.

If `IRCM_mm > 0`, stop. Do not compensate by changing the AWM water ledger or by
silently setting `nonpolicy_irrigation_mm`. First determine whether DSSAT's
historical automatic-irrigation controls are active and classify every source of
irrigation.

## 5. Yield and nitrogen interpretation

`HWAM_kg_ha`, `NICM`, and `ETCM` are returned as diagnostics. Their values do
not validate the agronomic protocol. In particular, this candidate does not yet
contain the formal fixed-N schedule required by the AWM paper.

A technically successful reset smoke therefore proves only:

```text
custom DSSAT binary
+ mulch-aware COX
+ AWM-owned assets
+ hashed worker
+ 13 OUT files
+ Summary.OUT parsing
```

It does not prove calibration or agronomic suitability.

## 6. Next gate

Proceed to the one-event 125-day engineering smoke only after:

- the reset smoke technically passes;
- all 13 outputs are present and non-empty;
- zero-policy `IRCM` has been classified;
- no hidden LRMB path is present in the worker.
