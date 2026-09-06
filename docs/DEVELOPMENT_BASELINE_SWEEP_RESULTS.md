# Canonical development agricultural-baseline sweep results

## Status and provenance

This document records the completed **development-weather** reference sweep for
`awm-agricultural-baselines-v1`. It is descriptive evidence for later protocol
design; it is **not** a final-test result and does not define `Y_ref`, `eta`, or
`alpha`.

Execution provenance:

- Git branch: `feat/formal-agricultural-baselines-v1`
- sweep execution commit: `e6c30a16008d62f172ca76bd749b0ef06fb3bb03`
- GitHub Actions run: `34026087638`
- GitHub Actions job: `101467190714`
- artifact: `awm-development-baseline-sweep-v1`
- artifact id: `9987133710`
- artifact size: `843292` bytes
- artifact digest: `sha256:d790703819205747c3a595583e37ad8569dd80ad5125682dd730bd1974b6e080`
- formal source COX SHA-256:
  `7984ea2ae684e8eb8e97919c5821f01fc35da3ff917ca7e9e267e52b8b30c274`

The artifact contains `manifest.json`, `results.jsonl`, `results.csv`,
`summary.json`, and the 207 complete episode audit JSON files.

## Hard integrity result

The canonical matrix is:

```text
23 ERA5 development years (2000-2022)
× 3 methods (conventional, ET, REW)
× 3 water treatments (W100, W80, W60)
= 207 episodes
```

The post-run validator reported:

- rows: **207 / 207**;
- unique `(year, method, treatment)` keys: **207 / 207**;
- missing keys: **0**;
- duplicate keys: **0**;
- unexpected keys: **0**;
- irrigation-accounting pass: **207 / 207**;
- train years: **18** (`2000-2017`);
- validation years: **5** (`2018-2022`);
- locked final-test years `2023, 2024, 2025`: **absent**;
- integrity problems: **none**.

No 2023-2025 station-weather result was generated or inspected by this sweep.

## Descriptive results — training-development weather (2000-2017)

Values are means across 18 ERA5 years. Yield ranges are shown in brackets.

| Method | Water | HWAM kg/ha, mean [min,max] | IRCM mm, mean [min,max] | IWP kg/m3 mean | Events mean |
|---|---|---:|---:|---:|---:|
| Conventional | W100 | 7087.9 [6149,8030] | 540.0 [540,540] | 1.313 | 11.00 |
| Conventional | W80 | 7096.0 [6156,7811] | 432.0 [432,432] | 1.643 | 11.00 |
| Conventional | W60 | 5767.2 [5030,6253] | 324.0 [324,324] | 1.780 | 11.00 |
| ET | W100 | 6204.4 [5246,6982] | 540.0 [540,540] | 1.149 | 11.00 |
| ET | W80 | 3690.7 [3010,4684] | 432.0 [432,432] | 0.854 | 9.00 |
| ET | W60 | 1912.3 [1450,2335] | 324.0 [324,324] | 0.590 | 7.00 |
| REW | W100 | 6965.9 [6089,7655] | 367.5 [315,405] | 1.897 | 7.17 |
| REW | W80 | 6965.9 [6089,7655] | 367.5 [315,405] | 1.897 | 7.17 |
| REW | W60 | 6449.2 [5899,6959] | 323.5 [315,324] | 1.993 | 6.94 |

## Descriptive results — validation-development weather (2018-2022)

Values are means across 5 ERA5 years.

| Method | Water | HWAM kg/ha, mean [min,max] | IRCM mm, mean [min,max] | IWP kg/m3 mean | Events mean |
|---|---|---:|---:|---:|---:|
| Conventional | W100 | 7558.0 [7102,8005] | 540.0 [540,540] | 1.400 | 11.00 |
| Conventional | W80 | 7284.4 [6849,7523] | 432.0 [432,432] | 1.686 | 11.00 |
| Conventional | W60 | 5592.6 [4666,6089] | 324.0 [324,324] | 1.726 | 11.00 |
| ET | W100 | 5729.8 [5064,6168] | 540.0 [540,540] | 1.061 | 11.00 |
| ET | W80 | 3426.2 [3166,4020] | 432.0 [432,432] | 0.793 | 9.00 |
| ET | W60 | 1670.0 [1298,1941] | 324.0 [324,324] | 0.515 | 7.00 |
| REW | W100 | 7285.2 [6677,7720] | 387.0 [360,405] | 1.884 | 7.60 |
| REW | W80 | 7285.2 [6677,7720] | 387.0 [360,405] | 1.884 | 7.60 |
| REW | W60 | 6341.4 [6127,6580] | 324.0 [324,324] | 1.957 | 7.00 |

## Structural observations from the development sweep

These observations describe the executed baseline rules; they are not claims
about learned-policy superiority or final-test performance.

### Conventional W80 can match W100 yield in some development years

The training-development means are 7096.0 kg/ha for W80 and 7087.9 kg/ha for
W100. This small reversal is a DSSAT/weather response under the frozen timing
rule; it must not be interpreted as a universal statement that 432 mm is better
than 540 mm. The per-year distributions, rather than only the means, must be
retained for risk analysis.

### REW does not use the W80/W100 quota in these years

For both train and validation development weather, REW W80 and REW W100 are
identical year-by-year because the threshold controller stops before either
quota binds. Executed total seasonal irrigation ranges from 315 to 405 mm in
training and 360 to 405 mm in validation, below the W80 cap of 432 mm.

This is expected threshold-rule behavior, not a water-accounting error.

### W60 binds the REW controller

REW W60 is effectively quota-limited in most development years:

- train mean IRCM: 323.5 mm, with range 315-324 mm;
- validation IRCM: exactly 324 mm in all five years.

Its baseline-level audit records quota-driven constraint adjustments while the
adapter-level projection remains separate.

### The frozen ET rule is aggressive early and quota-limited

ET uses the full total treatment cap in all development years:

- W100: 540 mm;
- W80: 432 mm;
- W60: 324 mm.

The three-layer audit shows many baseline-level adjustments because the ET
ledger can continue to desire replenishment above the 45-mm event limit and/or
after little seasonal quota remains. This is an intended audit distinction,
not an adapter/DSSAT accounting failure.

## Conventional W100 development yield trace — not `Y_ref`

The 23-year W100 conventional trace is retained because it is a useful input to
later reference-yield design, but this table does **not** define the reference.

| Year | Split | HWAM kg/ha |
|---:|---|---:|
| 2000 | train | 6831 |
| 2001 | train | 6722 |
| 2002 | train | 6149 |
| 2003 | train | 6306 |
| 2004 | train | 6310 |
| 2005 | train | 7689 |
| 2006 | train | 7485 |
| 2007 | train | 7202 |
| 2008 | train | 8030 |
| 2009 | train | 6153 |
| 2010 | train | 7330 |
| 2011 | train | 7134 |
| 2012 | train | 7634 |
| 2013 | train | 7303 |
| 2014 | train | 7060 |
| 2015 | train | 7220 |
| 2016 | train | 7474 |
| 2017 | train | 7551 |
| 2018 | validation | 7416 |
| 2019 | validation | 7102 |
| 2020 | validation | 7705 |
| 2021 | validation | 7562 |
| 2022 | validation | 8005 |

Summary only:

- training-development conventional W100 mean: **7087.9 kg/ha**;
- training-development median: **7211 kg/ha**;
- training-development range: **6149-8030 kg/ha**;
- validation-development mean: **7558.0 kg/ha**;
- validation-development median: **7562 kg/ha**;
- validation-development range: **7102-8005 kg/ha**.

Again, none of these scalars has been promoted to `Y_ref`.

## Boundary for the next scientific step

The completed 207-episode sweep supplies the development-only empirical input
needed to design and freeze the risk contract. The next stage must decide,
without accessing 2023-2025 station outcomes:

1. the formal definition of `Y_ref` (scalar versus weather-conditional reference);
2. the protection multiplier `eta`;
3. the lower-tail probability/risk level `alpha` and exact LCVaR convention;
4. how those choices are estimated/evaluated from training versus validation
   weather without leakage.

Until that stage is explicitly frozen, `Y_ref`, `eta`, and `alpha` remain TBD.
