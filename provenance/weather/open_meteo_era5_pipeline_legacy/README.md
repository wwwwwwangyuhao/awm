# Recovered Open-Meteo / ERA5 weather pipeline

This directory preserves the exact Python source recovered from the user-provided archive `wth_process.zip` on 2026-09-06. The source is archived here so the weather provenance cannot be lost again.

## Source archive integrity

Original ZIP SHA256:

```text
db150fda6d2b9f87314ef12584eac6e520ef7fd34f36b17346bd8d4e0714a768  wth_process.zip
```

Recovered source-file SHA256 values:

```text
dee9b775ba65df110386cb0eaca6f75955f24989dcb83852fffbbc991d840ed5  crawl_weather_reanalysis_v3.py
db0fe8712526b4350b1c0e1fc79c7044c7be8dfa525f293d0185df625d2d9d19  process_wth_reanalysis_v4.py
3cf13164630b26bc452df72d17a74e0c932ed49bc4acd26be892cdaaf78dd05d  process_wth_reanalysis_v5_scope_names.py
58c395669f4590c1d7a2fd11d358409b57258d26f12680bb67e747065d5e2b18  compare_reanalysis_station_v3.py
83db5b678830843d9793ad9b4ccc83aac0ec5e246660a8068d0823ddf582ba97  test_reanalysis_weather_pipeline_v2_scope_names.py
1d961586f17abbda9a5ff925b968df242ef7eab2a41530533cc671f67ac9c28a  weather_year_scheduler_reanalysis_eval_v7_scope_names.py
```

`__pycache__/` and `.pyc` files from the ZIP are deliberately excluded because they are generated artifacts, not source provenance.

## Recovered data-source contract

The downloader explicitly uses the Open-Meteo Historical Weather API:

```text
https://archive-api.open-meteo.com/v1/archive
```

The default main-paper configuration in the recovered downloader is:

```text
model=era5
latitude=44.223
longitude=87.305
start_year=2000
end_year=2025
timezone=Asia/Urumqi
wind_speed_unit=ms
precipitation_unit=mm
```

The downloader requests daily:

```text
temperature_2m_max
temperature_2m_min
temperature_2m_mean
shortwave_radiation_sum
precipitation_sum
wind_speed_10m_max
et0_fao_evapotranspiration
```

and hourly:

```text
wind_speed_10m
relative_humidity_2m
vapour_pressure_deficit
```

Hourly wind, relative humidity, and VPD are aggregated to daily means.

The recovered processor explicitly converts the requested Open-Meteo wind speed from m/s to the DSSAT WTH convention km/day:

```text
WIND_km_day = wind_m_s * 86.4
```

It writes DSSAT daily rows with:

```text
@YRDAY SRAD TMAX TMIN RAIN WIND
```

and validates complete daily coverage, source-model consistency, temperature ordering, non-negative finite values, WTH date coverage, and wind conversion after round-trip parsing.

## Important status

These files are an **unaltered provenance archive**, not the canonical AWM weather implementation.

In particular, the recovered `process_wth_reanalysis_v5_scope_names.py` and scheduler use the historical split:

```text
training:              2000-2018
reanalysis evaluation: 2019-2022
station evaluation:    2023-2025
```

The current AWM protocol decision is different:

```text
training:              2000-2017
reanalysis validation: 2018-2022
station final test:    2023-2025
```

Therefore this archived scheduler/split must not silently become the AWM experiment split. A later canonical AWM weather module should reuse the verified downloader and conversion semantics while implementing the current protocol explicitly.

## Local source check performed before archival

The recovered source passed:

```text
python -m py_compile *.py
python test_reanalysis_weather_pipeline_v2_scope_names.py
```

with:

```text
PASS: all-null fields rejected early
PASS: canonical JSON split names
PASS: deprecated JSON split keys removed
PASS: scheduler trains only on 2000-2018
PASS: evaluation years remain outside training pools
PASS: 5 m/s -> 432 km/day
```

No API keys, tokens, passwords, authorization headers, or embedded proxy credentials were found in the recovered Python source.
