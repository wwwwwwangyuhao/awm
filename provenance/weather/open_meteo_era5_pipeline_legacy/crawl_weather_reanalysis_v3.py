"""
Download reproducible daily reanalysis weather through the Open-Meteo
Historical Weather API.

Recommended main-paper configuration:
    --model era5

Supported model selections:
- era5: one internally consistent ERA5 source for the complete DSSAT weather
  variable set used by this pipeline.
- era5_seamless: Open-Meteo mixed product. Do not describe it as pure
  ERA5-Land.
- era5_land: accepted only for diagnostics; the Open-Meteo strict selection
  does not expose the complete variable combination used here and will fail
  early when arrays are null.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import aiohttp
import numpy as np
import pandas as pd


LOGGER = logging.getLogger("reanalysis_weather_downloader")
API_URL: Final[str] = "https://archive-api.open-meteo.com/v1/archive"
SUPPORTED_MODELS: Final[tuple[str, ...]] = (
    "era5",
    "era5_seamless",
    "era5_land",
)

DAILY_VARIABLES: Final[list[str]] = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "shortwave_radiation_sum",
    "precipitation_sum",
    "wind_speed_10m_max",
    "et0_fao_evapotranspiration",
]

HOURLY_VARIABLES: Final[list[str]] = [
    "wind_speed_10m",
    "relative_humidity_2m",
    "vapour_pressure_deficit",
]


@dataclass(frozen=True)
class DownloadConfig:
    latitude: float = 44.223
    longitude: float = 87.305
    start_year: int = 2000
    end_year: int = 2025
    timezone_name: str = "Asia/Urumqi"
    model: str = "era5"
    wind_speed_unit: str = "ms"
    max_concurrent: int = 3
    max_retries: int = 5
    timeout_seconds: int = 120


def _validate_config(config: DownloadConfig) -> None:
    if config.model not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model={config.model!r}; choose one of "
            f"{SUPPORTED_MODELS}"
        )
    if config.start_year > config.end_year:
        raise ValueError("start_year cannot be later than end_year")
    if config.max_concurrent < 1:
        raise ValueError("max_concurrent must be at least 1")
    if config.max_retries < 1:
        raise ValueError("max_retries must be at least 1")


def _validate_variable_array(
    *,
    year: int,
    section_name: str,
    variable: str,
    values: Any,
    expected_length: int,
) -> None:
    if not isinstance(values, list):
        raise ValueError(
            f"{year}: {section_name}.{variable} is not a list"
        )
    if len(values) != expected_length:
        raise ValueError(
            f"{year}: {section_name}.{variable} length={len(values)}, "
            f"expected={expected_length}"
        )
    if expected_length == 0:
        raise ValueError(f"{year}: {section_name}.{variable} is empty")

    valid_count = sum(value is not None for value in values)
    if valid_count == 0:
        raise ValueError(
            f"{year}: {section_name}.{variable} contains no valid values. "
            "The selected Open-Meteo model does not provide this variable "
            "for the requested DSSAT pipeline."
        )


def _validate_api_payload(
    payload: dict[str, Any],
    year: int,
    model: str,
) -> None:
    if payload.get("error"):
        raise RuntimeError(
            f"{year}: API error for model={model}: "
            f"{payload.get('reason', payload)}"
        )

    for section_name, variables in (
        ("daily", DAILY_VARIABLES),
        ("hourly", HOURLY_VARIABLES),
    ):
        section = payload.get(section_name)
        if not isinstance(section, dict):
            raise ValueError(
                f"{year}: API response has no {section_name} section"
            )

        times = section.get("time")
        if not isinstance(times, list) or not times:
            raise ValueError(
                f"{year}: {section_name}.time is missing or empty"
            )

        missing = [
            variable for variable in variables if variable not in section
        ]
        if missing:
            raise ValueError(
                f"{year}: missing {section_name} variables: {missing}"
            )

        for variable in variables:
            _validate_variable_array(
                year=year,
                section_name=section_name,
                variable=variable,
                values=section[variable],
                expected_length=len(times),
            )


async def _fetch_year(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    config: DownloadConfig,
    year: int,
) -> dict[str, Any]:
    params = {
        "latitude": config.latitude,
        "longitude": config.longitude,
        "start_date": f"{year}-01-01",
        "end_date": f"{year}-12-31",
        "daily": ",".join(DAILY_VARIABLES),
        "hourly": ",".join(HOURLY_VARIABLES),
        "models": config.model,
        "timezone": config.timezone_name,
        "wind_speed_unit": config.wind_speed_unit,
        "precipitation_unit": "mm",
    }

    async with semaphore:
        last_error: Exception | None = None
        for attempt in range(1, config.max_retries + 1):
            try:
                async with session.get(API_URL, params=params) as response:
                    body = await response.text()
                    if response.status != 200:
                        raise RuntimeError(
                            f"HTTP {response.status}: {body[:1000]}"
                        )
                    payload = json.loads(body)
                    _validate_api_payload(payload, year, config.model)
                    LOGGER.info(
                        "Downloaded and validated %s using model=%s",
                        year,
                        config.model,
                    )
                    return payload
            except Exception as exc:
                last_error = exc
                if isinstance(exc, ValueError):
                    break
                if attempt == config.max_retries:
                    break
                delay = min(2**attempt, 30)
                LOGGER.warning(
                    "%s attempt %s failed: %s; retry in %ss",
                    year,
                    attempt,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)

        model_hint = ""
        if config.model == "era5_land":
            model_hint = (
                "\nOpen-Meteo strict ERA5-Land does not provide the complete "
                "temperature/precipitation/radiation/wind/ET0 combination "
                "used here. Use --model era5 for the main experiment, "
                "--model era5_seamless for a mixed-source sensitivity run, "
                "or download native ERA5-Land variables directly from CDS."
            )
        raise RuntimeError(
            f"Failed to download {year} with model={config.model}."
            f"{model_hint}"
        ) from last_error


def _payload_to_daily_frame(
    payload: dict[str, Any],
    year: int,
    source_model: str,
) -> pd.DataFrame:
    daily = pd.DataFrame(payload["daily"])
    daily["date"] = pd.to_datetime(daily["time"], errors="raise")
    daily = daily.drop(columns=["time"])

    hourly = pd.DataFrame(payload["hourly"])
    hourly["timestamp"] = pd.to_datetime(
        hourly["time"], errors="raise"
    )
    hourly["date"] = hourly["timestamp"].dt.normalize()

    hourly_daily = (
        hourly.groupby("date", as_index=False)
        .agg(
            wind_speed_10m_mean=("wind_speed_10m", "mean"),
            relative_humidity_2m_mean=(
                "relative_humidity_2m",
                "mean",
            ),
            vapour_pressure_deficit_mean=(
                "vapour_pressure_deficit",
                "mean",
            ),
        )
    )

    frame = daily.merge(
        hourly_daily,
        on="date",
        how="left",
        validate="one_to_one",
    )
    frame["year"] = int(year)
    frame["source_model"] = source_model
    return frame


def _validate_complete_series(
    frame: pd.DataFrame,
    config: DownloadConfig,
) -> None:
    if frame.empty:
        raise ValueError("Downloaded weather frame is empty")

    if frame["date"].duplicated().any():
        duplicated = frame.loc[
            frame["date"].duplicated(), "date"
        ].astype(str).tolist()
        raise ValueError(f"Duplicate dates: {duplicated[:10]}")

    expected = pd.date_range(
        f"{config.start_year}-01-01",
        f"{config.end_year}-12-31",
        freq="D",
    )
    observed = pd.DatetimeIndex(frame["date"])
    missing = expected.difference(observed)
    extra = observed.difference(expected)
    if len(missing) or len(extra):
        raise ValueError(
            f"Date coverage mismatch: missing={len(missing)}, "
            f"extra={len(extra)}"
        )

    numeric_columns = DAILY_VARIABLES + [
        "wind_speed_10m_mean",
        "relative_humidity_2m_mean",
        "vapour_pressure_deficit_mean",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    missing_values = frame[numeric_columns].isna().sum()
    bad = missing_values[missing_values > 0]
    if not bad.empty:
        raise ValueError(
            "Missing weather values after aggregation:\n"
            + bad.to_string()
        )

    nonfinite = {
        column: int((~np.isfinite(frame[column])).sum())
        for column in numeric_columns
        if (~np.isfinite(frame[column])).any()
    }
    if nonfinite:
        raise ValueError(f"Non-finite weather values: {nonfinite}")

    actual_models = set(
        frame["source_model"].astype(str).str.lower()
    )
    if actual_models != {config.model}:
        raise ValueError(
            f"Unexpected source_model values: {sorted(actual_models)}; "
            f"expected only {config.model!r}"
        )


async def download(config: DownloadConfig) -> pd.DataFrame:
    _validate_config(config)
    timeout = aiohttp.ClientTimeout(total=config.timeout_seconds)
    connector = aiohttp.TCPConnector(
        limit=config.max_concurrent,
        limit_per_host=config.max_concurrent,
    )
    semaphore = asyncio.Semaphore(config.max_concurrent)

    async with aiohttp.ClientSession(
        timeout=timeout,
        connector=connector,
    ) as session:
        tasks = [
            _fetch_year(session, semaphore, config, year)
            for year in range(config.start_year, config.end_year + 1)
        ]
        payloads = await asyncio.gather(*tasks)

    frames = [
        _payload_to_daily_frame(payload, year, config.model)
        for year, payload in zip(
            range(config.start_year, config.end_year + 1),
            payloads,
        )
    ]
    result = (
        pd.concat(frames, ignore_index=True)
        .sort_values("date")
        .reset_index(drop=True)
    )
    _validate_complete_series(result, config)
    return result


def _unit_metadata() -> dict[str, str]:
    return {
        "temperature_2m_max": "degC",
        "temperature_2m_min": "degC",
        "temperature_2m_mean": "degC",
        "shortwave_radiation_sum": "MJ/m2/day",
        "precipitation_sum": "mm/day",
        "wind_speed_10m_max": "m/s",
        "et0_fao_evapotranspiration": "mm/day",
        "wind_speed_10m_mean": "m/s",
        "relative_humidity_2m_mean": "%",
        "vapour_pressure_deficit_mean": "kPa",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download complete DSSAT weather variables from a fixed "
            "Open-Meteo reanalysis model."
        )
    )
    parser.add_argument(
        "--model",
        choices=SUPPORTED_MODELS,
        default="era5",
    )
    parser.add_argument("--latitude", type=float, default=44.223)
    parser.add_argument("--longitude", type=float, default=87.305)
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Default: weather/<model>. Do not store ERA5 outputs under "
            "an era5_land directory."
        ),
    )
    parser.add_argument("--max-concurrent", type=int, default=3)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = parse_args()
    config = DownloadConfig(
        latitude=args.latitude,
        longitude=args.longitude,
        start_year=args.start_year,
        end_year=args.end_year,
        model=args.model,
        max_concurrent=args.max_concurrent,
        max_retries=args.max_retries,
        timeout_seconds=args.timeout_seconds,
    )
    _validate_config(config)

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else Path("weather") / config.model
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = asyncio.run(download(config))
    csv_path = output_dir / (
        f"{config.model}_daily_"
        f"{config.start_year}_{config.end_year}.csv"
    )
    metadata_path = output_dir / f"{config.model}_metadata.json"

    frame.to_csv(csv_path, index=False, encoding="utf-8")
    metadata = {
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "api_url": API_URL,
        "config": asdict(config),
        "data_source_statement": (
            "ERA5 reanalysis"
            if config.model == "era5"
            else (
                "Open-Meteo ERA5-Seamless mixed reanalysis"
                if config.model == "era5_seamless"
                else "Open-Meteo strict ERA5-Land selection"
            )
        ),
        "daily_variables": DAILY_VARIABLES,
        "hourly_variables": HOURLY_VARIABLES,
        "row_count": int(len(frame)),
        "date_min": str(frame["date"].min().date()),
        "date_max": str(frame["date"].max().date()),
        "units": _unit_metadata(),
        "validation": {
            "api_fields_present": "PASS",
            "all_null_fields": "PASS",
            "complete_daily_coverage": "PASS",
            "missing_values": "PASS",
            "source_model_consistency": "PASS",
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    LOGGER.info("Saved %s", csv_path)
    LOGGER.info("Saved %s", metadata_path)


if __name__ == "__main__":
    main()
