"""
Compare 2023--2025 fixed-source reanalysis daily weather with station data.

Canonical station columns:
    date, tmax_c, tmin_c, srad_mj_m2_day,
    precipitation_mm, wind_mean_ms, et0_mm

ET0 is optional in the station file. The script records the actual
source_model and rejects mislabeled files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE_COLUMNS = {
    "tmax_c": "temperature_2m_max",
    "tmin_c": "temperature_2m_min",
    "srad_mj_m2_day": "shortwave_radiation_sum",
    "precipitation_mm": "precipitation_sum",
    "wind_mean_ms": "wind_speed_10m_mean",
    "et0_mm": "et0_fao_evapotranspiration",
}


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported table format: {path}")


def _load_column_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {
            "date": "date",
            **{name: name for name in SOURCE_COLUMNS},
        }
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("column-map JSON must contain an object")
    return {str(key): str(item) for key, item in value.items()}


def _prepare_station(
    path: Path,
    column_map: dict[str, str],
) -> pd.DataFrame:
    source = _read_table(path)
    required_canonical = {
        "date",
        "tmax_c",
        "tmin_c",
        "srad_mj_m2_day",
        "precipitation_mm",
        "wind_mean_ms",
    }
    missing_mappings = required_canonical - set(column_map)
    if missing_mappings:
        raise KeyError(
            f"Column map misses: {sorted(missing_mappings)}"
        )

    missing_source = [
        source_name
        for canonical, source_name in column_map.items()
        if canonical != "et0_mm" and source_name not in source.columns
    ]
    if missing_source:
        raise KeyError(
            f"Station data misses columns: {missing_source}"
        )

    rename = {
        source_name: canonical
        for canonical, source_name in column_map.items()
        if source_name in source.columns
    }
    station = source.rename(columns=rename).copy()
    station["date"] = pd.to_datetime(
        station["date"], errors="raise"
    )
    station = station[
        station["date"].dt.year.between(2023, 2025)
    ].copy()

    if station["date"].duplicated().any():
        raise ValueError("Station data contains duplicate dates")
    return station


def _prepare_reanalysis(
    path: Path,
    expected_source_model: str,
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(
        frame["date"], errors="raise"
    )
    if "source_model" not in frame.columns:
        raise KeyError("Reanalysis data has no source_model column")

    actual = (
        frame["source_model"]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    if not (actual == expected_source_model).all():
        raise ValueError(
            f"Expected source_model={expected_source_model!r}, "
            f"found {sorted(actual.unique().tolist())}"
        )

    frame = frame[
        frame["date"].dt.year.between(2023, 2025)
    ].copy()
    rename = {
        source_column: f"reanalysis_{canonical}"
        for canonical, source_column in SOURCE_COLUMNS.items()
        if source_column in frame.columns
    }
    return frame[["date", *rename.keys()]].rename(columns=rename)


def _metric_row(
    station: pd.Series,
    reanalysis: pd.Series,
    variable: str,
    period: str,
    source_model: str,
) -> dict:
    pair = pd.concat([station, reanalysis], axis=1).dropna()
    pair.columns = ["station", "reanalysis"]

    if pair.empty:
        return {
            "source_model": source_model,
            "period": period,
            "variable": variable,
            "n": 0,
            "bias_reanalysis_minus_station": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
            "pearson_r": np.nan,
        }

    error = pair["reanalysis"] - pair["station"]
    return {
        "source_model": source_model,
        "period": period,
        "variable": variable,
        "n": int(len(pair)),
        "bias_reanalysis_minus_station": float(error.mean()),
        "mae": float(error.abs().mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "pearson_r": (
            float(pair["station"].corr(pair["reanalysis"]))
            if len(pair) > 1
            else np.nan
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reanalysis_csv", type=Path)
    parser.add_argument("station_file", type=Path)
    parser.add_argument(
        "--expected-source-model",
        choices=("era5", "era5_seamless", "era5_land"),
        default="era5",
    )
    parser.add_argument(
        "--column-map-json", type=Path, default=None
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("weather/overlap_comparison"),
    )
    args = parser.parse_args()

    station = _prepare_station(
        args.station_file,
        _load_column_map(args.column_map_json),
    )
    reanalysis = _prepare_reanalysis(
        args.reanalysis_csv,
        args.expected_source_model,
    )
    merged = station.merge(
        reanalysis,
        on="date",
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("No overlapping 2023--2025 dates")

    variables = [
        canonical
        for canonical in SOURCE_COLUMNS
        if canonical in merged.columns
        and f"reanalysis_{canonical}" in merged.columns
    ]

    metric_rows = []
    for variable in variables:
        metric_rows.append(
            _metric_row(
                merged[variable],
                merged[f"reanalysis_{variable}"],
                variable,
                "2023-2025_all",
                args.expected_source_model,
            )
        )
        for year, year_data in merged.groupby(
            merged["date"].dt.year
        ):
            metric_rows.append(
                _metric_row(
                    year_data[variable],
                    year_data[f"reanalysis_{variable}"],
                    variable,
                    str(int(year)),
                    args.expected_source_model,
                )
            )

    annual_rows = []
    for year, year_data in merged.groupby(
        merged["date"].dt.year
    ):
        annual = {
            "source_model": args.expected_source_model,
            "year": int(year),
            "n_days": int(len(year_data)),
        }
        for variable in variables:
            aggregation = (
                "sum"
                if variable
                in {
                    "srad_mj_m2_day",
                    "precipitation_mm",
                    "et0_mm",
                }
                else "mean"
            )
            annual[
                f"station_{variable}_{aggregation}"
            ] = float(getattr(year_data[variable], aggregation)())
            annual[
                f"reanalysis_{variable}_{aggregation}"
            ] = float(
                getattr(
                    year_data[f"reanalysis_{variable}"],
                    aggregation,
                )()
            )
        annual_rows.append(annual)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(
        args.output_dir
        / "reanalysis_station_daily_overlap.csv",
        index=False,
        encoding="utf-8",
    )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(
        args.output_dir / "reanalysis_station_metrics.csv",
        index=False,
        encoding="utf-8",
    )
    pd.DataFrame(annual_rows).to_csv(
        args.output_dir
        / "reanalysis_station_annual_summary.csv",
        index=False,
        encoding="utf-8",
    )
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
