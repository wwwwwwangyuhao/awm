"""
将明确指定来源的日尺度再分析天气数据转换为 DSSAT .WTH 文件，
并构建 2000–2018 年训练天气的 ET0 情景分组。

关键修正
--------
1. Open-Meteo 下载脚本使用 wind_speed_unit=ms，因此输入风速默认为 m/s。
2. DSSAT WTH 文件中的 WIND 单位必须为 km/day。
3. 写入前执行：
       WIND_km_day = wind_m_s * 86.4
4. 不再把 m/s 或 km/h 原值直接写入 WTH。
5. 风速列与输入单位必须显式指定，不允许静默猜测。
6. 生成 WTH 后进行回读校验，确认日期和风速转换正确。

主天气分组
----------
- 在水肥决策时域内累计 FAO-56 ET0；
- 只使用 2000–2018 年训练年份；
- 按等频三分组划分 low / medium / high ET0。

辅助诊断
--------
- 生育期累计降水；
- 气候水分亏缺 ET0 - precipitation；
- 高温日数；
- 累计太阳辐射；
- 平均 VPD；
- 平均风速。

本脚本只负责历史天气驱动与天气情景分组，不把未来 WTH 行作为智能体天气预报。
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Literal

import numpy as np
import pandas as pd


M_PER_S_TO_KM_PER_DAY: Final[float] = 86.4
KM_PER_HOUR_TO_KM_PER_DAY: Final[float] = 24.0

WindInputUnit = Literal["ms", "kmh", "kmday"]


@dataclass(frozen=True)
class SplitConfig:
    training_weather_start: int = 2000
    training_weather_end: int = 2018
    reanalysis_evaluation_start: int = 2019
    reanalysis_evaluation_end: int = 2022
    station_evaluation_start: int = 2023
    station_evaluation_end: int = 2025
    planting_doy: int = 119
    decision_days: int = 125
    hot_day_threshold_c: float = 35.0


@dataclass(frozen=True)
class WTHConfig:
    station: str = "XJHX"
    latitude: float = 44.223
    longitude: float = 87.305
    elevation_m: float = -99.0
    reference_height_m: float = 2.0
    wind_height_m: float = 10.0
    wind_source_column: str = "wind_speed_10m_mean"
    input_wind_unit: WindInputUnit = "ms"


REQUIRED_BASE_COLUMNS: Final[set[str]] = {
    "date",
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "shortwave_radiation_sum",
    "precipitation_sum",
    "et0_fao_evapotranspiration",
    "source_model",
}


def _validate_station_code(station: str) -> str:
    station = station.strip().upper()
    if len(station) != 4 or not station.isalnum():
        raise ValueError(
            "DSSAT station code must contain exactly four letters/digits; "
            f"received {station!r}."
        )
    return station


def _convert_wind_to_km_day(
    values: pd.Series,
    input_unit: WindInputUnit,
) -> pd.Series:
    """将输入风速统一转换为 DSSAT WTH 所需的 km/day。"""
    numeric = pd.to_numeric(values, errors="raise").astype(float)

    if input_unit == "ms":
        converted = numeric * M_PER_S_TO_KM_PER_DAY
    elif input_unit == "kmh":
        converted = numeric * KM_PER_HOUR_TO_KM_PER_DAY
    elif input_unit == "kmday":
        converted = numeric.copy()
    else:
        raise ValueError(f"Unsupported wind input unit: {input_unit!r}")

    if (~np.isfinite(converted)).any():
        bad = converted[~np.isfinite(converted)]
        raise ValueError(
            "Converted wind contains non-finite values at rows: "
            f"{bad.index[:10].tolist()}"
        )
    if (converted < 0).any():
        bad = converted[converted < 0]
        raise ValueError(
            "Wind speed cannot be negative. Bad rows: "
            f"{bad.index[:10].tolist()}"
        )
    return converted


def _validate_complete_daily_coverage(frame: pd.DataFrame) -> None:
    """确认输入从最小日期到最大日期每天连续且无重复。"""
    if frame.empty:
        raise ValueError("Weather CSV is empty.")

    if frame["date"].duplicated().any():
        duplicated = (
            frame.loc[frame["date"].duplicated(), "date"]
            .dt.strftime("%Y-%m-%d")
            .tolist()
        )
        raise ValueError(
            f"Weather CSV contains duplicate dates: {duplicated[:10]}"
        )

    expected = pd.date_range(
        frame["date"].min(),
        frame["date"].max(),
        freq="D",
    )
    observed = pd.DatetimeIndex(frame["date"])
    missing = expected.difference(observed)
    if len(missing):
        raise ValueError(
            "Weather CSV has missing dates, first missing values: "
            f"{missing[:10].strftime('%Y-%m-%d').tolist()}"
        )

    for year, year_data in frame.groupby(frame["date"].dt.year):
        expected_days = 366 if pd.Timestamp(year=int(year), month=12, day=31).is_leap_year else 365
        if len(year_data) != expected_days:
            raise ValueError(
                f"{int(year)} has {len(year_data)} rows; "
                f"expected {expected_days} complete daily rows."
            )


def _validate_physical_ranges(frame: pd.DataFrame) -> None:
    numeric_columns = [
        "temperature_2m_max",
        "temperature_2m_min",
        "temperature_2m_mean",
        "shortwave_radiation_sum",
        "precipitation_sum",
        "et0_fao_evapotranspiration",
        "wind_km_day",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if frame[column].isna().any():
            raise ValueError(f"{column} contains missing values.")
        if (~np.isfinite(frame[column])).any():
            raise ValueError(f"{column} contains non-finite values.")

    nonnegative_columns = [
        "shortwave_radiation_sum",
        "precipitation_sum",
        "et0_fao_evapotranspiration",
        "wind_km_day",
    ]
    for column in nonnegative_columns:
        if (frame[column] < 0).any():
            rows = frame.index[frame[column] < 0][:10].tolist()
            raise ValueError(
                f"{column} contains negative values at rows {rows}."
            )

    invalid_temperature = (
        frame["temperature_2m_max"]
        < frame["temperature_2m_min"]
    )
    if invalid_temperature.any():
        rows = frame.index[invalid_temperature][:10].tolist()
        raise ValueError(
            "TMAX is lower than TMIN at rows "
            f"{rows}."
        )


def _load_weather(
    path: Path,
    wind_source_column: str,
    input_wind_unit: WindInputUnit,
    expected_source_model: str,
) -> pd.DataFrame:
    frame = pd.read_csv(path)

    required = REQUIRED_BASE_COLUMNS | {wind_source_column}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(
            "Weather CSV is missing required columns: "
            f"{sorted(missing)}"
        )

    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame = frame.sort_values("date").reset_index(drop=True)

    source_model = (
        frame["source_model"]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    expected_source_model = expected_source_model.strip().lower()
    if not (source_model == expected_source_model).all():
        unexpected = sorted(source_model.unique().tolist())
        raise ValueError(
            f"Expected source_model={expected_source_model!r}, "
            f"found {unexpected}"
        )

    _validate_complete_daily_coverage(frame)

    frame["wind_source_value"] = pd.to_numeric(
        frame[wind_source_column],
        errors="raise",
    )
    frame["wind_km_day"] = _convert_wind_to_km_day(
        frame["wind_source_value"],
        input_wind_unit,
    )

    _validate_physical_ranges(frame)
    return frame


def _annual_tav_amp(
    year_data: pd.DataFrame,
) -> tuple[float, float]:
    """计算 DSSAT WTH 头部的年均温 TAV 和月均温振幅 AMP。"""
    tav = float(year_data["temperature_2m_mean"].mean())
    monthly_mean = (
        year_data.set_index("date")["temperature_2m_mean"]
        .resample("MS")
        .mean()
    )
    if len(monthly_mean) != 12:
        raise ValueError(
            "Cannot calculate AMP because the year does not contain "
            "12 monthly temperature means."
        )
    amp = float(monthly_mean.max() - monthly_mean.min())
    return tav, amp


def _format_wth_line(
    yrdoy: str,
    srad: float,
    tmax: float,
    tmin: float,
    rain: float,
    wind_km_day: float,
) -> str:
    # WIND may be several hundred km/day, so allocate enough width.
    return (
        f"{yrdoy:>5} "
        f"{srad:5.1f} "
        f"{tmax:6.1f} "
        f"{tmin:6.1f} "
        f"{rain:6.1f} "
        f"{wind_km_day:7.1f}\n"
    )


def _write_wth(
    year_data: pd.DataFrame,
    path: Path,
    config: WTHConfig,
    source_model: str,
) -> None:
    year_values = year_data["date"].dt.year.unique()
    if len(year_values) != 1:
        raise ValueError(
            f"WTH writer received multiple years: {year_values.tolist()}"
        )

    year = int(year_values[0])
    year_short = f"{year % 100:02d}"
    tav, amp = _annual_tav_amp(year_data)

    with path.open("w", encoding="utf-8", newline="") as file:
        file.write(
            f"*WEATHER : {config.station} "
            f"{source_model.upper()} (WIND in km/day)\n\n"
        )
        file.write(
            "@ INSI      LAT     LONG  ELEV   TAV   AMP REFHT WNDHT\n"
        )
        file.write(
            f"  {config.station:<4} "
            f"{config.latitude:8.3f} "
            f"{config.longitude:8.3f} "
            f"{config.elevation_m:5.0f} "
            f"{tav:5.1f} "
            f"{amp:5.1f} "
            f"{config.reference_height_m:5.1f} "
            f"{config.wind_height_m:5.1f}\n"
        )
        file.write("@YRDAY SRAD   TMAX   TMIN   RAIN    WIND\n")

        for row in year_data.itertuples(index=False):
            yrdoy = f"{year_short}{row.date.dayofyear:03d}"
            file.write(
                _format_wth_line(
                    yrdoy=yrdoy,
                    srad=float(row.shortwave_radiation_sum),
                    tmax=float(row.temperature_2m_max),
                    tmin=float(row.temperature_2m_min),
                    rain=float(row.precipitation_sum),
                    wind_km_day=float(row.wind_km_day),
                )
            )


def _read_wth_daily_table(path: Path) -> pd.DataFrame:
    """读取 @YRDAY 表头之后的逐日天气行。"""
    rows: list[dict[str, float | str]] = []
    in_daily_section = False

    with path.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()

            if not line:
                continue

            if line.upper().startswith("@YRDAY"):
                in_daily_section = True
                continue

            if not in_daily_section:
                continue

            if line.startswith(("*", "@", "!")):
                continue

            parts = line.split()
            if len(parts) != 6:
                raise ValueError(
                    f"{path}: expected six daily WTH columns, "
                    f"found {len(parts)} in line {line!r}."
                )

            rows.append(
                {
                    "yrdoy": parts[0],
                    "srad": float(parts[1]),
                    "tmax": float(parts[2]),
                    "tmin": float(parts[3]),
                    "rain": float(parts[4]),
                    "wind_km_day": float(parts[5]),
                }
            )

    if not in_daily_section:
        raise ValueError(f"{path}: @YRDAY header was not found.")
    if not rows:
        raise ValueError(f"{path}: no daily WTH rows found.")

    return pd.DataFrame(rows)


def _validate_written_wth(
    year_data: pd.DataFrame,
    path: Path,
) -> None:
    """回读 WTH，验证日期数量及风速单位转换结果。"""
    written = _read_wth_daily_table(path).reset_index(drop=True)
    expected = year_data.reset_index(drop=True)

    if len(written) != len(expected):
        raise ValueError(
            f"{path}: wrote {len(written)} daily rows; "
            f"expected {len(expected)}."
        )

    expected_yrdoy = expected["date"].dt.strftime("%y%j")
    if not written["yrdoy"].equals(expected_yrdoy):
        mismatch = np.flatnonzero(
            written["yrdoy"].to_numpy()
            != expected_yrdoy.to_numpy()
        )
        raise ValueError(
            f"{path}: YRDAY mismatch at rows "
            f"{mismatch[:10].tolist()}."
        )

    # WTH stores one decimal, so compare after the same rounding.
    expected_wind = expected["wind_km_day"].round(1).to_numpy()
    actual_wind = written["wind_km_day"].to_numpy()
    if not np.allclose(
        actual_wind,
        expected_wind,
        rtol=0.0,
        atol=0.05,
    ):
        mismatch = np.flatnonzero(
            ~np.isclose(
                actual_wind,
                expected_wind,
                rtol=0.0,
                atol=0.05,
            )
        )
        raise ValueError(
            f"{path}: WIND conversion mismatch at rows "
            f"{mismatch[:10].tolist()}."
        )


def _decision_horizon(
    frame: pd.DataFrame,
    year: int,
    config: SplitConfig,
) -> pd.DataFrame:
    yearly = frame[frame["date"].dt.year == year].copy()

    start = (
        pd.Timestamp(year=year, month=1, day=1)
        + pd.Timedelta(days=config.planting_doy - 1)
    )
    end = start + pd.Timedelta(
        days=config.decision_days - 1
    )

    selected = yearly[
        (yearly["date"] >= start)
        & (yearly["date"] <= end)
    ].copy()

    if len(selected) != config.decision_days:
        raise ValueError(
            f"{year}: expected {config.decision_days} decision days "
            f"from {start.date()} to {end.date()}, "
            f"found {len(selected)}."
        )

    return selected


def _optional_mean(
    frame: pd.DataFrame,
    column: str,
) -> float:
    if column not in frame.columns:
        return math.nan
    return float(
        pd.to_numeric(
            frame[column],
            errors="coerce",
        ).mean()
    )


def _build_year_features(
    frame: pd.DataFrame,
    config: SplitConfig,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []

    for year in sorted(frame["date"].dt.year.unique()):
        season = _decision_horizon(
            frame,
            int(year),
            config,
        )

        et0 = float(
            season["et0_fao_evapotranspiration"].sum()
        )
        precipitation = float(
            season["precipitation_sum"].sum()
        )

        rows.append(
            {
                "year": int(year),
                "season_start": (
                    season["date"].min().date().isoformat()
                ),
                "season_end": (
                    season["date"].max().date().isoformat()
                ),
                "season_et0_mm": et0,
                "season_precipitation_mm": precipitation,
                "season_climatic_water_deficit_mm": (
                    et0 - precipitation
                ),
                "season_tmax_mean_c": float(
                    season["temperature_2m_max"].mean()
                ),
                "season_tmin_mean_c": float(
                    season["temperature_2m_min"].mean()
                ),
                "hot_days_ge_35c": int(
                    (
                        season["temperature_2m_max"]
                        >= config.hot_day_threshold_c
                    ).sum()
                ),
                "season_shortwave_mj_m2": float(
                    season["shortwave_radiation_sum"].sum()
                ),
                "season_vpd_mean_kpa": _optional_mean(
                    season,
                    "vapour_pressure_deficit_mean",
                ),
                "season_wind_source_mean": float(
                    season["wind_source_value"].mean()
                ),
                "season_wind_mean_km_day": float(
                    season["wind_km_day"].mean()
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("year")
        .reset_index(drop=True)
    )


def _equal_frequency_thirds(
    train_features: pd.DataFrame,
    value_column: str,
    labels: tuple[str, str, str],
) -> tuple[pd.DataFrame, dict[str, float]]:
    ordered = (
        train_features
        .sort_values([value_column, "year"])
        .reset_index(drop=True)
        .copy()
    )

    groups = np.array_split(
        np.arange(len(ordered)),
        3,
    )
    ordered["group"] = ""

    for indices, label in zip(groups, labels):
        ordered.loc[indices, "group"] = label

    low_values = ordered.loc[
        ordered["group"] == labels[0],
        value_column,
    ]
    medium_values = ordered.loc[
        ordered["group"] == labels[1],
        value_column,
    ]

    thresholds = {
        "low_max": float(low_values.max()),
        "medium_max": float(medium_values.max()),
    }
    return ordered, thresholds


def _build_manifest(
    features: pd.DataFrame,
    split_config: SplitConfig,
    wth_config: WTHConfig,
    source_model: str,
) -> tuple[pd.DataFrame, dict]:
    train = features[
        features["year"].between(
            split_config.training_weather_start,
            split_config.training_weather_end,
        )
    ].copy()

    expected_training_years = (
        split_config.training_weather_end
        - split_config.training_weather_start
        + 1
    )
    if len(train) != expected_training_years:
        raise ValueError(
            f"Expected {expected_training_years} training years, "
            f"found {len(train)}."
        )

    grouped_et0, et0_thresholds = _equal_frequency_thirds(
        train,
        "season_et0_mm",
        ("low_et0", "medium_et0", "high_et0"),
    )
    grouped_cwd, cwd_thresholds = _equal_frequency_thirds(
        train,
        "season_climatic_water_deficit_mm",
        ("low_cwd", "medium_cwd", "high_cwd"),
    )

    result_features = features.merge(
        grouped_et0[["year", "group"]].rename(
            columns={"group": "et0_group"}
        ),
        on="year",
        how="left",
    )
    result_features = result_features.merge(
        grouped_cwd[["year", "group"]].rename(
            columns={"group": "cwd_group"}
        ),
        on="year",
        how="left",
    )

    follower_pools = {
        label: sorted(
            grouped_et0.loc[
                grouped_et0["group"] == label,
                "year",
            ]
            .astype(int)
            .tolist()
        )
        for label in (
            "low_et0",
            "medium_et0",
            "high_et0",
        )
    }

    manifest = {
        "version": "weather_scenarios_v5_scope_names",
        "source_model": source_model,
        "wth": {
            "station": wth_config.station,
            "latitude": wth_config.latitude,
            "longitude": wth_config.longitude,
            "elevation_m": wth_config.elevation_m,
            "wind_source_column": (
                wth_config.wind_source_column
            ),
            "input_wind_unit": (
                wth_config.input_wind_unit
            ),
            "dssat_wind_unit": "km/day",
            "wind_conversion": {
                "ms_to_km_day_multiplier": (
                    M_PER_S_TO_KM_PER_DAY
                ),
                "kmh_to_km_day_multiplier": (
                    KM_PER_HOUR_TO_KM_PER_DAY
                ),
            },
        },
        "decision_horizon": {
            "planting_doy": split_config.planting_doy,
            "decision_days": split_config.decision_days,
        },
        "splits": {
            "training_weather_years": list(
                range(
                    split_config.training_weather_start,
                    split_config.training_weather_end + 1,
                )
            ),
            "reanalysis_evaluation_years": list(
                range(
                    split_config.reanalysis_evaluation_start,
                    split_config.reanalysis_evaluation_end + 1,
                )
            ),
            "station_evaluation_years": list(
                range(
                    split_config.station_evaluation_start,
                    split_config.station_evaluation_end + 1,
                )
            ),
            "station_overlap_years": list(
                range(
                    split_config.station_evaluation_start,
                    split_config.station_evaluation_end + 1,
                )
            ),
        },
        "main_grouping": {
            "variable": "season_et0_mm",
            "method": (
                "equal_frequency_thirds_"
                "training_years_only"
            ),
            "thresholds_mm": et0_thresholds,
            "follower_pools": follower_pools,
        },
        "sensitivity_grouping": {
            "variable": (
                "season_climatic_water_deficit_mm"
            ),
            "method": (
                "equal_frequency_thirds_"
                "training_years_only"
            ),
            "thresholds_mm": cwd_thresholds,
            "groups": {
                label: sorted(
                    grouped_cwd.loc[
                        grouped_cwd["group"] == label,
                        "year",
                    ]
                    .astype(int)
                    .tolist()
                )
                for label in (
                    "low_cwd",
                    "medium_cwd",
                    "high_cwd",
                )
            },
        },
        "secondary_diagnostics": [
            "season_precipitation_mm",
            "hot_days_ge_35c",
            "season_shortwave_mj_m2",
            "season_vpd_mean_kpa",
            "season_wind_source_mean",
            "season_wind_mean_km_day",
        ],
        "policy_sampling": {
            "leader": (
                "balanced shuffled cycle over all training years"
            ),
            "follower_1": (
                "balanced shuffled cycle over low_et0"
            ),
            "follower_2": (
                "balanced shuffled cycle over medium_et0"
            ),
            "follower_3": (
                "balanced shuffled cycle over high_et0"
            ),
        },
    }

    return result_features, manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert fixed-source daily reanalysis weather to DSSAT "
            "WTH with correct WIND units and build ET0 groups."
        )
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help=(
            "CSV generated by crawl_weather_reanalysis_v3.py."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: weather/<expected-source-model>",
    )
    parser.add_argument(
        "--expected-source-model",
        choices=("era5", "era5_seamless", "era5_land"),
        default="era5",
    )
    parser.add_argument(
        "--station",
        default="XJHX",
    )
    parser.add_argument(
        "--latitude",
        type=float,
        default=44.223,
    )
    parser.add_argument(
        "--longitude",
        type=float,
        default=87.305,
    )
    parser.add_argument(
        "--elevation",
        type=float,
        default=-99.0,
    )
    parser.add_argument(
        "--reference-height",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--wind-height",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--wind-column",
        default="wind_speed_10m_mean",
        help=(
            "Input wind column. Use daily mean wind for DSSAT. "
            "Do not silently substitute daily maximum wind."
        ),
    )
    parser.add_argument(
        "--input-wind-unit",
        choices=("ms", "kmh", "kmday"),
        default="ms",
        help=(
            "Unit of --wind-column. crawl_weather_reanalysis_v3.py "
            "requests m/s, so the default is ms."
        ),
    )
    parser.add_argument(
        "--planting-doy",
        type=int,
        default=119,
    )
    parser.add_argument(
        "--decision-days",
        type=int,
        default=125,
    )
    parser.add_argument(
        "--hot-day-threshold",
        type=float,
        default=35.0,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    station = _validate_station_code(args.station)

    split_config = SplitConfig(
        planting_doy=args.planting_doy,
        decision_days=args.decision_days,
        hot_day_threshold_c=args.hot_day_threshold,
    )
    wth_config = WTHConfig(
        station=station,
        latitude=args.latitude,
        longitude=args.longitude,
        elevation_m=args.elevation,
        reference_height_m=args.reference_height,
        wind_height_m=args.wind_height,
        wind_source_column=args.wind_column,
        input_wind_unit=args.input_wind_unit,
    )

    frame = _load_weather(
        args.input_csv,
        wind_source_column=wth_config.wind_source_column,
        input_wind_unit=wth_config.input_wind_unit,
        expected_source_model=args.expected_source_model,
    )

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else Path("weather") / args.expected_source_model
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    wth_dir = output_dir / "WTH"
    wth_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    generated_wth_files: list[str] = []

    for year, year_data in frame.groupby(
        frame["date"].dt.year,
        sort=True,
    ):
        year_int = int(year)
        path = (
            wth_dir
            / f"{station}{year_int % 100:02d}01.WTH"
        )
        year_data = year_data.copy()
        _write_wth(
            year_data=year_data,
            path=path,
            config=wth_config,
            source_model=args.expected_source_model,
        )
        _validate_written_wth(
            year_data=year_data,
            path=path,
        )
        generated_wth_files.append(str(path))

    features = _build_year_features(
        frame,
        split_config,
    )
    features, manifest = _build_manifest(
        features,
        split_config,
        wth_config,
        args.expected_source_model,
    )

    features_path = (
        output_dir
        / "weather_year_features.csv"
    )
    manifest_path = (
        output_dir
        / "weather_scenarios_2000_2018.json"
    )
    conversion_report_path = (
        output_dir
        / "wth_conversion_report.json"
    )

    features.to_csv(
        features_path,
        index=False,
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    conversion_report = {
        "input_csv": str(args.input_csv.resolve()),
        "output_directory": str(
            output_dir.resolve()
        ),
        "source_model": args.expected_source_model,
        "wind_source_column": (
            wth_config.wind_source_column
        ),
        "input_wind_unit": (
            wth_config.input_wind_unit
        ),
        "dssat_wind_unit": "km/day",
        "wind_conversion": {
            "ms_to_km_day": "value * 86.4",
            "kmh_to_km_day": "value * 24.0",
            "kmday_to_km_day": "value",
        },
        "input_date_min": (
            frame["date"].min().date().isoformat()
        ),
        "input_date_max": (
            frame["date"].max().date().isoformat()
        ),
        "input_row_count": int(len(frame)),
        "generated_wth_count": len(
            generated_wth_files
        ),
        "generated_wth_files": generated_wth_files,
        "validation": {
            "daily_coverage": "PASS",
            "source_model": "PASS",
            "physical_ranges": "PASS",
            "wth_round_trip": "PASS",
        },
        "split_config": asdict(split_config),
        "wth_config": asdict(wth_config),
    }
    conversion_report_path.write_text(
        json.dumps(
            conversion_report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Reanalysis to DSSAT WTH conversion completed.")
    print(
        f"Input wind: {wth_config.wind_source_column} "
        f"[{wth_config.input_wind_unit}]"
    )
    print("DSSAT WTH WIND unit: km/day")
    print(
        "Applied wind conversion: "
        + (
            "m/s × 86.4"
            if wth_config.input_wind_unit == "ms"
            else (
                "km/h × 24.0"
                if wth_config.input_wind_unit == "kmh"
                else "no conversion"
            )
        )
    )
    print(f"Generated WTH directory: {wth_dir}")
    print(f"Annual features: {features_path}")
    print(f"Scenario manifest: {manifest_path}")
    print(f"Conversion report: {conversion_report_path}")
    print(
        json.dumps(
            manifest["main_grouping"][
                "follower_pools"
            ],
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
