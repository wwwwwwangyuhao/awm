"""Offline tests for the reanalysis weather pipeline and split naming."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_all_null_detection() -> None:
    module = load_module(
        "crawl_weather_reanalysis_v3",
        ROOT / "crawl_weather_reanalysis_v3.py",
    )
    payload = {
        "daily": {
            "time": ["2000-01-01"],
            **{
                name: [1.0]
                for name in module.DAILY_VARIABLES
            },
        },
        "hourly": {
            "time": ["2000-01-01T00:00"],
            **{
                name: [1.0]
                for name in module.HOURLY_VARIABLES
            },
        },
    }
    payload["daily"]["precipitation_sum"] = [None]

    try:
        module._validate_api_payload(
            payload,
            2000,
            "era5_land",
        )
    except ValueError as exc:
        assert "contains no valid values" in str(exc)
    else:
        raise AssertionError(
            "All-null variable was not rejected"
        )


def build_synthetic_weather() -> pd.DataFrame:
    dates = pd.date_range(
        "2000-01-01",
        "2025-12-31",
        freq="D",
    )
    day = dates.dayofyear.to_numpy()
    return pd.DataFrame(
        {
            "date": dates,
            "temperature_2m_max": (
                25
                + 10
                * np.sin(2 * np.pi * day / 365.25)
            ),
            "temperature_2m_min": (
                10
                + 8
                * np.sin(2 * np.pi * day / 365.25)
            ),
            "temperature_2m_mean": (
                17
                + 9
                * np.sin(2 * np.pi * day / 365.25)
            ),
            "shortwave_radiation_sum": 18.0,
            "precipitation_sum": 0.1,
            "wind_speed_10m_max": 7.0,
            "et0_fao_evapotranspiration": (
                3.0 + 0.03 * (dates.year - 2000)
            ),
            "wind_speed_10m_mean": 5.0,
            "relative_humidity_2m_mean": 45.0,
            "vapour_pressure_deficit_mean": 1.5,
            "year": dates.year,
            "source_model": "era5",
        }
    )


def test_processor_and_scheduler() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        input_csv = temporary / "era5_daily_2000_2025.csv"
        output_dir = temporary / "era5"
        build_synthetic_weather().to_csv(
            input_csv,
            index=False,
        )

        subprocess.run(
            [
                sys.executable,
                str(
                    ROOT
                    / "process_wth_reanalysis_v5_scope_names.py"
                ),
                str(input_csv),
                "--expected-source-model",
                "era5",
                "--output-dir",
                str(output_dir),
                "--input-wind-unit",
                "ms",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        wth = output_dir / "WTH" / "XJHX0001.WTH"
        lines = wth.read_text(
            encoding="utf-8"
        ).splitlines()
        header_index = next(
            index
            for index, line in enumerate(lines)
            if line.strip().upper().startswith("@YRDAY")
        )
        first_daily = lines[header_index + 1].split()
        assert float(first_daily[-1]) == 432.0

        manifest_path = (
            output_dir
            / "weather_scenarios_2000_2018.json"
        )
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        splits = manifest["splits"]

        assert splits["training_weather_years"] == list(
            range(2000, 2019)
        )
        assert splits["reanalysis_evaluation_years"] == list(
            range(2019, 2023)
        )
        assert splits["station_evaluation_years"] == [
            2023,
            2024,
            2025,
        ]
        assert splits["station_overlap_years"] == [
            2023,
            2024,
            2025,
        ]

        for deprecated_key in (
            "train_years",
            "validation_years",
            "holdout_test_years",
        ):
            assert deprecated_key not in splits

        scheduler_module = load_module(
            "weather_year_scheduler_scope_test",
            ROOT
            / "weather_year_scheduler_reanalysis_eval_v7_scope_names.py",
        )
        scheduler = scheduler_module.WeatherYearScheduler(
            manifest_path,
            seed=21,
        )

        assert scheduler.training_years() == tuple(
            str(year)
            for year in range(2000, 2019)
        )
        assert scheduler.reanalysis_evaluation_years() == (
            "2019",
            "2020",
            "2021",
            "2022",
        )
        assert scheduler.station_evaluation_years() == (
            "2023",
            "2024",
            "2025",
        )

        assignments = scheduler.sample_episode(0)
        assert len(assignments) == 4
        assert all(
            assignment["year"]
            in scheduler.training_years()
            for assignment in assignments
        )


def main() -> None:
    test_all_null_detection()
    test_processor_and_scheduler()
    print("PASS: all-null fields rejected early")
    print("PASS: canonical JSON split names")
    print("PASS: deprecated JSON split keys removed")
    print("PASS: scheduler trains only on 2000-2018")
    print("PASS: evaluation years remain outside training pools")
    print("PASS: 5 m/s -> 432 km/day")


if __name__ == "__main__":
    main()
