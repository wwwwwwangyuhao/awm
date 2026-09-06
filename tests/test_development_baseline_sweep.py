import json
from pathlib import Path
import tempfile

import pytest

from awm.baselines.development_sweep import (
    METHODS,
    TREATMENTS,
    development_split,
    prepare_sweep_inputs,
)


ROOT = Path(__file__).resolve().parents[1]


def _work_dir():
    runtime = ROOT / "runtime"
    runtime.mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(dir=runtime)


def test_prepare_one_cross_year_episode_retargets_weather_dates_not_soil():
    with _work_dir() as tmp:
        episodes = prepare_sweep_inputs(
            project_root=ROOT,
            years=(2001,),
            methods=("conventional",),
            treatments=("W100",),
            work_dir=tmp,
        )
        assert len(episodes) == 1
        episode = episodes[0]
        assert episode.year == 2001
        assert episode.split == "train"

        config = json.loads(episode.config_path.read_text(encoding="utf-8"))
        assert config["weather_year"] == 2001
        assert config["weather_split"] == "train"
        assert config["weather_source"] == "era5"
        assert config["weather_filename"] == "XJHX0101.WTH"
        assert config["plant_yrdoy"] == "01119"
        assert config["rendered_cox_name"] == "XJHX0101.COX"
        assert not Path(config["cox_template"]).is_absolute()

        cox = episode.generated_cox_path.read_text(encoding="utf-8")
        field_row = next(
            line
            for line in cox.splitlines()
            if line.strip().startswith("1 XJHX0001 XJHX0101")
        )
        fields = field_row.split()
        assert fields[2] == "XJHX0101"
        assert fields[11] == "XJHX0001"
        assert "01119 01133" in cox
        assert " 1 01116 IR005 45.00" in cox


def test_selected_train_validation_matrix_has_exact_cartesian_size():
    with _work_dir() as tmp:
        years = (2000, 2018, 2022)
        episodes = prepare_sweep_inputs(
            project_root=ROOT,
            years=years,
            work_dir=tmp,
        )
        assert len(episodes) == len(years) * len(METHODS) * len(TREATMENTS)
        assert {episode.split for episode in episodes if episode.year == 2000} == {"train"}
        assert {episode.split for episode in episodes if episode.year >= 2018} == {"validation"}
        assert len({episode.key for episode in episodes}) == len(episodes)


def test_development_split_boundary_is_frozen():
    assert development_split(2017) == "train"
    assert development_split(2018) == "validation"
    assert development_split(2022) == "validation"
    with pytest.raises(ValueError):
        development_split(2023)


def test_sweep_api_blocks_all_final_test_years():
    for year in (2023, 2024, 2025):
        with _work_dir() as tmp:
            with pytest.raises(ValueError, match="final-test years"):
                prepare_sweep_inputs(
                    project_root=ROOT,
                    years=(year,),
                    methods=("rew",),
                    treatments=("W60",),
                    work_dir=tmp,
                )
