from pathlib import Path

import pytest

from awm.dssat.management import DSSATExperimentRenderer
from awm.dssat.output_reader import CachedDSSATOutputReader
from awm.dssat.workspace import validate_dssatpro_record_width
from awm.envs.cotton_state import (
    CottonObservationBuilder,
    DIRECT_RAW_FEATURES,
    DRAINED_UPPER_LIMIT_FEATURES,
    EXPECTED_STATE_DIM,
    LOWER_LIMIT_FEATURES,
    SOIL_WATER_FEATURES,
)
from awm.envs.cotton_water_env import CottonWaterEnv
from awm.envs.dssat_irrigation import (
    DSSATDecisionCalendar,
    DSSATIrrigationAdapter,
)
from awm.envs.water_budget import IrrigationSystemSpec, WaterBudgetController


def _raw_state():
    raw = {name: 1.0 for name in DIRECT_RAW_FEATURES}
    for name in SOIL_WATER_FEATURES:
        raw[name] = 0.25
    for name in LOWER_LIMIT_FEATURES:
        raw[name] = 0.10
    for name in DRAINED_UPPER_LIMIT_FEATURES:
        raw[name] = 0.40
    return raw


def test_renderer_preserves_fixed_management_and_only_injects_irrigation(tmp_path: Path):
    template = tmp_path / "template.COX"
    output = tmp_path / "run.COX"
    template.write_text(
        "*FERTILIZERS\n"
        " 1 26120 FE010 AP005 2 90\n"
        "@I IDATE IROP IRVAL\n"
        "{{AWM_IRRIGATION_EVENTS}}\n"
        "*END\n",
        encoding="utf-8",
    )
    renderer = DSSATExperimentRenderer(
        template_path=str(template),
        output_cox_path=str(output),
    )
    renderer.reset()
    renderer.add_irrigation("26140", 12.34)
    text = output.read_text(encoding="utf-8")
    assert "FE010 AP005 2 90" in text
    assert "1 26140 IR005 12.34" in text
    assert "{{AWM_IRRIGATION_EVENTS}}" not in text


def test_output_reader_keeps_summary_separate_from_daily_state(tmp_path: Path):
    summary = tmp_path / "Summary.OUT"
    daily = tmp_path / "PlantGro.OUT"
    summary.write_text("@ HWAM IRCM\n 5000 120\n", encoding="utf-8")
    daily.write_text(
        "*DSSAT Cropping System\n"
        "@ YEAR DOY GWAD\n"
        " 2026 120 10\n"
        " 2026 121 11\n",
        encoding="utf-8",
    )
    reader = CachedDSSATOutputReader(
        summary_out=str(summary),
        out_files=[str(daily)],
    )
    reader.refresh()
    assert reader.daily_state("26120")["GWAD"] == 10
    assert "HWAM" not in reader.daily_state("26120")
    assert reader.season_summary()["HWAM"] == 5000


def test_workspace_rejects_long_l48_record(tmp_path: Path):
    (tmp_path / "DSSATPRO.L48").write_text(
        "CRD //" + "x" * 90 + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        validate_dssatpro_record_width(tmp_path)


def test_cotton_builder_remains_74d_and_ignores_terminal_fields():
    raw = _raw_state()
    raw["HWAM"] = 9999
    raw["IRCM"] = 999
    observation = CottonObservationBuilder(125).build(raw, 0)
    assert len(observation) == EXPECTED_STATE_DIM == 74
    assert 9999 not in observation


class FakeBackend:
    def __init__(self):
        self.writes = []
        self.reruns = 0
        self.resets = 0

    def reset_episode(self):
        self.writes.clear()
        self.reruns = 0
        self.resets += 1

    def write_irrigation(self, action_yrdoy, amount_mm):
        self.writes.append((action_yrdoy, amount_mm))

    def rerun_and_refresh(self):
        self.reruns += 1

    def daily_state(self, yrdoy):
        return _raw_state()

    def season_summary(self):
        return {"HWAM": 5000.0, "IRCM": 5.0}


def test_cotton_water_env_is_79d_and_summary_is_terminal_only():
    backend = FakeBackend()
    spec = IrrigationSystemSpec(
        seasonal_budget_mm=20.0,
        min_event_mm=5.0,
        max_event_mm=10.0,
        min_interval_days=0,
        horizon_days=2,
    )
    controller = WaterBudgetController(spec)
    calendar = DSSATDecisionCalendar.from_yrdoy("26119", horizon_days=2)
    adapter = DSSATIrrigationAdapter(
        controller=controller,
        backend=backend,
        calendar=calendar,
        execution_resolution_mm=0.01,
        nonpolicy_irrigation_mm=0.0,
        summary_tolerance_mm=1e-6,
    )
    env = CottonWaterEnv(
        backend=backend,
        adapter=adapter,
        plant_yrdoy="26119",
        yield_target_fraction=0.95,
    )

    observation, info = env.reset()
    assert len(observation.flat()) == 79
    assert info["seasonal_summary_exposed"] is False

    first = env.step(irrigate=False, amount_fraction=0.7)
    assert first.terminated is False
    assert first.info["seasonal_summary_exposed"] is False
    assert backend.reruns == 0

    second = env.step(irrigate=True, amount_fraction=0.0)
    assert second.terminated is True
    assert second.info["seasonal_summary_exposed"] is True
    assert second.info["HWAM"] == 5000.0
    assert second.info["IRCM"] == 5.0
    assert second.info["irrigation_accounting_passed"] is True
    assert backend.writes == [("26121", 5.0)]
    assert backend.reruns == 1
