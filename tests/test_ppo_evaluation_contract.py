from types import SimpleNamespace

import numpy as np

from awm.ppo.agent import PPOAgent
from awm.ppo.evaluation import evaluate_checkpoint
from awm.ppo.normalization import RunningObservationNormalizer
from awm.risk import VALIDATION_YEARS


class _Observation:
    def flat(self):
        return tuple(np.zeros(79, dtype=np.float32))


class _FakeEnv:
    def __init__(self, cell):
        self.calendar = SimpleNamespace(calendar_year=cell.weather_year)
        self.yield_target_fraction = cell.eta
        self._observation = _Observation()

    def reset(self):
        return self._observation, {}

    def step(self, *, irrigate, amount_fraction):
        del irrigate, amount_fraction
        audit = SimpleNamespace(event_applied=False)
        info = {
            "HWAM": 95.0,
            "IRCM": 45.0,
            "policy_irrigation_mm": 0.0,
            "irrigation_accounting_passed": True,
        }
        return SimpleNamespace(
            observation=self._observation,
            terminated=True,
            irrigation_audit=audit,
            info=info,
        )


def test_ppo_validation_report_is_selector_ready_json_shape():
    agent = PPOAgent(seed=21, device="cpu")
    normalizer = RunningObservationNormalizer(state_dim=79)
    references = {year: 100.0 for year in VALIDATION_YEARS}
    report = evaluate_checkpoint(
        checkpoint_id="ppo_seed21_update0000",
        training_seed=21,
        training_step=0,
        agent=agent,
        normalizer=normalizer,
        env_factory=_FakeEnv,
        reference_yield_by_year=references,
    )

    assert report["cell_count"] == 15
    assert report["normalizer_updated_during_validation"] is False
    assert report["final_test_station_results_present"] is False
    assert len(report["eta_metrics"]) == 3
    for metric in report["eta_metrics"]:
        assert metric["validation_years"] == list(VALIDATION_YEARS)
        assert isinstance(metric["validation_years"], list)
