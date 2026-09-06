from types import SimpleNamespace

import numpy as np
import torch

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
        audit = SimpleNamespace(event_applied=bool(irrigate))
        irrigation = 45.0 + (float(amount_fraction) * 10.0 if irrigate else 0.0)
        yield_value = 90.0 + (5.0 if irrigate else 0.0) + float(amount_fraction)
        info = {
            "HWAM": yield_value,
            "IRCM": irrigation,
            "policy_irrigation_mm": irrigation - 45.0,
            "irrigation_accounting_passed": True,
        }
        return SimpleNamespace(
            observation=self._observation,
            terminated=True,
            irrigation_audit=audit,
            info=info,
        )


def _report(agent: PPOAgent):
    normalizer = RunningObservationNormalizer(state_dim=79)
    references = {year: 100.0 for year in VALIDATION_YEARS}
    return evaluate_checkpoint(
        checkpoint_id="ppo_seed21_update0000",
        training_seed=21,
        training_step=0,
        agent=agent,
        normalizer=normalizer,
        env_factory=_FakeEnv,
        reference_yield_by_year=references,
    )


def test_ppo_validation_report_is_selector_ready_json_shape():
    agent = PPOAgent(seed=21, device="cpu")
    report = _report(agent)

    assert report["cell_count"] == 15
    assert report["validation_action_mode"] == "deterministic"
    assert "validation_draw_seeds_by_year" not in report
    assert report["normalizer_updated_during_validation"] is False
    assert report["final_test_station_results_present"] is False
    assert len(report["eta_metrics"]) == 3
    assert all(cell["action_mode"] == "deterministic" for cell in report["cell_results"])
    assert all("policy_draw_seed" not in cell for cell in report["cell_results"])
    for metric in report["eta_metrics"]:
        assert metric["validation_years"] == list(VALIDATION_YEARS)
        assert isinstance(metric["validation_years"], list)


def test_deterministic_validation_does_not_depend_on_ppo_rng_state():
    agent = PPOAgent(seed=21, device="cpu")
    first = _report(agent)
    # Consume and replace the stochastic PPO RNG state. Deterministic evaluation
    # must remain unchanged because weather-year risk excludes action-sampling noise.
    _ = torch.rand(100, generator=agent.generator)
    agent.generator.manual_seed(987654321)
    second = _report(agent)
    assert first["eta_metrics"] == second["eta_metrics"]
    assert first["cell_results"] == second["cell_results"]
