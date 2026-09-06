from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from awm.ppo import (
    PPOAgent,
    PPOEpisodeReward,
    PPOHyperparameters,
    PPORolloutBuffer,
    RunningObservationNormalizer,
    WeatherEtaCell,
    collect_episode,
)


@dataclass(frozen=True)
class FakeObservation:
    values: tuple[float, ...]

    def flat(self):
        return self.values


@dataclass(frozen=True)
class FakeAudit:
    applied_irrigation_mm: float
    event_applied: bool
    water_budget_projected: bool = False


@dataclass(frozen=True)
class FakeStep:
    observation: FakeObservation
    terminated: bool
    irrigation_audit: FakeAudit
    info: dict


class FakeEnv:
    def __init__(self):
        self.calendar = SimpleNamespace(calendar_year=2000, horizon_days=2)
        self.yield_target_fraction = 0.95
        self.current_day = 0
        self.policy_water = 0.0

    def reset(self):
        self.current_day = 0
        self.policy_water = 0.0
        return FakeObservation((1.0, 2.0, 3.0, 0.95)), {}

    def step(self, *, irrigate: bool, amount_fraction: float):
        amount = 10.0 * float(amount_fraction) if irrigate else 0.0
        self.policy_water += amount
        self.current_day += 1
        done = self.current_day == 2
        info = {}
        if done:
            info = {
                "HWAM": 940.0,
                "IRCM": 45.0 + self.policy_water,
                "policy_irrigation_mm": self.policy_water,
                "irrigation_accounting_passed": True,
            }
        return FakeStep(
            observation=FakeObservation(
                (1.0 + self.current_day, 2.0, 3.0, 0.95)
            ),
            terminated=done,
            irrigation_audit=FakeAudit(amount, amount > 0.0),
            info=info,
        )


def test_collect_episode_keeps_behavior_and_reward_accounting_separate():
    h = PPOHyperparameters(
        state_dim=4,
        actor_hidden_dims=(16, 8),
        critic_hidden_dims=(16, 8),
        minibatch_size=2,
        update_epochs=1,
    )
    agent = PPOAgent(hyperparameters=h, seed=7, device="cpu")
    normalizer = RunningObservationNormalizer(state_dim=4)
    buffer = PPORolloutBuffer(
        state_dim=4,
        expected_size=2,
        gamma=1.0,
        gae_lambda=0.95,
        policy_version=0,
    )
    reward = PPOEpisodeReward(
        weather_year=2000,
        eta=0.95,
        reference_yield_by_year={2000: 1000.0},
    )
    raw_states: list[np.ndarray] = []
    outcome = collect_episode(
        FakeEnv(),
        cell=WeatherEtaCell(2000, 0.95),
        agent=agent,
        normalizer=normalizer,
        reward_tracker=reward,
        buffer=buffer,
        raw_observation_sink=raw_states,
    )
    batch = buffer.finalize()
    assert batch.size == 2
    assert batch.policy_version == 0
    assert batch.dones.tolist() == [False, True]
    assert len(raw_states) == 2
    assert outcome.step_count == 2
    assert outcome.policy_irrigation_mm == outcome.reward.policy_irrigation_mm
    assert outcome.reward.target_feasible is False
    assert outcome.reward.violation_penalty < -1.0
    # The terminal penalty is applied exactly once to the final transition.
    expected_water_rewards = -outcome.policy_irrigation_mm / 495.0
    assert float(batch.rewards.sum()) == pytest.approx(
        expected_water_rewards + outcome.reward.violation_penalty,
        abs=1e-6,
    )


import pytest
