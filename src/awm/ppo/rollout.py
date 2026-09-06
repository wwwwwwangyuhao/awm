"""Policy/environment rollout collection for AWM PPO v1.

The collector is environment-agnostic. A real-DSSAT factory can supply
CottonWaterEnv instances, while unit tests can use fake environments. Running
observation statistics are frozen during one complete on-policy rollout and are
updated only after all 54 weather×eta episodes have been collected.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol

import numpy as np
import torch

from .agent import PPOAgent
from .buffer import PPORolloutBatch, PPORolloutBuffer
from .normalization import RunningObservationNormalizer
from .reward import PPOEpisodeReward, PPORewardBreakdown
from .scheduler import WeatherEtaCell, balanced_training_cycle


class ObservationLike(Protocol):
    def flat(self) -> tuple[float, ...]: ...


class StepLike(Protocol):
    observation: ObservationLike
    terminated: bool
    irrigation_audit: Any
    info: Mapping[str, Any]


class PPOEnvLike(Protocol):
    current_day: int
    calendar: Any
    yield_target_fraction: float

    def reset(self) -> tuple[ObservationLike, Mapping[str, Any]]: ...
    def step(self, *, irrigate: bool, amount_fraction: float) -> StepLike: ...


@dataclass(frozen=True, slots=True)
class PPOEpisodeOutcome:
    weather_year: int
    eta: float
    step_count: int
    sampled_event_count: int
    executed_event_count: int
    projected_event_count: int
    policy_irrigation_mm: float
    dssat_ircm_mm: float
    yield_kg_ha: float
    reward: PPORewardBreakdown


@dataclass(frozen=True, slots=True)
class BalancedRolloutResult:
    batch: PPORolloutBatch
    outcomes: tuple[PPOEpisodeOutcome, ...]
    normalizer_training_observation_count: int
    policy_version: int


def collect_episode(
    env: PPOEnvLike,
    *,
    cell: WeatherEtaCell,
    agent: PPOAgent,
    normalizer: RunningObservationNormalizer,
    reward_tracker: PPOEpisodeReward,
    buffer: PPORolloutBuffer,
    raw_observation_sink: list[np.ndarray],
) -> PPOEpisodeOutcome:
    if int(env.calendar.calendar_year) != int(cell.weather_year):
        raise ValueError("environment weather/calendar year does not match rollout cell")
    if abs(float(env.yield_target_fraction) - float(cell.eta)) > 1e-12:
        raise ValueError("environment eta does not match rollout cell")

    observation, _ = env.reset()
    sampled_events = 0
    executed_events = 0
    projected_events = 0
    step_count = 0
    terminal_info: Mapping[str, Any] | None = None

    while True:
        raw_state = np.asarray(observation.flat(), dtype=np.float32)
        if raw_state.shape != (agent.hparams.state_dim,):
            raise ValueError(
                f"environment produced state shape {raw_state.shape}; expected "
                f"({agent.hparams.state_dim},)"
            )
        raw_observation_sink.append(raw_state.copy())
        normalized = normalizer.normalize(raw_state)
        state_tensor = torch.from_numpy(normalized).unsqueeze(0)
        action, value = agent.act(state_tensor)
        irrigate = bool(action.irrigate.item())
        amount_fraction = float(action.amount_fraction.item())
        sampled_events += int(irrigate)

        step = env.step(irrigate=irrigate, amount_fraction=amount_fraction)
        audit = step.irrigation_audit
        executed_events += int(audit.event_applied)
        projected_events += int(audit.water_budget_projected)
        scalar_reward = reward_tracker.step_reward(float(audit.applied_irrigation_mm))
        buffer.append(
            state=normalized,
            irrigate=irrigate,
            raw_amount=float(action.raw_amount.item()),
            log_prob=float(action.log_prob.item()),
            value=float(value.item()),
            reward=scalar_reward,
            done=bool(step.terminated),
        )
        step_count += 1
        observation = step.observation
        if step.terminated:
            terminal_info = step.info
            break

    if terminal_info is None:
        raise AssertionError("PPO episode terminated without terminal info")
    if step_count != int(env.calendar.horizon_days):
        raise RuntimeError("PPO episode did not span the complete decision horizon")
    required = ("HWAM", "IRCM", "policy_irrigation_mm", "irrigation_accounting_passed")
    missing = [key for key in required if key not in terminal_info]
    if missing:
        raise KeyError("terminal PPO episode missing fields: " + ", ".join(missing))
    breakdown = reward_tracker.finish(
        yield_kg_ha=float(terminal_info["HWAM"]),
        irrigation_accounting_passed=bool(terminal_info["irrigation_accounting_passed"]),
    )
    buffer.add_terminal_reward(breakdown.violation_penalty)
    if abs(float(terminal_info["policy_irrigation_mm"]) - breakdown.policy_irrigation_mm) > 1e-8:
        raise RuntimeError("reward water ledger disagrees with environment policy irrigation")
    return PPOEpisodeOutcome(
        weather_year=int(cell.weather_year),
        eta=float(cell.eta),
        step_count=step_count,
        sampled_event_count=sampled_events,
        executed_event_count=executed_events,
        projected_event_count=projected_events,
        policy_irrigation_mm=float(terminal_info["policy_irrigation_mm"]),
        dssat_ircm_mm=float(terminal_info["IRCM"]),
        yield_kg_ha=float(terminal_info["HWAM"]),
        reward=breakdown,
    )


def collect_balanced_training_rollout(
    *,
    agent: PPOAgent,
    normalizer: RunningObservationNormalizer,
    env_factory: Callable[[WeatherEtaCell], PPOEnvLike],
    reference_yield_by_year: Mapping[int, float],
    training_seed: int,
    update_index: int,
) -> BalancedRolloutResult:
    """Collect one 54-episode, 6750-transition strictly on-policy rollout."""
    cells = balanced_training_cycle(seed=training_seed, update_index=update_index)
    expected_size = len(cells) * 125
    if expected_size != 6750:
        raise AssertionError("formal PPO rollout must contain 6750 transitions")
    buffer = PPORolloutBuffer(
        state_dim=agent.hparams.state_dim,
        expected_size=expected_size,
        gamma=agent.hparams.gamma,
        gae_lambda=agent.hparams.gae_lambda,
        policy_version=agent.policy_version,
    )
    raw_observations: list[np.ndarray] = []
    outcomes: list[PPOEpisodeOutcome] = []
    # The normalizer is intentionally frozen while this policy version acts.
    for cell in cells:
        env = env_factory(cell)
        reward = PPOEpisodeReward(
            weather_year=cell.weather_year,
            eta=cell.eta,
            reference_yield_by_year=reference_yield_by_year,
        )
        outcomes.append(
            collect_episode(
                env,
                cell=cell,
                agent=agent,
                normalizer=normalizer,
                reward_tracker=reward,
                buffer=buffer,
                raw_observation_sink=raw_observations,
            )
        )
    batch = buffer.finalize()
    # Statistics from this rollout become available only to the next rollout.
    normalizer.update(np.stack(raw_observations))
    return BalancedRolloutResult(
        batch=batch,
        outcomes=tuple(outcomes),
        normalizer_training_observation_count=len(raw_observations),
        policy_version=batch.policy_version,
    )


__all__ = [
    "BalancedRolloutResult",
    "PPOEpisodeOutcome",
    "collect_balanced_training_rollout",
    "collect_episode",
]
