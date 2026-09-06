"""Balanced real-environment rollout collection for PPO-Lagrangian v1."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
import torch

from awm.ppo.normalization import RunningObservationNormalizer
from awm.ppo.scheduler import WeatherEtaCell, balanced_training_cycle

from .agent import PPOLagrangianAgent
from .buffer import LagrangianRolloutBatch, LagrangianRolloutBuffer
from .signals import LagrangianEpisodeSignals, LagrangianSignalBreakdown


@dataclass(frozen=True, slots=True)
class LagrangianEpisodeOutcome:
    weather_year: int
    eta: float
    step_count: int
    sampled_event_count: int
    executed_event_count: int
    projected_event_count: int
    policy_irrigation_mm: float
    dssat_ircm_mm: float
    yield_kg_ha: float
    signals: LagrangianSignalBreakdown


@dataclass(frozen=True, slots=True)
class BalancedLagrangianRolloutResult:
    batch: LagrangianRolloutBatch
    outcomes: tuple[LagrangianEpisodeOutcome, ...]
    normalizer_training_observation_count: int
    policy_version: int


def collect_episode(
    env,
    *,
    cell: WeatherEtaCell,
    agent: PPOLagrangianAgent,
    normalizer: RunningObservationNormalizer,
    signals: LagrangianEpisodeSignals,
    buffer: LagrangianRolloutBuffer,
    raw_observation_sink: list[np.ndarray],
) -> LagrangianEpisodeOutcome:
    if int(env.calendar.calendar_year) != int(cell.weather_year):
        raise ValueError("environment weather/calendar year does not match rollout cell")
    if abs(float(env.yield_target_fraction) - float(cell.eta)) > 1e-12:
        raise ValueError("environment eta does not match rollout cell")

    observation, _ = env.reset()
    sampled_events = 0
    executed_events = 0
    projected_events = 0
    step_count = 0
    terminal_info = None

    while True:
        raw_state = np.asarray(observation.flat(), dtype=np.float32)
        if raw_state.shape != (agent.hparams.state_dim,):
            raise ValueError(
                f"environment produced state shape {raw_state.shape}; expected ({agent.hparams.state_dim},)"
            )
        raw_observation_sink.append(raw_state.copy())
        normalized = normalizer.normalize(raw_state)
        state_tensor = torch.from_numpy(normalized).unsqueeze(0)
        action, reward_value, cost_value = agent.act(state_tensor)
        irrigate = bool(action.irrigate.item())
        amount_fraction = float(action.amount_fraction.item())
        sampled_events += int(irrigate)

        step = env.step(irrigate=irrigate, amount_fraction=amount_fraction)
        audit = step.irrigation_audit
        executed_events += int(audit.event_applied)
        projected_events += int(audit.water_budget_projected)
        water_reward = signals.step_reward(float(audit.applied_irrigation_mm))
        buffer.append(
            state=normalized,
            irrigate=irrigate,
            raw_amount=float(action.raw_amount.item()),
            log_prob=float(action.log_prob.item()),
            eta=float(cell.eta),
            reward_value=float(reward_value.item()),
            cost_value=float(cost_value.item()),
            reward=water_reward,
            cost=0.0,
            done=bool(step.terminated),
        )
        step_count += 1
        observation = step.observation
        if step.terminated:
            terminal_info = dict(step.info)
            break

    if terminal_info is None:
        raise AssertionError("episode terminated without terminal info")
    if step_count != int(env.calendar.horizon_days):
        raise RuntimeError("episode did not span the complete decision horizon")
    required = ("HWAM", "IRCM", "policy_irrigation_mm", "irrigation_accounting_passed")
    missing = [key for key in required if key not in terminal_info]
    if missing:
        raise KeyError("terminal episode missing fields: " + ", ".join(missing))
    breakdown = signals.finish(
        yield_kg_ha=float(terminal_info["HWAM"]),
        irrigation_accounting_passed=bool(terminal_info["irrigation_accounting_passed"]),
    )
    buffer.set_terminal_cost(
        eta=float(cell.eta),
        cost=float(breakdown.signed_constraint_cost),
    )
    if abs(float(terminal_info["policy_irrigation_mm"]) - breakdown.policy_irrigation_mm) > 1e-8:
        raise RuntimeError("signal water ledger disagrees with environment policy irrigation")
    return LagrangianEpisodeOutcome(
        weather_year=int(cell.weather_year),
        eta=float(cell.eta),
        step_count=step_count,
        sampled_event_count=sampled_events,
        executed_event_count=executed_events,
        projected_event_count=projected_events,
        policy_irrigation_mm=float(terminal_info["policy_irrigation_mm"]),
        dssat_ircm_mm=float(terminal_info["IRCM"]),
        yield_kg_ha=float(terminal_info["HWAM"]),
        signals=breakdown,
    )


def collect_balanced_training_rollout(
    *,
    agent: PPOLagrangianAgent,
    normalizer: RunningObservationNormalizer,
    env_factory: Callable[[WeatherEtaCell], object],
    reference_yield_by_year: Mapping[int, float],
    training_seed: int,
    update_index: int,
) -> BalancedLagrangianRolloutResult:
    cells = balanced_training_cycle(seed=training_seed, update_index=update_index)
    expected_size = len(cells) * 125
    if expected_size != 6750:
        raise AssertionError("formal PPO-Lagrangian rollout must contain 6750 transitions")
    buffer = LagrangianRolloutBuffer(
        state_dim=agent.hparams.state_dim,
        expected_size=expected_size,
        gamma=agent.hparams.gamma,
        gae_lambda=agent.hparams.gae_lambda,
        cost_gamma=agent.hparams.cost_gamma,
        cost_gae_lambda=agent.hparams.cost_gae_lambda,
        policy_version=agent.policy_version,
    )
    raw_observations: list[np.ndarray] = []
    outcomes: list[LagrangianEpisodeOutcome] = []
    for cell in cells:
        env = env_factory(cell)
        try:
            tracker = LagrangianEpisodeSignals(
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
                    signals=tracker,
                    buffer=buffer,
                    raw_observation_sink=raw_observations,
                )
            )
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()
    batch = buffer.finalize()
    normalizer.update(np.stack(raw_observations))
    return BalancedLagrangianRolloutResult(
        batch=batch,
        outcomes=tuple(outcomes),
        normalizer_training_observation_count=len(raw_observations),
        policy_version=batch.policy_version,
    )


__all__ = [
    "BalancedLagrangianRolloutResult",
    "LagrangianEpisodeOutcome",
    "collect_balanced_training_rollout",
    "collect_episode",
]
