"""Frozen 15-cell deterministic validation evaluation for PPO checkpoints."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean
from typing import Callable, Mapping

import numpy as np
import torch

from awm.evaluation import EtaValidationMetrics, build_candidate_from_report
from awm.risk import REGISTERED_ETA_LEVELS, VALIDATION_YEARS, empirical_lower_cvar, yield_retention

from .agent import PPOAgent
from .normalization import RunningObservationNormalizer
from .scheduler import WeatherEtaCell, validation_cells


@dataclass(frozen=True, slots=True)
class PPOValidationCellResult:
    weather_year: int
    eta: float
    action_mode: str
    yield_kg_ha: float
    reference_yield_kg_ha: float
    yield_retention: float
    total_irrigation_mm: float
    policy_irrigation_mm: float
    irrigation_accounting_passed: bool
    sampled_event_count: int
    executed_event_count: int


def evaluate_validation_cell(
    env,
    *,
    cell: WeatherEtaCell,
    agent: PPOAgent,
    normalizer: RunningObservationNormalizer,
    reference_yield_kg_ha: float,
) -> PPOValidationCellResult:
    if int(env.calendar.calendar_year) != int(cell.weather_year):
        raise ValueError("validation environment year mismatch")
    if abs(float(env.yield_target_fraction) - float(cell.eta)) > 1e-12:
        raise ValueError("validation environment eta mismatch")
    observation, _ = env.reset()
    sampled_events = 0
    executed_events = 0
    terminal_info = None

    while True:
        raw = np.asarray(observation.flat(), dtype=np.float32)
        normalized = normalizer.normalize(raw)
        state = torch.from_numpy(normalized).unsqueeze(0).to(agent.device)
        with torch.no_grad():
            action, _ = agent.deterministic_action(state)
        irrigate = bool(action.irrigate.item())
        sampled_events += int(irrigate)
        step = env.step(
            irrigate=irrigate,
            amount_fraction=float(action.amount_fraction.item()),
        )
        executed_events += int(step.irrigation_audit.event_applied)
        observation = step.observation
        if step.terminated:
            terminal_info = dict(step.info)
            break

    if terminal_info is None:
        raise AssertionError("validation episode ended without terminal info")
    if terminal_info.get("irrigation_accounting_passed") is not True:
        raise RuntimeError("validation episode failed irrigation accounting")
    y = float(terminal_info["HWAM"])
    retention = yield_retention(y, float(reference_yield_kg_ha))
    return PPOValidationCellResult(
        weather_year=int(cell.weather_year),
        eta=float(cell.eta),
        action_mode="deterministic",
        yield_kg_ha=y,
        reference_yield_kg_ha=float(reference_yield_kg_ha),
        yield_retention=retention,
        total_irrigation_mm=float(terminal_info["IRCM"]),
        policy_irrigation_mm=float(terminal_info["policy_irrigation_mm"]),
        irrigation_accounting_passed=True,
        sampled_event_count=sampled_events,
        executed_event_count=executed_events,
    )


def evaluate_checkpoint(
    *,
    checkpoint_id: str,
    training_seed: int,
    training_step: int,
    agent: PPOAgent,
    normalizer: RunningObservationNormalizer,
    env_factory: Callable[[WeatherEtaCell], object],
    reference_yield_by_year: Mapping[int, float],
) -> dict[str, object]:
    """Evaluate all 5 validation years × 3 eta and emit selector-ready report."""
    if not checkpoint_id.strip():
        raise ValueError("checkpoint_id must be non-empty")
    if agent.seed != int(training_seed):
        raise ValueError("agent seed and validation training_seed disagree")
    missing_refs = sorted(set(VALIDATION_YEARS) - set(reference_yield_by_year))
    if missing_refs:
        raise KeyError(f"missing validation Y_ref values: {missing_refs}")

    actor_was_training = agent.actor.training
    critic_was_training = agent.critic.training
    agent.actor.eval()
    agent.critic.eval()
    cell_results: list[PPOValidationCellResult] = []
    try:
        for cell in validation_cells():
            env = env_factory(cell)
            try:
                cell_results.append(
                    evaluate_validation_cell(
                        env,
                        cell=cell,
                        agent=agent,
                        normalizer=normalizer,
                        reference_yield_kg_ha=float(reference_yield_by_year[cell.weather_year]),
                    )
                )
            finally:
                close = getattr(env, "close", None)
                if callable(close):
                    close()
    finally:
        agent.actor.train(actor_was_training)
        agent.critic.train(critic_was_training)

    eta_metrics: list[dict[str, object]] = []
    for eta in REGISTERED_ETA_LEVELS:
        group = sorted(
            (item for item in cell_results if abs(item.eta - float(eta)) <= 1e-12),
            key=lambda item: item.weather_year,
        )
        years = tuple(item.weather_year for item in group)
        if years != tuple(VALIDATION_YEARS):
            raise RuntimeError(f"incomplete validation group for eta={eta}: {years}")
        retentions = [item.yield_retention for item in group]
        metric = asdict(
            EtaValidationMetrics(
                eta=float(eta),
                validation_years=years,
                lcvar_retention=empirical_lower_cvar(retentions, alpha=0.20),
                mean_total_irrigation_mm=mean(item.total_irrigation_mm for item in group),
                minimum_retention=min(retentions),
            )
        )
        metric["validation_years"] = list(metric["validation_years"])
        eta_metrics.append(metric)

    report = {
        "checkpoint_id": checkpoint_id,
        "training_seed": int(training_seed),
        "training_step": int(training_step),
        "eta_metrics": eta_metrics,
        "cell_count": len(cell_results),
        "validation_action_mode": "deterministic",
        "cell_results": [asdict(item) for item in cell_results],
        "normalizer_updated_during_validation": False,
        "final_test_station_results_present": False,
    }
    build_candidate_from_report(report)
    return report


__all__ = [
    "PPOValidationCellResult",
    "evaluate_checkpoint",
    "evaluate_validation_cell",
]
