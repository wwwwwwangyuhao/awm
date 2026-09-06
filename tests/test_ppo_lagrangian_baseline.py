from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from awm.ppo.agent import PPOAgent
from awm.ppo_lagrangian import (
    LagrangianEpisodeSignals,
    LagrangianRolloutBuffer,
    PPOLagrangianAgent,
    PPOLagrangianHyperparameters,
)


ROOT = Path(__file__).resolve().parents[1]


def _references() -> dict[int, float]:
    payload = json.loads(
        (ROOT / "configs" / "yield_reference_v1_development.json").read_text(
            encoding="utf-8"
        )
    )
    return {int(k): float(v) for k, v in payload["training_years"].items()}


def test_signed_constraint_signal_is_expected_retention_not_positive_shortfall():
    tracker = LagrangianEpisodeSignals(
        weather_year=2000,
        eta=0.95,
        reference_yield_by_year=_references(),
    )
    assert tracker.step_reward(49.5) == pytest.approx(-0.1)
    result = tracker.finish(
        yield_kg_ha=1.10 * _references()[2000],
        irrigation_accounting_passed=True,
    )
    assert result.yield_retention == pytest.approx(1.10)
    assert result.signed_constraint_cost == pytest.approx(-0.15)
    assert result.water_return == pytest.approx(-0.1)
    assert result.per_year_expected_constraint_satisfied is True


def test_cost_buffer_does_not_leak_terminal_cost_across_episode_boundary():
    buffer = LagrangianRolloutBuffer(
        state_dim=2,
        expected_size=4,
        gamma=1.0,
        gae_lambda=1.0,
        cost_gamma=1.0,
        cost_gae_lambda=1.0,
    )
    for episode, terminal_cost in ((0, 0.2), (1, -0.3)):
        eta = 0.90 if episode == 0 else 0.95
        for step in range(2):
            buffer.append(
                state=np.array([episode, step], dtype=np.float32),
                irrigate=False,
                raw_amount=0.0,
                log_prob=-0.1,
                eta=eta,
                reward_value=0.0,
                cost_value=0.0,
                reward=-0.01,
                cost=0.0,
                done=step == 1,
            )
        buffer.set_terminal_cost(eta=eta, cost=terminal_cost)
    batch = buffer.finalize()
    assert batch.cost_returns.tolist() == pytest.approx([0.2, 0.2, -0.3, -0.3])
    assert batch.reward_returns.tolist() == pytest.approx([-0.02, -0.01, -0.02, -0.01])
    assert batch.episode_costs.tolist() == pytest.approx([0.2, -0.3])


def test_eta_specific_dual_projection_is_independent():
    agent = PPOLagrangianAgent(seed=21, device="cpu")
    etas = torch.tensor([0.90] * 18 + [0.95] * 18 + [0.98] * 18)
    costs = torch.tensor([0.20] * 18 + [-0.40] * 18 + [0.0] * 18)
    before, after, means = agent._dual_update(episode_etas=etas, episode_costs=costs)
    assert before == {"0.90": 1.0, "0.95": 1.0, "0.98": 1.0}
    assert means["0.90"] == pytest.approx(0.20)
    assert means["0.95"] == pytest.approx(-0.40)
    assert means["0.98"] == pytest.approx(0.0)
    assert after["0.90"] == pytest.approx(1.01)
    assert after["0.95"] == pytest.approx(0.98)
    assert after["0.98"] == pytest.approx(1.0)


def test_actor_architecture_and_initialization_match_standard_ppo():
    ppo = PPOAgent(seed=21, device="cpu")
    lag = PPOLagrangianAgent(seed=21, device="cpu")
    ppo_state = ppo.actor.state_dict()
    lag_state = lag.actor.state_dict()
    assert ppo_state.keys() == lag_state.keys()
    for key in ppo_state:
        assert torch.equal(ppo_state[key], lag_state[key]), key


def test_checkpoint_roundtrip_preserves_duals_and_rng():
    agent = PPOLagrangianAgent(seed=31, device="cpu")
    agent.dual_by_eta[0.90] = 1.23
    agent.dual_by_eta[0.95] = 0.45
    agent.policy_version = 7
    agent.update_index = 7
    payload = agent.checkpoint_payload()
    expected_draw = torch.rand(5, generator=agent.generator)

    restored = PPOLagrangianAgent(seed=31, device="cpu")
    restored.load_checkpoint_payload(payload)
    actual_draw = torch.rand(5, generator=restored.generator)
    assert actual_draw.tolist() == pytest.approx(expected_draw.tolist())
    assert restored.dual_by_eta == pytest.approx(agent.dual_by_eta)
    assert restored.policy_version == 7
    assert restored.update_index == 7


def test_protocol_matches_ppo_interaction_actor_and_validation_budgets():
    lag = json.loads(
        (ROOT / "configs" / "ppo_lagrangian_baseline_v1.json").read_text(encoding="utf-8")
    )
    ppo = json.loads(
        (ROOT / "configs" / "ppo_baseline_v1.json").read_text(encoding="utf-8")
    )
    assert lag["policy"]["actor_hidden_dims"] == ppo["policy"]["actor_hidden_dims"]
    assert lag["policy"]["observation_dim"] == ppo["environment"]["observation_dim"]
    assert lag["rollout"]["episodes_per_update"] == ppo["rollout"]["balanced_weather_eta_cells_per_update"]
    assert lag["rollout"]["transitions_per_update"] == ppo["rollout"]["transitions_per_update"]
    assert lag["interaction_budget"]["training_episodes_per_seed"] == ppo["interaction_budget"]["training_episodes_per_seed"]
    assert lag["interaction_budget"]["decision_transitions_per_seed"] == ppo["interaction_budget"]["decision_transitions_per_seed"]
    assert lag["training"]["candidate_checkpoint_count_per_seed"] == ppo["training"]["candidate_checkpoint_count_per_seed"]
    assert lag["training"]["validation_episodes_per_seed"] == ppo["training"]["validation_episodes_per_seed"]
    assert lag["evaluation"]["validation_action_mode"] == "deterministic"
    assert lag["lagrangian"]["extra_environment_episodes"] == 0
    assert "LCVaR" not in lag["constraint"]["expected_constraint"]


def test_frozen_hyperparameters_have_separate_full_episode_cost_critic():
    h = PPOLagrangianHyperparameters()
    assert h.gamma == h.gae_lambda == 1.0
    assert h.cost_gamma == h.cost_gae_lambda == 1.0
    assert h.dual_initial_value == pytest.approx(1.0)
    assert h.dual_learning_rate == pytest.approx(0.05)
    assert h.reward_value_loss_coefficient == pytest.approx(0.5)
    assert h.cost_value_loss_coefficient == pytest.approx(0.5)
