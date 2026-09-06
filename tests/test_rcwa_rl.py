from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from awm.ppo.agent import PPOAgent
from awm.risk import empirical_lower_cvar
from awm.rcwa import (
    RCWAAgent,
    RCWAHyperparameters,
    RCWARolloutBuffer,
    empirical_lower_quantile,
    evaluate_eta_tail,
    evaluate_registered_eta_groups,
)


ROOT = Path(__file__).resolve().parents[1]


def test_empirical_tau_and_ru_value_match_frozen_fractional_lcvar():
    values = [0.70 + 0.02 * i for i in range(18)]
    metric = evaluate_eta_tail(values, eta=0.95, alpha=0.20)
    assert metric.sample_count == 18
    assert metric.tau == pytest.approx(sorted(values)[3])
    assert metric.empirical_lcvar == pytest.approx(empirical_lower_cvar(values, alpha=0.20))
    assert metric.ru_lcvar == pytest.approx(metric.empirical_lcvar)
    assert metric.violation == pytest.approx(0.95 - metric.empirical_lcvar)
    assert sum(x > 0.0 for x in metric.hinge_costs) == 3


def test_empirical_tau_handles_integer_tail_mass_and_duplicates():
    values = [0.8, 0.8, 0.9, 1.0, 1.1]
    assert empirical_lower_quantile(values, alpha=0.20) == pytest.approx(0.8)
    metric = evaluate_eta_tail(values, eta=0.9, alpha=0.20)
    assert metric.empirical_lcvar == pytest.approx(0.8)
    assert metric.ru_lcvar == pytest.approx(0.8)


def test_registered_eta_groups_are_independent_and_balanced():
    etas = [0.90] * 18 + [0.95] * 18 + [0.98] * 18
    retentions = (
        [0.80 + 0.01 * i for i in range(18)]
        + [0.75 + 0.01 * i for i in range(18)]
        + [0.70 + 0.01 * i for i in range(18)]
    )
    costs, metrics = evaluate_registered_eta_groups(
        episode_etas=etas,
        retentions=retentions,
        alpha=0.20,
        expected_per_eta=18,
    )
    assert len(costs) == 54
    assert set(metrics) == {"0.90", "0.95", "0.98"}
    assert all(item.sample_count == 18 for item in metrics.values())
    assert metrics["0.90"].empirical_lcvar > metrics["0.95"].empirical_lcvar > metrics["0.98"].empirical_lcvar
    with pytest.raises(RuntimeError, match="requires 18"):
        evaluate_registered_eta_groups(
            episode_etas=etas[:-1],
            retentions=retentions[:-1],
            alpha=0.20,
            expected_per_eta=18,
        )


def test_buffer_assigns_tail_cost_only_after_all_episode_retentions_known():
    buffer = RCWARolloutBuffer(
        state_dim=2,
        expected_size=54,
        gamma=1.0,
        gae_lambda=1.0,
        risk_gamma=1.0,
        risk_gae_lambda=1.0,
        alpha=0.20,
    )
    for eta, base in ((0.90, 0.80), (0.95, 0.75), (0.98, 0.70)):
        for i in range(18):
            buffer.append(
                state=np.array([eta, i], dtype=np.float32),
                irrigate=False,
                raw_amount=0.0,
                log_prob=-0.1,
                eta=eta,
                reward_value=0.0,
                risk_value=0.0,
                reward=-0.01,
                done=True,
            )
            buffer.register_terminal_retention(eta=eta, retention=base + 0.01 * i)
    batch = buffer.finalize()
    assert batch.size == 54
    assert int(batch.dones.sum().item()) == 54
    assert set(batch.tail_metrics) == {"0.90", "0.95", "0.98"}
    assert torch.equal(batch.risk_returns, batch.risk_costs)
    assert batch.risk_costs.shape == (54,)
    # Exactly the observations strictly below tau have positive RU hinge cost.
    assert int((batch.risk_costs > 0).sum().item()) == 9


def test_actor_and_reward_critic_initialization_match_ppo_for_same_seed():
    ppo = PPOAgent(seed=21, device="cpu")
    rcwa = RCWAAgent(seed=21, device="cpu")
    for key, value in ppo.actor.state_dict().items():
        assert torch.equal(value, rcwa.actor.state_dict()[key]), key
    for key, value in ppo.critic.state_dict().items():
        assert torch.equal(value, rcwa.reward_critic.state_dict()[key]), key


def test_checkpoint_roundtrip_preserves_duals_and_rng():
    agent = RCWAAgent(seed=31, device="cpu")
    agent.dual_by_eta[0.90] = 1.2
    agent.dual_by_eta[0.95] = 0.4
    agent.policy_version = 3
    agent.update_index = 3
    payload = agent.checkpoint_payload()
    expected_draw = torch.rand(5, generator=agent.generator)
    restored = RCWAAgent(seed=31, device="cpu")
    restored.load_checkpoint_payload(payload)
    actual_draw = torch.rand(5, generator=restored.generator)
    assert actual_draw.tolist() == pytest.approx(expected_draw.tolist())
    assert restored.dual_by_eta == pytest.approx(agent.dual_by_eta)
    assert restored.policy_version == 3
    assert restored.update_index == 3


def test_protocol_matches_ppo_fairness_and_risk_contract():
    rcwa = json.loads((ROOT / "configs" / "rcwa_rl_v1.json").read_text(encoding="utf-8"))
    ppo = json.loads((ROOT / "configs" / "ppo_baseline_v1.json").read_text(encoding="utf-8"))
    risk = json.loads((ROOT / "configs" / "risk_contract_v1.json").read_text(encoding="utf-8"))
    assert rcwa["policy"]["actor_hidden_dims"] == ppo["policy"]["actor_hidden_dims"]
    assert rcwa["rollout"]["episodes_per_update"] == ppo["rollout"]["balanced_weather_eta_cells_per_update"]
    assert rcwa["rollout"]["transitions_per_update"] == ppo["rollout"]["transitions_per_update"]
    assert rcwa["interaction_budget"]["training_episodes_per_seed"] == ppo["interaction_budget"]["training_episodes_per_seed"]
    assert rcwa["interaction_budget"]["decision_transitions_per_seed"] == ppo["interaction_budget"]["decision_transitions_per_seed"]
    assert rcwa["risk_constraint"]["alpha"] == risk["risk_measure"]["alpha"] == 0.20
    assert rcwa["lagrangian"]["extra_environment_episodes"] == 0
    assert rcwa["evaluation"]["validation_action_mode"] == "deterministic"
    assert rcwa["training"]["seeds"] == ppo["training"]["seeds"]


def test_frozen_rcwa_hyperparameters_use_complete_episode_risk_credit():
    h = RCWAHyperparameters()
    assert h.gamma == h.gae_lambda == 1.0
    assert h.risk_gamma == h.risk_gae_lambda == 1.0
    assert h.alpha == pytest.approx(0.20)
    assert h.dual_initial_value == pytest.approx(1.0)
    assert h.dual_learning_rate == pytest.approx(0.05)
