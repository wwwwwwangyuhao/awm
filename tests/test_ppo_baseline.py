from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from awm.ppo import (
    HierarchicalIrrigationActor,
    PPOAgent,
    PPOEpisodeReward,
    PPOHyperparameters,
    PPORolloutBatch,
    PPORolloutBuffer,
    RunningObservationNormalizer,
    balanced_training_cycle,
    training_cells,
    validation_cells,
)

ROOT = Path(__file__).resolve().parents[1]


def test_hierarchical_actor_exact_noop_and_logprob_recompute():
    torch.manual_seed(123)
    actor = HierarchicalIrrigationActor(state_dim=4, hidden_dims=(16, 8))
    states = torch.randn(128, 4)
    generator = torch.Generator().manual_seed(321)
    with torch.no_grad():
        action = actor.sample(states, generator=generator)
        recomputed, _ = actor.evaluate_behavior(states, action.irrigate, action.raw_amount)
    assert torch.allclose(action.log_prob, recomputed, atol=1e-6, rtol=0.0)
    assert torch.all(action.amount_fraction[~action.irrigate] == 0.0)
    assert torch.all(action.amount_fraction[action.irrigate] > 0.0)
    assert torch.all(action.amount_fraction[action.irrigate] < 1.0)
    # Neutral gate initialization is exactly p=0.5 for every state.
    _, logits = actor.components(states)
    assert torch.allclose(torch.sigmoid(logits), torch.full_like(logits, 0.5))


def test_inactive_amount_latent_does_not_change_log_probability():
    actor = HierarchicalIrrigationActor(state_dim=3, hidden_dims=(8, 4))
    states = torch.randn(5, 3)
    inactive = torch.zeros(5, dtype=torch.bool)
    raw_a = torch.zeros(5)
    raw_b = torch.full((5,), 100.0)
    lp_a, _ = actor.evaluate_behavior(states, inactive, raw_a)
    lp_b, _ = actor.evaluate_behavior(states, inactive, raw_b)
    assert torch.allclose(lp_a, lp_b, atol=1e-6, rtol=0.0)


def test_balanced_scheduler_is_exact_and_never_uses_final_test_years():
    canonical = training_cells()
    assert len(canonical) == 54
    assert len(set(canonical)) == 54
    assert {cell.weather_year for cell in canonical} == set(range(2000, 2018))
    assert {cell.eta for cell in canonical} == {0.90, 0.95, 0.98}
    assert not ({2023, 2024, 2025} & {cell.weather_year for cell in canonical})

    first = balanced_training_cycle(seed=21, update_index=1)
    same = balanced_training_cycle(seed=21, update_index=1)
    second = balanced_training_cycle(seed=21, update_index=2)
    assert first == same
    assert set(first) == set(canonical)
    assert first != second

    validation = validation_cells()
    assert len(validation) == 15
    assert {cell.weather_year for cell in validation} == set(range(2018, 2023))
    assert not ({2023, 2024, 2025} & {cell.weather_year for cell in validation})


def test_ppo_reward_is_feasibility_first_without_tunable_penalty():
    refs = {2000: 1000.0}
    feasible = PPOEpisodeReward(weather_year=2000, eta=0.95, reference_yield_by_year=refs)
    # Full policy quota costs exactly -1.0, but target is satisfied.
    assert feasible.step_reward(495.0) == pytest.approx(-1.0)
    feasible_result = feasible.finish(yield_kg_ha=950.0, irrigation_accounting_passed=True)
    assert feasible_result.episode_return == pytest.approx(-1.0)
    assert feasible_result.target_feasible is True

    infeasible = PPOEpisodeReward(weather_year=2000, eta=0.95, reference_yield_by_year=refs)
    # Even with zero water cost, any strict shortfall is below -1.
    infeasible_result = infeasible.finish(yield_kg_ha=949.0, irrigation_accounting_passed=True)
    assert infeasible_result.episode_return < -1.0
    assert infeasible_result.target_feasible is False
    assert feasible_result.episode_return > infeasible_result.episode_return


def test_ppo_reward_rejects_validation_final_test_and_bad_accounting():
    refs = {2018: 1000.0, 2023: 1000.0, 2000: 1000.0}
    with pytest.raises(ValueError, match="2000-2017"):
        PPOEpisodeReward(weather_year=2018, eta=0.90, reference_yield_by_year=refs)
    with pytest.raises(ValueError, match="2000-2017"):
        PPOEpisodeReward(weather_year=2023, eta=0.90, reference_yield_by_year=refs)
    tracker = PPOEpisodeReward(weather_year=2000, eta=0.90, reference_yield_by_year=refs)
    with pytest.raises(RuntimeError, match="accounting"):
        tracker.finish(yield_kg_ha=900.0, irrigation_accounting_passed=False)


def test_rollout_gae_resets_at_episode_boundaries():
    buffer = PPORolloutBuffer(
        state_dim=2, expected_size=4, gamma=1.0, gae_lambda=1.0, policy_version=0
    )
    rewards = [1.0, 2.0, 3.0, 4.0]
    dones = [False, True, False, True]
    for idx, (reward, done) in enumerate(zip(rewards, dones)):
        buffer.append(
            state=np.asarray([idx, idx + 1], dtype=np.float32),
            irrigate=False,
            raw_amount=0.0,
            log_prob=-0.5,
            value=0.0,
            reward=reward,
            done=done,
        )
    batch = buffer.finalize()
    assert batch.returns.tolist() == pytest.approx([3.0, 2.0, 7.0, 4.0])
    assert batch.advantages.tolist() == pytest.approx([3.0, 2.0, 7.0, 4.0])


def _synthetic_batch(agent: PPOAgent, size: int) -> PPORolloutBatch:
    states = torch.randn(size, agent.hparams.state_dim)
    with torch.no_grad():
        action = agent.actor.sample(states, generator=agent.generator)
        values = agent.critic(states)
    returns = values + torch.linspace(-0.5, 0.5, size)
    advantages = returns - values
    return PPORolloutBatch(
        states=states,
        irrigate=action.irrigate,
        raw_amount=action.raw_amount,
        old_log_probs=action.log_prob,
        values=values,
        returns=returns,
        advantages=advantages,
        rewards=torch.zeros(size),
        dones=torch.zeros(size, dtype=torch.bool),
        policy_version=agent.policy_version,
    )


def test_ppo_update_is_finite_strictly_on_policy_and_versioned():
    h = PPOHyperparameters(
        state_dim=4,
        actor_hidden_dims=(16, 8),
        critic_hidden_dims=(16, 8),
        learning_rate=1e-4,
        update_epochs=2,
        minibatch_size=4,
    )
    agent = PPOAgent(hyperparameters=h, seed=21, device="cpu")
    batch = _synthetic_batch(agent, 8)
    stats = agent.update(batch)
    assert stats.sample_count == 8
    assert stats.rollout_policy_version == 0
    assert stats.new_policy_version == 1
    for value in (
        stats.actor_loss,
        stats.value_loss,
        stats.total_loss,
        stats.entropy,
        stats.approx_kl,
        stats.clip_fraction,
        stats.grad_norm,
    ):
        assert np.isfinite(value)
    with pytest.raises(RuntimeError, match="strict on-policy"):
        agent.update(batch)
    payload = agent.checkpoint_payload()
    assert payload["protocol_id"] == "awm-ppo-baseline-v1"
    assert payload["hyperparameters"]["clip_epsilon"] == pytest.approx(0.2)


def test_running_observation_normalizer_round_trip_and_freeze_usage():
    normalizer = RunningObservationNormalizer(state_dim=3)
    data = np.asarray([[1, 2, 3], [3, 4, 5], [5, 6, 7]], dtype=np.float32)
    normalizer.update(data)
    normalized = normalizer.normalize(data)
    assert normalized.shape == data.shape
    assert np.isfinite(normalized).all()
    state = normalizer.state()
    clone = RunningObservationNormalizer(state_dim=3)
    clone.load_state(state)
    assert np.allclose(normalized, clone.normalize(data))


def test_machine_readable_ppo_protocol_matches_implementation_constants():
    config = json.loads((ROOT / "configs" / "ppo_baseline_v1.json").read_text())
    assert config["environment"]["observation_dim"] == 79
    assert config["environment"]["training_years"] == list(range(2000, 2018))
    assert config["environment"]["validation_years"] == list(range(2018, 2023))
    assert config["environment"]["locked_final_test_station_years"] == [2023, 2024, 2025]
    assert config["environment"]["eta_levels"] == [0.90, 0.95, 0.98]
    assert config["rollout"]["balanced_weather_eta_cells_per_update"] == 54
    assert config["rollout"]["transitions_per_update"] == 6750
    assert config["optimizer"]["minibatch_size"] == 450
    assert 6750 % 450 == 0
    assert config["optimizer"]["gamma"] == pytest.approx(1.0)
    assert config["training"]["seeds"] == [11, 21, 31, 41, 51]
    assert config["reference_provenance"]["read_only_repository"] == "wwwwwwangyuhao/lrmb"
