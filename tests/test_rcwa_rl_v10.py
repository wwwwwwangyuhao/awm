from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from awm.ppo.agent import PPOAgent
from awm.rcwa.agent_v10 import RCWAV10Agent, V10_PROTOCOL_ID
from awm.rcwa.buffer import RCWARolloutBatch
from awm.rcwa.risk_batch import evaluate_eta_tail
from awm.rcwa.trainer_v10 import PROTOCOL_PATH

ROOT = Path(__file__).resolve().parents[1]


def test_v10_preserves_ppo_actor_and_reward_critic_initialization():
    ppo = PPOAgent(seed=21, device="cpu")
    agent = RCWAV10Agent(seed=21, device="cpu")
    for key, value in ppo.actor.state_dict().items():
        assert torch.equal(value, agent.actor.state_dict()[key]), key
    for key, value in ppo.critic.state_dict().items():
        assert torch.equal(value, agent.reward_critic.state_dict()[key]), key
    assert torch.count_nonzero(agent.risk_critic.value_head.weight).item() == 0
    assert torch.count_nonzero(agent.risk_critic.value_head.bias).item() == 0


def test_v10_actor_advantage_is_exact_raw_primal_dual_signal():
    agent = RCWAV10Agent(seed=11, device="cpu")
    agent.dual_by_eta = {0.90: 1.0, 0.95: 2.0, 0.98: 3.0}
    etas = torch.tensor([0.90, 0.90, 0.95, 0.95, 0.98, 0.98])
    reward = torch.tensor([1.0, -2.0, 3.0, -4.0, 5.0, -6.0])
    risk = torch.tensor([0.1, 0.2, -0.3, 0.4, 0.5, -0.6])
    combined, diagnostics = agent._condition_actor_advantages(
        reward_adv=reward,
        risk_adv=risk,
        etas=etas,
    )
    expected = reward - torch.tensor([1.0, 1.0, 2.0, 2.0, 3.0, 3.0]) * risk
    assert torch.equal(combined, expected)
    assert diagnostics["reward_advantage_mean_before_conditioning"] == pytest.approx(
        reward.mean().item()
    )
    assert diagnostics["combined_advantage_std_after_conditioning"] == pytest.approx(
        expected.std(unbiased=False).item()
    )


def test_v10_does_not_amplify_tiny_mc_tail_signal():
    agent = RCWAV10Agent(seed=11, device="cpu")
    reward = torch.zeros(12)
    etas = torch.tensor([0.90] * 4 + [0.95] * 4 + [0.98] * 4)
    tiny = torch.tensor([0.0, 0.0, 0.0, 1e-7] * 3)
    combined, _ = agent._condition_actor_advantages(
        reward_adv=reward, risk_adv=tiny, etas=etas
    )
    assert combined.abs().max().item() == pytest.approx(1e-7, rel=1e-5)


def test_v10_protocol_is_raw_mc_and_preserves_formal_contracts():
    v2 = json.loads((ROOT / "configs" / "rcwa_rl_v2.json").read_text())
    v10 = json.loads((ROOT / "configs" / "rcwa_rl_v10.json").read_text())
    assert v10["rcwa_protocol_id"] == V10_PROTOCOL_ID
    assert v10["environment"] == v2["environment"]
    assert v10["rollout"] == v2["rollout"]
    assert v10["interaction_budget"] == v2["interaction_budget"]
    assert v10["training"] == v2["training"]
    assert v10["risk_constraint"]["alpha"] == v2["risk_constraint"]["alpha"]
    assert v10["risk_constraint"]["constraint"] == v2["risk_constraint"]["constraint"]
    assert v10["optimizer"]["combined_actor_advantage"] == (
        "A_reward - lambda_eta * A_tail_risk_MC"
    )
    assert v10["optimizer"]["advantage_normalization"].startswith("none")
    assert PROTOCOL_PATH == "configs/rcwa_rl_v10.json"


def test_v10_checkpoint_protocol_is_distinct_and_roundtrips_cpu():
    agent = RCWAV10Agent(seed=31, device="cpu")
    payload = agent.checkpoint_payload()
    assert payload["protocol_id"] == V10_PROTOCOL_ID
    restored = RCWAV10Agent(seed=31, device="cpu")
    restored.load_checkpoint_payload(payload)
    assert restored.policy_version == agent.policy_version
    assert restored.update_index == agent.update_index
    assert torch.equal(restored.generator.get_state(), agent.generator.get_state())
    bad = dict(payload)
    bad["protocol_id"] = "awm-rcwa-rl-v2"
    with pytest.raises(ValueError, match="protocol_id mismatch"):
        restored.load_checkpoint_payload(bad)


def test_v10_synthetic_update_uses_direct_mc_credit_without_tail_q():
    from awm.rcwa.agent import RCWAHyperparameters

    h = RCWAHyperparameters(
        state_dim=2,
        actor_hidden_dims=(8, 4),
        reward_critic_hidden_dims=(8, 4),
        risk_critic_hidden_dims=(8, 4),
        update_epochs=1,
        minibatch_size=450,
    )
    agent = RCWAV10Agent(hyperparameters=h, seed=7, device="cpu")
    n = 450
    states = torch.randn(n, 2)
    irrigate = torch.zeros(n, dtype=torch.bool)
    raw_amount = torch.zeros(n)
    with torch.no_grad():
        old_log_probs, _ = agent.actor.evaluate_behavior(states, irrigate, raw_amount)
    etas = torch.tensor([0.90] * 150 + [0.95] * 150 + [0.98] * 150)
    reward_adv = torch.linspace(-0.2, 0.2, n)
    risk_adv = torch.linspace(0.3, -0.1, n)
    tail_metrics = {
        "0.90": evaluate_eta_tail([0.80 + 0.005 * i for i in range(18)], eta=0.90),
        "0.95": evaluate_eta_tail([0.82 + 0.005 * i for i in range(18)], eta=0.95),
        "0.98": evaluate_eta_tail([0.84 + 0.005 * i for i in range(18)], eta=0.98),
    }
    batch = RCWARolloutBatch(
        states=states,
        irrigate=irrigate,
        raw_amount=raw_amount,
        old_log_probs=old_log_probs,
        etas=etas,
        reward_values=torch.zeros(n),
        risk_values=torch.zeros(n),
        reward_returns=reward_adv.clone(),
        risk_returns=risk_adv.clone(),
        reward_advantages=reward_adv,
        risk_advantages=risk_adv,
        rewards=torch.zeros(n),
        risk_costs=torch.zeros(n),
        dones=torch.zeros(n, dtype=torch.bool),
        episode_etas=torch.tensor([0.90] * 18 + [0.95] * 18 + [0.98] * 18),
        episode_retentions=torch.ones(54),
        episode_risk_costs=torch.zeros(54),
        tail_metrics=tail_metrics,
        policy_version=0,
    )
    stats = agent.update(batch)
    assert stats.update_index == 1
    assert stats.actor_grad_norm >= 0.0
    assert stats.combined_advantage_std_after_conditioning > 0.0
    assert not hasattr(agent, "tail_q_critic")
