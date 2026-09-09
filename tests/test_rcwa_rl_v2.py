from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from awm.ppo.agent import PPOAgent
from awm.rcwa import RCWAAgent
from awm.rcwa.trainer import PROTOCOL_PATH


ROOT = Path(__file__).resolve().parents[1]


def test_v2_keeps_actor_and_reward_critic_fairness_but_zeroes_extra_risk_head():
    ppo = PPOAgent(seed=21, device="cpu")
    rcwa = RCWAAgent(seed=21, device="cpu")
    for key, value in ppo.actor.state_dict().items():
        assert torch.equal(value, rcwa.actor.state_dict()[key]), key
    for key, value in ppo.critic.state_dict().items():
        assert torch.equal(value, rcwa.reward_critic.state_dict()[key]), key
    assert torch.count_nonzero(rcwa.risk_critic.value_head.weight).item() == 0
    assert torch.count_nonzero(rcwa.risk_critic.value_head.bias).item() == 0


def test_component_conditioning_preserves_dual_scale_for_tiny_tail_signal():
    agent = RCWAAgent(seed=11, device="cpu")
    etas = torch.tensor([0.90] * 4 + [0.95] * 4 + [0.98] * 4)
    reward_adv = torch.zeros(12)
    tiny_tail = torch.tensor([0.0, 0.0, 0.0, 1e-7] * 3)

    combined_one, diagnostics = agent._condition_actor_advantages(
        reward_adv=reward_adv,
        risk_adv=tiny_tail,
        etas=etas,
    )
    std_one = combined_one.std(unbiased=False).item()
    assert std_one > 0.5
    assert all(
        diagnostics["risk_advantage_std_before_conditioning_by_eta"][key] > 0.0
        for key in ("0.90", "0.95", "0.98")
    )

    agent.dual_by_eta = {0.90: 8.0, 0.95: 8.0, 0.98: 8.0}
    combined_eight, _ = agent._condition_actor_advantages(
        reward_adv=reward_adv,
        risk_adv=tiny_tail,
        etas=etas,
    )
    assert combined_eight.std(unbiased=False).item() == pytest.approx(8.0 * std_one)


def test_v2_protocol_preserves_risk_contract_and_budget():
    v1 = json.loads((ROOT / "configs" / "rcwa_rl_v1.json").read_text())
    v2 = json.loads((ROOT / "configs" / "rcwa_rl_v2.json").read_text())
    assert v2["rcwa_protocol_id"] == "awm-rcwa-rl-v2"
    assert v2["risk_constraint"]["alpha"] == v1["risk_constraint"]["alpha"] == 0.20
    assert v2["risk_constraint"]["constraint"] == v1["risk_constraint"]["constraint"]
    assert v2["risk_constraint"]["tau_estimator"] == v1["risk_constraint"]["tau_estimator"]
    assert v2["environment"] == v1["environment"]
    assert v2["rollout"] == v1["rollout"]
    assert v2["interaction_budget"] == v1["interaction_budget"]
    assert v2["training"] == v1["training"]
    assert "do not standardize the combined advantage" in v2["optimizer"]["advantage_normalization"]
    assert PROTOCOL_PATH == "configs/rcwa_rl_v2.json"


def test_v2_checkpoint_protocol_is_not_loadable_as_v1():
    agent = RCWAAgent(seed=31, device="cpu")
    payload = agent.checkpoint_payload()
    assert payload["protocol_id"] == "awm-rcwa-rl-v2"
    bad = dict(payload)
    bad["protocol_id"] = "awm-rcwa-rl-v1"
    with pytest.raises(ValueError, match="protocol_id mismatch"):
        agent.load_checkpoint_payload(bad)


def test_v2_update_reports_separate_gradient_norms():
    from awm.rcwa import RCWAHyperparameters, RCWARolloutBatch, evaluate_eta_tail

    h = RCWAHyperparameters(
        state_dim=2,
        actor_hidden_dims=(8, 4),
        reward_critic_hidden_dims=(8, 4),
        risk_critic_hidden_dims=(8, 4),
        update_epochs=1,
        minibatch_size=450,
    )
    agent = RCWAAgent(hyperparameters=h, seed=7, device="cpu")
    n = 450
    states = torch.randn(n, 2)
    irrigate = torch.zeros(n, dtype=torch.bool)
    raw_amount = torch.zeros(n)
    with torch.no_grad():
        old_log_probs, _ = agent.actor.evaluate_behavior(states, irrigate, raw_amount)
    etas = torch.tensor([0.90] * 150 + [0.95] * 150 + [0.98] * 150)
    reward_adv = torch.linspace(-1.0, 1.0, n)
    risk_adv = torch.tensor(([0.0] * 149 + [1e-4]) * 3)
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
    assert stats.reward_critic_grad_norm >= 0.0
    assert stats.risk_critic_grad_norm >= 0.0
    assert stats.combined_advantage_std_after_conditioning > 0.0
    assert set(stats.risk_advantage_std_before_conditioning_by_eta) == {
        "0.90",
        "0.95",
        "0.98",
    }
