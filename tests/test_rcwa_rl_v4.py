"""Engineering tests for RCWA-RL v4 action-conditioned tail-Q credit."""
from __future__ import annotations

from dataclasses import asdict, replace
import json
import math
from pathlib import Path
from types import MethodType

import pytest
import torch

from awm.rcwa import RCWARolloutBatch, evaluate_eta_tail
from awm.rcwa.agent_v3 import RCWAV3Agent, RCWAV3Hyperparameters
from awm.rcwa.agent_v4 import (
    RCWAV4Agent,
    RCWAV4Hyperparameters,
    RCWAV4UpdateStats,
    TailActionValueNetwork,
)
from awm.rcwa.trainer_v4 import RCWAV4Trainer, V4_PROTOCOL_PATH, v4_hyperparameters_from_protocol


ROOT = Path(__file__).resolve().parents[1]
EPISODE_LENGTH = 25
EPISODES_PER_ETA = 18
ETA_LEVELS = (0.90, 0.95, 0.98)


def _synthetic_batch(*, seed: int = 0, state_dim: int = 2) -> RCWARolloutBatch:
    generator = torch.Generator().manual_seed(seed)
    episode_count = EPISODES_PER_ETA * len(ETA_LEVELS)
    size = episode_count * EPISODE_LENGTH
    states = torch.randn(size, state_dim, generator=generator)
    dones = torch.zeros(size, dtype=torch.bool)
    risk_costs = torch.zeros(size)
    risk_returns = torch.zeros(size)
    etas = torch.zeros(size)
    episode_etas = torch.zeros(episode_count)
    rewards = -torch.rand(size, generator=generator) * 0.01
    irrigate = torch.rand(size, generator=generator) < 0.4
    raw_amount = torch.zeros(size)
    raw_amount[irrigate] = torch.randn(int(irrigate.sum()), generator=generator)
    for index in range(episode_count):
        start = index * EPISODE_LENGTH
        stop = start + EPISODE_LENGTH
        dones[stop - 1] = True
        cost = 1.0 + 0.03 * index
        risk_costs[stop - 1] = cost
        risk_returns[start:stop] = cost
        eta = ETA_LEVELS[index // EPISODES_PER_ETA]
        etas[start:stop] = eta
        episode_etas[index] = eta
    tail_metrics = {
        f"{eta:.2f}": evaluate_eta_tail(
            [0.80 + 0.005 * (i + 5 * level) for i in range(EPISODES_PER_ETA)], eta=eta
        )
        for level, eta in enumerate(ETA_LEVELS)
    }
    return RCWARolloutBatch(
        states=states,
        irrigate=irrigate,
        raw_amount=raw_amount,
        old_log_probs=torch.full((size,), -0.1),
        etas=etas,
        reward_values=torch.zeros(size),
        risk_values=torch.zeros(size),
        reward_returns=torch.zeros(size),
        risk_returns=risk_returns,
        reward_advantages=torch.linspace(-1.0, 1.0, size),
        risk_advantages=risk_returns.clone(),
        rewards=rewards,
        risk_costs=risk_costs,
        dones=dones,
        episode_etas=episode_etas,
        episode_retentions=torch.ones(episode_count),
        episode_risk_costs=torch.zeros(episode_count),
        tail_metrics=tail_metrics,
        policy_version=0,
    )


def _v3_small() -> RCWAV3Hyperparameters:
    return RCWAV3Hyperparameters(
        state_dim=2,
        actor_hidden_dims=(8, 4),
        reward_critic_hidden_dims=(8, 4),
        risk_critic_hidden_dims=(8, 4),
        update_epochs=2,
        minibatch_size=450,
        risk_credit_prefit_epochs=2,
    )


def _v4_small(**overrides) -> RCWAV4Hyperparameters:
    values = dict(
        state_dim=2,
        actor_hidden_dims=(8, 4),
        reward_critic_hidden_dims=(8, 4),
        risk_critic_hidden_dims=(8, 4),
        update_epochs=2,
        minibatch_size=450,
        risk_credit_prefit_epochs=2,
        tail_q_prefit_epochs=2,
    )
    values.update(overrides)
    return RCWAV4Hyperparameters(**values)


def _state_dict_equal(left, right) -> bool:
    return set(left) == set(right) and all(torch.equal(left[k], right[k]) for k in left)


# ---------------------------------------------------------------------------
# Protocol and identity
# ---------------------------------------------------------------------------
def test_v4_protocol_preserves_v3_frozen_invariants_and_changes_only_credit_capacity():
    v3 = json.loads((ROOT / "configs" / "rcwa_rl_v3.json").read_text())
    v4 = json.loads((ROOT / "configs" / "rcwa_rl_v4.json").read_text())
    assert v4["rcwa_protocol_id"] == "awm-rcwa-rl-v4"
    assert v4["supersedes"] == "awm-rcwa-rl-v3"
    for section in (
        "base_contracts",
        "environment",
        "objective",
        "risk_constraint",
        "lagrangian",
        "optimizer",
        "rollout",
        "interaction_budget",
        "training",
        "evaluation",
    ):
        assert v4[section] == v3[section], section
    for key, value in v3["policy"].items():
        if key != "fairness_rule":
            assert v4["policy"][key] == value, key
    credit = v4["tail_credit"]
    assert credit["mechanism"] == "action_conditioned_monte_carlo_tail_q_advantage"
    assert credit["tail_q_hidden_dims"] == v3["policy"]["risk_critic_hidden_dims"]
    assert credit["risk_credit_prefit_epochs"] == v4["optimizer"]["update_epochs"]
    assert credit["tail_q_prefit_epochs"] == v4["optimizer"]["update_epochs"]
    assert credit["extra_environment_episodes"] == 0
    assert credit["no_new_risk_objective"] is True
    assert credit["no_distributional_critic"] is True


def test_v4_trainer_identity_and_hyperparameters_are_separate():
    assert V4_PROTOCOL_PATH == "configs/rcwa_rl_v4.json"
    assert RCWAV4Trainer.TRAINER_PROTOCOL_ID == "awm-rcwa-trainer-v4"
    assert RCWAV4Trainer.RUNTIME_SUBDIR == "rcwa_rl_v4"
    assert RCWAV4Trainer.MANIFEST_ID == "awm-rcwa-run-manifest-v4"
    assert RCWAV4Trainer.AGENT_CLS is RCWAV4Agent
    protocol = json.loads((ROOT / V4_PROTOCOL_PATH).read_text())
    h = v4_hyperparameters_from_protocol(protocol)
    assert isinstance(h, RCWAV4Hyperparameters)
    assert h.risk_credit_prefit_epochs == 10
    assert h.tail_q_prefit_epochs == 10
    assert h.alpha == pytest.approx(0.20)


# ---------------------------------------------------------------------------
# Network/action representation
# ---------------------------------------------------------------------------
def test_v4_preserves_v3_actor_reward_and_risk_value_initialization():
    v3 = RCWAV3Agent(hyperparameters=_v3_small(), seed=17, device="cpu")
    v4 = RCWAV4Agent(hyperparameters=_v4_small(), seed=17, device="cpu")
    assert _state_dict_equal(v3.actor.state_dict(), v4.actor.state_dict())
    assert _state_dict_equal(v3.reward_critic.state_dict(), v4.reward_critic.state_dict())
    assert _state_dict_equal(v3.risk_critic.state_dict(), v4.risk_critic.state_dict())
    assert torch.count_nonzero(v4.tail_q_critic.value_head.weight) == 0
    assert torch.count_nonzero(v4.tail_q_critic.value_head.bias) == 0


def test_tail_action_value_network_validates_state_and_action_shapes():
    q = TailActionValueNetwork(state_dim=3, hidden_dims=(8, 4))
    assert q(torch.zeros(5, 3), torch.zeros(5, 2)).shape == (5,)
    with pytest.raises(ValueError, match="action_features"):
        q(torch.zeros(5, 3), torch.zeros(5, 3))


def test_v4_action_features_encode_noop_gate_and_active_amount_fraction_exactly():
    batch = _synthetic_batch(seed=4)
    gate = batch.irrigate.clone()
    batch = replace(
        batch,
        irrigate=torch.tensor([False, True, True] + gate[3:].tolist()),
        raw_amount=torch.tensor([9.0, 0.0, math.atanh(0.5)] + batch.raw_amount[3:].tolist()),
    )
    features = RCWAV4Agent._action_features_from_batch(batch, device=torch.device("cpu"))
    assert features[0].tolist() == pytest.approx([0.0, 0.0])
    assert features[1].tolist() == pytest.approx([1.0, 0.5])
    assert features[2].tolist() == pytest.approx([1.0, 0.75])


# ---------------------------------------------------------------------------
# Tail-Q prefit and action attribution
# ---------------------------------------------------------------------------
def test_tail_q_prefit_reduces_action_dependent_mc_loss_and_freezes_v3_parameters():
    h = _v4_small(learning_rate=1e-3, tail_q_prefit_epochs=40)
    agent = RCWAV4Agent(hyperparameters=h, seed=9, device="cpu")
    g = torch.Generator().manual_seed(12)
    states = torch.randn(1350, 2, generator=g)
    gate = (torch.rand(1350, generator=g) > 0.5).float()
    amount = torch.rand(1350, generator=g) * gate
    action = torch.stack((gate, amount), dim=-1)
    targets = 0.2 * states[:, 0] - 0.1 * states[:, 1] + 1.5 * gate + 2.0 * amount
    actor_before = {k: v.clone() for k, v in agent.actor.state_dict().items()}
    reward_before = {k: v.clone() for k, v in agent.reward_critic.state_dict().items()}
    risk_before = {k: v.clone() for k, v in agent.risk_critic.state_dict().items()}
    rng_before = agent.generator.get_state().clone()
    d = agent._prefit_tail_q_critic(states=states, action_features=action, risk_returns=targets)
    assert d["tail_q_prefit_loss_after"] < d["tail_q_prefit_loss_before"]
    assert d["tail_q_prefit_grad_norm_mean"] > 0.0
    assert torch.equal(agent.generator.get_state(), rng_before)
    assert _state_dict_equal(actor_before, agent.actor.state_dict())
    assert _state_dict_equal(reward_before, agent.reward_critic.state_dict())
    assert _state_dict_equal(risk_before, agent.risk_critic.state_dict())


def test_tail_q_minus_v_localizes_action_amount_at_same_state():
    h = _v4_small(learning_rate=2e-3, risk_credit_prefit_epochs=50, tail_q_prefit_epochs=50)
    agent = RCWAV4Agent(hyperparameters=h, seed=23, device="cpu")
    n = 1350
    states = torch.zeros(n, 2)
    gate = torch.zeros(n)
    gate[n // 3 :] = 1.0
    amount = torch.zeros(n)
    amount[n // 3 :] = torch.linspace(0.1, 0.9, n - n // 3)
    action = torch.stack((gate, amount), dim=-1)
    targets = 0.2 + gate * (0.5 + 2.5 * amount)
    agent._prefit_risk_critic(states=states, risk_returns=targets)
    agent._prefit_tail_q_critic(states=states, action_features=action, risk_returns=targets)
    with torch.no_grad():
        credit = agent.tail_q_critic(states, action) - agent.risk_critic(states)
    noop = gate == 0
    active = gate == 1
    assert float(credit[active].mean()) > float(credit[noop].mean()) + 0.4
    assert agent._safe_correlation(amount[active], credit[active]) > 0.75


def test_v4_update_emits_tail_q_diagnostics_and_uses_action_conditioned_credit():
    agent = RCWAV4Agent(hyperparameters=_v4_small(), seed=13, device="cpu")
    batch = _synthetic_batch(seed=6)
    stats = agent.update(batch)
    assert isinstance(stats, RCWAV4UpdateStats)
    assert stats.tail_q_prefit_steps == 2 * (1350 // 450)
    assert stats.tail_q_prefit_loss_after <= stats.tail_q_prefit_loss_before
    assert set(stats.tail_q_credit_std_by_eta) == {"0.90", "0.95", "0.98"}
    assert set(stats.tail_q_credit_early_window_amount_correlation_by_eta) == {"0.90", "0.95", "0.98"}
    assert stats.actor_grad_norm > 0.0


# ---------------------------------------------------------------------------
# Attribution isolation relative to v3
# ---------------------------------------------------------------------------
def test_v4_extra_q_prefit_does_not_change_v3_update_when_credit_is_forced_to_v3_td():
    batch = _synthetic_batch(seed=18)
    v3 = RCWAV3Agent(hyperparameters=_v3_small(), seed=21, device="cpu")
    v4 = RCWAV4Agent(hyperparameters=_v4_small(), seed=21, device="cpu")

    def v3_equivalent_credit(self, *, batch, states, etas):
        risk_returns = batch.risk_returns.to(self.device)
        self._prefit_risk_critic(states=states, risk_returns=risk_returns)
        action = self._action_features_from_batch(batch, device=self.device)
        self._prefit_tail_q_critic(states=states, action_features=action, risk_returns=risk_returns)
        credit, _values, _error = self._frozen_tail_td_credit(
            states=states,
            risk_costs=batch.risk_costs.to(self.device),
            dones=batch.dones.to(self.device),
        )
        return credit, {}

    v4._actor_risk_credit = MethodType(v3_equivalent_credit, v4)
    v3.update(batch)
    v4.update(batch)
    assert _state_dict_equal(v3.actor.state_dict(), v4.actor.state_dict())
    assert _state_dict_equal(v3.reward_critic.state_dict(), v4.reward_critic.state_dict())
    assert _state_dict_equal(v3.risk_critic.state_dict(), v4.risk_critic.state_dict())
    assert torch.equal(v3.generator.get_state(), v4.generator.get_state())
    assert v3.dual_by_eta == v4.dual_by_eta


# ---------------------------------------------------------------------------
# Checkpoint isolation
# ---------------------------------------------------------------------------
def test_v4_checkpoint_roundtrip_includes_tail_q_and_rejects_v3_payload():
    h4 = _v4_small()
    source = RCWAV4Agent(hyperparameters=h4, seed=31, device="cpu")
    source.update(_synthetic_batch(seed=7))
    payload = source.checkpoint_payload()
    assert payload["protocol_id"] == "awm-rcwa-rl-v4"
    assert "tail_q_critic_state_dict" in payload
    restored = RCWAV4Agent(hyperparameters=h4, seed=31, device="cpu")
    restored.load_checkpoint_payload(payload)
    assert _state_dict_equal(source.tail_q_critic.state_dict(), restored.tail_q_critic.state_dict())
    assert _state_dict_equal(source.actor.state_dict(), restored.actor.state_dict())
    assert source.dual_by_eta == restored.dual_by_eta
    assert torch.equal(source.generator.get_state(), restored.generator.get_state())

    v3 = RCWAV3Agent(hyperparameters=_v3_small(), seed=31, device="cpu")
    with pytest.raises(ValueError, match="protocol_id mismatch"):
        restored.load_checkpoint_payload(v3.checkpoint_payload())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for v4 device regression")
def test_v4_full_update_runs_on_cuda():
    agent = RCWAV4Agent(hyperparameters=_v4_small(), seed=41, device="cuda:0")
    stats = agent.update(_synthetic_batch(seed=11))
    assert stats.update_index == 1
    assert math.isfinite(stats.actor_grad_norm)
    assert math.isfinite(stats.tail_q_prefit_loss_after)
    assert agent.tail_q_critic.value_head.weight.device.type == "cuda"
