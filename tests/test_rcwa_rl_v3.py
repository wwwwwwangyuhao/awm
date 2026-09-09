"""RCWA-RL v3 tests: frozen one-step tail TD credit plus risk-critic prefit.

These tests are engineering evidence only.  They show that the temporal-credit
mechanism is implemented as specified; they say nothing about whether v3
improves lower-CVaR retention on DSSAT, which is NOT TESTED here.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from awm.ppo.agent import PPOAgent
from awm.rcwa import (
    RCWAAgent,
    RCWARolloutBatch,
    RCWAV3Agent,
    RCWAV3Hyperparameters,
    RCWAV3UpdateStats,
    evaluate_eta_tail,
    next_risk_values,
    tail_credit_telescoping_errors,
    tail_credit_telescoping_terms,
    tail_td_credit,
    verify_tail_credit_telescoping,
)
from awm.rcwa.trainer_v3 import RCWAV3Trainer, V3_PROTOCOL_PATH, v3_hyperparameters_from_protocol


ROOT = Path(__file__).resolve().parents[1]
EPISODE_LENGTH = 25
EPISODES_PER_ETA = 18
ETA_LEVELS = (0.90, 0.95, 0.98)


def _synthetic_batch(
    *,
    state_dim: int = 2,
    terminal_cost: float = 2.0,
    seed: int = 0,
) -> RCWARolloutBatch:
    """54 complete synthetic episodes of 25 steps (1350 transitions).

    Mirrors the formal layout: contiguous complete episodes, terminal-only
    risk cost, and the Monte-Carlo risk return repeated over each episode.
    """
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
    for index in range(episode_count):
        start = index * EPISODE_LENGTH
        stop = start + EPISODE_LENGTH
        dones[stop - 1] = True
        cost = terminal_cost * (1.0 + 0.05 * index)
        risk_costs[stop - 1] = cost
        risk_returns[start:stop] = cost
        eta = ETA_LEVELS[index // EPISODES_PER_ETA]
        etas[start:stop] = eta
        episode_etas[index] = eta
    irrigate = torch.rand(size, generator=generator) < 0.3
    tail_metrics = {
        f"{eta:.2f}": evaluate_eta_tail(
            [0.80 + 0.005 * (i + 5 * level) for i in range(EPISODES_PER_ETA)], eta=eta
        )
        for level, eta in enumerate(ETA_LEVELS)
    }
    return RCWARolloutBatch(
        states=states,
        irrigate=irrigate,
        raw_amount=torch.zeros(size),
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


def _small_hyperparameters(**overrides) -> RCWAV3Hyperparameters:
    defaults = dict(
        state_dim=2,
        actor_hidden_dims=(8, 4),
        reward_critic_hidden_dims=(8, 4),
        risk_critic_hidden_dims=(8, 4),
        update_epochs=2,
        minibatch_size=450,
    )
    defaults.update(overrides)
    return RCWAV3Hyperparameters(**defaults)


# ---------------------------------------------------------------------------
# 1. Protocol invariants
# ---------------------------------------------------------------------------
def test_v3_protocol_preserves_every_frozen_v2_invariant():
    v2 = json.loads((ROOT / "configs" / "rcwa_rl_v2.json").read_text(encoding="utf-8"))
    v3 = json.loads((ROOT / "configs" / "rcwa_rl_v3.json").read_text(encoding="utf-8"))
    assert v3["rcwa_protocol_id"] == "awm-rcwa-rl-v3"
    assert v3["supersedes"] == "awm-rcwa-rl-v2"
    for section in (
        "base_contracts",
        "environment",
        "policy",
        "objective",
        "risk_constraint",
        "lagrangian",
        "optimizer",
        "rollout",
        "interaction_budget",
        "training",
        "evaluation",
    ):
        assert v3[section] == v2[section], section
    assert v3["risk_constraint"]["alpha"] == 0.20
    assert v3["environment"]["policy_water_budget_mm"] == 495.0
    assert v3["interaction_budget"]["max_updates_per_seed"] == 200
    assert v3["training"]["seeds"] == [11, 21, 31, 41, 51]
    assert v3["training"]["candidate_checkpoint_interval_updates"] == 10
    assert v3["lagrangian"]["extra_environment_episodes"] == 0
    assert v3["tail_credit"]["extra_environment_episodes"] == 0
    assert v3["tail_credit"]["risk_credit_prefit_epochs"] == v3["optimizer"]["update_epochs"]
    assert v3["tail_credit"]["risk_credit_prefit_minibatch_size"] == v3["optimizer"]["minibatch_size"]
    assert v3["tail_credit"]["risk_credit_prefit_trainable_parameters"] == ["risk_critic"]
    assert V3_PROTOCOL_PATH == "configs/rcwa_rl_v3.json"


def test_v3_trainer_identity_is_separate_from_v2():
    from awm.rcwa.trainer import RCWATrainer

    assert RCWAV3Trainer.PROTOCOL_RELPATH == "configs/rcwa_rl_v3.json"
    assert RCWAV3Trainer.TRAINER_PROTOCOL_ID == "awm-rcwa-trainer-v3"
    assert RCWAV3Trainer.RUNTIME_SUBDIR == "rcwa_rl_v3"
    assert RCWAV3Trainer.MANIFEST_ID == "awm-rcwa-run-manifest-v3"
    assert RCWAV3Trainer.AGENT_CLS is RCWAV3Agent
    assert (RCWATrainer.PROTOCOL_RELPATH, RCWATrainer.RUNTIME_SUBDIR) == (
        "configs/rcwa_rl_v2.json",
        "rcwa_rl_v2",
    )
    protocol = json.loads((ROOT / V3_PROTOCOL_PATH).read_text(encoding="utf-8"))
    hparams = v3_hyperparameters_from_protocol(protocol)
    assert isinstance(hparams, RCWAV3Hyperparameters)
    assert hparams.risk_credit_prefit_epochs == 10
    assert hparams.alpha == pytest.approx(0.20)
    assert hparams.risk_gamma == 1.0


# ---------------------------------------------------------------------------
# 2-4. Pure TD-credit helpers
# ---------------------------------------------------------------------------
def test_tail_td_credit_telescopes_exactly_on_synthetic_complete_episodes():
    generator = torch.Generator().manual_seed(11)
    for _ in range(20):
        length = int(torch.randint(2, 30, (1,), generator=generator).item())
        values = torch.randn(length, generator=generator, dtype=torch.float64)
        costs = torch.zeros(length, dtype=torch.float64)
        costs[-1] = float(torch.rand((1,), generator=generator).item()) * 5.0
        dones = torch.zeros(length, dtype=torch.bool)
        dones[-1] = True
        credit = tail_td_credit(
            costs=costs,
            values=values,
            next_values=next_risk_values(values, dones),
            dones=dones,
            gamma=1.0,
        )
        errors = tail_credit_telescoping_errors(
            credits=credit, costs=costs, values=values, dones=dones, gamma=1.0
        )
        assert errors == pytest.approx([0.0], abs=1e-9)
        achieved, expected = tail_credit_telescoping_terms(
            credits=credit, costs=costs, values=values, dones=dones, gamma=1.0
        )
        # sum_t delta_t = h_i - V_r(s_0)
        assert float(achieved[0]) == pytest.approx(float(costs.sum() - values[0]), abs=1e-12)
        assert float(expected[0]) == pytest.approx(float(costs.sum() - values[0]), abs=1e-12)


def test_tail_td_credit_telescopes_within_float32_summation_noise():
    values = torch.randn(125, generator=torch.Generator().manual_seed(5))
    costs = torch.zeros(125)
    costs[-1] = 3.0
    dones = torch.zeros(125, dtype=torch.bool)
    dones[-1] = True
    credit = tail_td_credit(
        costs=costs,
        values=values,
        next_values=next_risk_values(values, dones),
        dones=dones,
        gamma=1.0,
    )
    errors = tail_credit_telescoping_errors(
        credits=credit, costs=costs, values=values, dones=dones, gamma=1.0
    )
    assert errors[0] < 1e-5
    assert credit.dtype is torch.float32


def test_tail_td_credit_localizes_a_value_jump_instead_of_spreading_it():
    values = torch.tensor([0.0, 0.0, 0.0, 3.0, 3.0])
    costs = torch.tensor([0.0, 0.0, 0.0, 0.0, 2.0])
    dones = torch.tensor([False, False, False, False, True])
    credit = tail_td_credit(
        costs=costs,
        values=values,
        next_values=next_risk_values(values, dones),
        dones=dones,
        gamma=1.0,
    )
    assert float(credit[2]) == pytest.approx(3.0)
    assert float(credit[0]) == pytest.approx(0.0)
    assert float(credit[1]) == pytest.approx(0.0)
    # The Monte-Carlo credit would have put h/5 = 0.4 on every transition.
    assert float(credit[2]) > 5.0 * (float(costs.sum()) / 5.0)
    assert float(credit.sum()) == pytest.approx(2.0)


def test_terminal_only_cost_does_not_bootstrap_across_done_boundaries():
    values = torch.tensor([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    costs = torch.tensor([0.0, 0.0, 5.0, 0.0, 0.0, 7.0])
    dones = torch.tensor([False, False, True, False, False, True])
    next_values = next_risk_values(values, dones)
    assert float(next_values[2]) == 0.0
    assert float(next_values[5]) == 0.0
    credit = tail_td_credit(
        costs=costs, values=values, next_values=next_values, dones=dones, gamma=1.0
    )
    assert float(credit[2]) == pytest.approx(5.0 - 3.0)
    assert float(credit[5]) == pytest.approx(7.0 - 3.0)
    errors = tail_credit_telescoping_errors(
        credits=credit, costs=costs, values=values, dones=dones, gamma=1.0
    )
    assert errors == pytest.approx([0.0, 0.0], abs=1e-9)


def test_telescoping_verifier_rejects_a_broken_credit():
    values = torch.tensor([1.0, 2.0])
    costs = torch.tensor([0.0, 3.0])
    dones = torch.tensor([False, True])
    credit = tail_td_credit(
        costs=costs,
        values=values,
        next_values=next_risk_values(values, dones),
        dones=dones,
        gamma=1.0,
    )
    assert verify_tail_credit_telescoping(
        credits=credit, costs=costs, values=values, dones=dones, gamma=1.0
    ) == pytest.approx(0.0, abs=1e-9)
    with pytest.raises(RuntimeError, match="telescoping identity"):
        verify_tail_credit_telescoping(
            credits=credit + 0.5, costs=costs, values=values, dones=dones, gamma=1.0
        )


# ---------------------------------------------------------------------------
# 5. Prefit behavior
# ---------------------------------------------------------------------------
def test_prefit_reduces_risk_value_mse_and_freezes_actor_and_reward_critic():
    agent = RCWAV3Agent(hyperparameters=_small_hyperparameters(), seed=5, device="cpu")
    generator = torch.Generator().manual_seed(3)
    states = torch.randn(1350, 2, generator=generator)
    risk_returns = 2.0 * states[:, 0] - states[:, 1] + 1.5
    actor_before = {k: v.clone() for k, v in agent.actor.state_dict().items()}
    reward_before = {k: v.clone() for k, v in agent.reward_critic.state_dict().items()}
    risk_before = {k: v.clone() for k, v in agent.risk_critic.state_dict().items()}

    diagnostics = agent._prefit_risk_critic(states=states, risk_returns=risk_returns)

    assert diagnostics["risk_credit_prefit_loss_after"] < diagnostics["risk_credit_prefit_loss_before"]
    assert diagnostics["risk_credit_prefit_grad_norm_mean"] > 0.0
    assert diagnostics["risk_credit_prefit_steps"] == 10 * (1350 // 450)
    for key, value in agent.actor.state_dict().items():
        assert torch.equal(value, actor_before[key]), key
    for key, value in agent.reward_critic.state_dict().items():
        assert torch.equal(value, reward_before[key]), key
    changed = [
        key
        for key, value in agent.risk_critic.state_dict().items()
        if not torch.equal(value, risk_before[key])
    ]
    assert changed, "prefit must move the risk critic"


def test_prefit_isolates_the_shared_agent_generator_state():
    """The prefit must not advance the actor/behavior RNG timeline.

    ``self.generator`` is shared by actor action sampling and by the main PPO
    minibatch permutations, so a prefit that leaves it advanced would make a
    v3 update differ from v2 even when the actor's risk credit is identical.
    """
    agent = RCWAV3Agent(hyperparameters=_small_hyperparameters(), seed=5, device="cpu")
    generator = torch.Generator().manual_seed(3)
    states = torch.randn(1350, 2, generator=generator)
    risk_returns = 2.0 * states[:, 0] - states[:, 1] + 1.5
    risk_before = {k: v.clone() for k, v in agent.risk_critic.state_dict().items()}
    generator_state_before = agent.generator.get_state().clone()

    diagnostics = agent._prefit_risk_critic(states=states, risk_returns=risk_returns)

    assert torch.equal(agent.generator.get_state(), generator_state_before)
    # The prefit still has to do real work: lower MSE and moved parameters.
    assert diagnostics["risk_credit_prefit_loss_after"] < diagnostics["risk_credit_prefit_loss_before"]
    assert diagnostics["risk_credit_prefit_steps"] == 10 * (1350 // 450)
    assert [
        key
        for key, value in agent.risk_critic.state_dict().items()
        if not torch.equal(value, risk_before[key])
    ]


def test_prefit_generator_isolation_holds_on_the_exception_path():
    agent = RCWAV3Agent(hyperparameters=_small_hyperparameters(), seed=5, device="cpu")
    states = torch.randn(1350, 2, generator=torch.Generator().manual_seed(3))
    risk_returns = torch.full((1350,), float("nan"))
    generator_state_before = agent.generator.get_state().clone()

    with pytest.raises(FloatingPointError, match="NaN/Inf"):
        agent._prefit_risk_critic(states=states, risk_returns=risk_returns)

    assert torch.equal(agent.generator.get_state(), generator_state_before)


def test_v3_with_v2_risk_credit_is_bit_identical_to_v2_actor_and_generator():
    """Controlled attribution test for the only v2->v3 mechanism change.

    v3 performs its real risk-critic prefit but is monkeypatched to hand the
    actor the v2 Monte-Carlo risk advantage.  Under that same-credit condition
    the actor parameters and the shared generator must be bit-identical to a
    v2 update.  Before the RNG isolation this failed: the prefit advanced the
    shared generator, so the PPO minibatch permutations differed.
    """
    hyperparameters = _small_hyperparameters()
    batch = _synthetic_batch(seed=1)
    v2 = RCWAAgent(hyperparameters=hyperparameters, seed=21, device="cpu")
    v3 = RCWAV3Agent(hyperparameters=hyperparameters, seed=21, device="cpu")

    def v2_credit_after_prefit(**kwargs):
        v3._prefit_risk_critic(
            states=kwargs["states"],
            risk_returns=kwargs["batch"].risk_returns.to(v3.device),
        )
        return kwargs["batch"].risk_advantages.to(v3.device), {}

    v3._actor_risk_credit = v2_credit_after_prefit  # type: ignore[method-assign]

    v2.update(batch)
    v3.update(batch)

    v2_actor = v2.actor.state_dict()
    v3_actor = v3.actor.state_dict()
    assert set(v2_actor) == set(v3_actor)
    for key, value in v2_actor.items():
        assert torch.equal(value, v3_actor[key]), key
    assert torch.equal(v2.generator.get_state(), v3.generator.get_state())


# ---------------------------------------------------------------------------
# 6. Frozen, detached actor credit
# ---------------------------------------------------------------------------
def test_actor_credit_is_detached_and_computed_once_per_update():
    agent = RCWAV3Agent(
        hyperparameters=_small_hyperparameters(risk_credit_prefit_epochs=2),
        seed=7,
        device="cpu",
    )
    batch = _synthetic_batch()
    observed: list[torch.Tensor] = []
    original = agent._actor_risk_credit

    def spy(**kwargs):
        credit, diagnostics = original(**kwargs)
        observed.append(credit)
        return credit, diagnostics

    agent._actor_risk_credit = spy  # type: ignore[method-assign]
    stats = agent.update(batch)

    assert len(observed) == 1
    credit = observed[0]
    assert credit.requires_grad is False
    assert credit.grad_fn is None
    assert isinstance(stats, RCWAV3UpdateStats)
    assert stats.update_index == 1


# ---------------------------------------------------------------------------
# 7-8. Initialization fairness
# ---------------------------------------------------------------------------
def test_v3_keeps_actor_and_reward_critic_initialization_equal_to_ppo():
    ppo = PPOAgent(seed=21, device="cpu")
    rcwa = RCWAV3Agent(seed=21, device="cpu")
    for key, value in ppo.actor.state_dict().items():
        assert torch.equal(value, rcwa.actor.state_dict()[key]), key
    for key, value in ppo.critic.state_dict().items():
        assert torch.equal(value, rcwa.reward_critic.state_dict()[key]), key


def test_v3_risk_critic_output_head_is_zero_initialized_before_prefit():
    agent = RCWAV3Agent(seed=41, device="cpu")
    assert torch.count_nonzero(agent.risk_critic.value_head.weight).item() == 0
    assert torch.count_nonzero(agent.risk_critic.value_head.bias).item() == 0
    with torch.no_grad():
        values = agent.risk_critic(torch.randn(4, 79))
    assert torch.count_nonzero(values).item() == 0


# ---------------------------------------------------------------------------
# 9. Checkpoint identity
# ---------------------------------------------------------------------------
def test_v3_checkpoint_roundtrip_preserves_protocol_duals_and_rng_and_rejects_v2():
    agent = RCWAV3Agent(hyperparameters=_small_hyperparameters(), seed=31, device="cpu")
    agent.update(_synthetic_batch(seed=2))
    agent.dual_by_eta[0.90] = 1.2
    agent.dual_by_eta[0.95] = 0.4
    payload = agent.checkpoint_payload()
    assert payload["protocol_id"] == "awm-rcwa-rl-v3"
    assert payload["hyperparameters"]["risk_credit_prefit_epochs"] == 10
    expected_draw = torch.rand(5, generator=agent.generator)

    restored = RCWAV3Agent(hyperparameters=_small_hyperparameters(), seed=31, device="cpu")
    restored.load_checkpoint_payload(payload)
    assert torch.rand(5, generator=restored.generator).tolist() == pytest.approx(
        expected_draw.tolist()
    )
    assert restored.dual_by_eta == pytest.approx(agent.dual_by_eta)
    assert restored.policy_version == agent.policy_version
    assert restored.update_index == agent.update_index
    assert restored.hparams == agent.hparams
    original_state = agent.optimizer.state_dict()["state"]
    restored_state = restored.optimizer.state_dict()["state"]
    assert set(restored_state) == set(original_state)
    for index, group in restored_state.items():
        assert int(group["step"]) == int(original_state[index]["step"])

    v2_payload = RCWAAgent(seed=31, device="cpu").checkpoint_payload()
    assert v2_payload["protocol_id"] == "awm-rcwa-rl-v2"
    with pytest.raises(ValueError, match="protocol_id mismatch"):
        restored.load_checkpoint_payload(v2_payload)


# ---------------------------------------------------------------------------
# 10. End-to-end update diagnostics
# ---------------------------------------------------------------------------
def test_v3_update_reports_tail_credit_diagnostics():
    agent = RCWAV3Agent(
        hyperparameters=_small_hyperparameters(risk_credit_prefit_epochs=3),
        seed=11,
        device="cpu",
    )
    stats = agent.update(_synthetic_batch(seed=4))
    assert isinstance(stats, RCWAV3UpdateStats)
    assert stats.risk_credit_prefit_loss_before > 0.0
    assert stats.risk_credit_prefit_loss_after >= 0.0
    assert stats.risk_credit_prefit_grad_norm_mean > 0.0
    assert stats.risk_td_telescoping_max_abs_error < 1e-4
    for mapping in (
        stats.risk_td_credit_mean_by_eta,
        stats.risk_td_credit_std_by_eta,
        stats.risk_td_credit_positive_fraction_by_eta,
        stats.risk_td_credit_early_window_mean_irrigate_by_eta,
        stats.risk_td_credit_early_window_mean_noop_by_eta,
        stats.risk_td_credit_early_window_mean_applied_irrigation_mm_by_eta,
    ):
        assert set(mapping) == {"0.90", "0.95", "0.98"}
    for value in stats.risk_td_credit_positive_fraction_by_eta.values():
        assert 0.0 <= value <= 1.0
    for value in stats.risk_td_credit_early_window_mean_applied_irrigation_mm_by_eta.values():
        assert 0.0 <= value <= 45.0
    assert stats.dual_after["0.90"] >= 0.0
    assert set(stats.tau_by_eta) == {"0.90", "0.95", "0.98"}


def test_v3_update_uses_td_credit_not_the_monte_carlo_risk_advantage():
    """A constant MC risk advantage gives zero within-eta contrast; TD does not."""
    batch = _synthetic_batch(seed=6)
    agent = RCWAV3Agent(
        hyperparameters=_small_hyperparameters(risk_credit_prefit_epochs=3),
        seed=13,
        device="cpu",
    )
    agent.update(batch)
    credit, _ = agent._actor_risk_credit(
        batch=batch,
        states=batch.states.to(agent.device),
        etas=batch.etas.to(agent.device),
    )
    assert float(credit.std(unbiased=False).item()) > 0.0
    assert not torch.allclose(credit, batch.risk_advantages.to(agent.device))
