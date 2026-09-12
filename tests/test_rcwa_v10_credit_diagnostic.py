from __future__ import annotations

import torch

from awm.rcwa.agent_v10 import RCWAV10Agent, RCWAHyperparameters
from awm.rcwa.buffer import RCWARolloutBatch
from awm.rcwa.credit_diagnostic_v10 import DIAGNOSTIC_ACTION_SEED, gradient_diagnostics
from awm.rcwa.risk_batch import EtaTailRiskMetrics


def _batch(agent: RCWAV10Agent) -> tuple[RCWARolloutBatch, torch.Tensor]:
    n = 12
    g = torch.Generator().manual_seed(7)
    states = torch.randn(n, agent.hparams.state_dim, generator=g)
    with torch.no_grad():
        action = agent.actor.sample(states, generator=g)
    etas = torch.tensor([0.90, 0.95, 0.98] * 4)
    reward_adv = torch.linspace(-0.3, 0.4, n)
    risk_adv = torch.linspace(0.2, -0.1, n)
    dummy_metric = EtaTailRiskMetrics(0.9, 4, 0.2, 0.2, 0.2, 0.2, 0.7, (0.0,) * 4)
    batch = RCWARolloutBatch(
        states=states,
        irrigate=action.irrigate,
        raw_amount=action.raw_amount,
        old_log_probs=action.log_prob,
        etas=etas,
        reward_values=torch.zeros(n),
        risk_values=torch.zeros(n),
        reward_returns=torch.zeros(n),
        risk_returns=torch.zeros(n),
        reward_advantages=reward_adv,
        risk_advantages=risk_adv,
        rewards=torch.zeros(n),
        risk_costs=torch.zeros(n),
        dones=torch.tensor([False, False, True] * 4),
        episode_etas=torch.tensor([0.90, 0.95, 0.98, 0.90]),
        episode_retentions=torch.ones(4),
        episode_risk_costs=torch.zeros(4),
        tail_metrics={"0.90": dummy_metric, "0.95": dummy_metric, "0.98": dummy_metric},
        policy_version=agent.policy_version,
    )
    weather = torch.tensor([2000] * 3 + [2001] * 3 + [2002] * 3 + [2003] * 3)
    return batch, weather


def test_gradient_diagnostic_is_finite_and_blocked() -> None:
    agent = RCWAV10Agent(hyperparameters=RCWAHyperparameters(minibatch_size=3), seed=21)
    batch, weather = _batch(agent)
    before = {k: v.detach().clone() for k, v in agent.actor.state_dict().items()}
    summary, tensors = gradient_diagnostics(agent=agent, batch=batch, weather_by_transition=weather)
    assert set(summary["full"]) == {"trunk", "gate", "amount"}
    assert tensors["years"].tolist() == [2000, 2001, 2002, 2003]
    for block in ("trunk", "gate", "amount"):
        item = summary["full"][block]
        assert item["total_norm"] >= 0.0
        assert torch.isfinite(tensors["weather_total_gradients"][block]).all()
    for key, value in agent.actor.state_dict().items():
        assert torch.equal(value, before[key])


def test_diagnostic_action_seed_is_frozen() -> None:
    assert DIAGNOSTIC_ACTION_SEED == 20260913
