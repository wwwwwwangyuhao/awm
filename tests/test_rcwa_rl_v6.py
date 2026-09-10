"""Engineering tests for RCWA-v6 policy-consistent Tail-Q baseline."""
from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from awm.rcwa import RCWARolloutBatch, evaluate_eta_tail
from awm.rcwa.agent_v4 import RCWAV4Agent, RCWAV4Hyperparameters
from awm.rcwa.agent_v6 import (
    POLICY_BASELINE_GH_ORDER, RCWAV6Agent, RCWAV6Hyperparameters, RCWAV6UpdateStats
)
from awm.rcwa.trainer_v6 import RCWAV6Trainer, V6_PROTOCOL_PATH, v6_hyperparameters_from_protocol

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


def _v6_small(**overrides) -> RCWAV6Hyperparameters:
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
    return RCWAV6Hyperparameters(**values)


def _state_dict_equal(left, right) -> bool:
    return set(left) == set(right) and all(torch.equal(left[k], right[k]) for k in left)



class _GateOnlyQ(nn.Module):
    def forward(self, state: torch.Tensor, action_features: torch.Tensor) -> torch.Tensor:
        return 0.125 * state[:, 0] + 2.0 * action_features[:, 0]


def test_v6_protocol_changes_only_actor_credit_baseline_from_v4():
    v4=json.loads((ROOT/'configs/rcwa_rl_v4.json').read_text())
    v6=json.loads((ROOT/'configs/rcwa_rl_v6.json').read_text())
    assert v6['rcwa_protocol_id']=='awm-rcwa-rl-v6'
    assert v6['supersedes']=='awm-rcwa-rl-v4'
    for section in ('base_contracts','environment','objective','risk_constraint','lagrangian','optimizer','rollout','interaction_budget','training','evaluation','policy'):
        assert v6[section]==v4[section], section
    for key in ('risk_credit_prefit_epochs','tail_q_prefit_epochs','tail_q_target','tail_q_action','tail_q_hidden_dims'):
        assert v6['tail_credit'][key]==v4['tail_credit'][key]
    assert 'E_{a_prime~pi' in v6['tail_credit']['actor_credit']
    assert '128-point Gauss-Hermite' in v6['tail_credit']['policy_q_baseline_quadrature']


def test_v6_identity_initialization_and_parameter_set_match_v4_exactly():
    h6=_v6_small(); h4=RCWAV4Hyperparameters(**asdict(h6))
    a4=RCWAV4Agent(hyperparameters=h4,seed=17,device='cpu')
    a6=RCWAV6Agent(hyperparameters=h6,seed=17,device='cpu')
    assert _state_dict_equal(a4.actor.state_dict(),a6.actor.state_dict())
    assert _state_dict_equal(a4.reward_critic.state_dict(),a6.reward_critic.state_dict())
    assert _state_dict_equal(a4.risk_critic.state_dict(),a6.risk_critic.state_dict())
    assert _state_dict_equal(a4.tail_q_critic.state_dict(),a6.tail_q_critic.state_dict())
    mods4=(a4.actor,a4.reward_critic,a4.risk_critic,a4.tail_q_critic)
    mods6=(a6.actor,a6.reward_critic,a6.risk_critic,a6.tail_q_critic)
    assert sum(p.numel() for m in mods4 for p in m.parameters())==sum(p.numel() for m in mods6 for p in m.parameters())


def test_policy_q_baseline_is_exact_for_gate_only_q_and_consumes_no_rng():
    a=RCWAV6Agent(hyperparameters=_v6_small(),seed=3,device='cpu')
    a.tail_q_critic=_GateOnlyQ()
    states=torch.randn(37,2,generator=torch.Generator().manual_seed(5))
    rng=a.generator.get_state().clone()
    baseline,q0,q1,p=a._policy_q_baseline(states)
    assert torch.equal(a.generator.get_state(),rng)
    expected=0.125*states[:,0]+2.0*p
    assert torch.allclose(q0,0.125*states[:,0],atol=1e-7,rtol=0)
    assert torch.allclose(q1,0.125*states[:,0]+2.0,atol=1e-6,rtol=0)
    assert torch.allclose(baseline,expected,atol=1e-6,rtol=0)
    residual=(1-p)*(q0-baseline)+p*(q1-baseline)
    assert float(residual.abs().max())<1e-6


def test_128_point_quadrature_matches_256_point_reference_for_tail_q():
    a=RCWAV6Agent(hyperparameters=_v6_small(),seed=11,device='cpu')
    g=torch.Generator().manual_seed(13)
    with torch.no_grad():
        a.tail_q_critic.value_head.weight.copy_(torch.randn(a.tail_q_critic.value_head.weight.shape,generator=g)*0.2)
        a.tail_q_critic.value_head.bias.copy_(torch.tensor([0.07]))
    states=torch.randn(53,2,generator=g)
    b128,_,q128,_=a._policy_q_baseline(states,quadrature_order=128)
    b256,_,q256,_=a._policy_q_baseline(states,quadrature_order=256)
    assert float((q128-q256).abs().max())<1e-5
    assert float((b128-b256).abs().max())<1e-5


def test_v6_full_update_emits_policy_baseline_identity_and_keeps_q_prefit_v4_style():
    a=RCWAV6Agent(hyperparameters=_v6_small(),seed=13,device='cpu')
    stats=a.update(_synthetic_batch(seed=6))
    assert isinstance(stats,RCWAV6UpdateStats)
    assert stats.tail_q_prefit_steps==2*(1350//450)
    assert stats.tail_q_prefit_loss_after<=stats.tail_q_prefit_loss_before
    assert stats.tail_q_policy_quadrature_order==POLICY_BASELINE_GH_ORDER==128
    assert set(stats.tail_q_policy_baseline_mean_by_eta)=={'0.90','0.95','0.98'}
    assert max(stats.tail_q_policy_expected_advantage_max_abs_error_by_eta.values())<1e-6
    assert math.isfinite(stats.actor_grad_norm) and stats.actor_grad_norm>0


def test_v6_checkpoint_roundtrip_and_trainer_identity():
    h=_v6_small(); a=RCWAV6Agent(hyperparameters=h,seed=31,device='cpu')
    a.update(_synthetic_batch(seed=7)); payload=a.checkpoint_payload()
    assert payload['protocol_id']=='awm-rcwa-rl-v6'
    b=RCWAV6Agent(hyperparameters=h,seed=31,device='cpu'); b.load_checkpoint_payload(payload)
    assert _state_dict_equal(a.actor.state_dict(),b.actor.state_dict())
    assert _state_dict_equal(a.tail_q_critic.state_dict(),b.tail_q_critic.state_dict())
    assert torch.equal(a.generator.get_state(),b.generator.get_state())
    assert V6_PROTOCOL_PATH=='configs/rcwa_rl_v6.json'
    assert RCWAV6Trainer.TRAINER_PROTOCOL_ID=='awm-rcwa-trainer-v6'
    assert RCWAV6Trainer.RUNTIME_SUBDIR=='rcwa_rl_v6'
    proto=json.loads((ROOT/V6_PROTOCOL_PATH).read_text())
    hp=v6_hyperparameters_from_protocol(proto)
    assert hp.tail_q_prefit_epochs==10 and hp.risk_credit_prefit_epochs==10


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_v6_full_update_and_policy_baseline_run_on_cuda():
    a=RCWAV6Agent(hyperparameters=_v6_small(),seed=41,device='cuda:0')
    stats=a.update(_synthetic_batch(seed=11))
    assert stats.update_index==1
    assert max(stats.tail_q_policy_expected_advantage_max_abs_error_by_eta.values())<1e-6
    assert a.tail_q_critic.value_head.weight.device.type=='cuda'
