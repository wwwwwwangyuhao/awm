"""Engineering tests for RCWA-v8 twin Tail-Q disagreement reliability."""
from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path

import pytest
import torch

from awm.rcwa import RCWARolloutBatch, evaluate_eta_tail
from awm.rcwa.agent_v6 import RCWAV6Agent, RCWAV6Hyperparameters
from awm.rcwa.agent_v8 import RCWAV8Agent, RCWAV8Hyperparameters, RCWAV8UpdateStats
from awm.rcwa.trainer_v8 import RCWAV8Trainer, V8_PROTOCOL_PATH, v8_hyperparameters_from_protocol

ROOT=Path(__file__).resolve().parents[1]
EPISODE_LENGTH=25
EPISODES_PER_ETA=18
ETA_LEVELS=(0.90,0.95,0.98)


def _synthetic_batch(*,seed:int=0,state_dim:int=2)->RCWARolloutBatch:
    g=torch.Generator().manual_seed(seed)
    episodes=EPISODES_PER_ETA*len(ETA_LEVELS); size=episodes*EPISODE_LENGTH
    states=torch.randn(size,state_dim,generator=g); dones=torch.zeros(size,dtype=torch.bool)
    risk_costs=torch.zeros(size); risk_returns=torch.zeros(size); etas=torch.zeros(size)
    episode_etas=torch.zeros(episodes); rewards=-torch.rand(size,generator=g)*0.01
    irrigate=torch.rand(size,generator=g)<0.4; raw=torch.zeros(size)
    raw[irrigate]=torch.randn(int(irrigate.sum()),generator=g)
    for i in range(episodes):
        start=i*EPISODE_LENGTH; stop=start+EPISODE_LENGTH; dones[stop-1]=True
        level=i//EPISODES_PER_ETA; eta=ETA_LEVELS[level]; cost=0.05*level+0.03*(i%EPISODES_PER_ETA)
        risk_costs[stop-1]=cost; risk_returns[start:stop]=cost; etas[start:stop]=eta; episode_etas[i]=eta
    metrics={f'{eta:.2f}':evaluate_eta_tail([0.80+0.005*(j+5*k) for j in range(18)],eta=eta) for k,eta in enumerate(ETA_LEVELS)}
    return RCWARolloutBatch(states=states,irrigate=irrigate,raw_amount=raw,old_log_probs=torch.full((size,),-0.1),
        etas=etas,reward_values=torch.zeros(size),risk_values=torch.zeros(size),reward_returns=torch.zeros(size),
        risk_returns=risk_returns,reward_advantages=torch.linspace(-1,1,size),risk_advantages=risk_returns.clone(),
        rewards=rewards,risk_costs=risk_costs,dones=dones,episode_etas=episode_etas,
        episode_retentions=torch.ones(episodes),episode_risk_costs=torch.zeros(episodes),tail_metrics=metrics,policy_version=0)


def _small(**overrides)->RCWAV8Hyperparameters:
    d=dict(state_dim=2,actor_hidden_dims=(8,4),reward_critic_hidden_dims=(8,4),risk_critic_hidden_dims=(8,4),
           update_epochs=2,minibatch_size=450,risk_credit_prefit_epochs=2,tail_q_prefit_epochs=2)
    d.update(overrides); return RCWAV8Hyperparameters(**d)


def _same(a,b):
    return set(a)==set(b) and all(torch.equal(a[k],b[k]) for k in a)


def _eta_tensor(n=12):
    return torch.cat([torch.full((n,),x) for x in ETA_LEVELS])


def _pair_diagnostics(agent, a1, a2, etas):
    z=torch.zeros_like(a1); p=torch.zeros_like(a1)
    return agent._twin_reliability_diagnostics(
        primary_credit=a1,aux_credit=a2,aux_baseline=z,aux_q_noop=z,aux_q_active=z,gate_p=p,etas=etas
    )


def test_v8_protocol_is_v6_plus_twin_reliability_only():
    v6=json.loads((ROOT/'configs/rcwa_rl_v6.json').read_text()); v8=json.loads((ROOT/'configs/rcwa_rl_v8.json').read_text())
    assert v8['rcwa_protocol_id']=='awm-rcwa-rl-v8' and v8['supersedes']=='awm-rcwa-rl-v6'
    for section in ('base_contracts','environment','objective','risk_constraint','lagrangian','rollout','interaction_budget','training','evaluation','policy'):
        assert v8[section]==v6[section],section
    numeric_optimizer={k:v for k,v in v6['optimizer'].items() if not isinstance(v,str)}
    assert {k:v8['optimizer'][k] for k in numeric_optimizer}==numeric_optimizer
    for key in v6['tail_credit']:
        if key not in ('mechanism','actor_credit_combination','no_schedule_prior'):
            assert v8['tail_credit'][key]==v6['tail_credit'][key],key
    assert v8['tail_credit']['twin_q_auxiliary_count']==1
    assert v8['tail_credit']['weather_split_for_uncertainty']=='none; both Q estimators use all 18 weather years x 3 eta cells so v5 tail-label starvation is not reintroduced.'
    assert v8['tail_credit']['primary_credit_unchanged_from_v6'] is True
    assert v8['tail_credit']['extra_environment_episodes']==0


def test_v8_primary_initialization_and_optimizer_match_v6_exactly():
    h8=_small(); h6=RCWAV6Hyperparameters(**asdict(h8))
    a6=RCWAV6Agent(hyperparameters=h6,seed=17,device='cpu'); a8=RCWAV8Agent(hyperparameters=h8,seed=17,device='cpu')
    for name in ('actor','reward_critic','risk_critic','tail_q_critic'):
        assert _same(getattr(a6,name).state_dict(),getattr(a8,name).state_dict())
    primary_ids={id(p) for g in a8.optimizer.param_groups for p in g['params']}
    aux_ids={id(p) for g in a8.aux_tail_q_optimizer.param_groups for p in g['params']}
    assert primary_ids.isdisjoint(aux_ids)
    assert len(primary_ids)==len({id(p) for g in a6.optimizer.param_groups for p in g['params']})
    assert not _same(a8.tail_q_critic.net.state_dict(),a8.aux_tail_q_critic.net.state_dict())
    assert torch.count_nonzero(a8.aux_tail_q_critic.value_head.weight)==0
    assert a8.auxiliary_seed==18


def test_v8_primary_credit_prefit_is_bitwise_v6_path_before_conditioning():
    b=_synthetic_batch(seed=5); h8=_small(); h6=RCWAV6Hyperparameters(**asdict(h8))
    a6=RCWAV6Agent(hyperparameters=h6,seed=13,device='cpu'); a8=RCWAV8Agent(hyperparameters=h8,seed=13,device='cpu')
    c6,_=a6._actor_risk_credit(batch=b,states=b.states,etas=b.etas)
    c8,_=a8._actor_risk_credit(batch=b,states=b.states,etas=b.etas)
    assert torch.equal(c6,c8)
    assert _same(a6.tail_q_critic.state_dict(),a8.tail_q_critic.state_dict())
    assert _same(a6.risk_critic.state_dict(),a8.risk_critic.state_dict())
    assert _same(a6.actor.state_dict(),a8.actor.state_dict())


def test_small_but_consistent_twin_credit_keeps_full_reliability():
    a=RCWAV8Agent(hyperparameters=_small(),seed=3,device='cpu'); etas=_eta_tensor()
    base=torch.linspace(-1,1,12)*1e-5; a1=torch.cat([base,base,base]); a2=a1.clone()
    d=_pair_diagnostics(a,a1,a2,etas)
    assert all(x==pytest.approx(1.0,abs=1e-7) for x in d['twin_q_reliability_by_eta'].values())
    assert all(x==pytest.approx(1.0,abs=1e-6) for x in d['twin_q_credit_correlation_by_eta'].values())


def test_disagreement_dominated_opposite_credit_has_zero_reliability():
    a=RCWAV8Agent(hyperparameters=_small(),seed=4,device='cpu'); etas=_eta_tensor()
    base=torch.linspace(-1,1,12)*0.2; a1=torch.cat([base,base,base]); a2=-a1
    d=_pair_diagnostics(a,a1,a2,etas)
    assert all(x==pytest.approx(0.0,abs=1e-8) for x in d['twin_q_reliability_by_eta'].values())
    assert all(x==pytest.approx(-1.0,abs=1e-6) for x in d['twin_q_credit_correlation_by_eta'].values())


def test_v8_full_update_emits_twin_diagnostics_and_aux_fit():
    a=RCWAV8Agent(hyperparameters=_small(),seed=19,device='cpu'); s=a.update(_synthetic_batch(seed=6))
    assert isinstance(s,RCWAV8UpdateStats) and s.update_index==1
    assert s.aux_tail_q_prefit_steps==2*(1350//450)
    assert s.aux_tail_q_prefit_loss_after<=s.aux_tail_q_prefit_loss_before
    assert all(0.0<=x<=1.0 for x in s.twin_q_reliability_by_eta.values())
    assert all(-1.0<=x<=1.0 for x in s.twin_q_credit_correlation_by_eta.values())
    assert max(s.twin_q_aux_expected_advantage_max_abs_error_by_eta.values())<1e-6
    assert s.twin_q_auxiliary_seed==20
    assert math.isfinite(s.actor_grad_norm) and s.actor_grad_norm>0


def test_v8_checkpoint_roundtrip_restores_auxiliary_estimator():
    h=_small(); a=RCWAV8Agent(hyperparameters=h,seed=31,device='cpu'); a.update(_synthetic_batch(seed=7)); p=a.checkpoint_payload()
    assert p['protocol_id']=='awm-rcwa-rl-v8' and p['twin_q_auxiliary_seed']==32
    b=RCWAV8Agent(hyperparameters=h,seed=31,device='cpu'); b.load_checkpoint_payload(p)
    assert _same(a.actor.state_dict(),b.actor.state_dict())
    assert _same(a.tail_q_critic.state_dict(),b.tail_q_critic.state_dict())
    assert _same(a.aux_tail_q_critic.state_dict(),b.aux_tail_q_critic.state_dict())
    assert torch.equal(a.generator.get_state(),b.generator.get_state())
    assert torch.equal(a.aux_generator.get_state(),b.aux_generator.get_state())
    assert a.aux_tail_q_optimizer.state_dict()['state'].keys()==b.aux_tail_q_optimizer.state_dict()['state'].keys()
    assert V8_PROTOCOL_PATH=='configs/rcwa_rl_v8.json'
    assert RCWAV8Trainer.TRAINER_PROTOCOL_ID=='awm-rcwa-trainer-v8' and RCWAV8Trainer.RUNTIME_SUBDIR=='rcwa_rl_v8'
    hp=v8_hyperparameters_from_protocol(json.loads((ROOT/V8_PROTOCOL_PATH).read_text()))
    assert hp.tail_q_prefit_epochs==10 and hp.risk_credit_prefit_epochs==10


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_v8_full_update_and_checkpoint_run_on_cuda():
    h=_small(); a=RCWAV8Agent(hyperparameters=h,seed=41,device='cuda:0'); s=a.update(_synthetic_batch(seed=11))
    assert s.update_index==1 and math.isfinite(s.actor_grad_norm)
    assert a.tail_q_critic.value_head.weight.device.type=='cuda'
    assert a.aux_tail_q_critic.value_head.weight.device.type=='cuda'
    assert all(0.0<=x<=1.0 for x in s.twin_q_reliability_by_eta.values())
    p=a.checkpoint_payload(); b=RCWAV8Agent(hyperparameters=h,seed=41,device='cuda:0'); b.load_checkpoint_payload(p)
    assert _same(a.aux_tail_q_critic.state_dict(),b.aux_tail_q_critic.state_dict())
    assert torch.equal(a.aux_generator.get_state(),b.aux_generator.get_state())
