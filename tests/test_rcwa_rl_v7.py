"""Engineering tests for RCWA-v7 episode-aware risk-conditioning noise floor."""
from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path

import pytest
import torch

from awm.rcwa import RCWARolloutBatch, evaluate_eta_tail
from awm.rcwa.agent_v6 import RCWAV6Agent, RCWAV6Hyperparameters
from awm.rcwa.agent_v7 import RCWAV7Agent, RCWAV7Hyperparameters, RCWAV7UpdateStats
from awm.rcwa.trainer_v7 import RCWAV7Trainer, V7_PROTOCOL_PATH, v7_hyperparameters_from_protocol

ROOT=Path(__file__).resolve().parents[1]
EPISODE_LENGTH=25
EPISODES_PER_ETA=18
ETA_LEVELS=(0.90,0.95,0.98)


def _synthetic_batch(*,seed:int=0,state_dim:int=2,constant_tail:bool=False)->RCWARolloutBatch:
    g=torch.Generator().manual_seed(seed)
    episodes=EPISODES_PER_ETA*len(ETA_LEVELS); size=episodes*EPISODE_LENGTH
    states=torch.randn(size,state_dim,generator=g); dones=torch.zeros(size,dtype=torch.bool)
    risk_costs=torch.zeros(size); risk_returns=torch.zeros(size); etas=torch.zeros(size)
    episode_etas=torch.zeros(episodes); rewards=-torch.rand(size,generator=g)*0.01
    irrigate=torch.rand(size,generator=g)<0.4; raw=torch.zeros(size)
    raw[irrigate]=torch.randn(int(irrigate.sum()),generator=g)
    for i in range(episodes):
        start=i*EPISODE_LENGTH; stop=start+EPISODE_LENGTH; dones[stop-1]=True
        level=i//EPISODES_PER_ETA; eta=ETA_LEVELS[level]
        cost=1.0 if constant_tail else 0.05*level+0.03*(i%EPISODES_PER_ETA)
        risk_costs[stop-1]=cost; risk_returns[start:stop]=cost; etas[start:stop]=eta; episode_etas[i]=eta
    metrics={f'{eta:.2f}':evaluate_eta_tail([0.80+0.005*(j+5*k) for j in range(18)],eta=eta) for k,eta in enumerate(ETA_LEVELS)}
    return RCWARolloutBatch(states=states,irrigate=irrigate,raw_amount=raw,old_log_probs=torch.full((size,),-0.1),
        etas=etas,reward_values=torch.zeros(size),risk_values=torch.zeros(size),reward_returns=torch.zeros(size),
        risk_returns=risk_returns,reward_advantages=torch.linspace(-1,1,size),risk_advantages=risk_returns.clone(),
        rewards=rewards,risk_costs=risk_costs,dones=dones,episode_etas=episode_etas,
        episode_retentions=torch.ones(episodes),episode_risk_costs=torch.zeros(episodes),tail_metrics=metrics,policy_version=0)


def _small(**overrides)->RCWAV7Hyperparameters:
    d=dict(state_dim=2,actor_hidden_dims=(8,4),reward_critic_hidden_dims=(8,4),risk_critic_hidden_dims=(8,4),
           update_epochs=2,minibatch_size=450,risk_credit_prefit_epochs=2,tail_q_prefit_epochs=2)
    d.update(overrides); return RCWAV7Hyperparameters(**d)


def _same(a,b):
    return set(a)==set(b) and all(torch.equal(a[k],b[k]) for k in a)


def test_v7_protocol_changes_only_conditioning_from_v6():
    v6=json.loads((ROOT/'configs/rcwa_rl_v6.json').read_text()); v7=json.loads((ROOT/'configs/rcwa_rl_v7.json').read_text())
    assert v7['rcwa_protocol_id']=='awm-rcwa-rl-v7' and v7['supersedes']=='awm-rcwa-rl-v6'
    for section in ('base_contracts','environment','objective','risk_constraint','lagrangian','optimizer','rollout','interaction_budget','training','evaluation','policy'):
        assert v7[section]==v6[section],section
    for key in v6['tail_credit']:
        if key not in ('mechanism','actor_credit_combination','no_schedule_prior'):
            assert v7['tail_credit'][key]==v6['tail_credit'][key],key
    assert 'std(h_episode_eta)/sqrt(N_eta)' in v7['tail_credit']['actor_credit_combination']
    assert 'risk_conditioning_noise_floor' in v7['tail_credit']


def test_v7_initialization_and_parameter_set_match_v6_exactly():
    h7=_small(); h6=RCWAV6Hyperparameters(**asdict(h7))
    a6=RCWAV6Agent(hyperparameters=h6,seed=17,device='cpu'); a7=RCWAV7Agent(hyperparameters=h7,seed=17,device='cpu')
    for name in ('actor','reward_critic','risk_critic','tail_q_critic'):
        assert _same(getattr(a6,name).state_dict(),getattr(a7,name).state_dict())


def _eta_tensor(n=12):
    return torch.cat([torch.full((n,),x) for x in ETA_LEVELS])


def test_noise_floor_attenuates_small_credit_instead_of_unit_standardizing():
    a=RCWAV7Agent(hyperparameters=_small(),seed=3,device='cpu'); etas=_eta_tensor()
    base=torch.linspace(-1,1,12); risk=torch.cat([base,base,base]); reward=torch.zeros_like(risk)
    raw_std=float(base.std(unbiased=False)); floor=2.0*raw_std
    a._v7_noise_floor_by_eta={f'{e:.2f}':floor for e in ETA_LEVELS}
    combined,_=a._condition_actor_advantages(reward_adv=reward,risk_adv=risk,etas=etas)
    for e in ETA_LEVELS:
        m=torch.isclose(etas,torch.tensor(e)); assert float(combined[m].std(unbiased=False))==pytest.approx(0.5,abs=1e-6)


def test_noise_floor_recovers_v6_full_strength_when_signal_exceeds_floor():
    a=RCWAV7Agent(hyperparameters=_small(),seed=4,device='cpu'); etas=_eta_tensor()
    base=torch.linspace(-1,1,12); risk=torch.cat([base,base,base]); reward=torch.zeros_like(risk)
    raw_std=float(base.std(unbiased=False)); a._v7_noise_floor_by_eta={f'{e:.2f}':0.25*raw_std for e in ETA_LEVELS}
    combined,_=a._condition_actor_advantages(reward_adv=reward,risk_adv=risk,etas=etas)
    for e in ETA_LEVELS:
        m=torch.isclose(etas,torch.tensor(e)); assert float(combined[m].std(unbiased=False))==pytest.approx(1.0,abs=1e-6)


def test_zero_episode_tail_variance_forces_zero_reliability():
    a=RCWAV7Agent(hyperparameters=_small(),seed=5,device='cpu'); b=_synthetic_batch(seed=8,constant_tail=True)
    etas=b.etas; fake_credit=torch.randn(b.size,generator=torch.Generator().manual_seed(9))*0.3
    d=a._episode_noise_floor_diagnostics(batch=b,credit=fake_credit,etas=etas)
    assert d['risk_conditioning_episode_count_by_eta']=={'0.90':18,'0.95':18,'0.98':18}
    assert all(x==0.0 for x in d['risk_conditioning_episode_tail_std_by_eta'].values())
    assert all(x==0.0 for x in d['risk_conditioning_reliability_by_eta'].values())
    combined,_=a._condition_actor_advantages(reward_adv=torch.zeros_like(fake_credit),risk_adv=fake_credit,etas=etas)
    assert float(combined.abs().max())==0.0


def test_v7_full_update_emits_episode_aware_reliability_diagnostics():
    a=RCWAV7Agent(hyperparameters=_small(),seed=13,device='cpu'); stats=a.update(_synthetic_batch(seed=6))
    assert isinstance(stats,RCWAV7UpdateStats); assert stats.update_index==1
    assert stats.risk_conditioning_episode_count_by_eta=={'0.90':18,'0.95':18,'0.98':18}
    assert all(x>0 for x in stats.risk_conditioning_noise_floor_by_eta.values())
    assert all(0.0<=x<=1.0 for x in stats.risk_conditioning_reliability_by_eta.values())
    assert max(stats.tail_q_policy_expected_advantage_max_abs_error_by_eta.values())<1e-6


def test_v7_checkpoint_roundtrip_and_trainer_identity():
    h=_small(); a=RCWAV7Agent(hyperparameters=h,seed=31,device='cpu'); a.update(_synthetic_batch(seed=7)); p=a.checkpoint_payload()
    assert p['protocol_id']=='awm-rcwa-rl-v7'; b=RCWAV7Agent(hyperparameters=h,seed=31,device='cpu'); b.load_checkpoint_payload(p)
    assert _same(a.actor.state_dict(),b.actor.state_dict()); assert _same(a.tail_q_critic.state_dict(),b.tail_q_critic.state_dict())
    assert torch.equal(a.generator.get_state(),b.generator.get_state())
    assert V7_PROTOCOL_PATH=='configs/rcwa_rl_v7.json'; assert RCWAV7Trainer.TRAINER_PROTOCOL_ID=='awm-rcwa-trainer-v7'
    hp=v7_hyperparameters_from_protocol(json.loads((ROOT/V7_PROTOCOL_PATH).read_text())); assert hp.tail_q_prefit_epochs==10


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_v7_full_update_runs_on_cuda():
    a=RCWAV7Agent(hyperparameters=_small(),seed=41,device='cuda:0'); s=a.update(_synthetic_batch(seed=11))
    assert s.update_index==1 and math.isfinite(s.actor_grad_norm)
    assert all(0.0<=x<=1.0 for x in s.risk_conditioning_reliability_by_eta.values())
