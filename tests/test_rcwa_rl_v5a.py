from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import pytest
import torch

from awm.rcwa import RCWARolloutBatch, evaluate_eta_tail
from awm.rcwa.agent_v4 import RCWAV4Agent, RCWAV4Hyperparameters
from awm.rcwa.agent_v5a import RCWAV5AAgent, RCWAV5AHyperparameters, RCWAV5AUpdateStats
from awm.rcwa.real_smoke import _smoke_protocol_spec
from awm.rcwa.trainer_v5a import RCWAV5ATrainer, V5A_PROTOCOL_PATH, v5a_hyperparameters_from_protocol

ROOT=Path(__file__).resolve().parents[1]
ETAS=(0.90,0.95,0.98)

def _small(**overrides):
    values=dict(state_dim=2, actor_hidden_dims=(8,4), reward_critic_hidden_dims=(8,4),
                risk_critic_hidden_dims=(8,4), update_epochs=2, minibatch_size=450,
                risk_credit_prefit_epochs=2, tail_q_prefit_epochs=2)
    values.update(overrides)
    return RCWAV5AHyperparameters(**values)

def _batch(seed=0):
    g=torch.Generator().manual_seed(seed); ep_len=25; eps_per_eta=18; ne=54; n=ne*ep_len
    states=torch.randn(n,2,generator=g); irrigate=torch.rand(n,generator=g)<0.45
    raw=torch.zeros(n); raw[irrigate]=torch.randn(int(irrigate.sum()),generator=g)
    dones=torch.zeros(n,dtype=torch.bool); risk_returns=torch.zeros(n); risk_costs=torch.zeros(n); etas=torch.zeros(n)
    episode_etas=[]
    for e in range(ne):
        a=e*ep_len; b=a+ep_len; dones[b-1]=True; eta=ETAS[e//eps_per_eta]; etas[a:b]=eta; episode_etas.append(eta)
        target=0.05+0.02*(e%eps_per_eta); risk_returns[a:b]=target; risk_costs[b-1]=target
    tail={f'{eta:.2f}':evaluate_eta_tail([0.70+0.01*i for i in range(eps_per_eta)],eta=eta) for eta in ETAS}
    return RCWARolloutBatch(states=states,irrigate=irrigate,raw_amount=raw,old_log_probs=torch.zeros(n),etas=etas,
        reward_values=torch.zeros(n),risk_values=torch.zeros(n),reward_returns=torch.zeros(n),risk_returns=risk_returns,
        reward_advantages=torch.linspace(-1,1,n),risk_advantages=risk_returns.clone(),rewards=-torch.rand(n,generator=g)*.01,
        risk_costs=risk_costs,dones=dones,episode_etas=torch.tensor(episode_etas),episode_retentions=torch.ones(ne),
        episode_risk_costs=torch.zeros(ne),tail_metrics=tail,policy_version=0)

def _eq(a,b): return set(a)==set(b) and all(torch.equal(a[k],b[k]) for k in a)

def test_protocol_is_single_factor_from_v4():
    v4=json.loads((ROOT/'configs/rcwa_rl_v4.json').read_text()); v5a=json.loads((ROOT/V5A_PROTOCOL_PATH).read_text())
    assert v5a['rcwa_protocol_id']=='awm-rcwa-rl-v5a'
    for section in ('base_contracts','environment','objective','risk_constraint','lagrangian','optimizer','rollout','interaction_budget','training','evaluation'):
        assert v5a[section]==v4[section], section
    for key,val in v4['policy'].items():
        if key!='fairness_rule': assert v5a['policy'][key]==val
    assert v5a['tail_credit']['coverage_scope'].startswith('full 18-weather-year')
    assert 'cross-fitting' in v5a['tail_credit']['coverage_scope']
    assert v5a['tail_credit']['extra_environment_episodes']==0
    assert v5a['tail_credit']['no_validation_feedback'] is True

def test_identity_and_initialization_match_v4_exactly():
    h5=_small(); h4=RCWAV4Hyperparameters(**asdict(h5))
    a4=RCWAV4Agent(hyperparameters=h4,seed=17); a5=RCWAV5AAgent(hyperparameters=h5,seed=17)
    assert _eq(a4.actor.state_dict(),a5.actor.state_dict())
    assert _eq(a4.reward_critic.state_dict(),a5.reward_critic.state_dict())
    assert _eq(a4.risk_critic.state_dict(),a5.risk_critic.state_dict())
    assert _eq(a4.tail_q_critic.state_dict(),a5.tail_q_critic.state_dict())
    assert not hasattr(a5,'tail_q_critic_fold1')
    mods4=(a4.actor,a4.reward_critic,a4.risk_critic,a4.tail_q_critic)
    mods5=(a5.actor,a5.reward_critic,a5.risk_critic,a5.tail_q_critic)
    assert sum(p.numel() for m in mods4 for p in m.parameters())==sum(p.numel() for m in mods5 for p in m.parameters())

def test_coverage_weights_equalize_present_stratum_mass_per_eta():
    a=RCWAV5AAgent(hyperparameters=_small(),seed=1)
    etas=torch.tensor([.90]*10+[.95]*10+[.98]*10)
    strata=torch.tensor(([0]*6+[1]*2+[4]*2)*3)
    w,counts,maxw,present=a._coverage_weights(etas=etas,strata=strata)
    for j,eta in enumerate(ETAS):
        group=torch.arange(j*10,(j+1)*10)
        masses=[]
        for s in (0,1,4): masses.append(float(w[group][strata[group]==s].sum()))
        assert masses[0]==pytest.approx(masses[1]); assert masses[1]==pytest.approx(masses[2])
        assert float(w[group].mean())==pytest.approx(1.0)
        assert present[f'{eta:.2f}']==3
        assert counts[f'{eta:.2f}']['active_025_050']==0
        assert maxw[f'{eta:.2f}']>1

def test_weighted_prefit_reduces_loss_and_freezes_non_q_parameters():
    a=RCWAV5AAgent(hyperparameters=_small(learning_rate=1e-3,tail_q_prefit_epochs=20),seed=3)
    b=_batch(3); states=b.states; af=a._action_features_from_batch(b,device=a.device); etas=b.etas
    before_actor={k:v.clone() for k,v in a.actor.state_dict().items()}; before_v={k:v.clone() for k,v in a.risk_critic.state_dict().items()}
    d=a._prefit_coverage_tail_q(states=states,action_features=af,risk_returns=b.risk_returns,etas=etas)
    assert d['tail_q_prefit_loss_after']<d['tail_q_prefit_loss_before']
    assert d['tail_q_prefit_steps']==20*(b.size//450)
    assert _eq(before_actor,a.actor.state_dict()); assert _eq(before_v,a.risk_critic.state_dict())

def test_full_update_and_checkpoint_roundtrip():
    a=RCWAV5AAgent(hyperparameters=_small(),seed=4); stats=a.update(_batch(4))
    assert isinstance(stats,RCWAV5AUpdateStats); assert stats.tail_q_prefit_steps==2*(1350//450)
    assert set(stats.tail_q_coverage_stratum_counts)=={'0.90','0.95','0.98'}
    payload=a.checkpoint_payload(); assert payload['protocol_id']=='awm-rcwa-rl-v5a'
    restored=RCWAV5AAgent(hyperparameters=_small(),seed=4); restored.load_checkpoint_payload(payload)
    assert _eq(a.tail_q_critic.state_dict(),restored.tail_q_critic.state_dict())

def test_trainer_and_smoke_identity():
    assert RCWAV5ATrainer.TRAINER_PROTOCOL_ID=='awm-rcwa-trainer-v5a'
    assert RCWAV5ATrainer.RUNTIME_SUBDIR=='rcwa_rl_v5a'; assert RCWAV5ATrainer.MANIFEST_ID=='awm-rcwa-run-manifest-v5a'
    assert _smoke_protocol_spec('awm-rcwa-rl-v5a')[0] is RCWAV5AAgent
    p=json.loads((ROOT/V5A_PROTOCOL_PATH).read_text()); h=v5a_hyperparameters_from_protocol(p)
    assert h.tail_q_prefit_epochs==10 and h.risk_credit_prefit_epochs==10
