from __future__ import annotations
from dataclasses import replace
import json
from pathlib import Path
from types import MethodType
import pytest
import torch
from awm.ppo.scheduler import balanced_training_cycle
from awm.rcwa import RCWARolloutBatch, evaluate_eta_tail
from awm.rcwa.agent_v4 import RCWAV4Agent, RCWAV4Hyperparameters
from awm.rcwa.agent_v5 import (
    ACTION_STRATUM_NAMES, RCWAV5Agent, RCWAV5Hyperparameters, V5_PROTOCOL_ID, YEAR_FOLD,
)
from awm.rcwa.trainer_v5 import RCWAV5Trainer, V5_PROTOCOL_PATH, v5_hyperparameters_from_protocol
ROOT=Path(__file__).resolve().parents[1]
ETA_LEVELS=(0.90,0.95,0.98)
EPISODE_LENGTH=25

def _v5_small(**overrides):
    values=dict(state_dim=2,actor_hidden_dims=(8,4),reward_critic_hidden_dims=(8,4),risk_critic_hidden_dims=(8,4),update_epochs=1,minibatch_size=225,risk_credit_prefit_epochs=1,tail_q_prefit_epochs=1)
    values.update(overrides); return RCWAV5Hyperparameters(**values)

def _state_dict_equal(a,b):
    return set(a)==set(b) and all(torch.equal(a[k],b[k]) for k in a)

def _synthetic_batch(*, seed=21, update_index=1, state_dim=2):
    g=torch.Generator().manual_seed(9000+seed+update_index)
    cells=balanced_training_cycle(seed=seed,update_index=update_index)
    n=len(cells)*EPISODE_LENGTH
    states=torch.randn(n,state_dim,generator=g)
    irrigate=torch.rand(n,generator=g)<0.55
    raw=torch.zeros(n); raw[irrigate]=torch.randn(int(irrigate.sum()),generator=g)
    dones=torch.zeros(n,dtype=torch.bool); etas=torch.zeros(n); risk_returns=torch.zeros(n); risk_costs=torch.zeros(n)
    episode_etas=[]; episode_retentions=[]
    for i,cell in enumerate(cells):
        a=i*EPISODE_LENGTH; b=a+EPISODE_LENGTH; dones[b-1]=True; etas[a:b]=cell.eta
        # Finite, weather-dependent MC target; no validation data involved.
        cost=0.2+0.015*((cell.weather_year-2000)%7)+0.1*(cell.eta-0.90)
        risk_returns[a:b]=cost; risk_costs[b-1]=cost; episode_etas.append(cell.eta)
        episode_retentions.append(0.78+0.01*((cell.weather_year-2000)%10))
    metrics={}
    for eta in ETA_LEVELS:
        vals=[r for r,c in zip(episode_retentions,cells) if abs(c.eta-eta)<1e-9]
        metrics[f'{eta:.2f}']=evaluate_eta_tail(vals,eta=eta)
    return RCWARolloutBatch(states=states,irrigate=irrigate,raw_amount=raw,old_log_probs=torch.zeros(n),etas=etas,reward_values=torch.zeros(n),risk_values=torch.zeros(n),reward_returns=torch.zeros(n),risk_returns=risk_returns,reward_advantages=torch.linspace(-1,1,n),risk_advantages=risk_returns.clone(),rewards=-torch.rand(n,generator=g)*0.01,risk_costs=risk_costs,dones=dones,episode_etas=torch.tensor(episode_etas),episode_retentions=torch.tensor(episode_retentions),episode_risk_costs=torch.zeros(len(cells)),tail_metrics=metrics,policy_version=0)

def test_v5_protocol_preserves_v4_frozen_sections():
    v4=json.loads((ROOT/'configs/rcwa_rl_v4.json').read_text()); v5=json.loads((ROOT/'configs/rcwa_rl_v5.json').read_text())
    assert v5['rcwa_protocol_id']=='awm-rcwa-rl-v5' and v5['supersedes']=='awm-rcwa-rl-v4'
    for section in ('base_contracts','environment','objective','risk_constraint','lagrangian','optimizer','rollout','interaction_budget','training','evaluation'):
        assert v5[section]==v4[section],section
    for key,value in v4['policy'].items():
        if key!='fairness_rule': assert v5['policy'][key]==value,key
    c=v5['tail_credit']; assert c['cross_weather_folds']==2; assert c['behavior_policy_changed'] is False
    assert c['validation_used_for_training_or_tuning'] is False; assert c['extra_environment_episodes']==0
    assert len(c['coverage_strata'])==5 and c['tail_q_prefit_epochs']==v5['optimizer']['update_epochs']

def test_v5_trainer_identity_and_no_new_scalar_hparams():
    assert V5_PROTOCOL_PATH=='configs/rcwa_rl_v5.json'; assert RCWAV5Trainer.AGENT_CLS is RCWAV5Agent
    assert RCWAV5Trainer.RUNTIME_SUBDIR=='rcwa_rl_v5'; assert RCWAV5Trainer.MANIFEST_ID=='awm-rcwa-run-manifest-v5'
    h=v5_hyperparameters_from_protocol(json.loads((ROOT/V5_PROTOCOL_PATH).read_text()))
    assert isinstance(h,RCWAV5Hyperparameters); assert h.tail_q_prefit_epochs==10; assert h.entropy_coefficient==0.0

def test_v5_preserves_v4_base_initialization_and_identical_fold_q_initialization():
    h4=RCWAV4Hyperparameters(state_dim=2,actor_hidden_dims=(8,4),reward_critic_hidden_dims=(8,4),risk_critic_hidden_dims=(8,4),update_epochs=1,minibatch_size=225,risk_credit_prefit_epochs=1,tail_q_prefit_epochs=1)
    v4=RCWAV4Agent(hyperparameters=h4,seed=17,device='cpu'); v5=RCWAV5Agent(hyperparameters=_v5_small(),seed=17,device='cpu')
    assert _state_dict_equal(v4.actor.state_dict(),v5.actor.state_dict()); assert _state_dict_equal(v4.reward_critic.state_dict(),v5.reward_critic.state_dict()); assert _state_dict_equal(v4.risk_critic.state_dict(),v5.risk_critic.state_dict())
    assert _state_dict_equal(v5.tail_q_critic.state_dict(),v5.tail_q_critic_fold1.state_dict())

def test_v5_weather_folds_are_permanent_balanced_and_reconstructed_from_formal_cycle():
    assert set(YEAR_FOLD)==set(range(2000,2018)); assert sum(v==0 for v in YEAR_FOLD.values())==9; assert sum(v==1 for v in YEAR_FOLD.values())==9
    a=RCWAV5Agent(hyperparameters=_v5_small(),seed=21,device='cpu'); b=_synthetic_batch(seed=21)
    years,folds=a._transition_weather_folds(b)
    assert years.numel()==b.size; assert int((folds==0).sum())==675; assert int((folds==1).sum())==675
    for y,f in YEAR_FOLD.items(): assert torch.all(folds[years==y]==f)

def test_v5_coverage_weights_equalize_nonempty_action_strata_within_eta():
    a=RCWAV5Agent(hyperparameters=_v5_small(),seed=21,device='cpu')
    strata_one=torch.tensor([0,0,0,0,0,1,1,2,3,4]); strata=strata_one.repeat(3)
    etas=torch.tensor([0.90]*10+[0.95]*10+[0.98]*10); mask=torch.ones(30,dtype=torch.bool)
    w,counts,maxw=a._coverage_weights(train_mask=mask,etas=etas,strata=strata)
    for j,eta in enumerate(ETA_LEVELS):
        group=torch.arange(j*10,(j+1)*10); key=f'{eta:.2f}'
        assert counts[key][ACTION_STRATUM_NAMES[0]]==5 and maxw[key]==pytest.approx(2.0)
        masses=[]
        for s in range(5): masses.append(float(w[group[strata[group]==s]].sum()))
        assert masses==pytest.approx([2.0]*5); assert float(w[group].mean())==pytest.approx(1.0)

def test_v5_fold_prefit_updates_only_allowed_q_and_restores_shared_generator():
    a=RCWAV5Agent(hyperparameters=_v5_small(),seed=21,device='cpu'); b=_synthetic_batch(seed=21)
    states=b.states; af=a._action_features_from_batch(b,device=a.device); _,folds=a._transition_weather_folds(b); strata=a._coverage_strata(af)
    q0_before={k:v.clone() for k,v in a.tail_q_critic.state_dict().items()}; q1_before={k:v.clone() for k,v in a.tail_q_critic_fold1.state_dict().items()}; actor_before={k:v.clone() for k,v in a.actor.state_dict().items()}; rng=a.generator.get_state().clone()
    r=a._prefit_fold_tail_q(fold=0,critic=a.tail_q_critic,states=states,action_features=af,risk_returns=b.risk_returns,etas=b.etas,fold_ids=folds,strata=strata)
    assert r['loss_after']<r['loss_before']; assert not _state_dict_equal(q0_before,a.tail_q_critic.state_dict()); assert _state_dict_equal(q1_before,a.tail_q_critic_fold1.state_dict()); assert _state_dict_equal(actor_before,a.actor.state_dict()); assert torch.equal(rng,a.generator.get_state())

def test_v5_actor_credit_uses_opposite_weather_fold_q():
    a=RCWAV5Agent(hyperparameters=_v5_small(),seed=21,device='cpu'); b=_synthetic_batch(seed=21)
    with torch.no_grad():
        for q,bias in ((a.tail_q_critic,1.0),(a.tail_q_critic_fold1,2.0)):
            for p in q.parameters(): p.zero_()
            q.value_head.bias.fill_(bias)
        for p in a.risk_critic.parameters(): p.zero_()
    a._prefit_risk_critic=MethodType(lambda self,**kwargs: {},a)
    dummy=lambda fold: {'loss_before':0.0,'loss_after':0.0,'grad_mean':0.0,'grad_max':0.0,'steps':0,'counts':{},'max_weights':{},'transition_count':675}
    a._prefit_fold_tail_q=MethodType(lambda self,fold,**kwargs: dummy(fold),a)
    credit,_=a._actor_risk_credit(batch=b,states=b.states,etas=b.etas); _,folds=a._transition_weather_folds(b)
    assert torch.allclose(credit[folds==0],torch.full_like(credit[folds==0],2.0)); assert torch.allclose(credit[folds==1],torch.full_like(credit[folds==1],1.0))

def test_v5_checkpoint_roundtrip_includes_both_fold_critics_and_rejects_v4():
    a=RCWAV5Agent(hyperparameters=_v5_small(),seed=31,device='cpu'); p=a.checkpoint_payload(); assert p['protocol_id']==V5_PROTOCOL_ID and 'tail_q_critic_fold1_state_dict' in p
    b=RCWAV5Agent(hyperparameters=_v5_small(),seed=31,device='cpu'); b.load_checkpoint_payload(p); assert _state_dict_equal(a.tail_q_critic_fold1.state_dict(),b.tail_q_critic_fold1.state_dict())
    v4=RCWAV4Agent(hyperparameters=RCWAV4Hyperparameters(state_dim=2,actor_hidden_dims=(8,4),reward_critic_hidden_dims=(8,4),risk_critic_hidden_dims=(8,4),update_epochs=1,minibatch_size=225,risk_credit_prefit_epochs=1,tail_q_prefit_epochs=1),seed=31,device='cpu')
    with pytest.raises(ValueError,match='protocol_id mismatch'): b.load_checkpoint_payload(v4.checkpoint_payload())

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_v5_full_update_and_cuda_checkpoint_map_location_resume():
    a=RCWAV5Agent(hyperparameters=_v5_small(),seed=41,device='cuda:0'); b=_synthetic_batch(seed=41)
    with torch.no_grad(): lp,_=a.actor.evaluate_behavior(b.states.to(a.device),b.irrigate.to(a.device),b.raw_amount.to(a.device))
    b=replace(b,old_log_probs=lp.cpu()); s=a.update(b); assert s.update_index==1; assert s.tail_q_cross_weather_transition_count=={'fold0_credit_from_fold1':675,'fold1_credit_from_fold0':675}
    assert s.tail_q_prefit_loss_after<s.tail_q_prefit_loss_before
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pt') as f:
        torch.save(a.checkpoint_payload(),f.name); p=torch.load(f.name,map_location='cuda:0',weights_only=False); assert p['generator_state'].device.type=='cuda'
        restored=RCWAV5Agent(hyperparameters=_v5_small(),seed=41,device='cuda:0'); restored.load_checkpoint_payload(p)
    assert torch.equal(a.generator.get_state(),restored.generator.get_state()); assert _state_dict_equal(a.tail_q_critic_fold1.state_dict(),restored.tail_q_critic_fold1.state_dict())
