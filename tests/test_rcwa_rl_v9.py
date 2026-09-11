"""Engineering tests for RCWA-v9 exact-KL trust-region actor optimization."""
from __future__ import annotations
from dataclasses import replace
import json, math
from pathlib import Path
import pytest
import torch
from torch.distributions import Bernoulli, Normal, kl_divergence

from awm.rcwa.agent_v6 import RCWAV6Agent, RCWAV6Hyperparameters
from awm.rcwa.agent_v9 import (
    RCWAV9Agent, RCWAV9Hyperparameters, RCWAV9UpdateStats,
    clip_envelope_kl_radius,
)
from awm.rcwa.trainer_v9 import RCWAV9Trainer, V9_PROTOCOL_PATH, v9_hyperparameters_from_protocol
from test_rcwa_rl_v6 import _synthetic_batch

ROOT=Path(__file__).resolve().parents[1]

def _small(cls=RCWAV9Hyperparameters, **kw):
    values=dict(state_dim=2,actor_hidden_dims=(8,4),reward_critic_hidden_dims=(8,4),risk_critic_hidden_dims=(8,4),update_epochs=1,minibatch_size=450,risk_credit_prefit_epochs=1,tail_q_prefit_epochs=1)
    values.update(kw); return cls(**values)

def _matched_batch(agent, seed=6):
    b=_synthetic_batch(seed=seed)
    with torch.no_grad(): lp,_=agent.actor.evaluate_behavior(b.states.to(agent.device),b.irrigate.to(agent.device),b.raw_amount.to(agent.device))
    return replace(b,old_log_probs=lp.detach().cpu())

def _maxdiff(left,right):
    return max(float((left.state_dict()[k].cpu()-right.state_dict()[k].cpu()).abs().max()) for k in left.state_dict())

def test_v9_protocol_is_single_optimizer_replacement_from_v6():
    v6=json.loads((ROOT/'configs/rcwa_rl_v6.json').read_text())
    v9=json.loads((ROOT/'configs/rcwa_rl_v9.json').read_text())
    assert v9['rcwa_protocol_id']=='awm-rcwa-rl-v9' and v9['supersedes']=='awm-rcwa-rl-v6'
    for section in ('base_contracts','environment','policy','objective','risk_constraint','lagrangian','tail_credit','rollout','interaction_budget','training','evaluation'):
        if section=='tail_credit':
            a=dict(v6[section]); b=dict(v9[section]); a.pop('actor_credit_combination',None); b.pop('actor_credit_combination',None); assert a==b
        else: assert v9[section]==v6[section], section
    assert v9['optimizer']['ppo_clipping_used_for_actor'] is False
    assert math.isclose(v9['optimizer']['trust_region_radius'],clip_envelope_kl_radius(0.2),rel_tol=0,abs_tol=1e-15)

def test_clip_envelope_radius_and_exact_hierarchical_kl():
    expected=-0.5*math.log(1.0-0.2**2)
    assert math.isclose(clip_envelope_kl_radius(.2),expected,rel_tol=0,abs_tol=1e-15)
    old_g=torch.tensor([-1.2,0.3,1.1]); new_g=torch.tensor([-0.8,-0.2,0.7])
    om=torch.tensor([-.2,.4,.8]); nm=torch.tensor([.1,.2,.6]); os=torch.tensor([.7,1.2,.5]); ns=torch.tensor([.9,.8,.6])
    got=RCWAV9Agent._hierarchical_kl_from_old(old_gate_logits=old_g,old_mean=om,old_scale=os,new_gate_logits=new_g,new_mean=nm,new_scale=ns)
    ref=kl_divergence(Bernoulli(logits=old_g),Bernoulli(logits=new_g))+torch.sigmoid(old_g)*kl_divergence(Normal(om,os),Normal(nm,ns))
    assert torch.allclose(got,ref,atol=1e-7,rtol=0)

def test_v9_full_update_respects_exact_kl_and_improves_surrogate():
    a=RCWAV9Agent(hyperparameters=_small(),seed=13,device='cpu')
    stats=a.update(_matched_batch(a))
    assert isinstance(stats,RCWAV9UpdateStats)
    assert stats.trust_region_exact_kl <= stats.trust_region_radius + 1e-6
    assert stats.trust_region_surrogate_after >= stats.trust_region_surrogate_before
    assert stats.trust_region_surrogate_improvement >= 0.0
    assert stats.trust_region_old_logprob_max_abs_error < 1e-6
    assert math.isclose(stats.approx_kl,stats.trust_region_exact_kl,abs_tol=1e-12)
    assert math.isclose(stats.clip_fraction,stats.reference_clip_fraction,abs_tol=1e-12)
    assert stats.trust_region_policy_gradient_norm>0 and stats.trust_region_cg_iterations>0

def test_v9_changes_only_actor_optimizer_path_critics_and_rng_match_v6():
    h6=_small(RCWAV6Hyperparameters); h9=_small(RCWAV9Hyperparameters)
    a6=RCWAV6Agent(hyperparameters=h6,seed=17,device='cpu'); a9=RCWAV9Agent(hyperparameters=h9,seed=17,device='cpu')
    b=_matched_batch(a6,seed=8)
    a6.update(b); s9=a9.update(b)
    assert _maxdiff(a6.reward_critic,a9.reward_critic)==0.0
    assert _maxdiff(a6.risk_critic,a9.risk_critic)==0.0
    assert _maxdiff(a6.tail_q_critic,a9.tail_q_critic)==0.0
    assert torch.equal(a6.generator.get_state(),a9.generator.get_state())
    assert _maxdiff(a6.actor,a9.actor)>0.0
    assert s9.trust_region_exact_kl<=s9.trust_region_radius+1e-6

def test_v9_checkpoint_roundtrip_and_trainer_identity():
    h=_small(); a=RCWAV9Agent(hyperparameters=h,seed=31,device='cpu'); a.update(_matched_batch(a,seed=9)); payload=a.checkpoint_payload()
    assert payload['protocol_id']=='awm-rcwa-rl-v9'
    b=RCWAV9Agent(hyperparameters=h,seed=31,device='cpu'); b.load_checkpoint_payload(payload)
    assert _maxdiff(a.actor,b.actor)==0.0 and _maxdiff(a.tail_q_critic,b.tail_q_critic)==0.0
    assert torch.equal(a.generator.get_state(),b.generator.get_state())
    assert V9_PROTOCOL_PATH=='configs/rcwa_rl_v9.json'
    assert RCWAV9Trainer.TRAINER_PROTOCOL_ID=='awm-rcwa-trainer-v9' and RCWAV9Trainer.RUNTIME_SUBDIR=='rcwa_rl_v9'
    proto=json.loads((ROOT/V9_PROTOCOL_PATH).read_text()); hp=v9_hyperparameters_from_protocol(proto)
    assert hp.tail_q_prefit_epochs==10 and hp.risk_credit_prefit_epochs==10

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_v9_full_update_runs_on_cuda_and_respects_kl():
    a=RCWAV9Agent(hyperparameters=_small(),seed=41,device='cuda:0')
    b=_matched_batch(a,seed=11)
    stats=a.update(b)
    assert stats.trust_region_exact_kl<=stats.trust_region_radius+1e-6
    assert a.tail_q_critic.value_head.weight.device.type=='cuda'
