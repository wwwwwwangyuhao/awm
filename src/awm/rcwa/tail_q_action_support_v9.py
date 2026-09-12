"""Action-support / identifiability diagnostic for RCWA-v9 Tail-Q.

Diagnostic only: exact formal rollout replay, formal Tail-Q refit, then
behavior-support and counterfactual-baseline sensitivity analysis.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

import torch

from awm.rcwa.agent_v6 import POLICY_BASELINE_GH_ORDER, _gauss_hermite_rule
from awm.rcwa.rollout import collect_balanced_training_rollout
from awm.rcwa.tail_credit import episode_step_indices
from awm.rcwa.tail_q_weather_jackknife_v9 import _fit_credit, _combined_gradient
from awm.rcwa.trajectory_directional_validation import (
    _replay_identity, _restore_source_checkpoint, _training_row,
)
from awm.rcwa.trainer_v9 import RCWAV9Trainer

CENTRAL_MASSES = (0.50, 0.80, 0.90, 0.95, 0.99)
AMOUNT_EDGES = (0.0, 0.25, 0.50, 0.75, 1.0000001)
def _quantiles(x: torch.Tensor, probs=(0.01,0.05,0.25,0.5,0.75,0.95,0.99)):
    if x.numel() == 0:
        return {str(p): None for p in probs}
    q = torch.quantile(x.detach().float().cpu(), torch.tensor(probs))
    return {str(p): float(v) for p, v in zip(probs, q.tolist(), strict=True)}


def _amount_bin_counts(amount: torch.Tensor) -> list[int]:
    vals = amount.detach().cpu()
    counts=[]
    for lo, hi in zip(AMOUNT_EDGES[:-1], AMOUNT_EDGES[1:], strict=True):
        counts.append(int(((vals >= lo) & (vals < hi)).sum().item()))
    return counts


def _eta_masks(agent, etas):
    return agent._eta_masks(etas)


def _behavior_support(agent, *, batch, states, etas) -> dict[str, Any]:
    irrigate=batch.irrigate.to(agent.device)
    raw=batch.raw_amount.to(agent.device)
    with torch.no_grad():
        dist, logits=agent.actor.components(states)
        p=torch.sigmoid(logits)
        active=irrigate.bool()
        z=(raw[active]-dist.loc[active])/dist.scale[active]
        amount=(torch.tanh(raw[active])+1.0)*0.5
    out={"global":{},"by_eta":{}}
    def one(mask):
        a=active & mask
        za=(raw[a]-dist.loc[a])/dist.scale[a]
        aa=(torch.tanh(raw[a])+1.0)*0.5
        pm=p[mask]
        return {
            "transitions":int(mask.sum().item()),
            "active_count":int(a.sum().item()),
            "observed_active_fraction":float(a.float().sum().item()/max(int(mask.sum().item()),1)),
            "expected_active_fraction":float(pm.mean().item()),
            "gate_probability_quantiles":_quantiles(pm),
            "active_amount_quantiles":_quantiles(aa),
            "active_amount_quartile_bin_counts":_amount_bin_counts(aa),
            "active_z_quantiles":_quantiles(za),
            "active_z_fraction_abs_le_1":float((za.abs()<=1).float().mean().item()) if za.numel() else 0.0,
            "active_z_fraction_abs_le_2":float((za.abs()<=2).float().mean().item()) if za.numel() else 0.0,
            "active_z_fraction_abs_le_3":float((za.abs()<=3).float().mean().item()) if za.numel() else 0.0,
        }
    allmask=torch.ones_like(active,dtype=torch.bool)
    out["global"]=one(allmask)
    for key,mask in _eta_masks(agent,etas).items(): out["by_eta"][key]=one(mask)
    return out


def _day_eta_support(agent, *, batch, etas) -> dict[str, Any]:
    steps=episode_step_indices(batch.dones).to(agent.device)
    active=batch.irrigate.to(agent.device).bool()
    raw=batch.raw_amount.to(agent.device)
    amount=torch.where(active,(torch.tanh(raw)+1.0)*0.5,torch.zeros_like(raw))
    rows=[]
    for key,emask in _eta_masks(agent,etas).items():
        for day in range(125):
            mask=emask & (steps==day)
            a=active & mask
            counts=_amount_bin_counts(amount[a])
            rows.append({
                "eta":key,"policy_day":day,"group_size":int(mask.sum().item()),
                "active_count":int(a.sum().item()),"noop_count":int((mask & ~active).sum().item()),
                "occupied_active_amount_bins":sum(x>0 for x in counts),
                "active_amount_bin_counts":counts,
            })
    both=sum(r["active_count"]>0 and r["noop_count"]>0 for r in rows)
    occupied=torch.tensor([r["occupied_active_amount_bins"] for r in rows],dtype=torch.float32)
    active_counts=torch.tensor([r["active_count"] for r in rows],dtype=torch.float32)
    return {
        "natural_group":"same_policy_day_x_eta_across_18_weather_years",
        "group_count":len(rows),
        "groups_with_both_gate_outcomes":both,
        "fraction_groups_with_both_gate_outcomes":both/len(rows),
        "active_count_quantiles":_quantiles(active_counts),
        "occupied_active_amount_bins_quantiles":_quantiles(occupied),
        "groups_with_at_least_3_active_amount_bins":sum(r["occupied_active_amount_bins"]>=3 for r in rows),
        "groups_with_all_4_active_amount_bins":sum(r["occupied_active_amount_bins"]==4 for r in rows),
    }


def _truncated_baseline(agent, states: torch.Tensor, central_mass: float):
    nodes_raw,weights_raw=_gauss_hermite_rule(POLICY_BASELINE_GH_ORDER)
    nodes=torch.as_tensor(nodes_raw,dtype=states.dtype,device=agent.device)
    weights=torch.as_tensor(weights_raw,dtype=states.dtype,device=agent.device)/math.sqrt(math.pi)
    z=math.sqrt(2.0)*nodes
    normal=torch.distributions.Normal(torch.tensor(0.,device=agent.device),torch.tensor(1.,device=agent.device))
    zmax=float(normal.icdf(torch.tensor((1.0+central_mass)/2.0,device=agent.device)).item())
    keep=z.abs()<=zmax
    kept_weight=float(weights[keep].sum().item())
    w=weights[keep]/weights[keep].sum()
    zkeep=z[keep]
    chunk=int(agent.hparams.minibatch_size)
    baselines=[]; active_means=[]; noop_vals=[]; gate_probs=[]
    for start in range(0,int(states.shape[0]),chunk):
        s=states[start:start+chunk]
        dist,logits=agent.actor.components(s); gp=torch.sigmoid(logits)
        noop_action=torch.zeros((s.shape[0],2),dtype=s.dtype,device=agent.device)
        q0=agent.tail_q_critic(s,noop_action)
        raw=dist.loc[:,None]+dist.scale[:,None]*zkeep[None,:]
        amt=(torch.tanh(raw)+1.0)*0.5
        rs=s[:,None,:].expand(-1,zkeep.numel(),-1).reshape(-1,s.shape[1])
        aa=torch.stack((torch.ones_like(amt),amt),dim=-1).reshape(-1,2)
        qa=agent.tail_q_critic(rs,aa).reshape(-1,zkeep.numel())
        qam=(qa*w[None,:]).sum(dim=1)
        baselines.append((1-gp)*q0+gp*qam); active_means.append(qam)
        noop_vals.append(q0); gate_probs.append(gp)
    return torch.cat(baselines).detach(),torch.cat(noop_vals).detach(),torch.cat(active_means).detach(),torch.cat(gate_probs).detach(),kept_weight


def _formal_refit(agent, *, batch, states, etas):
    action_features=agent._action_features_from_batch(batch,device=agent.device)
    risk_returns=batch.risk_returns.to(agent.device)
    mask=torch.ones(batch.size,dtype=torch.bool,device=agent.device)
    credit=_fit_credit(agent,states=states,action_features=action_features,risk_returns=risk_returns,fit_mask=mask,etas=etas)
    qobs=agent.tail_q_critic(states,action_features).detach()
    full_baseline,q0,qa,gp=agent._policy_q_baseline(states)
    return action_features,credit,qobs,full_baseline,q0,qa,gp
def diagnose(*, root:Path, checkpoint:Path, metrics:Path, rollout_update:int,
             output:Path, runtime_base:Path, device:str)->None:
    if output.exists() and any(output.iterdir()): raise FileExistsError(output)
    output.mkdir(parents=True,exist_ok=True)
    trainer=RCWAV9Trainer(project_root=root,seed=21,device=device,
        output_dir=output/'rollout_source',runtime_base=runtime_base/'r')
    _restore_source_checkpoint(trainer,checkpoint,'v9')
    rollout=collect_balanced_training_rollout(agent=trainer.agent,normalizer=trainer.normalizer,
        env_factory=trainer.env_factory,reference_yield_by_year=trainer.references,
        training_seed=21,update_index=rollout_update)
    formal_row=_training_row(metrics,rollout_update)
    replay=_replay_identity(rollout,formal_row)
    batch=rollout.batch
    agent=copy.deepcopy(trainer.agent)
    states=batch.states.to(agent.device); etas=batch.etas.to(agent.device)
    action_features,credit,qobs,full_baseline,q0,qa,gp=_formal_refit(agent,batch=batch,states=states,etas=etas)
    irrigate=batch.irrigate.to(agent.device); raw=batch.raw_amount.to(agent.device)
    old_lp=batch.old_log_probs.to(agent.device); reward_adv=batch.reward_advantages.to(agent.device)
    full_grad,_=_combined_gradient(agent,states=states,irrigate=irrigate,raw_amount=raw,
        old_log_probs=old_lp,reward_adv=reward_adv,risk_credit=credit,etas=etas)
    full_diag=agent._tail_q_diagnostics(q_values=qobs,credit=credit,etas=etas)
    early_diag=agent._tail_q_early_diagnostics(batch=batch,action_features=action_features,credit=credit,etas=etas)
    formal_opt=formal_row['optimizer']; reproduced=True; diffs={}
    for key,source in (
        ('tail_q_credit_std_by_eta',full_diag),
        ('tail_q_credit_early_window_amount_correlation_by_eta',early_diag),
    ):
        diffs[key]={}
        for eta_key,formal_value in formal_opt[key].items():
            diff=abs(float(source[key][eta_key])-float(formal_value))
            diffs[key][eta_key]=diff; reproduced=reproduced and diff<=1e-8
    if not replay['passed'] or not reproduced:
        raise RuntimeError(f'reproduction gate failed replay={replay["passed"]} diffs={diffs}')

    behavior=_behavior_support(agent,batch=batch,states=states,etas=etas)
    local_support=_day_eta_support(agent,batch=batch,etas=etas)
    path=[]
    full_credit_norm=float(torch.linalg.vector_norm(credit).item())
    full_grad_norm=float(torch.linalg.vector_norm(full_grad).item())
    for mass in CENTRAL_MASSES:
        b,_,_,_,retained=_truncated_baseline(agent,states,mass)
        c=(qobs-b).detach()
        g,_=_combined_gradient(agent,states=states,irrigate=irrigate,raw_amount=raw,
            old_log_probs=old_lp,reward_adv=reward_adv,risk_credit=c,etas=etas)
        path.append({
            'requested_central_mass':mass,'quadrature_retained_mass':retained,
            'baseline_mean_abs_diff':float((b-full_baseline).abs().mean().item()),
            'credit_cosine_to_full':float(torch.dot(c,credit).item()/(torch.linalg.vector_norm(c).item()*full_credit_norm)),
            'gradient_cosine_to_full':float(torch.dot(g,full_grad).item()/(torch.linalg.vector_norm(g).item()*full_grad_norm)),
            'gradient_norm_ratio':float(torch.linalg.vector_norm(g).item()/full_grad_norm),
        })
    gate_contrast={}
    for key,mask in _eta_masks(agent,etas).items():
        diff=(qa[mask]-q0[mask]).detach()
        gate_contrast[key]={
            'mean_active_minus_noop_q':float(diff.mean().item()),
            'mean_abs_active_minus_noop_q':float(diff.abs().mean().item()),
            'positive_fraction':float((diff>0).float().mean().item()),
            'gate_probability_mean':float(gp[mask].mean().item()),
        }
    payload={
        'diagnostic_id':'awm-rcwa-v9-tail-q-action-support-v1',
        'source_checkpoint':str(checkpoint.resolve()),
        'source_checkpoint_update':int(trainer.agent.update_index),
        'rollout_update_index':int(rollout_update),
        'replay_identity':replay,
        'full_refit_reproduced_formal_diagnostics':reproduced,
        'full_refit_reproduction_abs_diffs':diffs,
        'behavior_support':behavior,
        'same_day_eta_empirical_support':local_support,
        'gate_q_contrast_by_eta':gate_contrast,
        'central_mass_baseline_sensitivity':path,
        'formal_tail_q_credit_std_by_eta':full_diag['tail_q_credit_std_by_eta'],
        'formal_early_amount_correlation_by_eta':early_diag['tail_q_credit_early_window_amount_correlation_by_eta'],
    }
    (output/'tail_q_action_support.json').write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({
        'rollout_update_index':rollout_update,
        'replay':replay['passed'],'refit':reproduced,
        'global_behavior':behavior['global'],
        'same_day_eta_support':local_support,
        'central_mass_path':path,
    },indent=2))


def main()->None:
    p=argparse.ArgumentParser()
    p.add_argument('--project-root',required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--source-metrics',required=True)
    p.add_argument('--rollout-update',type=int,required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--runtime-base',required=True)
    p.add_argument('--device',default='cpu')
    a=p.parse_args()
    diagnose(
        root=Path(a.project_root).resolve(),checkpoint=Path(a.checkpoint).resolve(),
        metrics=Path(a.source_metrics).resolve(),rollout_update=a.rollout_update,
        output=Path(a.output).resolve(),runtime_base=Path(a.runtime_base).resolve(),
        device=a.device,
    )


if __name__=='__main__':
    main()
