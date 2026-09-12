"""Diagnostic decomposition of damped v9 actor directions."""
from __future__ import annotations
import argparse, json, math
from dataclasses import asdict
from pathlib import Path
import torch
from awm.rcwa.agent_v9 import clip_envelope_kl_radius
from awm.rcwa.directional_validation_v9 import DIAG_RADIUS_FRACTION, _cpu_state_dict, _risk_z, _signed_alpha_for_kl
from awm.rcwa.fisher_cg_diagnostic_v9 import _cg_trace, _cosine, _flat_tensors
from awm.rcwa.fisher_damped_cg_diagnostic_v9 import _true_snapshot
from awm.rcwa.fisher_damped_directional_prepare_v9 import _path_rows, _state_dict_identical
from awm.rcwa.rollout import collect_balanced_training_rollout
from awm.rcwa.trajectory_directional_validation import _replay_identity, _restore_source_checkpoint, _training_row
from awm.rcwa.trainer_v9 import RCWAV9Trainer


def _grad(agent, states, irrigate, raw_amount, old_log_probs, advantage):
    params=list(agent.actor.parameters())
    new_lp,_=agent.actor.evaluate_behavior(states,irrigate,raw_amount)
    surrogate=(torch.exp(new_lp-old_log_probs)*advantage.detach()).mean()
    grads=torch.autograd.grad(surrogate,params,allow_unused=True)
    return _flat_tensors(list(grads),params).detach(), float(surrogate.detach().item())

def probe(*, root:Path, checkpoint:Path, metrics:Path, rollout_update:int, output:Path, runtime_base:Path, device:str, regularization_path:Path, reference_base:Path)->None:
    if output.exists() and any(output.iterdir()): raise FileExistsError(output)
    output.mkdir(parents=True,exist_ok=True)
    trainer=RCWAV9Trainer(project_root=root,seed=21,device=device,output_dir=output/'rollout_source',runtime_base=runtime_base/'r')
    _restore_source_checkpoint(trainer,checkpoint,'v9')
    if trainer.agent.update_index+1 != rollout_update: raise ValueError('checkpoint/update mismatch')
    pre_norm=trainer.normalizer.state()
    rollout=collect_balanced_training_rollout(agent=trainer.agent,normalizer=trainer.normalizer,env_factory=trainer.env_factory,reference_yield_by_year=trainer.references,training_seed=21,update_index=rollout_update)
    replay=_replay_identity(rollout,_training_row(metrics,rollout_update))
    agent=trainer.agent; batch=rollout.batch
    states=batch.states.to(agent.device); etas=batch.etas.to(agent.device)
    irrigate=batch.irrigate.to(agent.device); raw_amount=batch.raw_amount.to(agent.device)
    old_log_probs=batch.old_log_probs.to(agent.device); reward_adv=batch.reward_advantages.to(agent.device)
    risk_credit,credit_diag=agent._actor_risk_credit(batch=batch,states=states,etas=etas)
    reward_z,_,_=agent._standardize(reward_adv); risk_z=_risk_z(agent,risk_credit,etas); lambdas=agent._lambda_tensor(etas)
    advantages={'water':reward_z.detach(),'risk':(-lambdas*risk_z).detach(),'combined':(reward_z-lambdas*risk_z).detach()}
    params=list(agent.actor.parameters()); base=agent._flat_parameters(params).to(agent.device); base_state=_cpu_state_dict(agent.actor)
    if not _state_dict_identical(base_state,reference_base): raise RuntimeError('base actor mismatch')
    with torch.no_grad():
        old_dist,old_gate=agent.actor.components(states); old_mean=old_dist.loc.detach().clone(); old_scale=old_dist.scale.detach().clone(); old_gate=old_gate.detach().clone()
        recomputed,_=agent.actor.evaluate_behavior(states,irrigate,raw_amount)
    lp_err=float((recomputed-old_log_probs).abs().max().item())
    if lp_err>2e-5: raise RuntimeError(f'old logprob mismatch {lp_err}')

    def f_hvp(vector:torch.Tensor)->torch.Tensor:
        kl=agent._mean_exact_kl(states,old_gate_logits=old_gate,old_mean=old_mean,old_scale=old_scale)
        first=torch.autograd.grad(kl,params,create_graph=True,allow_unused=True)
        flat_first=_flat_tensors(list(first),params)
        second=torch.autograd.grad(torch.dot(flat_first,vector),params,allow_unused=True)
        out=_flat_tensors(list(second),params).detach()
        if not torch.isfinite(out).all(): raise FloatingPointError('Fisher HVP NaN/Inf')
        return out
    row=_path_rows(regularization_path)[0]; xi=float(row['xi']); target_tol=float(row['tolerance'])
    def damped_hvp(v): return f_hvp(v)+xi*v
    gradients={}; directions={}; solve={}
    for name,adv in advantages.items():
        g,s0=_grad(agent,states,irrigate,raw_amount,old_log_probs,adv); gradients[name]=g
        snaps,reason,last=_cg_trace(damped_hvp,g,marks=(50,))
        if 50 not in snaps: raise RuntimeError(f'{name} damped CG missing: {reason}')
        d=snaps[50].pop('solution').to(agent.device); directions[name]=d
        detail=_true_snapshot(f_hvp=f_hvp,damped_hvp=damped_hvp,g=g,d=d,xi=xi)
        detail.update({'surrogate_at_base':s0,'cg_stop_reason':reason,'cg_last_iteration':last})
        if detail['true_damped_relative_residual']>target_tol: raise RuntimeError(f'{name} residual missed target')
        solve[name]=detail
    def norm(x): return float(torch.linalg.vector_norm(x).item())
    def cos(a,b): return _cosine(a,b)
    gradient_geometry={'norms':{k:norm(v) for k,v in gradients.items()},'cosines':{'water_risk':cos(gradients['water'],gradients['risk']),'water_combined':cos(gradients['water'],gradients['combined']),'risk_combined':cos(gradients['risk'],gradients['combined'])}}
    direction_geometry={'norms':{k:norm(v) for k,v in directions.items()},'cosines':{'water_risk':cos(directions['water'],directions['risk']),'water_combined':cos(directions['water'],directions['combined']),'risk_combined':cos(directions['risk'],directions['combined'])},'cosine_to_gradient':{k:cos(directions[k],gradients[k]) for k in directions}}

    radius=clip_envelope_kl_radius(agent.hparams.clip_epsilon); target_kl=radius*DIAG_RADIUS_FRACTION
    variants=output/'variants'; variants.mkdir(parents=True,exist_ok=True); torch.save(base_state,variants/'base.pt')
    signed={}
    for name in ('water','risk'):
        d=directions[name]; signed[name]={}
        for label,sign in (('plus',1.0),('minus',-1.0)):
            alpha,achieved=_signed_alpha_for_kl(agent,states=states,base=base,direction=d,target_kl=target_kl,sign=sign)
            agent._set_flat_parameters(params,base+alpha*d)
            torch.save(_cpu_state_dict(agent.actor),variants/f'{name}_{label}.pt')
            signed[name][label]={'alpha':alpha,'exact_kl':achieved}
            agent._set_flat_parameters(params,base)
    (output/'normalizer_pre_rollout.json').write_text(json.dumps(asdict(pre_norm),indent=2)+'\n')
    payload={'diagnostic_id':'awm-rcwa-v9-damped-component-probe-v1','rollout_update_index':rollout_update,'source_checkpoint':str(checkpoint),'replay_identity':replay,'base_actor_identical':True,'xi':xi,'target_tolerance':target_tol,'target_kl':target_kl,'old_logprob_max_abs_error':lp_err,'gradient_geometry':gradient_geometry,'direction_geometry':direction_geometry,'solve':solve,'signed_variants':signed,'credit_diagnostics':credit_diag}
    (output/'diagnostic_manifest.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(payload,indent=2))


def main():
    p=argparse.ArgumentParser(); p.add_argument('--project-root',required=True); p.add_argument('--checkpoint',required=True); p.add_argument('--source-metrics',required=True); p.add_argument('--rollout-update',type=int,required=True); p.add_argument('--output',required=True); p.add_argument('--runtime-base',required=True); p.add_argument('--regularization-path',required=True); p.add_argument('--reference-base',required=True); p.add_argument('--device',default='cpu')
    a=p.parse_args(); probe(root=Path(a.project_root).resolve(),checkpoint=Path(a.checkpoint),metrics=Path(a.source_metrics),rollout_update=a.rollout_update,output=Path(a.output),runtime_base=Path(a.runtime_base),device=a.device,regularization_path=Path(a.regularization_path),reference_base=Path(a.reference_base))
if __name__=='__main__': main()
