"""Lanczos spectrum diagnostic for the v9 exact-KL Fisher operator."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import torch
from awm.rcwa.fisher_cg_diagnostic_v9 import _flat_tensors
from awm.rcwa.trajectory_directional_validation import _restore_source_checkpoint, _training_row, _replay_identity
from awm.rcwa.rollout import collect_balanced_training_rollout
from awm.rcwa.trainer_v9 import RCWAV9Trainer

MARKS=(20,50,100)
SEED=9122026
def _build(root:Path, checkpoint:Path, metrics:Path, update:int, output:Path, runtime:Path, device:str):
    trainer=RCWAV9Trainer(project_root=root,seed=21,device=device,
        output_dir=output/'rollout_source',runtime_base=runtime/'r')
    _restore_source_checkpoint(trainer,checkpoint,'v9')
    rollout=collect_balanced_training_rollout(
        agent=trainer.agent,normalizer=trainer.normalizer,env_factory=trainer.env_factory,
        reference_yield_by_year=trainer.references,training_seed=21,update_index=update)
    replay=_replay_identity(rollout,_training_row(metrics,update))
    agent=trainer.agent; batch=rollout.batch; states=batch.states.to(agent.device)
    params=list(agent.actor.parameters())
    with torch.no_grad():
        dist,gate=agent.actor.components(states)
        old_mean=dist.loc.detach().clone(); old_scale=dist.scale.detach().clone(); old_gate=gate.detach().clone()
    def hvp(v:torch.Tensor)->torch.Tensor:
        kl=agent._mean_exact_kl(states,old_gate_logits=old_gate,old_mean=old_mean,old_scale=old_scale)
        first=torch.autograd.grad(kl,params,create_graph=True,allow_unused=True)
        ff=_flat_tensors(list(first),params)
        second=torch.autograd.grad(torch.dot(ff,v),params,allow_unused=True)
        return _flat_tensors(list(second),params).detach()
    return agent,hvp,replay
def _ritz_summary(alphas:list[float], betas:list[float], m:int)->dict:
    a=np.asarray(alphas[:m],dtype=np.float64)
    b=np.asarray(betas[:max(0,m-1)],dtype=np.float64)
    T=np.diag(a)
    if m>1:
        T += np.diag(b,1)+np.diag(b,-1)
    eig=np.linalg.eigvalsh(T)
    maxeig=float(eig[-1]); tol=max(1e-12,abs(maxeig)*1e-8)
    pos=eig[eig>tol]
    return {
        'm':m,'min_ritz':float(eig[0]),'max_ritz':maxeig,
        'negative_ritz_count':int(np.sum(eig < -tol)),
        'near_zero_ritz_count':int(np.sum(np.abs(eig)<=tol)),
        'min_positive_ritz':float(pos[0]) if pos.size else None,
        'effective_condition':float(maxeig/pos[0]) if pos.size else None,
        'ritz_values':eig.tolist(),
    }


def _lanczos(hvp, n:int, device:torch.device, max_m:int=100):
    gen=torch.Generator(device=device); gen.manual_seed(SEED)
    q=torch.randn(n,generator=gen,device=device); q=q/torch.linalg.vector_norm(q)
    Q=[]; alphas=[]; betas=[]; qprev=None; beta_prev=torch.tensor(0.,device=device)
    for j in range(max_m):
        z=hvp(q)
        if qprev is not None: z=z-beta_prev*qprev
        alpha=torch.dot(q,z); z=z-alpha*q
        # full re-orthogonalization for numerical spectrum diagnostics
        for qi in Q:
            z=z-torch.dot(qi,z)*qi
        beta=torch.linalg.vector_norm(z)
        Q.append(q.detach().clone()); alphas.append(float(alpha.item()))
        if j < max_m-1: betas.append(float(beta.item()))
        if (not torch.isfinite(beta)) or float(beta.item()) < 1e-12: break
        qprev=q; beta_prev=beta; q=z/beta
    return alphas,betas,Q
def diagnose(root:Path, checkpoint:Path, metrics:Path, update:int, output:Path, runtime:Path, device:str):
    if output.exists() and any(output.iterdir()): raise FileExistsError(output)
    output.mkdir(parents=True,exist_ok=True)
    agent,hvp,replay=_build(root,checkpoint,metrics,update,output,runtime,device)
    n=sum(p.numel() for p in agent.actor.parameters())
    dev=next(agent.actor.parameters()).device
    # basic symmetry check
    gen=torch.Generator(device=dev); gen.manual_seed(SEED+update)
    x=torch.randn(n,generator=gen,device=dev); y=torch.randn(n,generator=gen,device=dev)
    Fx=hvp(x); Fy=hvp(y)
    xy=float(torch.dot(x,Fy).item()); yx=float(torch.dot(y,Fx).item())
    sym_rel=abs(xy-yx)/max(1.0,abs(xy),abs(yx))
    alphas,betas,Q=_lanczos(hvp,n,dev,max_m=max(MARKS))
    summaries={str(m):_ritz_summary(alphas,betas,m) for m in MARKS if len(alphas)>=m}
    payload={
        'diagnostic_id':'awm-rcwa-v9-fisher-spectrum-v1','rollout_update_index':update,
        'actor_parameter_count':n,'replay_identity':replay,'symmetry_relative_error':sym_rel,
        'lanczos_steps':len(alphas),'summaries':summaries,
    }
    (output/'fisher_spectrum_diagnostic.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps({k:v for k,v in payload.items() if k!='replay_identity'},indent=2))


def main():
    p=argparse.ArgumentParser(); p.add_argument('--project-root',required=True); p.add_argument('--checkpoint',required=True)
    p.add_argument('--source-metrics',required=True); p.add_argument('--rollout-update',required=True,type=int)
    p.add_argument('--output',required=True); p.add_argument('--runtime-base',required=True); p.add_argument('--device',default='cpu')
    a=p.parse_args(); diagnose(Path(a.project_root).resolve(),Path(a.checkpoint).resolve(),Path(a.source_metrics).resolve(),
        a.rollout_update,Path(a.output).resolve(),Path(a.runtime_base).resolve(),a.device)
if __name__=='__main__': main()
