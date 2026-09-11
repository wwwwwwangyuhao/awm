"""Numerical Fisher/CG diagnostic for frozen RCWA-v9 trajectory nodes.

Diagnostic only: exact-replay a formal rollout, reconstruct the same combined
policy gradient and exact-KL Hessian-vector product, then inspect CG convergence
without updating the policy or changing the training algorithm.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from awm.rcwa.agent_v9 import clip_envelope_kl_radius
from awm.rcwa.rollout import collect_balanced_training_rollout
from awm.rcwa.trajectory_directional_validation import (
    _replay_identity,
    _restore_source_checkpoint,
    _training_row,
)
from awm.rcwa.trainer_v9 import RCWAV9Trainer

MARKS = (50, 100, 200, 500)


def _flat_tensors(
    tensors: tuple[torch.Tensor | None, ...] | list[torch.Tensor | None],
    params: list[torch.nn.Parameter],
) -> torch.Tensor:
    pieces=[]
    for tensor,param in zip(tensors,params):
        pieces.append(torch.zeros_like(param).reshape(-1) if tensor is None else tensor.reshape(-1))
    return torch.cat(pieces)


def _snapshot(
    *, hvp: Callable[[torch.Tensor], torch.Tensor], b: torch.Tensor,
    x: torch.Tensor, recursive_r: torch.Tensor, iteration: int,
) -> dict[str, Any]:
    hx=hvp(x)
    true_r=b-hx
    bnorm=max(float(torch.linalg.vector_norm(b).item()),1e-30)
    q_gd=float(torch.dot(b,x).item())
    q_fd=float(torch.dot(x,hx).item())
    return {
        "iteration":int(iteration),
        "recursive_relative_residual":float(torch.linalg.vector_norm(recursive_r).item())/bnorm,
        "true_relative_residual":float(torch.linalg.vector_norm(true_r).item())/bnorm,
        "g_dot_d":q_gd,
        "d_dot_Fd":q_fd,
        "quadratic_ratio_Fd_over_gd":q_fd/q_gd if abs(q_gd)>1e-30 else None,
        "direction_norm":float(torch.linalg.vector_norm(x).item()),
    }


def _cg_trace(
    hvp: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    *,
    marks: tuple[int,...]=MARKS,
) -> tuple[dict[int,dict[str,Any]],str,int]:
    x=torch.zeros_like(b); r=b.clone(); p=r.clone(); rr=torch.dot(r,r)
    rr0=float(rr.item())
    if rr0<=0.0:
        return {0:_snapshot(hvp=hvp,b=b,x=x,recursive_r=r,iteration=0)},"zero_rhs",0
    out:dict[int,dict[str,Any]]={}
    reason="max_iterations"
    last=0
    max_iter=max(marks)
    for k in range(1,max_iter+1):
        hp=hvp(p); curvature=torch.dot(p,hp)
        c=float(curvature.item())
        if (not math.isfinite(c)) or c<=0.0:
            reason=f"nonpositive_curvature:{c}"
            last=k-1
            break
        alpha=rr/curvature
        x=x+alpha*p
        r=r-alpha*hp
        rr_new=torch.dot(r,r)
        if not torch.isfinite(rr_new):
            raise FloatingPointError("CG recursive residual became NaN/Inf")
        last=k
        if k in marks:
            snap=_snapshot(hvp=hvp,b=b,x=x,recursive_r=r,iteration=k)
            snap["solution"] = x.detach().cpu().clone()
            out[k]=snap
        beta=rr_new/rr
        p=r+beta*p
        rr=rr_new
    return out,reason,last


def _cosine(a:torch.Tensor,b:torch.Tensor)->float:
    na=float(torch.linalg.vector_norm(a).item()); nb=float(torch.linalg.vector_norm(b).item())
    if na<=0.0 or nb<=0.0: return float("nan")
    return float(torch.dot(a,b).item()/(na*nb))


def _exact_kl_for_scaled_direction(
    agent, *, states:torch.Tensor, base:torch.Tensor, direction:torch.Tensor,
    alpha:float, old_gate:torch.Tensor, old_mean:torch.Tensor, old_scale:torch.Tensor,
)->float:
    params=list(agent.actor.parameters())
    agent._set_flat_parameters(params,base+float(alpha)*direction)
    try:
        return float(agent._mean_exact_kl(
            states,old_gate_logits=old_gate,old_mean=old_mean,old_scale=old_scale,
        ).item())
    finally:
        agent._set_flat_parameters(params,base)


def diagnose_node(
    *, root:Path, checkpoint:Path, metrics:Path, rollout_update:int,
    output:Path, runtime_base:Path, device:str,
)->None:
    if output.exists() and any(output.iterdir()): raise FileExistsError(output)
    output.mkdir(parents=True,exist_ok=True)
    trainer=RCWAV9Trainer(project_root=root,seed=21,device=device,
        output_dir=output/"rollout_source",runtime_base=runtime_base/"r")
    payload=_restore_source_checkpoint(trainer,checkpoint,"v9")
    if trainer.agent.update_index+1 != int(rollout_update): raise ValueError("checkpoint/update mismatch")
    rollout=collect_balanced_training_rollout(
        agent=trainer.agent,normalizer=trainer.normalizer,env_factory=trainer.env_factory,
        reference_yield_by_year=trainer.references,training_seed=21,update_index=rollout_update,
    )
    replay=_replay_identity(rollout,_training_row(metrics,rollout_update))
    agent=trainer.agent; batch=rollout.batch
    states=batch.states.to(agent.device); etas=batch.etas.to(agent.device)
    irrigate=batch.irrigate.to(agent.device); raw_amount=batch.raw_amount.to(agent.device)
    old_log_probs=batch.old_log_probs.to(agent.device); reward_adv=batch.reward_advantages.to(agent.device)
    risk_credit,credit_diag=agent._actor_risk_credit(batch=batch,states=states,etas=etas)
    combined_adv,adv_diag=agent._condition_actor_advantages(
        reward_adv=reward_adv,risk_adv=risk_credit,etas=etas,
    )
    params=list(agent.actor.parameters()); base=agent._flat_parameters(params).to(agent.device)
    with torch.no_grad():
        old_dist,old_gate=agent.actor.components(states)
        old_mean=old_dist.loc.detach().clone(); old_scale=old_dist.scale.detach().clone()
        old_gate=old_gate.detach().clone()
        recomputed,_=agent.actor.evaluate_behavior(states,irrigate,raw_amount)
        lp_error=float((recomputed-old_log_probs).abs().max().item())
    if lp_error>2e-5: raise RuntimeError(f"old logprob mismatch {lp_error}")
    new_lp,_=agent.actor.evaluate_behavior(states,irrigate,raw_amount)
    surrogate=(torch.exp(new_lp-old_log_probs)*combined_adv.detach()).mean()
    grads=torch.autograd.grad(surrogate,params,allow_unused=True)
    g=_flat_tensors(list(grads),params).detach()

    def hvp(vector:torch.Tensor)->torch.Tensor:
        kl=agent._mean_exact_kl(
            states,old_gate_logits=old_gate,old_mean=old_mean,old_scale=old_scale,
        )
        first=torch.autograd.grad(kl,params,create_graph=True,allow_unused=True)
        flat_first=_flat_tensors(list(first),params)
        second=torch.autograd.grad(torch.dot(flat_first,vector),params,allow_unused=True)
        out=_flat_tensors(list(second),params).detach()
        if not torch.isfinite(out).all(): raise FloatingPointError("HVP became NaN/Inf")
        return out

    snapshots,stop_reason,last_iter=_cg_trace(hvp,g)
    if not snapshots: raise RuntimeError(f"CG produced no requested snapshots: {stop_reason}")
    final_k=max(snapshots); final_solution=snapshots[final_k].pop("solution")
    radius=clip_envelope_kl_radius(agent.hparams.clip_epsilon)
    for k,snap in snapshots.items():
        sol=final_solution if k==final_k else snap.pop("solution")
        snap["cosine_to_final"]=_cosine(sol,final_solution)
        q_gd=float(snap["g_dot_d"]); q_fd=float(snap["d_dot_Fd"])
        for label,q in (("gd",q_gd),("Fd",q_fd)):
            if q>0.0 and math.isfinite(q):
                alpha=math.sqrt(2.0*radius/q)
                snap[f"quadratic_alpha_{label}"]=alpha
                snap[f"exact_kl_at_alpha_{label}"]=_exact_kl_for_scaled_direction(
                    agent,states=states,base=base,direction=sol,alpha=alpha,
                    old_gate=old_gate,old_mean=old_mean,old_scale=old_scale,
                )
            else:
                snap[f"quadratic_alpha_{label}"]=None
                snap[f"exact_kl_at_alpha_{label}"]=None

    payload={
        "diagnostic_id":"awm-rcwa-v9-fisher-cg-v1",
        "source_checkpoint":str(checkpoint),
        "source_checkpoint_update":int(payload["agent"]["update_index"]),
        "rollout_update_index":int(rollout_update),
        "replay_identity":replay,
        "policy_gradient_norm":float(torch.linalg.vector_norm(g).item()),
        "surrogate_at_base":float(surrogate.detach().item()),
        "old_logprob_max_abs_error":lp_error,
        "trust_region_radius":radius,
        "cg_stop_reason":stop_reason,
        "cg_last_iteration":int(last_iter),
        "snapshots":{str(k):v for k,v in snapshots.items()},
        "tail_q_credit_std_by_eta":credit_diag["tail_q_credit_std_by_eta"],
        "tail_q_early_amount_correlation_by_eta":credit_diag[
            "tail_q_credit_early_window_amount_correlation_by_eta"
        ],
        "advantage_diagnostics":adv_diag,
    }
    destination=output/"fisher_cg_diagnostic.json"
    destination.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(payload,indent=2))


def main()->None:
    p=argparse.ArgumentParser()
    p.add_argument("--project-root",required=True)
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--source-metrics",required=True)
    p.add_argument("--rollout-update",required=True,type=int)
    p.add_argument("--output",required=True)
    p.add_argument("--runtime-base",required=True)
    p.add_argument("--device",default="cpu")
    a=p.parse_args()
    diagnose_node(root=Path(a.project_root).resolve(),checkpoint=Path(a.checkpoint).resolve(),
        metrics=Path(a.source_metrics).resolve(),rollout_update=a.rollout_update,
        output=Path(a.output).resolve(),runtime_base=Path(a.runtime_base).resolve(),device=a.device)


if __name__=="__main__": main()
