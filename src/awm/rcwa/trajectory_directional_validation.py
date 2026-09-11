"""Trajectory-level real-DSSAT directional validation for RCWA-v6/v9.

Diagnostic only. Restore a frozen recovery checkpoint, reproduce the next formal
54-cell rollout exactly, verify replay identity against the original training
record, then evaluate symmetric combined-natural-gradient perturbations in a
small exact-KL neighborhood using common random numbers.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from awm.ppo.normalization import NormalizerState
from awm.ppo.scheduler import balanced_training_cycle
from awm.rcwa.agent_v9 import RCWAV9Agent, RCWAV9Hyperparameters, clip_envelope_kl_radius
from awm.rcwa.directional_validation_v9 import (
    DIAG_CRN_SEED,
    DIAG_RADIUS_FRACTION,
    _cpu_state_dict,
    _fixed_draws,
    _natural_direction,
    _signed_alpha_for_kl,
)
from awm.rcwa.risk_batch import evaluate_eta_tail
from awm.rcwa.rollout import collect_balanced_training_rollout
from awm.rcwa.trainer_v6 import RCWAV6Trainer
from awm.rcwa.trainer_v9 import RCWAV9Trainer

ETA_LEVELS = (0.90, 0.95, 0.98)
SOURCE_COMMITS = {
    "v6": "09cfd8242a50c1fcc0424d14735b2cfc3f06d957",
    "v9": "108d7c573fd49f81cb7ebb529df7534db334f773",
}
SOURCE_PROTOCOLS = {"v6": "awm-rcwa-rl-v6", "v9": "awm-rcwa-rl-v9"}
SOURCE_TRAINER_PROTOCOLS = {"v6": "awm-rcwa-trainer-v6", "v9": "awm-rcwa-trainer-v9"}


def _source_trainer(source: str):
    return {"v6": RCWAV6Trainer, "v9": RCWAV9Trainer}[source]


def _restore_source_checkpoint(trainer, checkpoint: Path, source: str) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location=trainer.agent.device, weights_only=False)
    if payload.get("trainer_protocol_id") != SOURCE_TRAINER_PROTOCOLS[source]:
        raise ValueError("source trainer protocol mismatch")
    manifest = payload.get("run_manifest", {})
    if manifest.get("rcwa_protocol_id") != SOURCE_PROTOCOLS[source]:
        raise ValueError("source RCWA protocol mismatch")
    if manifest.get("git_commit") != SOURCE_COMMITS[source]:
        raise ValueError("source git commit mismatch")
    trainer.agent.load_checkpoint_payload(payload["agent"])
    raw = payload["normalizer"]
    trainer.normalizer.load_state(NormalizerState(
        count=float(raw["count"]),
        mean=tuple(float(x) for x in raw["mean"]),
        variance=tuple(float(x) for x in raw["variance"]),
    ))
    expected = trainer.agent.update_index * int(trainer.protocol["rollout"]["transitions_per_update"])
    if int(payload["transition_count"]) != expected:
        raise ValueError("source checkpoint transition_count mismatch")
    return payload


def _training_row(metrics_path: Path, update_index: int) -> dict[str, Any]:
    rows = [json.loads(x) for x in metrics_path.read_text().splitlines() if x.strip()]
    matches = [x for x in rows if int(x["update_index"]) == int(update_index)]
    if len(matches) != 1:
        raise ValueError(f"expected one training row for update {update_index}, got {len(matches)}")
    return matches[0]


def _replay_identity(rollout, original: dict[str, Any]) -> dict[str, Any]:
    outcomes = rollout.outcomes
    actual = {
        "mean_policy_irrigation_mm": float(np.mean([x.policy_irrigation_mm for x in outcomes])),
        "mean_total_irrigation_mm": float(np.mean([x.dssat_ircm_mm for x in outcomes])),
        "mean_yield_kg_ha": float(np.mean([x.yield_kg_ha for x in outcomes])),
        "eta_lcvar": {},
    }
    for eta in ETA_LEVELS:
        vals = [x.signals.yield_retention for x in outcomes if abs(x.eta - eta) <= 1e-12]
        actual["eta_lcvar"][f"{eta:.2f}"] = float(evaluate_eta_tail(vals, eta=eta).empirical_lcvar)
    expected = {
        "mean_policy_irrigation_mm": float(original["mean_policy_irrigation_mm"]),
        "mean_total_irrigation_mm": float(original["mean_total_irrigation_mm"]),
        "mean_yield_kg_ha": float(original["mean_yield_kg_ha"]),
        "eta_lcvar": {k: float(v["lcvar"]) for k, v in original["eta_summary"].items()},
    }
    diffs = {
        "mean_policy_irrigation_mm": actual["mean_policy_irrigation_mm"] - expected["mean_policy_irrigation_mm"],
        "mean_total_irrigation_mm": actual["mean_total_irrigation_mm"] - expected["mean_total_irrigation_mm"],
        "mean_yield_kg_ha": actual["mean_yield_kg_ha"] - expected["mean_yield_kg_ha"],
        "eta_lcvar": {k: actual["eta_lcvar"][k] - expected["eta_lcvar"][k] for k in actual["eta_lcvar"]},
    }
    ok = (
        abs(diffs["mean_policy_irrigation_mm"]) <= 1e-9
        and abs(diffs["mean_total_irrigation_mm"]) <= 1e-9
        and abs(diffs["mean_yield_kg_ha"]) <= 1e-9
        and max(abs(x) for x in diffs["eta_lcvar"].values()) <= 1e-10
    )
    if not ok:
        raise RuntimeError(f"formal rollout replay identity failed: {diffs}")
    return {"passed": True, "actual": actual, "expected": expected, "diffs": diffs}


def _geometry_agent(source_agent, *, seed: int, device: str) -> RCWAV9Agent:
    hp = RCWAV9Hyperparameters(**asdict(source_agent.hparams))
    geom = RCWAV9Agent(hyperparameters=hp, seed=seed, device=device)
    geom.actor.load_state_dict(source_agent.actor.state_dict(), strict=True)
    return geom


def prepare_node(
    *, root: Path, source: str, checkpoint: Path, source_metrics: Path,
    output: Path, runtime_base: Path, rollout_update: int, device: str,
) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    trainer_cls = _source_trainer(source)
    trainer = trainer_cls(
        project_root=root, seed=21, device=device,
        output_dir=output / "rollout_source", runtime_base=runtime_base / "r",
    )
    payload = _restore_source_checkpoint(trainer, checkpoint, source)
    if trainer.agent.update_index + 1 != int(rollout_update):
        raise ValueError("checkpoint/update node mismatch")
    pre_norm = trainer.normalizer.state()
    rollout = collect_balanced_training_rollout(
        agent=trainer.agent, normalizer=trainer.normalizer, env_factory=trainer.env_factory,
        reference_yield_by_year=trainer.references, training_seed=21, update_index=rollout_update,
    )
    original = _training_row(source_metrics, rollout_update)
    replay = _replay_identity(rollout, original)

    batch = rollout.batch
    source_agent = trainer.agent
    states = batch.states.to(source_agent.device)
    etas = batch.etas.to(source_agent.device)
    reward_adv = batch.reward_advantages.to(source_agent.device)
    risk_credit, credit_diag = source_agent._actor_risk_credit(batch=batch, states=states, etas=etas)
    combined_adv, adv_diag = source_agent._condition_actor_advantages(
        reward_adv=reward_adv, risk_adv=risk_credit, etas=etas,
    )

    geom = _geometry_agent(source_agent, seed=21, device=device)
    gstates = batch.states.to(geom.device)
    irrigate = batch.irrigate.to(geom.device)
    raw_amount = batch.raw_amount.to(geom.device)
    old_log_probs = batch.old_log_probs.to(geom.device)
    direction, dmeta = _natural_direction(
        geom, states=gstates, irrigate=irrigate, raw_amount=raw_amount,
        old_log_probs=old_log_probs, advantage=combined_adv.to(geom.device).detach(),
    )
    params = list(geom.actor.parameters())
    base = geom._flat_parameters(params).to(geom.device)
    radius = clip_envelope_kl_radius(geom.hparams.clip_epsilon)
    target = radius * DIAG_RADIUS_FRACTION
    variants = output / "variants"; variants.mkdir(parents=True, exist_ok=True)
    torch.save(_cpu_state_dict(geom.actor), variants / "base.pt")
    signed = {}
    for label, sign in (("plus", 1.0), ("minus", -1.0)):
        alpha, achieved = _signed_alpha_for_kl(
            geom, states=gstates, base=base, direction=direction,
            target_kl=target, sign=sign,
        )
        geom._set_flat_parameters(params, base + alpha * direction)
        torch.save(_cpu_state_dict(geom.actor), variants / f"combined_{label}.pt")
        signed[label] = {"alpha": alpha, "exact_kl": achieved}
        geom._set_flat_parameters(params, base)

    (output / "normalizer_pre_rollout.json").write_text(
        json.dumps(asdict(pre_norm), indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "diagnostic_id": "awm-rcwa-trajectory-directional-v1",
        "source_algorithm": source,
        "source_protocol": SOURCE_PROTOCOLS[source],
        "source_git_commit": SOURCE_COMMITS[source],
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_update": int(payload["agent"]["update_index"]),
        "rollout_update_index": int(rollout_update),
        "replay_identity": replay,
        "target_exact_kl": target,
        "direction": {**dmeta, **signed},
        "dual_by_eta": {f"{k:.2f}": float(v) for k, v in source_agent.dual_by_eta.items()},
        "tail_q_credit_std_by_eta": credit_diag["tail_q_credit_std_by_eta"],
        "tail_q_early_amount_correlation_by_eta": credit_diag[
            "tail_q_credit_early_window_amount_correlation_by_eta"
        ],
        "combined_advantage_mean": float(combined_adv.mean().item()),
        "combined_advantage_std": float(combined_adv.std(unbiased=False).item()),
        "advantage_diagnostics": adv_diag,
    }
    (output / "diagnostic_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def evaluate_variant(*, root: Path, output: Path, runtime_base: Path, variant: str, device: str) -> None:
    destination = output / "evaluations" / f"{variant}.json"
    if destination.exists(): raise FileExistsError(destination)
    actor_path = output / "variants" / f"{variant}.pt"
    if not actor_path.exists(): raise FileNotFoundError(actor_path)
    alias = {"base":"b", "combined_plus":"p", "combined_minus":"m"}[variant]
    trainer = RCWAV9Trainer(
        project_root=root, seed=21, device=device,
        output_dir=output / "evaluation_work" / alias,
        runtime_base=runtime_base / alias,
    )
    trainer.agent.actor.load_state_dict(torch.load(actor_path, map_location=trainer.agent.device, weights_only=True))
    raw = json.loads((output / "normalizer_pre_rollout.json").read_text())
    trainer.normalizer.load_state(NormalizerState(
        count=float(raw["count"]), mean=tuple(float(x) for x in raw["mean"]),
        variance=tuple(float(x) for x in raw["variance"]),
    ))
    cells = balanced_training_cycle(seed=21, update_index=int(json.loads((output/"diagnostic_manifest.json").read_text())["rollout_update_index"]))
    rows = []
    for cell in cells:
        env = trainer.env_factory(cell)
        uniforms, normals = _fixed_draws(cell.weather_year, cell.eta)
        try:
            obs, _ = env.reset(); terminal = None
            for step_index in range(125):
                state_np = trainer.normalizer.normalize(np.asarray(obs.flat(), dtype=np.float32))
                state = torch.from_numpy(state_np).unsqueeze(0).to(trainer.agent.device)
                with torch.no_grad():
                    dist, logits = trainer.agent.actor.components(state)
                    p = float(torch.sigmoid(logits).item())
                    active = bool(uniforms[step_index] < p)
                    raw_a = float(dist.loc.item() + dist.scale.item() * normals[step_index])
                    amount = (math.tanh(raw_a)+1.0)*0.5 if active else 0.0
                result = env.step(irrigate=active, amount_fraction=amount)
                obs = result.observation
                if result.terminated:
                    terminal = dict(result.info); break
            if terminal is None: raise RuntimeError("episode did not terminate")
            retention = float(terminal["HWAM"]) / float(trainer.references[int(cell.weather_year)])
            rows.append({
                "year": int(cell.weather_year), "eta": float(cell.eta), "retention": retention,
                "policy_irrigation_mm": float(terminal["policy_irrigation_mm"]),
                "total_irrigation_mm": float(terminal["IRCM"]), "yield_kg_ha": float(terminal["HWAM"]),
            })
        finally:
            env.close()
    eta_summary = {}
    for eta in ETA_LEVELS:
        group=[r for r in rows if abs(r["eta"]-eta)<=1e-12]
        metric=evaluate_eta_tail([r["retention"] for r in group],eta=eta)
        eta_summary[f"{eta:.2f}"]={
            "mean_retention":float(np.mean([r["retention"] for r in group])),
            "lcvar":float(metric.empirical_lcvar),
            "violation":float(metric.violation),
            "mean_policy_irrigation_mm":float(np.mean([r["policy_irrigation_mm"] for r in group])),
        }
    payload={
        "variant":variant,"episode_count":len(rows),
        "mean_retention":float(np.mean([r["retention"] for r in rows])),
        "mean_policy_irrigation_mm":float(np.mean([r["policy_irrigation_mm"] for r in rows])),
        "mean_lcvar":float(np.mean([eta_summary[f"{e:.2f}"]["lcvar"] for e in ETA_LEVELS])),
        "eta_summary":eta_summary,"cells":rows,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:payload[k] for k in ("variant","mean_retention","mean_policy_irrigation_mm","mean_lcvar")},indent=2))


def aggregate(output: Path) -> None:
    data={v:json.loads((output/"evaluations"/f"{v}.json").read_text()) for v in ("base","combined_plus","combined_minus")}
    base=data["base"]
    rows=[]
    for v,d in data.items():
        rows.append({
            "variant":v,
            "mean_retention":d["mean_retention"],
            "delta_mean_retention":d["mean_retention"]-base["mean_retention"],
            "mean_lcvar":d["mean_lcvar"],
            "delta_mean_lcvar":d["mean_lcvar"]-base["mean_lcvar"],
            "mean_policy_irrigation_mm":d["mean_policy_irrigation_mm"],
            "delta_policy_irrigation_mm":d["mean_policy_irrigation_mm"]-base["mean_policy_irrigation_mm"],
        })
    plus=data["combined_plus"]; minus=data["combined_minus"]
    by_weather=[]
    for year in range(2000,2018):
        dp=[]
        for eta in ETA_LEVELS:
            p=next(r for r in plus["cells"] if r["year"]==year and abs(r["eta"]-eta)<1e-12)
            m=next(r for r in minus["cells"] if r["year"]==year and abs(r["eta"]-eta)<1e-12)
            dp.append(p["retention"]-m["retention"])
        by_weather.append({"year":year,"mean_retention_plus_minus":float(np.mean(dp))})
    payload={
        "rows":rows,
        "combined_plus_minus":{
            "delta_mean_retention":plus["mean_retention"]-minus["mean_retention"],
            "delta_mean_lcvar":plus["mean_lcvar"]-minus["mean_lcvar"],
            "positive_cells":sum(
                1 for p,m in zip(sorted(plus["cells"],key=lambda r:(r["year"],r["eta"])), sorted(minus["cells"],key=lambda r:(r["year"],r["eta"])))
                if p["retention"]>m["retention"]
            ),
            "positive_weather_years":sum(x["mean_retention_plus_minus"]>0 for x in by_weather),
            "by_weather":by_weather,
        },
    }
    (output/"directional_summary.json").write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(payload,indent=2))


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("prepare","evaluate","aggregate"))
    p.add_argument("--project-root",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--runtime-base",required=True)
    p.add_argument("--device",default="cpu")
    p.add_argument("--source",choices=("v6","v9"))
    p.add_argument("--checkpoint")
    p.add_argument("--source-metrics")
    p.add_argument("--rollout-update",type=int)
    p.add_argument("--variant",choices=("base","combined_plus","combined_minus"))
    a=p.parse_args(); root=Path(a.project_root).resolve(); output=Path(a.output).resolve(); runtime=Path(a.runtime_base).resolve()
    if a.mode=="prepare":
        if not all((a.source,a.checkpoint,a.source_metrics,a.rollout_update)): p.error("prepare requires source/checkpoint/source-metrics/rollout-update")
        prepare_node(root=root,source=a.source,checkpoint=Path(a.checkpoint),source_metrics=Path(a.source_metrics),output=output,runtime_base=runtime,rollout_update=a.rollout_update,device=a.device)
    elif a.mode=="evaluate":
        if not a.variant: p.error("evaluate requires variant")
        evaluate_variant(root=root,output=output,runtime_base=runtime,variant=a.variant,device=a.device)
    else: aggregate(output)

if __name__=="__main__": main()
