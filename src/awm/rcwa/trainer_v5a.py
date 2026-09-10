"""Sequential, resumable real-DSSAT trainer for RCWA-v5a coverage-only ablation."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from typing import Any
from .agent_v5a import RCWAV5AAgent, RCWAV5AHyperparameters
from .trainer import RCWATrainer, hyperparameters_from_protocol

V5A_PROTOCOL_PATH = "configs/rcwa_rl_v5a.json"

def v5a_hyperparameters_from_protocol(protocol: dict[str, Any]) -> RCWAV5AHyperparameters:
    base = hyperparameters_from_protocol(protocol)
    credit = protocol["tail_credit"]
    return RCWAV5AHyperparameters(
        **asdict(base),
        risk_credit_prefit_epochs=int(credit["risk_credit_prefit_epochs"]),
        tail_q_prefit_epochs=int(credit["tail_q_prefit_epochs"]),
    )

class RCWAV5ATrainer(RCWATrainer):
    PROTOCOL_RELPATH = V5A_PROTOCOL_PATH
    TRAINER_PROTOCOL_ID = "awm-rcwa-trainer-v5a"
    RUNTIME_SUBDIR = "rcwa_rl_v5a"
    MANIFEST_ID = "awm-rcwa-run-manifest-v5a"
    AGENT_CLS = RCWAV5AAgent
    @staticmethod
    def hyperparameters_from_protocol(protocol: dict[str, Any]) -> RCWAV5AHyperparameters:
        return v5a_hyperparameters_from_protocol(protocol)

def main() -> None:
    p=argparse.ArgumentParser(description="Train AWM RCWA-v5a coverage-only ablation on real DSSAT")
    p.add_argument("--project-root", default=".")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--runtime-base")
    p.add_argument("--output-dir")
    p.add_argument("--resume")
    p.add_argument("--stop-after-update", type=int, required=True)
    a=p.parse_args()
    trainer=RCWAV5ATrainer(project_root=a.project_root, seed=a.seed, device=a.device, output_dir=a.output_dir, runtime_base=a.runtime_base)
    if a.resume: trainer.load_checkpoint(a.resume)
    print(json.dumps(trainer.train_until(a.stop_after_update), ensure_ascii=False, indent=2))

if __name__ == "__main__": main()

__all__=["RCWAV5ATrainer","V5A_PROTOCOL_PATH","v5a_hyperparameters_from_protocol"]
