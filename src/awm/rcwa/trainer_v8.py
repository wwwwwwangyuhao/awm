"""Sequential, resumable real-DSSAT trainer for RCWA-v8."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from typing import Any
from .agent_v8 import RCWAV8Agent, RCWAV8Hyperparameters
from .trainer import RCWATrainer, hyperparameters_from_protocol

V8_PROTOCOL_PATH = "configs/rcwa_rl_v8.json"

def v8_hyperparameters_from_protocol(protocol: dict[str, Any]) -> RCWAV8Hyperparameters:
    base = hyperparameters_from_protocol(protocol)
    credit = protocol["tail_credit"]
    return RCWAV8Hyperparameters(
        **asdict(base),
        risk_credit_prefit_epochs=int(credit["risk_credit_prefit_epochs"]),
        tail_q_prefit_epochs=int(credit["tail_q_prefit_epochs"]),
    )

class RCWAV8Trainer(RCWATrainer):
    PROTOCOL_RELPATH = V8_PROTOCOL_PATH
    TRAINER_PROTOCOL_ID = "awm-rcwa-trainer-v8"
    RUNTIME_SUBDIR = "rcwa_rl_v8"
    MANIFEST_ID = "awm-rcwa-run-manifest-v8"
    AGENT_CLS = RCWAV8Agent
    @staticmethod
    def hyperparameters_from_protocol(protocol: dict[str, Any]) -> RCWAV8Hyperparameters:
        return v8_hyperparameters_from_protocol(protocol)

def main() -> None:
    p=argparse.ArgumentParser(description="Train AWM RCWA-v8 twin-Q reliability ablation on real DSSAT")
    p.add_argument("--project-root", default=".")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--runtime-base")
    p.add_argument("--output-dir")
    p.add_argument("--resume")
    p.add_argument("--stop-after-update", type=int, required=True)
    a=p.parse_args()
    trainer=RCWAV8Trainer(project_root=a.project_root,seed=a.seed,device=a.device,output_dir=a.output_dir,runtime_base=a.runtime_base)
    if a.resume: trainer.load_checkpoint(a.resume)
    print(json.dumps(trainer.train_until(a.stop_after_update),ensure_ascii=False,indent=2))

if __name__ == "__main__": main()

__all__=["RCWAV8Trainer","V8_PROTOCOL_PATH","v8_hyperparameters_from_protocol"]
