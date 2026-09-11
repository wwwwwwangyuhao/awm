"""Sequential, resumable real-DSSAT trainer for RCWA-v9."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from typing import Any
from .agent_v9 import RCWAV9Agent, RCWAV9Hyperparameters
from .trainer import RCWATrainer, hyperparameters_from_protocol

V9_PROTOCOL_PATH = "configs/rcwa_rl_v9.json"

def v9_hyperparameters_from_protocol(protocol: dict[str, Any]) -> RCWAV9Hyperparameters:
    base = hyperparameters_from_protocol(protocol)
    credit = protocol["tail_credit"]
    return RCWAV9Hyperparameters(
        **asdict(base),
        risk_credit_prefit_epochs=int(credit["risk_credit_prefit_epochs"]),
        tail_q_prefit_epochs=int(credit["tail_q_prefit_epochs"]),
    )

class RCWAV9Trainer(RCWATrainer):
    PROTOCOL_RELPATH = V9_PROTOCOL_PATH
    TRAINER_PROTOCOL_ID = "awm-rcwa-trainer-v9"
    RUNTIME_SUBDIR = "rcwa_rl_v9"
    MANIFEST_ID = "awm-rcwa-run-manifest-v9"
    AGENT_CLS = RCWAV9Agent
    @staticmethod
    def hyperparameters_from_protocol(protocol: dict[str, Any]) -> RCWAV9Hyperparameters:
        return v9_hyperparameters_from_protocol(protocol)

def main() -> None:
    p=argparse.ArgumentParser(description="Train AWM RCWA-v9 exact-KL trust-region Tail-Q policy on real DSSAT")
    p.add_argument("--project-root", default=".")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--runtime-base")
    p.add_argument("--output-dir")
    p.add_argument("--resume")
    p.add_argument("--stop-after-update", type=int, required=True)
    a=p.parse_args()
    trainer=RCWAV9Trainer(project_root=a.project_root,seed=a.seed,device=a.device,output_dir=a.output_dir,runtime_base=a.runtime_base)
    if a.resume: trainer.load_checkpoint(a.resume)
    print(json.dumps(trainer.train_until(a.stop_after_update),ensure_ascii=False,indent=2))

if __name__ == "__main__": main()

__all__=["RCWAV9Trainer","V9_PROTOCOL_PATH","v9_hyperparameters_from_protocol"]
