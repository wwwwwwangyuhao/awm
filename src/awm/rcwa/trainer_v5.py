"""Sequential, resumable real-DSSAT trainer for RCWA-RL v5."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from typing import Any
from .agent_v5 import RCWAV5Agent, RCWAV5Hyperparameters
from .trainer_v4 import v4_hyperparameters_from_protocol
from .trainer import RCWATrainer
V5_PROTOCOL_PATH = "configs/rcwa_rl_v5.json"
def v5_hyperparameters_from_protocol(protocol: dict[str, Any]) -> RCWAV5Hyperparameters:
    return RCWAV5Hyperparameters(**asdict(v4_hyperparameters_from_protocol(protocol)))
class RCWAV5Trainer(RCWATrainer):
    PROTOCOL_RELPATH = V5_PROTOCOL_PATH
    TRAINER_PROTOCOL_ID = "awm-rcwa-trainer-v5"
    RUNTIME_SUBDIR = "rcwa_rl_v5"
    MANIFEST_ID = "awm-rcwa-run-manifest-v5"
    AGENT_CLS = RCWAV5Agent
    @staticmethod
    def hyperparameters_from_protocol(protocol: dict[str, Any]) -> RCWAV5Hyperparameters:
        return v5_hyperparameters_from_protocol(protocol)
def main() -> None:
    p=argparse.ArgumentParser(description="Train AWM RCWA-RL v5 on real DSSAT")
    p.add_argument("--project-root",default="."); p.add_argument("--seed",type=int,required=True)
    p.add_argument("--device",default="cpu"); p.add_argument("--runtime-base"); p.add_argument("--output-dir")
    p.add_argument("--resume"); p.add_argument("--stop-after-update",type=int,required=True); a=p.parse_args()
    t=RCWAV5Trainer(project_root=a.project_root,seed=a.seed,device=a.device,output_dir=a.output_dir,runtime_base=a.runtime_base)
    if a.resume: t.load_checkpoint(a.resume)
    print(json.dumps(t.train_until(a.stop_after_update),ensure_ascii=False,indent=2))
if __name__=="__main__": main()
__all__=["RCWAV5Trainer","V5_PROTOCOL_PATH","v5_hyperparameters_from_protocol"]
