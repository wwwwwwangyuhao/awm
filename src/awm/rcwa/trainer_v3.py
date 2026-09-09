"""Sequential, resumable real-DSSAT trainer for RCWA-RL v3.

Identical to the RCWA v2 trainer except for version identity: v3 reads
``configs/rcwa_rl_v3.json``, writes ``runtime/rcwa_rl_v3``, emits the v3 run
manifest and checkpoint protocol ids, and trains an :class:`RCWAV3Agent`.
No v2 checkpoint payload is accepted and no run directory is shared with v2.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from typing import Any

import torch

from .agent_v3 import RCWAV3Agent, RCWAV3Hyperparameters
from .trainer import RCWATrainer, hyperparameters_from_protocol


V3_PROTOCOL_PATH = "configs/rcwa_rl_v3.json"


def v3_hyperparameters_from_protocol(protocol: dict[str, Any]) -> RCWAV3Hyperparameters:
    base = hyperparameters_from_protocol(protocol)
    return RCWAV3Hyperparameters(
        **asdict(base),
        risk_credit_prefit_epochs=int(protocol["tail_credit"]["risk_credit_prefit_epochs"]),
    )


class RCWAV3Trainer(RCWATrainer):
    PROTOCOL_RELPATH = V3_PROTOCOL_PATH
    TRAINER_PROTOCOL_ID = "awm-rcwa-trainer-v3"
    RUNTIME_SUBDIR = "rcwa_rl_v3"
    MANIFEST_ID = "awm-rcwa-run-manifest-v3"
    AGENT_CLS = RCWAV3Agent

    @staticmethod
    def hyperparameters_from_protocol(protocol: dict[str, Any]) -> RCWAV3Hyperparameters:
        return v3_hyperparameters_from_protocol(protocol)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AWM RCWA-RL v3 on real DSSAT")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--runtime-base")
    parser.add_argument("--output-dir")
    parser.add_argument("--resume")
    parser.add_argument("--stop-after-update", type=int, required=True)
    args = parser.parse_args()
    trainer = RCWAV3Trainer(
        project_root=args.project_root,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
        runtime_base=args.runtime_base,
    )
    if args.resume:
        trainer.load_checkpoint(args.resume)
    result = trainer.train_until(args.stop_after_update)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


__all__ = ["RCWAV3Trainer", "V3_PROTOCOL_PATH", "v3_hyperparameters_from_protocol"]
