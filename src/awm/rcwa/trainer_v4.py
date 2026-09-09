"""Sequential, resumable real-DSSAT trainer for RCWA-RL v4."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from typing import Any

from .agent_v4 import RCWAV4Agent, RCWAV4Hyperparameters
from .trainer import RCWATrainer, hyperparameters_from_protocol


V4_PROTOCOL_PATH = "configs/rcwa_rl_v4.json"


def v4_hyperparameters_from_protocol(protocol: dict[str, Any]) -> RCWAV4Hyperparameters:
    base = hyperparameters_from_protocol(protocol)
    credit = protocol["tail_credit"]
    return RCWAV4Hyperparameters(
        **asdict(base),
        risk_credit_prefit_epochs=int(credit["risk_credit_prefit_epochs"]),
        tail_q_prefit_epochs=int(credit["tail_q_prefit_epochs"]),
    )


class RCWAV4Trainer(RCWATrainer):
    PROTOCOL_RELPATH = V4_PROTOCOL_PATH
    TRAINER_PROTOCOL_ID = "awm-rcwa-trainer-v4"
    RUNTIME_SUBDIR = "rcwa_rl_v4"
    MANIFEST_ID = "awm-rcwa-run-manifest-v4"
    AGENT_CLS = RCWAV4Agent

    @staticmethod
    def hyperparameters_from_protocol(protocol: dict[str, Any]) -> RCWAV4Hyperparameters:
        return v4_hyperparameters_from_protocol(protocol)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AWM RCWA-RL v4 on real DSSAT")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--runtime-base")
    parser.add_argument("--output-dir")
    parser.add_argument("--resume")
    parser.add_argument("--stop-after-update", type=int, required=True)
    args = parser.parse_args()
    trainer = RCWAV4Trainer(
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


__all__ = ["RCWAV4Trainer", "V4_PROTOCOL_PATH", "v4_hyperparameters_from_protocol"]
