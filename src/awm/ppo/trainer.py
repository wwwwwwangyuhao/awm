"""Sequential, resumable real-DSSAT trainer for AWM PPO baseline v1."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from statistics import mean
from typing import Any

import torch

from .agent import PPOAgent, PPOHyperparameters
from .evaluation import evaluate_checkpoint
from .normalization import NormalizerState, RunningObservationNormalizer
from .real_env import PPORealEnvFactory
from .rollout import collect_balanced_training_rollout


PROTOCOL_PATH = "configs/ppo_baseline_v1.json"
REFERENCE_PATH = "configs/yield_reference_v1_development.json"


def _load_protocol(root: Path) -> dict[str, Any]:
    return json.loads((root / PROTOCOL_PATH).read_text(encoding="utf-8"))


def _load_references(root: Path) -> dict[int, float]:
    payload = json.loads((root / REFERENCE_PATH).read_text(encoding="utf-8"))
    merged = {**payload["training_years"], **payload["validation_years"]}
    return {int(year): float(value) for year, value in merged.items()}


def hyperparameters_from_protocol(protocol: dict[str, Any]) -> PPOHyperparameters:
    policy = protocol["policy"]
    optimizer = protocol["optimizer"]
    environment = protocol["environment"]
    return PPOHyperparameters(
        state_dim=int(environment["observation_dim"]),
        actor_hidden_dims=tuple(int(x) for x in policy["actor_hidden_dims"]),
        critic_hidden_dims=tuple(int(x) for x in policy["critic_hidden_dims"]),
        learning_rate=float(optimizer["learning_rate"]),
        gamma=float(optimizer["gamma"]),
        gae_lambda=float(optimizer["gae_lambda"]),
        clip_epsilon=float(optimizer["clip_epsilon"]),
        update_epochs=int(optimizer["update_epochs"]),
        minibatch_size=int(optimizer["minibatch_size"]),
        value_loss_coefficient=float(optimizer["value_loss_coefficient"]),
        entropy_coefficient=float(optimizer["entropy_coefficient"]),
        max_grad_norm=float(optimizer["max_grad_norm"]),
        adam_beta1=float(optimizer["adam_beta1"]),
        adam_beta2=float(optimizer["adam_beta2"]),
        adam_eps=float(optimizer["adam_eps"]),
    )


class PPOTrainer:
    def __init__(
        self,
        *,
        project_root: str | Path,
        seed: int,
        device: str = "cpu",
        output_dir: str | Path | None = None,
        runtime_base: str | Path | None = None,
    ) -> None:
        self.root = Path(project_root).expanduser().resolve()
        self.protocol = _load_protocol(self.root)
        registered_seeds = tuple(int(x) for x in self.protocol["training"]["seeds"])
        if int(seed) not in registered_seeds:
            raise ValueError(f"seed must be one of preregistered values {registered_seeds}")
        self.seed = int(seed)
        self.references = _load_references(self.root)
        self.hparams = hyperparameters_from_protocol(self.protocol)
        self.agent = PPOAgent(hyperparameters=self.hparams, seed=self.seed, device=device)
        self.normalizer = RunningObservationNormalizer(state_dim=self.hparams.state_dim)
        self.output_dir = (
            Path(output_dir).expanduser().resolve()
            if output_dir is not None
            else (self.root / "runtime" / "ppo_baseline_v1" / f"seed_{self.seed}").resolve()
        )
        try:
            self.output_dir.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("PPO output_dir must remain inside the AWM project root") from exc
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.env_factory = PPORealEnvFactory(
            project_root=self.root,
            work_dir=self.output_dir / "env",
            runtime_base=runtime_base,
            env_idx=0,
        )
        self.metrics_path = self.output_dir / "train_updates.jsonl"
        self.validation_dir = self.output_dir / "validation"
        self.recovery_checkpoint_dir = self.output_dir / "recovery_checkpoints"
        self.candidate_checkpoint_dir = self.output_dir / "candidate_checkpoints"
        self.validation_dir.mkdir(parents=True, exist_ok=True)
        self.recovery_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.candidate_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @property
    def transition_count(self) -> int:
        return self.agent.update_index * int(self.protocol["rollout"]["transitions_per_update"])

    def run_update(self) -> dict[str, Any]:
        update_number = self.agent.update_index + 1
        rollout = collect_balanced_training_rollout(
            agent=self.agent,
            normalizer=self.normalizer,
            env_factory=self.env_factory,
            reference_yield_by_year=self.references,
            training_seed=self.seed,
            update_index=update_number,
        )
        stats = self.agent.update(rollout.batch)
        if stats.update_index != update_number:
            raise RuntimeError("PPO update index drifted from rollout scheduler index")
        outcomes = rollout.outcomes
        payload = {
            "status": "passed",
            "ppo_protocol_id": self.protocol["ppo_protocol_id"],
            "seed": self.seed,
            "update_index": stats.update_index,
            "transition_count": self.transition_count,
            "policy_version": stats.new_policy_version,
            "rollout_episode_count": len(outcomes),
            "rollout_transition_count": stats.sample_count,
            "normalizer_training_observation_count": rollout.normalizer_training_observation_count,
            "mean_policy_irrigation_mm": mean(x.policy_irrigation_mm for x in outcomes),
            "mean_total_irrigation_mm": mean(x.dssat_ircm_mm for x in outcomes),
            "mean_yield_kg_ha": mean(x.yield_kg_ha for x in outcomes),
            "mean_episode_return": mean(x.reward.episode_return for x in outcomes),
            "target_feasible_episode_fraction": mean(
                1.0 if x.reward.target_feasible else 0.0 for x in outcomes
            ),
            "optimizer": asdict(stats),
            "eta_summary": {
                f"{eta:.2f}": {
                    "episode_count": sum(abs(x.eta - eta) <= 1e-12 for x in outcomes),
                    "mean_policy_irrigation_mm": mean(
                        x.policy_irrigation_mm for x in outcomes if abs(x.eta - eta) <= 1e-12
                    ),
                    "mean_retention": mean(
                        x.reward.yield_retention for x in outcomes if abs(x.eta - eta) <= 1e-12
                    ),
                    "mean_return": mean(
                        x.reward.episode_return for x in outcomes if abs(x.eta - eta) <= 1e-12
                    ),
                }
                for eta in (0.90, 0.95, 0.98)
            },
        }
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return payload

    def _checkpoint_payload(self) -> dict[str, object]:
        state = self.normalizer.state()
        return {
            "trainer_protocol_id": "awm-ppo-trainer-v1",
            "agent": self.agent.checkpoint_payload(),
            "normalizer": asdict(state),
            "transition_count": self.transition_count,
        }

    def checkpoint_path(self, update_index: int | None = None) -> Path:
        """Backward-compatible recovery checkpoint path."""
        idx = self.agent.update_index if update_index is None else int(update_index)
        return self.recovery_checkpoint_dir / f"ppo_seed{self.seed}_update{idx:04d}.pt"

    def candidate_checkpoint_path(self, update_index: int | None = None) -> Path:
        idx = self.agent.update_index if update_index is None else int(update_index)
        return self.candidate_checkpoint_dir / f"ppo_seed{self.seed}_update{idx:04d}.pt"

    def save_checkpoint(self) -> Path:
        """Save a recovery checkpoint; recovery files are never selector candidates."""
        path = self.checkpoint_path()
        torch.save(self._checkpoint_payload(), path)
        return path

    def save_candidate_checkpoint(self) -> Path:
        """Save a scientific candidate checkpoint on the frozen candidate cadence."""
        interval = int(self.protocol["training"]["candidate_checkpoint_interval_updates"])
        if self.agent.update_index <= 0 or self.agent.update_index % interval != 0:
            raise ValueError(
                f"update {self.agent.update_index} is not on the candidate checkpoint cadence {interval}"
            )
        path = self.candidate_checkpoint_path()
        torch.save(self._checkpoint_payload(), path)
        return path

    def load_checkpoint(self, path: str | Path) -> None:
        payload = torch.load(Path(path), map_location=self.agent.device, weights_only=False)
        if payload.get("trainer_protocol_id") != "awm-ppo-trainer-v1":
            raise ValueError("trainer checkpoint protocol mismatch")
        self.agent.load_checkpoint_payload(payload["agent"])
        raw = payload["normalizer"]
        self.normalizer.load_state(
            NormalizerState(
                count=float(raw["count"]),
                mean=tuple(float(x) for x in raw["mean"]),
                variance=tuple(float(x) for x in raw["variance"]),
            )
        )
        if int(payload["transition_count"]) != self.transition_count:
            raise ValueError("checkpoint transition_count is inconsistent with update index")

    def validate_current_checkpoint(self) -> dict[str, Any]:
        checkpoint_id = f"ppo_seed{self.seed}_update{self.agent.update_index:04d}"
        report = evaluate_checkpoint(
            checkpoint_id=checkpoint_id,
            training_seed=self.seed,
            training_step=self.transition_count,
            agent=self.agent,
            normalizer=self.normalizer,
            env_factory=self.env_factory,
            reference_yield_by_year=self.references,
        )
        destination = self.validation_dir / f"{checkpoint_id}.json"
        destination.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return report

    def train_until(self, stop_after_update: int) -> dict[str, Any]:
        formal_max = int(self.protocol["training"]["max_updates"])
        target = int(stop_after_update)
        if not self.agent.update_index < target <= formal_max:
            raise ValueError(
                f"stop_after_update must lie in ({self.agent.update_index}, {formal_max}]"
            )
        recovery_interval = int(self.protocol["training"]["recovery_checkpoint_interval_updates"])
        candidate_interval = int(self.protocol["training"]["candidate_checkpoint_interval_updates"])
        validation_interval = int(self.protocol["training"]["validation_interval_updates"])
        if candidate_interval != validation_interval:
            raise RuntimeError("candidate and validation cadence must match in PPO Protocol v1")
        validation_reports: list[str] = []
        candidate_checkpoints: list[str] = []
        last_recovery_checkpoint: str | None = None
        while self.agent.update_index < target:
            self.run_update()
            if self.agent.update_index % recovery_interval == 0:
                last_recovery_checkpoint = str(self.save_checkpoint())
            if self.agent.update_index % candidate_interval == 0:
                candidate_checkpoints.append(str(self.save_candidate_checkpoint()))
                report = self.validate_current_checkpoint()
                validation_reports.append(str(self.validation_dir / f"{report['checkpoint_id']}.json"))
        return {
            "status": "formal_complete" if target == formal_max else "partial_training_stop",
            "seed": self.seed,
            "update_index": self.agent.update_index,
            "transition_count": self.transition_count,
            "last_recovery_checkpoint": last_recovery_checkpoint,
            "candidate_checkpoints": candidate_checkpoints,
            "validation_reports": validation_reports,
            "formal_max_updates": formal_max,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AWM standard PPO v1 on real DSSAT")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--runtime-base")
    parser.add_argument("--output-dir")
    parser.add_argument("--resume")
    parser.add_argument("--stop-after-update", type=int, required=True)
    args = parser.parse_args()
    trainer = PPOTrainer(
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


__all__ = ["PPOTrainer", "hyperparameters_from_protocol"]
