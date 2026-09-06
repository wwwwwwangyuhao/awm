from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from awm.ppo.trainer import PPOTrainer, hyperparameters_from_protocol


ROOT = Path(__file__).resolve().parents[1]


def _protocol():
    return json.loads((ROOT / "configs" / "ppo_baseline_v1.json").read_text(encoding="utf-8"))


def test_protocol_hyperparameters_freeze_full_episode_gae_and_adam():
    h = hyperparameters_from_protocol(_protocol())
    assert h.gamma == pytest.approx(1.0)
    assert h.gae_lambda == pytest.approx(1.0)
    assert h.adam_beta1 == pytest.approx(0.9)
    assert h.adam_beta2 == pytest.approx(0.999)
    assert h.adam_eps == pytest.approx(1e-8)


def test_recovery_and_candidate_checkpoint_namespaces_are_separate():
    output = ROOT / "runtime" / "pytest_ppo_checkpoint_semantics"
    shutil.rmtree(output, ignore_errors=True)
    trainer = PPOTrainer(
        project_root=ROOT,
        seed=21,
        device="cpu",
        output_dir=output,
        runtime_base=ROOT / "r",
    )
    try:
        trainer.agent.update_index = 9
        trainer.agent.policy_version = 9
        recovery = trainer.save_checkpoint()
        assert recovery.parent.name == "recovery_checkpoints"
        assert recovery.name.endswith("update0009.pt")
        with pytest.raises(ValueError, match="candidate checkpoint cadence"):
            trainer.save_candidate_checkpoint()

        trainer.agent.update_index = 10
        trainer.agent.policy_version = 10
        recovery10 = trainer.save_checkpoint()
        candidate10 = trainer.save_candidate_checkpoint()
        assert recovery10.parent.name == "recovery_checkpoints"
        assert candidate10.parent.name == "candidate_checkpoints"
        assert recovery10.name == candidate10.name == "ppo_seed21_update0010.pt"
        assert recovery10 != candidate10
        assert recovery10.is_file()
        assert candidate10.is_file()
    finally:
        shutil.rmtree(output, ignore_errors=True)


def test_interaction_and_candidate_budgets_are_exact():
    config = _protocol()
    budget = config["interaction_budget"]
    training = config["training"]
    assert budget["training_episodes_per_update"] == 54
    assert budget["decision_transitions_per_update"] == 6750
    assert budget["max_updates_per_seed"] == 200
    assert budget["training_episodes_per_seed"] == 54 * 200
    assert budget["decision_transitions_per_seed"] == 6750 * 200
    assert training["candidate_checkpoint_count_per_seed"] == 20
    assert training["validation_episodes_per_candidate"] == 15
    assert training["validation_episodes_per_seed"] == 20 * 15
    assert training["recovery_checkpoint_interval_updates"] == 1
    assert training["candidate_checkpoint_interval_updates"] == 10
    assert training["validation_interval_updates"] == 10
