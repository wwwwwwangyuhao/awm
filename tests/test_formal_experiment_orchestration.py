from __future__ import annotations

import json
from pathlib import Path

import pytest

from awm.experiments.development_export import build_selected_validation_tables
from awm.experiments.formal_monitor import inspect_seed_run
from awm.experiments.formal_selection import select_seed_checkpoint


ROOT = Path(__file__).resolve().parents[1]


def _eta_metrics(lcvar: float = 0.96, irrigation: float = 400.0) -> list[dict[str, object]]:
    return [
        {
            "eta": eta,
            "validation_years": [2018, 2019, 2020, 2021, 2022],
            "lcvar_retention": lcvar,
            "mean_total_irrigation_mm": irrigation,
            "minimum_retention": lcvar,
        }
        for eta in (0.90, 0.95, 0.98)
    ]


def _validation_report(seed: int, update: int, *, lcvar: float, irrigation: float) -> dict[str, object]:
    checkpoint_id = f"ppo_seed{seed}_update{update:04d}"
    cells = []
    for eta in (0.90, 0.95, 0.98):
        for year in range(2018, 2023):
            cells.append(
                {
                    "weather_year": year,
                    "eta": eta,
                    "action_mode": "deterministic",
                    "yield_kg_ha": 7000.0,
                    "reference_yield_kg_ha": 7000.0,
                    "yield_retention": 1.0,
                    "total_irrigation_mm": irrigation,
                    "policy_irrigation_mm": irrigation - 45.0,
                    "irrigation_accounting_passed": True,
                    "sampled_event_count": 10,
                    "executed_event_count": 10,
                }
            )
    return {
        "checkpoint_id": checkpoint_id,
        "training_seed": seed,
        "training_step": update * 6750,
        "eta_metrics": _eta_metrics(lcvar=lcvar, irrigation=irrigation),
        "cell_count": 15,
        "validation_action_mode": "deterministic",
        "cell_results": cells,
        "normalizer_updated_during_validation": False,
        "final_test_station_results_present": False,
    }


def test_monitor_distinguishes_training_complete_from_selection_ready(tmp_path: Path) -> None:
    run_root = tmp_path / "formal"
    seed_dir = run_root / "seed_11"
    (seed_dir / "candidate_checkpoints").mkdir(parents=True)
    (seed_dir / "validation").mkdir()
    (seed_dir / "recovery_checkpoints").mkdir()
    (seed_dir / "run_manifest.json").write_text(
        json.dumps({"training_seed": 11, "git_commit": "abc", "ppo_protocol_id": "awm-ppo-baseline-v1"}),
        encoding="utf-8",
    )
    with (seed_dir / "train_updates.jsonl").open("w", encoding="utf-8") as handle:
        for update in range(1, 201):
            handle.write(json.dumps({"update_index": update}) + "\n")
    status = inspect_seed_run(
        method_id="ppo",
        expected_protocol_id="awm-ppo-baseline-v1",
        seed=11,
        run_root=run_root,
        max_updates=200,
        candidate_updates=(10, 20),
        process_snapshot="",
    )
    assert status.training_complete is True
    assert status.selection_ready is False
    assert status.status == "artifact_incomplete"

    for update in (10, 20):
        stem = f"ppo_seed11_update{update:04d}"
        (seed_dir / "candidate_checkpoints" / f"{stem}.pt").write_bytes(b"checkpoint")
        (seed_dir / "validation" / f"{stem}.json").write_text("{}", encoding="utf-8")
    ready = inspect_seed_run(
        method_id="ppo",
        expected_protocol_id="awm-ppo-baseline-v1",
        seed=11,
        run_root=run_root,
        max_updates=200,
        candidate_updates=(10, 20),
        process_snapshot="",
    )
    assert ready.selection_ready is True
    assert ready.status == "selection_ready"


def test_selector_uses_frozen_rule_and_hashes_selected_checkpoint(tmp_path: Path) -> None:
    seed_dir = tmp_path / "seed_11"
    (seed_dir / "validation").mkdir(parents=True)
    (seed_dir / "candidate_checkpoints").mkdir()
    (seed_dir / "run_manifest.json").write_text(
        json.dumps({"training_seed": 11, "git_commit": "abc", "ppo_protocol_id": "awm-ppo-baseline-v1"}),
        encoding="utf-8",
    )
    for update, irrigation in ((10, 410.0), (20, 390.0)):
        report = _validation_report(11, update, lcvar=0.99, irrigation=irrigation)
        stem = report["checkpoint_id"]
        (seed_dir / "validation" / f"{stem}.json").write_text(json.dumps(report), encoding="utf-8")
        (seed_dir / "candidate_checkpoints" / f"{stem}.pt").write_bytes(f"ckpt-{update}".encode())
    selected = select_seed_checkpoint(
        method_id="ppo",
        expected_protocol_id="awm-ppo-baseline-v1",
        seed=11,
        seed_dir=seed_dir,
        candidate_updates=(10, 20),
        transitions_per_update=6750,
    )
    assert selected["selection"]["selected_checkpoint_id"] == "ppo_seed11_update0020"
    assert selected["selection"]["jointly_feasible"] is True
    assert len(selected["selected_checkpoint_sha256"]) == 64
    assert selected["final_test_station_results_present"] is False


def _selection_document(method_id: str) -> dict[str, object]:
    selected = []
    for seed in (11, 21, 31, 41, 51):
        report = _validation_report(seed, 20, lcvar=0.96, irrigation=400.0)
        selected.append(
            {
                "training_seed": seed,
                "selection": {
                    "training_seed": seed,
                    "selected_checkpoint_id": report["checkpoint_id"],
                    "selected_training_step": 135000,
                    "selection_status": "selected_infeasible",
                    "jointly_feasible": False,
                    "mean_total_irrigation_mm": 400.0,
                    "minimum_risk_margin": -0.02,
                    "mean_risk_margin": 0.0,
                    "worst_shortfall": 0.02,
                    "mean_shortfall": 0.01,
                    "candidate_count": 20,
                },
                "selected_checkpoint_sha256": "a" * 64,
                "selected_validation_report": report,
            }
        )
    return {
        "method_id": method_id,
        "selected_checkpoints": selected,
        "final_test_station_results_present": False,
    }


def test_development_export_requires_five_seeds_and_rejects_final_test() -> None:
    cells, seeds, summary = build_selected_validation_tables([_selection_document("ppo")])
    assert len(cells) == 75
    assert len(seeds) == 15
    assert summary["final_test_station_results_present"] is False
    assert summary["validation_years"] == [2018, 2019, 2020, 2021, 2022]

    contaminated = _selection_document("ppo")
    contaminated["selected_checkpoints"][0]["selected_validation_report"]["cell_results"][0][
        "weather_year"
    ] = 2023
    with pytest.raises(RuntimeError, match="final-test"):
        build_selected_validation_tables([contaminated])


def test_orchestration_and_ablation_protocols_are_frozen_and_fair() -> None:
    orchestration = json.loads(
        (ROOT / "configs" / "formal_experiment_orchestration_v1.json").read_text(encoding="utf-8")
    )
    assert orchestration["training"]["training_episodes_per_seed"] == 10800
    assert orchestration["training"]["training_transitions_per_seed"] == 1350000
    assert len(orchestration["selection"]["candidate_updates"]) == 20
    assert orchestration["development_export"]["final_test_access_allowed"] is False

    ablation = json.loads(
        (ROOT / "configs" / "rcwa_ablation_protocol_v1.json").read_text(encoding="utf-8")
    )
    independent = ablation["primary_ablation"]
    assert independent["total_episodes_per_seed_across_three_policies"] == 10800
    assert independent["total_decisions_per_seed_across_three_policies"] == 1350000
    assert len(independent["candidate_bundle_updates"]) == 20
    assert ablation["primary_component_control"]["new_training_run_required"] is False
    assert all(item["development_only"] is True for item in ablation["sensitivity"])
    assert "forbidden" in ablation["governance"]["station_final_test_access"]
