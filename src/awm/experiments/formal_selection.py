"""Automatic formal checkpoint selection using the frozen evaluation contract."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
from typing import Mapping

from awm.evaluation import build_candidate_from_report, select_checkpoint


_CONFIG_PATH = "configs/formal_experiment_orchestration_v1.json"
_UPDATE_RE = re.compile(r"update(?P<update>\d{4})")


def _load_config(project_root: str | Path) -> dict[str, object]:
    root = Path(project_root).expanduser().resolve()
    return json.loads((root / _CONFIG_PATH).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _update_from_checkpoint_id(checkpoint_id: str) -> int:
    match = _UPDATE_RE.search(checkpoint_id)
    if not match:
        raise ValueError(f"checkpoint_id does not contain update index: {checkpoint_id}")
    return int(match.group("update"))


def _protocol_id(manifest: Mapping[str, object]) -> str | None:
    for key in ("ppo_protocol_id", "rcwa_protocol_id", "protocol_id"):
        if key in manifest:
            return str(manifest[key])
    return None


def select_seed_checkpoint(
    *,
    method_id: str,
    expected_protocol_id: str,
    seed: int,
    seed_dir: str | Path,
    candidate_updates: tuple[int, ...],
    transitions_per_update: int,
) -> dict[str, object]:
    root = Path(seed_dir).expanduser().resolve()
    manifest_path = root / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing run manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("training_seed", -1)) != int(seed):
        raise ValueError(f"run manifest seed mismatch for {method_id} seed {seed}")
    if _protocol_id(manifest) != expected_protocol_id:
        raise ValueError(f"run manifest protocol mismatch for {method_id} seed {seed}")

    report_paths = sorted((root / "validation").glob("*.json"))
    checkpoint_paths = sorted((root / "candidate_checkpoints").glob("*.pt"))
    if len(report_paths) != len(candidate_updates):
        raise RuntimeError(
            f"{method_id} seed {seed} requires {len(candidate_updates)} validation reports, "
            f"found {len(report_paths)}"
        )
    if len(checkpoint_paths) != len(candidate_updates):
        raise RuntimeError(
            f"{method_id} seed {seed} requires {len(candidate_updates)} candidate checkpoints, "
            f"found {len(checkpoint_paths)}"
        )

    checkpoint_by_stem = {path.stem: path for path in checkpoint_paths}
    if len(checkpoint_by_stem) != len(checkpoint_paths):
        raise RuntimeError("duplicate candidate checkpoint stems")

    candidates = []
    report_by_id: dict[str, tuple[Path, Mapping[str, object]]] = {}
    observed_updates: list[int] = []
    for report_path in report_paths:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise TypeError(f"validation report must be an object: {report_path}")
        if payload.get("final_test_station_results_present") is not False:
            raise RuntimeError(f"final-test contamination flag in {report_path}")
        candidate = build_candidate_from_report(payload)
        if candidate.training_seed != int(seed):
            raise ValueError(f"validation seed mismatch in {report_path}")
        update = _update_from_checkpoint_id(candidate.checkpoint_id)
        if candidate.training_step != update * int(transitions_per_update):
            raise ValueError(f"training_step/update mismatch in {report_path}")
        if candidate.checkpoint_id not in checkpoint_by_stem:
            raise FileNotFoundError(
                f"validation report has no matching candidate checkpoint: {candidate.checkpoint_id}"
            )
        observed_updates.append(update)
        candidates.append(candidate)
        report_by_id[candidate.checkpoint_id] = (report_path, payload)

    if tuple(sorted(observed_updates)) != tuple(candidate_updates):
        raise RuntimeError(
            f"candidate update set mismatch for {method_id} seed {seed}: {sorted(observed_updates)}"
        )

    selection = select_checkpoint(candidates, expected_seed=int(seed))
    selected_id = selection.selected_checkpoint_id
    selected_checkpoint = checkpoint_by_stem[selected_id]
    selected_report, selected_payload = report_by_id[selected_id]
    return {
        "method_id": method_id,
        "training_seed": int(seed),
        "selection": asdict(selection),
        "selected_checkpoint_path": str(selected_checkpoint),
        "selected_checkpoint_sha256": _sha256(selected_checkpoint),
        "selected_validation_report_path": str(selected_report),
        "selected_validation_report_sha256": _sha256(selected_report),
        "selected_validation_report": selected_payload,
        "source_run_manifest_path": str(manifest_path),
        "source_run_manifest_sha256": _sha256(manifest_path),
        "source_git_commit": str(manifest["git_commit"]),
        "source_protocol_id": expected_protocol_id,
        "candidate_updates": list(candidate_updates),
        "candidate_count": len(candidates),
        "final_test_station_results_present": False,
    }


def select_formal_checkpoints(
    *,
    project_root: str | Path,
    method_id: str,
    run_root: str | Path,
) -> dict[str, object]:
    config = _load_config(project_root)
    methods = {str(item["method_id"]): item for item in config["methods"]}
    if method_id not in methods:
        raise KeyError(f"unknown formal method: {method_id}")
    method = methods[method_id]
    seeds = tuple(int(x) for x in config["seeds"])
    candidate_updates = tuple(int(x) for x in config["selection"]["candidate_updates"])
    transitions_per_update = int(config["training"]["transitions_per_update"])
    root = Path(run_root).expanduser().resolve()
    selections = [
        select_seed_checkpoint(
            method_id=method_id,
            expected_protocol_id=str(method["protocol_id"]),
            seed=seed,
            seed_dir=root / f"seed_{seed}",
            candidate_updates=candidate_updates,
            transitions_per_update=transitions_per_update,
        )
        for seed in seeds
    ]
    return {
        "status": "passed",
        "orchestration_id": config["orchestration_id"],
        "evaluation_contract_id": config["selection"]["evaluation_contract_id"],
        "method_id": method_id,
        "seed_count": len(selections),
        "seed_is_not_selection_axis": True,
        "selected_checkpoints": selections,
        "final_test_station_results_present": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Select formal AWM checkpoints after training")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--method-id", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = select_formal_checkpoints(
        project_root=args.project_root,
        method_id=args.method_id,
        run_root=args.run_root,
    )
    destination = Path(args.output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()


__all__ = ["select_formal_checkpoints", "select_seed_checkpoint"]
