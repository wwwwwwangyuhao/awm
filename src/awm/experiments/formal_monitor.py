"""Filesystem/process monitor for the three formal learned-method runs.

This module is read-only with respect to training outputs.  It never mutates
checkpoints, validation reports, manifests, tmux sessions, or trainer state.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import subprocess
from typing import Iterable, Mapping


_CONFIG_PATH = "configs/formal_experiment_orchestration_v1.json"
_UPDATE_RE = re.compile(r"update(?P<update>\d{4})")
_PROTOCOL_KEYS = ("ppo_protocol_id", "rcwa_protocol_id", "protocol_id")


@dataclass(frozen=True, slots=True)
class SeedRunStatus:
    method_id: str
    seed: int
    run_dir: str
    manifest_present: bool
    manifest_git_commit: str | None
    manifest_protocol_id: str | None
    manifest_protocol_matches: bool
    updates_completed: int
    latest_update_index: int
    recovery_checkpoint_count: int
    candidate_checkpoint_count: int
    validation_report_count: int
    candidate_updates: tuple[int, ...]
    validation_updates: tuple[int, ...]
    trainer_process_alive: bool
    training_complete: bool
    selection_ready: bool
    status: str


def _load_config(project_root: str | Path) -> dict[str, object]:
    root = Path(project_root).expanduser().resolve()
    return json.loads((root / _CONFIG_PATH).read_text(encoding="utf-8"))


def _load_updates(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    rows: list[dict[str, object]] = []
    for line_no, text in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not text.strip():
            continue
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise TypeError(f"{path}:{line_no} must contain a JSON object")
        rows.append(payload)
    observed = [int(row["update_index"]) for row in rows]
    expected = list(range(1, len(rows) + 1))
    if observed != expected:
        raise RuntimeError(f"non-contiguous update history in {path}: {observed[:8]} ...")
    return rows


def _extract_update(path: Path) -> int:
    match = _UPDATE_RE.search(path.stem)
    if not match:
        raise ValueError(f"cannot infer update index from {path.name}")
    return int(match.group("update"))


def _manifest_protocol_id(manifest: Mapping[str, object]) -> str | None:
    found = [str(manifest[key]) for key in _PROTOCOL_KEYS if key in manifest]
    if len(found) > 1 and len(set(found)) != 1:
        raise ValueError(f"ambiguous protocol ids in run manifest: {found}")
    return found[0] if found else None


def inspect_seed_run(
    *,
    method_id: str,
    expected_protocol_id: str,
    seed: int,
    run_root: str | Path,
    max_updates: int,
    candidate_updates: Iterable[int],
    process_snapshot: str,
) -> SeedRunStatus:
    root = Path(run_root).expanduser().resolve()
    run_dir = root / f"seed_{int(seed)}"
    manifest_path = run_dir / "run_manifest.json"
    manifest: dict[str, object] | None = None
    if manifest_path.is_file():
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise TypeError(f"run manifest must be an object: {manifest_path}")
        manifest = raw
        if int(manifest.get("training_seed", -1)) != int(seed):
            raise ValueError(f"manifest seed mismatch for {method_id} seed {seed}")

    updates = _load_updates(run_dir / "train_updates.jsonl")
    latest_update = len(updates)
    recovery = sorted((run_dir / "recovery_checkpoints").glob("*.pt"))
    candidates = sorted((run_dir / "candidate_checkpoints").glob("*.pt"))
    reports = sorted((run_dir / "validation").glob("*.json"))
    candidate_set = tuple(sorted(_extract_update(path) for path in candidates))
    validation_set = tuple(sorted(_extract_update(path) for path in reports))
    expected_candidates = tuple(int(x) for x in candidate_updates)
    process_alive = str(run_dir) in process_snapshot
    training_complete = latest_update == int(max_updates)
    selection_ready = (
        training_complete
        and candidate_set == expected_candidates
        and validation_set == expected_candidates
    )
    if latest_update > int(max_updates):
        raise RuntimeError(f"{method_id} seed {seed} exceeded max updates")

    if selection_ready:
        status = "selection_ready"
    elif training_complete:
        status = "artifact_incomplete"
    elif process_alive:
        status = "running"
    elif latest_update == 0 and manifest is None:
        status = "not_started"
    else:
        status = "stopped_incomplete"

    actual_protocol = _manifest_protocol_id(manifest) if manifest is not None else None
    return SeedRunStatus(
        method_id=str(method_id),
        seed=int(seed),
        run_dir=str(run_dir),
        manifest_present=manifest is not None,
        manifest_git_commit=(str(manifest.get("git_commit")) if manifest else None),
        manifest_protocol_id=actual_protocol,
        manifest_protocol_matches=(actual_protocol == expected_protocol_id),
        updates_completed=latest_update,
        latest_update_index=latest_update,
        recovery_checkpoint_count=len(recovery),
        candidate_checkpoint_count=len(candidates),
        validation_report_count=len(reports),
        candidate_updates=candidate_set,
        validation_updates=validation_set,
        trainer_process_alive=process_alive,
        training_complete=training_complete,
        selection_ready=selection_ready,
        status=status,
    )


def inspect_formal_runs(
    *,
    project_root: str | Path,
    run_roots: Mapping[str, str | Path],
    process_snapshot: str = "",
) -> dict[str, object]:
    config = _load_config(project_root)
    seeds = tuple(int(x) for x in config["seeds"])
    training = config["training"]
    selection = config["selection"]
    methods = config["methods"]
    statuses: list[SeedRunStatus] = []
    for method in methods:
        method_id = str(method["method_id"])
        if method_id not in run_roots:
            raise KeyError(f"missing run root for method {method_id}")
        for seed in seeds:
            statuses.append(
                inspect_seed_run(
                    method_id=method_id,
                    expected_protocol_id=str(method["protocol_id"]),
                    seed=seed,
                    run_root=run_roots[method_id],
                    max_updates=int(training["max_updates_per_seed"]),
                    candidate_updates=selection["candidate_updates"],
                    process_snapshot=process_snapshot,
                )
            )
    protocol_ok = all(item.manifest_protocol_matches for item in statuses if item.manifest_present)
    return {
        "status": "passed" if protocol_ok else "failed_protocol_provenance",
        "orchestration_id": config["orchestration_id"],
        "seed_run_count": len(statuses),
        "running_count": sum(item.status == "running" for item in statuses),
        "selection_ready_count": sum(item.selection_ready for item in statuses),
        "all_selection_ready": all(item.selection_ready for item in statuses),
        "final_test_station_results_present": False,
        "runs": [asdict(item) for item in statuses],
    }


def _parse_run_root(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--run-root must use METHOD=/absolute/path")
        method, path = value.split("=", 1)
        method = method.strip()
        if not method or method in result:
            raise ValueError(f"invalid or duplicate method in --run-root: {value}")
        result[method] = path
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor formal AWM learned-method training")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--run-root", action="append", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    process_snapshot = subprocess.run(
        ["ps", "-eo", "args="],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    report = inspect_formal_runs(
        project_root=args.project_root,
        run_roots=_parse_run_root(args.run_root),
        process_snapshot=process_snapshot,
    )
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()


__all__ = ["SeedRunStatus", "inspect_formal_runs", "inspect_seed_run"]
