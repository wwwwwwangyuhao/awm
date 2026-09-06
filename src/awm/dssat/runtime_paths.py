"""Length-controlled, project-namespaced DSSAT runtime paths for AWM.

AWM keeps immutable DSSAT assets inside the Git checkout but runs mutable DSSAT
workers below a short user-level runtime root.  The default root deliberately
lives under ``~/.dssat_rt/awm`` so AWM can coexist with LRMB without sharing
registry files, locks, archives, or worker directories.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import socket
import sys
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .workspace import managed_workspace_name

RUNTIME_NAMESPACE_HEX_LENGTH = 10
DEFAULT_RUNTIME_PARENT = Path.home() / ".dssat_rt"
DEFAULT_AWM_RUNTIME_BASE = DEFAULT_RUNTIME_PARENT / "awm"
RUNTIME_BASE_ENV = "AWM_DSSAT_RUNTIME_BASE"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _runtime_base(runtime_base: str | Path | None = None) -> Path:
    if runtime_base is not None:
        return Path(runtime_base).expanduser().resolve()
    configured = os.environ.get(RUNTIME_BASE_ENV)
    if configured:
        return Path(configured).expanduser().resolve()
    return DEFAULT_AWM_RUNTIME_BASE.expanduser().resolve()


def runtime_namespace_for_project(project_root: str | Path) -> str:
    """Return a stable 10-hex namespace derived from the resolved checkout path."""
    project = str(Path(project_root).expanduser().resolve())
    return hashlib.sha256(project.encode("utf-8")).hexdigest()[
        :RUNTIME_NAMESPACE_HEX_LENGTH
    ]


def runtime_root_for_project(
    project_root: str | Path,
    *,
    runtime_base: str | Path | None = None,
) -> Path:
    return _runtime_base(runtime_base) / runtime_namespace_for_project(project_root)


def worker_root_for_project(
    project_root: str | Path,
    *,
    runtime_base: str | Path | None = None,
) -> Path:
    return runtime_root_for_project(project_root, runtime_base=runtime_base) / "w"


def evaluation_root_for_project(
    project_root: str | Path,
    *,
    runtime_base: str | Path | None = None,
) -> Path:
    return runtime_root_for_project(project_root, runtime_base=runtime_base) / "e"


def archive_root_for_project(
    project_root: str | Path,
    *,
    runtime_base: str | Path | None = None,
) -> Path:
    return runtime_root_for_project(project_root, runtime_base=runtime_base) / "archives"


def worker_workspace_for_project(
    project_root: str | Path,
    *,
    policy_idx: int,
    env_idx: int,
    runtime_base: str | Path | None = None,
) -> Path:
    return worker_root_for_project(project_root, runtime_base=runtime_base) / managed_workspace_name(
        policy_idx,
        env_idx,
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{os.urandom(4).hex()}")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def register_dssat_runtime(
    *,
    project_root: str | Path,
    runtime_base: str | Path | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Register one AWM checkout under its short deterministic runtime namespace."""
    project = Path(project_root).expanduser().resolve()
    base = _runtime_base(runtime_base)
    namespace = runtime_namespace_for_project(project)
    runtime_root = base / namespace
    registry_path = base / "registry.json"
    lock_path = base / ".registry.lock"

    base.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            now = _utc_now()
            registry = _read_json_object(registry_path)
            registry.setdefault("schema_version", 1)
            registry.setdefault("runtime_family", "awm")
            projects = registry.setdefault("projects", {})
            if not isinstance(projects, dict):
                projects = {}
                registry["projects"] = projects

            previous = projects.get(namespace, {})
            if previous and str(previous.get("project_root", "")) not in {
                "",
                str(project),
            }:
                raise RuntimeError(
                    "AWM DSSAT runtime namespace collision: "
                    f"{namespace} is already registered to "
                    f"{previous.get('project_root')!r}."
                )

            record: dict[str, Any] = dict(previous) if isinstance(previous, dict) else {}
            record.update(
                {
                    "runtime_family": "awm",
                    "runtime_id": namespace,
                    "project_root": str(project),
                    "project_name": project.name,
                    "runtime_base": str(base),
                    "runtime_root": str(runtime_root),
                    "worker_root": str(runtime_root / "w"),
                    "evaluation_root": str(runtime_root / "e"),
                    "archive_root": str(runtime_root / "archives"),
                    "first_seen_utc": previous.get("first_seen_utc", now),
                    "last_seen_utc": now,
                    "hash_basis": "resolved_project_root",
                    "hash_algorithm": "sha256",
                    "hash_hex_length": RUNTIME_NAMESPACE_HEX_LENGTH,
                }
            )
            if metadata:
                record.update(dict(metadata))

            projects[namespace] = record
            registry["last_updated_utc"] = now
            _write_json_atomic(registry_path, registry)
            runtime_root.mkdir(parents=True, exist_ok=True)
            _write_json_atomic(runtime_root / "project.json", record)
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    return record


def ensure_runtime_roots(
    *,
    project_root: str | Path,
    runtime_base: str | Path | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    record = register_dssat_runtime(
        project_root=project_root,
        runtime_base=runtime_base,
        metadata=metadata,
    )
    for key in ("worker_root", "evaluation_root", "archive_root"):
        Path(str(record[key])).mkdir(parents=True, exist_ok=True)
    return record


class WorkspaceRootLock(AbstractContextManager["WorkspaceRootLock"]):
    """Hold an exclusive lock for one AWM checkout's generated DSSAT runtime."""

    def __init__(
        self,
        *,
        project_root: str | Path,
        runtime_base: str | Path | None = None,
    ) -> None:
        self.project_root = Path(project_root).expanduser().resolve()
        self.runtime_root = runtime_root_for_project(
            self.project_root,
            runtime_base=runtime_base,
        )
        self.lock_path = self.runtime_root / ".workspace.lock"
        self._handle = None

    def acquire(self) -> "WorkspaceRootLock":
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        handle = self.lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.seek(0)
            owner = handle.read().strip() or "<owner metadata unavailable>"
            handle.close()
            raise RuntimeError(
                "Another AWM process is already using DSSAT runtime_root="
                f"{self.runtime_root}. Lock owner: {owner}"
            ) from exc

        payload = {
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "started_at_utc": _utc_now(),
            "project_root": str(self.project_root),
            "runtime_root": str(self.runtime_root),
            "command": sys.argv,
        }
        handle.seek(0)
        handle.truncate()
        json.dump(payload, handle, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
        self._handle = handle
        return self

    def release(self) -> None:
        if self._handle is None:
            return
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "WorkspaceRootLock":
        return self.acquire()

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.release()
        return False


__all__ = [
    "DEFAULT_AWM_RUNTIME_BASE",
    "DEFAULT_RUNTIME_PARENT",
    "RUNTIME_BASE_ENV",
    "RUNTIME_NAMESPACE_HEX_LENGTH",
    "WorkspaceRootLock",
    "archive_root_for_project",
    "ensure_runtime_roots",
    "evaluation_root_for_project",
    "register_dssat_runtime",
    "runtime_namespace_for_project",
    "runtime_root_for_project",
    "worker_root_for_project",
    "worker_workspace_for_project",
]
