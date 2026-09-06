"""Immutable provenance manifest for formal RCWA-RL runs."""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
import subprocess
from typing import Any

import torch

from awm.dssat.runtime_assets import CUSTOM_DSSAT_BUILD_LABEL, CUSTOM_DSSAT_EXECUTABLE


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    if len(value) != 40:
        raise RuntimeError(f"unexpected git HEAD value: {value!r}")
    return value


def build_run_manifest(
    *,
    project_root: str | Path,
    protocol: dict[str, Any],
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    protocol_path = root / "configs" / "rcwa_rl_v1.json"
    executable = root / "dssat_workspace_template" / CUSTOM_DSSAT_EXECUTABLE
    if not protocol_path.is_file():
        raise FileNotFoundError(protocol_path)
    if not executable.is_file():
        raise FileNotFoundError(executable)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but torch.cuda is unavailable")
        device_index = 0 if device.index is None else int(device.index)
        device_name = torch.cuda.get_device_name(device_index)
    else:
        device_index = None
        device_name = f"cpu:{platform.machine()}"
    return {
        "manifest_id": "awm-rcwa-run-manifest-v1",
        "rcwa_protocol_id": str(protocol["rcwa_protocol_id"]),
        "git_commit": _git_head(root),
        "training_seed": int(seed),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "cuda_runtime_version": torch.version.cuda,
        "device_type": device.type,
        "device_index": device_index,
        "device_name": device_name,
        "rcwa_protocol_sha256": _sha256_file(protocol_path),
        "dssat_build_label": CUSTOM_DSSAT_BUILD_LABEL,
        "dssat_executable": CUSTOM_DSSAT_EXECUTABLE,
        "dssat_executable_sha256": _sha256_file(executable),
    }


def ensure_run_manifest(path: str | Path, expected: dict[str, Any]) -> Path:
    destination = Path(path)
    if destination.exists():
        current = json.loads(destination.read_text(encoding="utf-8"))
        if current != expected:
            keys = sorted(set(current) | set(expected))
            mismatches = {
                key: {"existing": current.get(key), "expected": expected.get(key)}
                for key in keys
                if current.get(key) != expected.get(key)
            }
            raise RuntimeError(
                "RCWA run manifest mismatch; refusing to mix provenance in one output directory: "
                + json.dumps(mismatches, sort_keys=True)
            )
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


__all__ = ["build_run_manifest", "ensure_run_manifest"]
