"""Minimal worker-workspace safety checks independent of any RL algorithm."""
from __future__ import annotations

from pathlib import Path

DSSATPRO_RECORD_WIDTH = 80


def managed_workspace_name(policy_idx: int, env_idx: int) -> str:
    policy = int(policy_idx)
    env = int(env_idx)
    if policy < 0 or env < 0:
        raise ValueError("policy_idx and env_idx must be non-negative")
    return f"p{policy}e{env}"


def validate_dssatpro_record_width(
    workspace: str | Path,
    max_width: int = DSSATPRO_RECORD_WIDTH,
) -> dict[str, object]:
    root = Path(workspace).resolve()
    path = root / "DSSATPRO.L48"
    if not path.is_file():
        raise FileNotFoundError(path)

    checked = 0
    max_observed = 0
    violations: list[str] = []
    for line_no, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        stripped = line.strip()
        if not stripped or stripped.startswith("***") or "//" not in line:
            continue
        checked += 1
        try:
            width = len(line.encode("ascii"))
        except UnicodeEncodeError:
            violations.append(f"line {line_no}: non-ASCII DSSATPRO record")
            continue
        max_observed = max(max_observed, width)
        if width > int(max_width):
            violations.append(f"line {line_no}: width={width} > {int(max_width)}")

    if violations:
        raise ValueError(
            "DSSATPRO.L48 fixed-width preflight failed: " + "; ".join(violations)
        )
    return {
        "workspace": str(root),
        "checked_record_count": checked,
        "max_observed_record_width": max_observed,
        "status": "passed",
    }


__all__ = [
    "DSSATPRO_RECORD_WIDTH",
    "managed_workspace_name",
    "validate_dssatpro_record_width",
]
