#!/usr/bin/env python3
"""One-time, read-only migration of the validated LRMB DSSAT assets into AWM.

This script NEVER checks out, resets, writes, or commits anything in the source
LRMB repository. It reads one fixed Git commit with ``git archive`` and writes
only inside the current AWM repository.

After the generated ``dssat_workspace_template/`` is committed to AWM, future
AWM clones no longer need LRMB to build DSSAT workers.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Iterable

SOURCE_REPOSITORY = "wwwwwwangyuhao/lrmb"
SOURCE_COMMIT = "d56336e09fdb9a9aea60ae61eaa892833314ab33"
SOURCE_SUBDIR = "dssat_workspace_template"

CUSTOM_DSSAT_BASE_VERSION = "4.8.5"
CUSTOM_DSSAT_BUILD_LABEL = "lab-dssat-4.8.5-mulch"
EXPECTED_DSSAT_EXEC_GIT_BLOB_SHA1 = "37c88710d0518d4e20a02881e884652973f559c3"
EXPECTED_DSSAT_EXEC_SIZE = 14_603_784

ERA5_WEATHER_FILENAMES = tuple(f"XJHX{year:02d}01.WTH" for year in range(26))
STATION_WEATHER_FILENAMES = (
    "XJHX2301.WTH",
    "XJHX2401.WTH",
    "XJHX2501.WTH",
)
CANONICAL_GENOTYPE_FILES = (
    "COGRO048.CUL",
    "COGRO048.ECO",
    "COGRO048.SPE",
)
STANDARD_DATA_FILES = (
    "CO2048.WDA",
    "FERCH048.SDA",
)
CDE_FILES = (
    "DATA.CDE",
    "DETAIL.CDE",
    "ECONOMIC.CDE",
    "GCOEFF.CDE",
    "GRSTAGE.CDE",
    "JDATE.CDE",
    "OUTPUT.CDE",
    "PEST.CDE",
    "SIMULATION.CDE",
    "SOIL.CDE",
    "WEATHER.CDE",
)
ROOT_RUNTIME_FILES = (
    "DATA.CDE",
    "DETAIL.CDE",
    "SIMULATION.CDE",
)


def _run(command: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _repo_root() -> Path:
    proc = _run(["git", "rev-parse", "--show-toplevel"], cwd=Path.cwd())
    if proc.returncode != 0:
        raise RuntimeError(
            "Run this script from inside the AWM Git repository. "
            + proc.stderr.decode("utf-8", errors="replace")
        )
    return Path(proc.stdout.decode().strip()).resolve()


def _require_commit(source_repo: Path) -> None:
    if not (source_repo / ".git").exists():
        raise FileNotFoundError(f"Source Git repository not found: {source_repo}")
    proc = _run(
        ["git", "-C", str(source_repo), "cat-file", "-e", f"{SOURCE_COMMIT}^{{commit}}"]
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "The required LRMB source commit is not available locally. "
            f"Required commit: {SOURCE_COMMIT}. "
            "Fetch it in the LRMB repository without checking it out, then rerun. "
            + proc.stderr.decode("utf-8", errors="replace")
        )


def _safe_extract_tar(payload: bytes, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
        for member in archive.getmembers():
            target = (root / member.name).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"Unsafe archive member: {member.name}")
        archive.extractall(root)
    return root / SOURCE_SUBDIR


def _archive_source(source_repo: Path, destination: Path) -> Path:
    _require_commit(source_repo)
    proc = _run(
        [
            "git",
            "-C",
            str(source_repo),
            "archive",
            "--format=tar",
            SOURCE_COMMIT,
            SOURCE_SUBDIR,
        ]
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "git archive failed: " + proc.stderr.decode("utf-8", errors="replace")
        )
    return _safe_extract_tar(proc.stdout, destination)


def _git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_files(root: Path, relative_paths: Iterable[str]) -> None:
    missing = [relative for relative in relative_paths if not (root / relative).is_file()]
    if missing:
        raise FileNotFoundError(
            "Source DSSAT template is incomplete; missing: " + ", ".join(missing)
        )


def _validate_source(source: Path) -> None:
    required = list(ROOT_RUNTIME_FILES)
    required += [f"Genotype/{name}" for name in CANONICAL_GENOTYPE_FILES]
    required += [f"StandardData/{name}" for name in STANDARD_DATA_FILES]
    required += [f"data/CDE/{name}" for name in CDE_FILES]
    required += ["data/soil/SOIL.SOL", "DSSATPRO.L48", "dscsm048"]
    required += [f"data/wth/era5/{name}" for name in ERA5_WEATHER_FILENAMES]
    required += [f"data/wth/station/{name}" for name in STATION_WEATHER_FILENAMES]
    _require_files(source, required)

    executable = source / "dscsm048"
    if executable.stat().st_size != EXPECTED_DSSAT_EXEC_SIZE:
        raise RuntimeError(
            "Custom DSSAT executable size mismatch: "
            f"expected={EXPECTED_DSSAT_EXEC_SIZE}, actual={executable.stat().st_size}"
        )
    blob_sha = _git_blob_sha1(executable)
    if blob_sha != EXPECTED_DSSAT_EXEC_GIT_BLOB_SHA1:
        raise RuntimeError(
            "Custom DSSAT executable Git blob mismatch: "
            f"expected={EXPECTED_DSSAT_EXEC_GIT_BLOB_SHA1}, actual={blob_sha}"
        )

    era5 = tuple(sorted(path.name for path in (source / "data/wth/era5").glob("*.WTH")))
    station = tuple(
        sorted(path.name for path in (source / "data/wth/station").glob("*.WTH"))
    )
    if era5 != tuple(sorted(ERA5_WEATHER_FILENAMES)):
        raise RuntimeError(f"Unexpected ERA5 weather inventory: {era5}")
    if station != tuple(sorted(STATION_WEATHER_FILENAMES)):
        raise RuntimeError(f"Unexpected station weather inventory: {station}")


def _copy_file(source_root: Path, destination_root: Path, relative: str) -> None:
    src = source_root / relative
    dst = destination_root / relative
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_canonical_assets(source: Path, destination: Path) -> None:
    for name in (*ROOT_RUNTIME_FILES, "dscsm048"):
        _copy_file(source, destination, name)

    for name in CANONICAL_GENOTYPE_FILES:
        _copy_file(source, destination, f"Genotype/{name}")
    for name in STANDARD_DATA_FILES:
        _copy_file(source, destination, f"StandardData/{name}")
    for name in CDE_FILES:
        _copy_file(source, destination, f"data/CDE/{name}")
    _copy_file(source, destination, "data/soil/SOIL.SOL")

    for name in ERA5_WEATHER_FILENAMES:
        _copy_file(source, destination, f"data/wth/era5/{name}")
    for name in STATION_WEATHER_FILENAMES:
        _copy_file(source, destination, f"data/wth/station/{name}")

    provenance = destination / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "DSSATPRO.L48", provenance / "source_DSSATPRO.L48")

    executable = destination / "dscsm048"
    executable.chmod(executable.stat().st_mode | 0o111)


def _manifest(destination: Path) -> dict:
    files = {}
    for path in sorted(p for p in destination.rglob("*") if p.is_file()):
        relative = path.relative_to(destination).as_posix()
        if relative in {"ASSET_MANIFEST.json", "README.md"}:
            continue
        files[relative] = {
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "git_blob_sha1": _git_blob_sha1(path),
        }

    return {
        "schema_version": 1,
        "status": "versioned_awm_runtime_template",
        "simulator": {
            "base_version": CUSTOM_DSSAT_BASE_VERSION,
            "build_label": CUSTOM_DSSAT_BUILD_LABEL,
            "executable": "dscsm048",
            "description": (
                "Laboratory-custom DSSAT 4.8.5 executable with mulch "
                "functionality used by the experiment COX configuration."
            ),
            "source_patch_and_compiler_provenance": "TBD",
            "expected_source_git_blob_sha1": EXPECTED_DSSAT_EXEC_GIT_BLOB_SHA1,
            "expected_size_bytes": EXPECTED_DSSAT_EXEC_SIZE,
        },
        "migration_source": {
            "repository": SOURCE_REPOSITORY,
            "commit": SOURCE_COMMIT,
            "path": SOURCE_SUBDIR,
            "method": "read-only git archive; no source checkout or writes",
        },
        "inventory": {
            "era5_weather_years": list(range(2000, 2026)),
            "station_weather_years": [2023, 2024, 2025],
            "excluded_noncanonical_source_items": [
                "Genotype/COGRO048_别人的参数.CUL",
                "data/wth/legacy_import/",
                "DSSATPRO_完整版.L48",
            ],
            "note": (
                "Excluded files are historical/reference items and are not "
                "required by the canonical AWM cotton runtime."
            ),
        },
        "files": files,
    }


def _readme() -> str:
    return f"""# AWM versioned DSSAT workspace template

This directory is part of the AWM repository and is the immutable source for
mutable DSSAT workers.

## Simulator identity

- DSSAT base version: {CUSTOM_DSSAT_BASE_VERSION}
- build label: `{CUSTOM_DSSAT_BUILD_LABEL}`
- executable: `dscsm048`
- source executable Git blob: `{EXPECTED_DSSAT_EXEC_GIT_BLOB_SHA1}`
- source executable size: `{EXPECTED_DSSAT_EXEC_SIZE}` bytes

The executable is the laboratory-custom DSSAT 4.8.5 build used with the
experiment's mulch-enabled COX configuration. It must not be described as an
unmodified official DSSAT 4.8.5 binary.

`ASSET_MANIFEST.json` records SHA256 and Git-blob hashes for every migrated
asset.

## Runtime rule

Never run episodes directly in this versioned directory. Build a mutable worker
with `awm.dssat.runtime_assets.prepare_worker_from_template(...)`; mutable
workers belong under ignored `runtime/` or another explicitly chosen short
runtime root.

The historical source `DSSATPRO.L48` is retained only under
`provenance/source_DSSATPRO.L48`. AWM renders a new worker-local profile so no
runtime path depends on LRMB.
"""


def _atomic_install(staging: Path, destination: Path, replace: bool) -> None:
    if destination.exists() and not replace:
        raise FileExistsError(
            f"Destination already exists: {destination}. "
            "Use --replace only after reviewing the existing AWM template."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        shutil.rmtree(destination)
    os.replace(staging, destination)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Migrate the fixed LRMB V3 DSSAT runtime assets into AWM."
    )
    parser.add_argument(
        "--source-repo",
        required=True,
        help="Local LRMB Git repository used read-only as the one-time source.",
    )
    parser.add_argument(
        "--destination",
        default="dssat_workspace_template",
        help="Destination inside the current AWM repository.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace an existing AWM template after validation succeeds.",
    )
    args = parser.parse_args()

    awm_root = _repo_root()
    source_repo = Path(args.source_repo).resolve()
    destination = Path(args.destination)
    if not destination.is_absolute():
        destination = (awm_root / destination).resolve()
    else:
        destination = destination.resolve()

    if destination == awm_root or awm_root not in destination.parents:
        raise ValueError(
            "Destination must be a subdirectory of the current AWM repository"
        )
    if source_repo == awm_root:
        raise ValueError("Source repository and AWM repository must be different")

    with tempfile.TemporaryDirectory(prefix="awm_dssat_import_") as temp_name:
        temp_root = Path(temp_name)
        source = _archive_source(source_repo, temp_root / "export")
        _validate_source(source)

        staging = awm_root / f".{destination.name}.staging.{os.getpid()}"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        try:
            _copy_canonical_assets(source, staging)
            manifest = _manifest(staging)
            (staging / "ASSET_MANIFEST.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            (staging / "README.md").write_text(_readme(), encoding="utf-8")
            _atomic_install(staging, destination, args.replace)
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    print("AWM DSSAT asset migration completed.")
    print(f"  source repository: {SOURCE_REPOSITORY}")
    print(f"  source commit:     {SOURCE_COMMIT}")
    print(f"  destination:       {destination}")
    print("  LRMB modifications: none (git archive read-only)")
    print()
    print("Next:")
    print(f"  git status --short -- {destination.relative_to(awm_root)}")
    print("  PYTHONPATH=\"$PWD/src\" python -m pytest")
    print(f"  git add {destination.relative_to(awm_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
