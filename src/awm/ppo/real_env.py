"""Materialize formal ERA5 PPO cells into managed real-DSSAT environments."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from awm.dssat.formal_year import materialize_formal_cox_year
from awm.dssat.runtime_paths import WorkspaceRootLock
from awm.risk import TRAIN_YEARS, VALIDATION_YEARS

from .real_smoke import _build_env
from .scheduler import WeatherEtaCell


FORMAL_SOURCE_COX = "run/formal/awm_protocol_v1_2000.COX.in"
FORMAL_SOURCE_COX_SHA256 = (
    "7984ea2ae684e8eb8e97919c5821f01fc35da3ff917ca7e9e267e52b8b30c274"
)
BASE_PPO_CONFIG = "configs/formal_ppo_smoke_2000.json"


@dataclass(frozen=True, slots=True)
class MaterializedPPOCell:
    cell: WeatherEtaCell
    split: str
    config_path: Path
    cox_path: Path
    cox_sha256: str


class ManagedRealPPOEnv:
    """CottonWaterEnv wrapper that owns one WorkspaceRootLock lifetime."""

    def __init__(self, *, env: Any, worker: Any, lock: WorkspaceRootLock) -> None:
        self._env = env
        self.worker = worker
        self._lock = lock
        self._closed = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def close(self) -> None:
        if not self._closed:
            self._lock.__exit__(None, None, None)
            self._closed = True


class PPORealEnvFactory:
    """Sequential real-DSSAT factory for development training/validation cells."""

    def __init__(
        self,
        *,
        project_root: str | Path,
        work_dir: str | Path | None = None,
        runtime_base: str | Path | None = None,
        env_idx: int = 0,
    ) -> None:
        self.root = Path(project_root).expanduser().resolve()
        if not (self.root / "pyproject.toml").is_file():
            raise FileNotFoundError(f"not an AWM project root: {self.root}")
        self.work_dir = (
            Path(work_dir).expanduser().resolve()
            if work_dir is not None
            else (self.root / "runtime" / "ppo_baseline_v1").resolve()
        )
        try:
            self.work_dir.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("PPO work_dir must remain inside the AWM project root") from exc
        self.runtime_base = (
            str(Path(runtime_base).expanduser().resolve())
            if runtime_base is not None
            else None
        )
        if env_idx < 0:
            raise ValueError("env_idx must be >= 0")
        self.env_idx = int(env_idx)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        source_text = (self.root / FORMAL_SOURCE_COX).read_text(encoding="utf-8")
        digest = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
        if digest != FORMAL_SOURCE_COX_SHA256:
            raise RuntimeError(
                "formal COX hash changed; PPO training is blocked until protocol review"
            )
        self._source_text = source_text
        self._base_config = json.loads(
            (self.root / BASE_PPO_CONFIG).read_text(encoding="utf-8")
        )

    @staticmethod
    def split_for_year(year: int) -> str:
        if year in TRAIN_YEARS:
            return "train"
        if year in VALIDATION_YEARS:
            return "validation"
        raise ValueError("PPO development factory permits only years 2000-2022")

    def materialize(self, cell: WeatherEtaCell) -> MaterializedPPOCell:
        split = self.split_for_year(int(cell.weather_year))
        materialized = materialize_formal_cox_year(
            self._source_text, target_year=int(cell.weather_year)
        )
        eta_token = f"{float(cell.eta):.2f}".replace(".", "p")
        cell_dir = self.work_dir / "inputs" / str(cell.weather_year) / f"eta_{eta_token}"
        cell_dir.mkdir(parents=True, exist_ok=True)
        cox_path = cell_dir / f"awm_protocol_v1_{cell.weather_year}.COX.in"
        cox_path.write_text(materialized.text, encoding="utf-8")
        cox_sha = hashlib.sha256(materialized.text.encode("utf-8")).hexdigest()

        config = json.loads(json.dumps(self._base_config))
        config.update(
            {
                "status": f"formal_ppo_{split}_cell",
                "weather_year": int(cell.weather_year),
                "weather_split": split,
                "eta": float(cell.eta),
                "weather_filename": materialized.weather_filename,
                "plant_yrdoy": materialized.plant_yrdoy,
                "rendered_cox_name": materialized.station_id + ".COX",
                "cox_template": cox_path.relative_to(self.root).as_posix(),
                "state_normalization": True,
            }
        )
        config["runtime"] = {
            "policy_idx": 0,
            "env_idx": self.env_idx,
            "replace_worker": True,
        }
        config_path = cell_dir / "ppo_env.json"
        config_path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return MaterializedPPOCell(
            cell=cell,
            split=split,
            config_path=config_path,
            cox_path=cox_path,
            cox_sha256=cox_sha,
        )

    def __call__(self, cell: WeatherEtaCell) -> ManagedRealPPOEnv:
        materialized = self.materialize(cell)
        config = json.loads(materialized.config_path.read_text(encoding="utf-8"))
        lock = WorkspaceRootLock(
            project_root=self.root,
            runtime_base=self.runtime_base,
        )
        lock.__enter__()
        try:
            worker, env = _build_env(
                str(materialized.config_path),
                config,
                root=self.root,
                runtime_base=self.runtime_base,
            )
        except Exception:
            lock.__exit__(None, None, None)
            raise
        return ManagedRealPPOEnv(env=env, worker=worker, lock=lock)


__all__ = [
    "BASE_PPO_CONFIG",
    "FORMAL_SOURCE_COX",
    "FORMAL_SOURCE_COX_SHA256",
    "ManagedRealPPOEnv",
    "MaterializedPPOCell",
    "PPORealEnvFactory",
]
