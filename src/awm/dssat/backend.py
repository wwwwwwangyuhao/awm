"""Concrete worker-local DSSAT backend used by the irrigation adapter."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .management import DSSATExperimentRenderer


class RunnerLike(Protocol):
    def run(self) -> tuple[int, str, str]: ...


class ReaderLike(Protocol):
    def invalidate(self) -> None: ...
    def refresh(self) -> None: ...
    def daily_state(self, yrdoy: str) -> dict: ...
    def season_summary(self) -> dict: ...


@dataclass(frozen=True, slots=True)
class DSSATWorkerPaths:
    workspace: str
    summary_out: str
    daily_out_files: tuple[str, ...]
    episode_artifacts: tuple[str, ...] = ()


class DSSATWorkerBackend:
    """Bind rendering, complete-season execution and cached DSSAT outputs.

    This class intentionally has no fertilizer write operation. Nitrogen is
    fixed by the externally validated COX template for the first AWM study.
    """

    def __init__(
        self,
        *,
        renderer: DSSATExperimentRenderer,
        runner: RunnerLike,
        reader: ReaderLike,
        paths: DSSATWorkerPaths,
    ) -> None:
        self.renderer = renderer
        self.runner = runner
        self.reader = reader
        self.paths = paths
        self._ready = False

    def _clean_episode_outputs(self) -> None:
        for item in self.paths.episode_artifacts:
            path = Path(item)
            if path.is_file() or path.is_symlink():
                path.unlink()

    def _run_and_refresh(self) -> None:
        status, stdout, stderr = self.runner.run()
        if status != 0:
            raise RuntimeError(
                f"DSSAT run failed: status={status}, reason={stderr or stdout}"
            )
        self.reader.invalidate()
        self.reader.refresh()
        self._ready = True

    def reset_episode(self) -> None:
        self._clean_episode_outputs()
        self.renderer.reset()
        self.reader.invalidate()
        self._run_and_refresh()

    def write_irrigation(self, action_yrdoy: str, amount_mm: float) -> None:
        self.renderer.add_irrigation(action_yrdoy, amount_mm)

    def rerun_and_refresh(self) -> None:
        self._run_and_refresh()

    def daily_state(self, yrdoy: str) -> dict:
        if not self._ready:
            raise RuntimeError("DSSAT worker has not been reset successfully")
        return self.reader.daily_state(yrdoy)

    def season_summary(self) -> dict:
        if not self._ready:
            raise RuntimeError("DSSAT worker has not been reset successfully")
        return self.reader.season_summary()


__all__ = ["DSSATWorkerBackend", "DSSATWorkerPaths"]
