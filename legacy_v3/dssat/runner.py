"""DSSAT process execution backend."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def log_message(verbose: int, message: str) -> None:
    if int(verbose or 0) > 0:
        print(message)


def ensure_directory_exists(directory: str | os.PathLike[str], verbose: int = 0) -> None:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    if verbose > 0:
        print(f"DSSAT directory ready: {path}")


class DSSATRunner:
    """Run one complete DSSAT season inside one isolated worker workspace."""

    def __init__(
        self,
        *,
        dssat_exec: str,
        output_dir: str,
        cox_path: str,
        weather_file: str | None = None,
        soil_file: str | None = None,
        verbose: int = 0,
        timeout_seconds: float = 1800.0,
    ) -> None:
        self.dssat_exec = str(dssat_exec)
        self.output_dir = str(output_dir)
        self.cox_path = str(cox_path)
        self.weather_file = str(weather_file) if weather_file else None
        self.soil_file = str(soil_file) if soil_file else None
        self.verbose = int(verbose or 0)
        self.timeout_seconds = float(timeout_seconds)
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

    def _copy_input(self, source: str, copied_files: list[Path]) -> Path:
        source_path = Path(source)
        if not source_path.is_file():
            raise FileNotFoundError(f"DSSAT input file not found: {source_path}")
        destination = Path(self.output_dir) / source_path.name
        shutil.copy2(source_path, destination)
        copied_files.append(destination)
        return destination

    def run(self) -> tuple[int, str, str]:
        """Execute DSSAT and return ``(returncode, stdout, stderr)``.

        DSSAT requires the output directory as the process working directory.
        ``subprocess.run(..., cwd=...)`` is used instead of process-global
        ``os.chdir`` so the runner has no hidden working-directory side effect.
        """
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        executable = Path(self.dssat_exec)
        if not executable.is_file():
            return -2, "", f"DSSAT executable not found: {executable}"

        copied_files: list[Path] = []
        try:
            cox = self._copy_input(self.cox_path, copied_files)
            if self.weather_file:
                self._copy_input(self.weather_file, copied_files)
            if self.soil_file:
                self._copy_input(self.soil_file, copied_files)

            command = [str(executable), "A", cox.name]
            log_message(self.verbose, f"DSSAT command: {' '.join(command)}")
            result = subprocess.run(
                command,
                cwd=str(output_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            return int(result.returncode), result.stdout, result.stderr
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout if isinstance(exc.stdout, str) else ""
            stderr = exc.stderr if isinstance(exc.stderr, str) else ""
            message = stderr or f"DSSAT timed out after {self.timeout_seconds}s"
            return -1, stdout, message
        except Exception as exc:
            return -2, "", f"DSSAT execution failed: {type(exc).__name__}: {exc}"
        finally:
            for path in copied_files:
                try:
                    if path.is_file():
                        path.unlink()
                except OSError as exc:
                    log_message(
                        self.verbose,
                        f"Failed to remove temporary DSSAT input {path}: {exc}",
                    )


__all__ = ["DSSATRunner", "ensure_directory_exists", "log_message"]
