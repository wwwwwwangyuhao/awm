"""Development-weather sweep for the three frozen agricultural baselines.

This module deliberately excludes the 2023--2025 station final-test years.
It materializes the immutable year-2000 agricultural protocol into ERA5
calendar years 2000--2022, runs every requested method/treatment sequentially
through the real DSSAT worker, and checkpoints one compact JSONL row per
completed episode.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from awm.dssat.formal_year import (
    DEVELOPMENT_YEARS,
    FormalYearMaterialization,
    materialize_formal_cox_year,
)

from .smoke import run_real_baseline


FORMAL_SOURCE_COX = "run/formal/awm_protocol_v1_2000.COX.in"
FORMAL_SOURCE_COX_SHA256 = (
    "7984ea2ae684e8eb8e97919c5821f01fc35da3ff917ca7e9e267e52b8b30c274"
)
METHODS = ("conventional", "et", "rew")
TREATMENTS = ("W100", "W80", "W60")
TRAIN_YEARS = tuple(range(2000, 2018))
VALIDATION_YEARS = tuple(range(2018, 2023))

_BASE_CONFIG_PATTERN = "configs/formal_baseline_{method}_{treatment}_2000.json"


@dataclass(frozen=True, slots=True)
class SweepEpisode:
    year: int
    split: str
    method: str
    treatment: str
    config_path: Path
    audit_path: Path
    generated_cox_path: Path
    generated_cox_sha256: str
    base_config_sha256: str

    @property
    def key(self) -> tuple[int, str, str]:
        return (self.year, self.method, self.treatment)


def development_split(year: int) -> str:
    if year in TRAIN_YEARS:
        return "train"
    if year in VALIDATION_YEARS:
        return "validation"
    raise ValueError("development sweep year must lie in 2000..2022")


def prepare_sweep_inputs(
    *,
    project_root: str | Path,
    years: Sequence[int] = DEVELOPMENT_YEARS,
    methods: Sequence[str] = METHODS,
    treatments: Sequence[str] = TREATMENTS,
    work_dir: str | Path | None = None,
) -> tuple[SweepEpisode, ...]:
    """Materialize year-specific COX/config inputs without executing DSSAT."""

    root = _validate_project_root(project_root)
    selected_years = _validate_years(years)
    selected_methods = _validate_methods(methods)
    selected_treatments = _validate_treatments(treatments)

    source_path = root / FORMAL_SOURCE_COX
    source_text = source_path.read_text(encoding="utf-8")
    source_sha = _sha256_text(source_text)
    if source_sha != FORMAL_SOURCE_COX_SHA256:
        raise RuntimeError(
            "frozen formal COX hash changed; development sweep is blocked until "
            "the protocol change is reviewed and FORMAL_SOURCE_COX_SHA256 is "
            f"updated intentionally: {source_sha}"
        )

    base = (
        Path(work_dir).expanduser().resolve()
        if work_dir is not None
        else (root / "runtime" / "development_baseline_sweep_v1").resolve()
    )
    try:
        base.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            "work_dir must stay inside the AWM project root because canonical "
            "DSSAT smoke configs require project-relative COX paths"
        ) from exc
    inputs_dir = base / "inputs"
    audits_dir = base / "audits"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    audits_dir.mkdir(parents=True, exist_ok=True)

    materialized_by_year: dict[int, tuple[FormalYearMaterialization, Path, str]] = {}
    episodes: list[SweepEpisode] = []

    for year in selected_years:
        materialized = materialize_formal_cox_year(source_text, target_year=year)
        year_dir = inputs_dir / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)
        cox_path = year_dir / f"awm_protocol_v1_{year}.COX.in"
        cox_path.write_text(materialized.text, encoding="utf-8")
        cox_sha = _sha256_text(materialized.text)
        materialized_by_year[year] = (materialized, cox_path, cox_sha)

        for method in selected_methods:
            for treatment in selected_treatments:
                base_config_path = root / _BASE_CONFIG_PATTERN.format(
                    method=method,
                    treatment=treatment.lower(),
                )
                raw = base_config_path.read_text(encoding="utf-8")
                base_config_sha = _sha256_text(raw)
                config = json.loads(raw)
                _retarget_config(
                    config,
                    year=year,
                    split=development_split(year),
                    materialized=materialized,
                    cox_path=cox_path,
                    project_root=root,
                )
                config_path = year_dir / f"{method}_{treatment.lower()}.json"
                config_path.write_text(
                    json.dumps(config, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                audit_path = audits_dir / f"{year}_{method}_{treatment.lower()}.json"
                episodes.append(
                    SweepEpisode(
                        year=year,
                        split=development_split(year),
                        method=method,
                        treatment=treatment,
                        config_path=config_path,
                        audit_path=audit_path,
                        generated_cox_path=cox_path,
                        generated_cox_sha256=cox_sha,
                        base_config_sha256=base_config_sha,
                    )
                )

    manifest = {
        "status": "prepared",
        "protocol": "awm-agricultural-baselines-v1",
        "formal_source_cox": FORMAL_SOURCE_COX,
        "formal_source_cox_sha256": FORMAL_SOURCE_COX_SHA256,
        "years": list(selected_years),
        "train_years": [y for y in selected_years if y in TRAIN_YEARS],
        "validation_years": [y for y in selected_years if y in VALIDATION_YEARS],
        "locked_final_test_years": [2023, 2024, 2025],
        "weather_source": "era5",
        "methods": list(selected_methods),
        "treatments": list(selected_treatments),
        "episode_count": len(episodes),
        "sequential_execution_required": True,
        "work_dir": str(base),
    }
    (base / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return tuple(episodes)


def run_development_sweep(
    *,
    project_root: str | Path,
    years: Sequence[int] = DEVELOPMENT_YEARS,
    methods: Sequence[str] = METHODS,
    treatments: Sequence[str] = TREATMENTS,
    work_dir: str | Path | None = None,
    resume: bool = False,
) -> dict[str, object]:
    """Execute selected development episodes and checkpoint compact results."""

    root = _validate_project_root(project_root)
    episodes = prepare_sweep_inputs(
        project_root=root,
        years=years,
        methods=methods,
        treatments=treatments,
        work_dir=work_dir,
    )
    base = episodes[0].config_path.parents[2] if episodes else (
        root / "runtime" / "development_baseline_sweep_v1"
    )
    jsonl_path = base / "results.jsonl"
    csv_path = base / "results.csv"

    existing_rows = _read_jsonl(jsonl_path) if resume else []
    if not resume and jsonl_path.exists():
        jsonl_path.unlink()
        existing_rows = []
    completed = {
        (int(row["year"]), str(row["method"]), str(row["water_treatment"]))
        for row in existing_rows
        if row.get("status") == "passed"
    }

    rows = list(existing_rows)
    for episode in episodes:
        if episode.key in completed:
            continue
        result = run_real_baseline(
            str(episode.config_path),
            audit_output=str(episode.audit_path),
            project_root=str(root),
        )
        row = _compact_result_row(episode, result)
        with jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        rows.append(row)
        _write_csv(csv_path, rows)

    _write_csv(csv_path, rows)
    return {
        "status": "passed",
        "episode_count": len(episodes),
        "completed_row_count": len(rows),
        "results_jsonl": str(jsonl_path),
        "results_csv": str(csv_path),
        "work_dir": str(base),
        "locked_final_test_years": [2023, 2024, 2025],
    }


def _retarget_config(
    config: dict[str, object],
    *,
    year: int,
    split: str,
    materialized: FormalYearMaterialization,
    cox_path: Path,
    project_root: Path,
) -> None:
    config["status"] = "formal_development_baseline_sweep"
    config["weather_year"] = year
    config["weather_split"] = split
    config["weather_source"] = "era5"
    config["weather_filename"] = materialized.weather_filename
    config["plant_yrdoy"] = materialized.plant_yrdoy
    config["rendered_cox_name"] = materialized.station_id + ".COX"
    config["cox_template"] = cox_path.relative_to(project_root).as_posix()
    runtime = config.setdefault("runtime", {})
    if not isinstance(runtime, dict):
        raise TypeError("baseline config runtime must be an object")
    runtime["policy_idx"] = 0
    runtime["env_idx"] = 0
    runtime["replace_worker"] = True


def _compact_result_row(
    episode: SweepEpisode,
    result: Mapping[str, object],
) -> dict[str, object]:
    events = list(result.get("executed_events", []))
    return {
        "status": result["status"],
        "year": episode.year,
        "split": episode.split,
        "method": episode.method,
        "baseline_name": result["baseline_name"],
        "water_treatment": episode.treatment,
        "weather_source": "era5",
        "weather_filename": f"XJHX{episode.year % 100:02d}01.WTH",
        "formal_source_cox_sha256": FORMAL_SOURCE_COX_SHA256,
        "generated_cox_sha256": episode.generated_cox_sha256,
        "base_config_sha256": episode.base_config_sha256,
        "total_seasonal_budget_mm": result["total_seasonal_budget_mm"],
        "policy_budget_mm": result["policy_budget_mm"],
        "fixed_nonpolicy_irrigation_mm": result["fixed_nonpolicy_irrigation_mm"],
        "HWAM_kg_ha": result["HWAM_kg_ha"],
        "IRCM_mm": result["IRCM_mm"],
        "policy_irrigation_mm": result["policy_irrigation_mm"],
        "IWP_kg_m3": result["IWP_kg_m3"],
        "baseline_desired_event_count": result["baseline_desired_event_count"],
        "requested_event_count": result["requested_event_count"],
        "baseline_constraint_adjusted_event_count": result[
            "baseline_constraint_adjusted_event_count"
        ],
        "adapter_projected_event_count": result["projected_event_count"],
        "irrigation_event_count": result["irrigation_event_count"],
        "event_daps": [int(event["action_dap"]) for event in events],
        "event_depths_mm": [float(event["applied_mm"]) for event in events],
        "irrigation_accounting_passed": result["irrigation_accounting_passed"],
        "audit_path": str(episode.audit_path),
    }


def _validate_project_root(project_root: str | Path) -> Path:
    root = Path(project_root).expanduser().resolve()
    if not (root / "pyproject.toml").is_file() or not (root / "src" / "awm").is_dir():
        raise FileNotFoundError(f"not an AWM project root: {root}")
    return root


def _validate_years(years: Sequence[int]) -> tuple[int, ...]:
    result = tuple(int(year) for year in years)
    if not result:
        raise ValueError("at least one development year is required")
    invalid = [year for year in result if year not in DEVELOPMENT_YEARS]
    if invalid:
        raise ValueError(
            "development sweep is locked to 2000..2022; final-test years "
            f"2023..2025 are forbidden here: {invalid}"
        )
    if len(set(result)) != len(result):
        raise ValueError("development years must not contain duplicates")
    return result


def _validate_methods(methods: Sequence[str]) -> tuple[str, ...]:
    result = tuple(str(method).strip().lower() for method in methods)
    if not result or any(method not in METHODS for method in result):
        raise ValueError(f"methods must be a non-empty subset of {METHODS}")
    if len(set(result)) != len(result):
        raise ValueError("methods must not contain duplicates")
    return result


def _validate_treatments(treatments: Sequence[str]) -> tuple[str, ...]:
    result = tuple(str(item).strip().upper() for item in treatments)
    if not result or any(item not in TREATMENTS for item in result):
        raise ValueError(f"treatments must be a non-empty subset of {TREATMENTS}")
    if len(set(result)) != len(result):
        raise ValueError("treatments must not contain duplicates")
    return result


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid results JSONL at line {line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"results JSONL line {line_number} is not an object")
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serializable = {
                key: (
                    json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                    if isinstance(value, (list, dict))
                    else value
                )
                for key, value in row.items()
            }
            writer.writerow(serializable)


def _parse_csv_tokens(text: str) -> tuple[str, ...]:
    return tuple(token.strip() for token in text.split(",") if token.strip())


def _parse_years(text: str) -> tuple[int, ...]:
    stripped = text.strip()
    if "-" in stripped and "," not in stripped:
        start_text, end_text = stripped.split("-", 1)
        start, end = int(start_text), int(end_text)
        if end < start:
            raise ValueError("year range end must be >= start")
        return tuple(range(start, end + 1))
    return tuple(int(token) for token in _parse_csv_tokens(stripped))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run frozen agricultural baselines on ERA5 development years only "
            "(2000-2022); 2023-2025 final-test years are intentionally blocked"
        )
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--years", default="2000-2022")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--treatments", default=",".join(TREATMENTS))
    parser.add_argument("--work-dir")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="materialize and validate inputs but do not execute DSSAT",
    )
    args = parser.parse_args()

    years = _parse_years(args.years)
    methods = _parse_csv_tokens(args.methods)
    treatments = _parse_csv_tokens(args.treatments)
    if args.prepare_only:
        episodes = prepare_sweep_inputs(
            project_root=args.project_root,
            years=years,
            methods=methods,
            treatments=treatments,
            work_dir=args.work_dir,
        )
        payload = {
            "status": "prepared",
            "episode_count": len(episodes),
            "years": sorted({episode.year for episode in episodes}),
            "locked_final_test_years": [2023, 2024, 2025],
            "work_dir": str(episodes[0].config_path.parents[2]) if episodes else None,
        }
    else:
        payload = run_development_sweep(
            project_root=args.project_root,
            years=years,
            methods=methods,
            treatments=treatments,
            work_dir=args.work_dir,
            resume=args.resume,
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "DEVELOPMENT_YEARS",
    "FORMAL_SOURCE_COX",
    "FORMAL_SOURCE_COX_SHA256",
    "METHODS",
    "SweepEpisode",
    "TRAIN_YEARS",
    "TREATMENTS",
    "VALIDATION_YEARS",
    "development_split",
    "prepare_sweep_inputs",
    "run_development_sweep",
]
