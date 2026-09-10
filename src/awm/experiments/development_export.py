"""Export selected-checkpoint development validation results without re-selection."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable, Mapping


_LOCKED_FINAL_TEST_YEARS = {2023, 2024, 2025}
_VALIDATION_YEARS = {2018, 2019, 2020, 2021, 2022}


def _sample_sd(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def _load_document(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"selection document must be an object: {path}")
    if payload.get("final_test_station_results_present") is not False:
        raise RuntimeError(f"selection document is not development-only: {path}")
    return payload


def build_selected_validation_tables(
    selection_documents: Iterable[Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    cell_rows: list[dict[str, object]] = []
    seed_rows: list[dict[str, object]] = []
    methods_seen: set[str] = set()
    for document in selection_documents:
        method_id = str(document["method_id"])
        if method_id in methods_seen:
            raise ValueError(f"duplicate selection document for method {method_id}")
        methods_seen.add(method_id)
        selected = document.get("selected_checkpoints")
        if not isinstance(selected, list) or len(selected) != 5:
            raise ValueError(f"method {method_id} must contain exactly 5 selected seeds")
        for item in selected:
            seed = int(item["training_seed"])
            selection = item["selection"]
            report = item["selected_validation_report"]
            if report.get("final_test_station_results_present") is not False:
                raise RuntimeError("selected validation report contains final-test contamination flag")
            if report.get("validation_action_mode") != "deterministic":
                raise ValueError("selected validation report must use deterministic actions")
            checkpoint_id = str(selection["selected_checkpoint_id"])
            if checkpoint_id != str(report["checkpoint_id"]):
                raise ValueError("selection/report checkpoint_id mismatch")
            eta_metrics = report.get("eta_metrics")
            cells = report.get("cell_results")
            if not isinstance(eta_metrics, list) or len(eta_metrics) != 3:
                raise ValueError("selected validation report must contain 3 eta metrics")
            if not isinstance(cells, list) or len(cells) != 15:
                raise ValueError("selected validation report must contain 15 cells")

            for metric in eta_metrics:
                years = {int(year) for year in metric["validation_years"]}
                if years != _VALIDATION_YEARS:
                    raise ValueError(f"validation years must be exactly {_VALIDATION_YEARS}")
                seed_rows.append(
                    {
                        "method_id": method_id,
                        "training_seed": seed,
                        "checkpoint_id": checkpoint_id,
                        "checkpoint_sha256": item["selected_checkpoint_sha256"],
                        "training_step": int(selection["selected_training_step"]),
                        "selection_status": selection["selection_status"],
                        "jointly_feasible": bool(selection["jointly_feasible"]),
                        "eta": float(metric["eta"]),
                        "lcvar_retention": float(metric["lcvar_retention"]),
                        "risk_margin": float(metric["lcvar_retention"]) - float(metric["eta"]),
                        "mean_total_irrigation_mm": float(metric["mean_total_irrigation_mm"]),
                        "minimum_retention": float(metric["minimum_retention"]),
                    }
                )

            for cell in cells:
                year = int(cell["weather_year"])
                if year in _LOCKED_FINAL_TEST_YEARS:
                    raise RuntimeError("final-test station year found in development export")
                if year not in _VALIDATION_YEARS:
                    raise ValueError(f"unexpected development validation year {year}")
                cell_rows.append(
                    {
                        "method_id": method_id,
                        "training_seed": seed,
                        "checkpoint_id": checkpoint_id,
                        "checkpoint_sha256": item["selected_checkpoint_sha256"],
                        "training_step": int(selection["selected_training_step"]),
                        "selection_status": selection["selection_status"],
                        "jointly_feasible": bool(selection["jointly_feasible"]),
                        "eta": float(cell["eta"]),
                        "weather_year": year,
                        "yield_kg_ha": float(cell["yield_kg_ha"]),
                        "reference_yield_kg_ha": float(cell["reference_yield_kg_ha"]),
                        "yield_retention": float(cell["yield_retention"]),
                        "total_irrigation_mm": float(cell["total_irrigation_mm"]),
                        "policy_irrigation_mm": float(cell["policy_irrigation_mm"]),
                        "sampled_event_count": int(cell["sampled_event_count"]),
                        "executed_event_count": int(cell["executed_event_count"]),
                    }
                )

    expected_cells = len(methods_seen) * 5 * 3 * 5
    expected_seed_rows = len(methods_seen) * 5 * 3
    if len(cell_rows) != expected_cells or len(seed_rows) != expected_seed_rows:
        raise RuntimeError("selected development export is incomplete")

    groups: dict[tuple[str, float], list[dict[str, object]]] = {}
    for row in seed_rows:
        groups.setdefault((str(row["method_id"]), float(row["eta"])), []).append(row)
    summary_rows: list[dict[str, object]] = []
    for (method_id, eta), rows in sorted(groups.items()):
        if len(rows) != 5:
            raise RuntimeError(f"expected 5 seed metrics for {method_id} eta={eta}")
        lcvars = [float(row["lcvar_retention"]) for row in rows]
        irrigation = [float(row["mean_total_irrigation_mm"]) for row in rows]
        margins = [float(row["risk_margin"]) for row in rows]
        summary_rows.append(
            {
                "method_id": method_id,
                "eta": eta,
                "seed_count": 5,
                "mean_lcvar_retention": mean(lcvars),
                "sample_sd_lcvar_retention": _sample_sd(lcvars),
                "mean_total_irrigation_mm": mean(irrigation),
                "sample_sd_total_irrigation_mm": _sample_sd(irrigation),
                "mean_risk_margin": mean(margins),
                "sample_sd_risk_margin": _sample_sd(margins),
                "feasible_seed_count": sum(float(row["risk_margin"]) >= -1e-12 for row in rows),
            }
        )

    summary = {
        "status": "passed",
        "scope": "development_validation_only",
        "methods": sorted(methods_seen),
        "validation_years": sorted(_VALIDATION_YEARS),
        "locked_final_test_years": sorted(_LOCKED_FINAL_TEST_YEARS),
        "final_test_station_results_present": False,
        "cell_row_count": len(cell_rows),
        "seed_metric_row_count": len(seed_rows),
        "summary_rows": summary_rows,
    }
    return cell_rows, seed_rows, summary


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def export_selected_validation(
    *,
    selection_paths: Iterable[str | Path],
    output_dir: str | Path,
) -> dict[str, object]:
    documents = [_load_document(path) for path in selection_paths]
    cells, seeds, summary = build_selected_validation_tables(documents)
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    _write_csv(destination / "selected_validation_cells.csv", cells)
    _write_csv(destination / "selected_seed_eta_metrics.csv", seeds)
    (destination / "selected_validation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Export selected development validation results")
    parser.add_argument("--selection", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    summary = export_selected_validation(
        selection_paths=args.selection,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


__all__ = ["build_selected_validation_tables", "export_selected_validation"]
