"""Offline damping path from saved RCWA-v9 Fisher spectra.

Diagnostic only: no DSSAT interaction and no policy update.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


DEFAULT_TOLS = (1e-3, 1e-4, 1e-5, 1e-6)


def cg_condition_limit(*, iterations: int, tolerance: float) -> float:
    if iterations <= 0 or not 0.0 < tolerance < 2.0:
        raise ValueError("invalid CG numerical requirement")
    q = (tolerance / 2.0) ** (1.0 / iterations)
    return ((1.0 + q) / (1.0 - q)) ** 2


def minimum_damping(lmin: float, lmax: float, kappa_max: float) -> float:
    if not 0.0 < lmin <= lmax or kappa_max <= 1.0:
        raise ValueError("invalid spectral inputs")
    return max(0.0, (lmax - kappa_max * lmin) / (kappa_max - 1.0))


def _mode_summary(values: np.ndarray, xi: float) -> dict[str, float | int]:
    filt = values / (values + xi)
    return {
        "modes": int(values.size),
        "lambda_below_xi": int(np.sum(values < xi)),
        "filter_below_0p1": int(np.sum(filt < 0.1)),
        "filter_below_0p5": int(np.sum(filt < 0.5)),
        "filter_median": float(np.median(filt)),
        "filter_q25": float(np.quantile(filt, 0.25)),
        "filter_q75": float(np.quantile(filt, 0.75)),
    }


def build_summary(spectrum_root: Path, iterations: int = 50) -> dict:
    nodes: dict[str, dict] = {}
    spectra: dict[str, tuple[float, float, np.ndarray]] = {}
    for update in (2, 3, 5):
        path = spectrum_root / f"u{update}" / "fisher_spectrum_diagnostic.json"
        payload = json.loads(path.read_text())
        if not payload["replay_identity"]["passed"]:
            raise RuntimeError(f"u{update} replay identity failed")
        spec = payload["random_start_lanczos"]
        vals = np.asarray(spec["ritz_values"], dtype=float)
        spectra[f"u{update}"] = (
            float(spec["min_positive_ritz"]),
            float(spec["ritz_max"]),
            vals,
        )
    rows = []
    for tol in DEFAULT_TOLS:
        kappa = cg_condition_limit(iterations=iterations, tolerance=tol)
        required = {
            node: minimum_damping(lmin, lmax, kappa)
            for node, (lmin, lmax, _) in spectra.items()
        }
        global_xi = max(required.values())
        per_node = {}
        for node, (lmin, lmax, vals) in spectra.items():
            conditioned = (lmax + global_xi) / (lmin + global_xi)
            per_node[node] = {
                "lambda_min_positive": lmin,
                "lambda_max": lmax,
                "condition_undamped": lmax / lmin,
                "condition_damped": conditioned,
                "minimum_node_damping": required[node],
                "mode_filter": _mode_summary(vals, global_xi),
            }
        rows.append({
            "tolerance": tol,
            "condition_limit": kappa,
            "global_minimum_damping": global_xi,
            "nodes": per_node,
        })
    return {
        "diagnostic_id": "awm-rcwa-v9-fisher-regularization-path-v1",
        "cg_iterations": iterations,
        "criterion": "standard SPD CG worst-case energy-norm bound",
        "rows": rows,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--spectrum-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--iterations", type=int, default=50)
    args = p.parse_args()
    summary = build_summary(args.spectrum_root.resolve(), args.iterations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
