from __future__ import annotations

import torch

from awm.rcwa.fisher_spectrum_diagnostic_v9 import _lanczos, _symmetry_probe


def test_lanczos_recovers_diagonal_spd_spectrum() -> None:
    diag = torch.tensor([1.0, 2.0, 4.0, 8.0], dtype=torch.float64)
    hvp = lambda v: diag * v
    start = torch.ones(4, dtype=torch.float64)
    out = _lanczos(hvp, start, steps=4)
    vals = torch.tensor(out["ritz_values"], dtype=torch.float64)
    assert torch.allclose(vals, diag, atol=1e-10, rtol=1e-10)
    assert out["negative_ritz_count"] == 0
    assert abs(out["condition_estimate"] - 8.0) < 1e-10


def test_symmetry_probe_detects_symmetric_operator() -> None:
    diag = torch.tensor([1.0, 3.0, 5.0], dtype=torch.float32)
    out = _symmetry_probe(lambda v: diag * v, 3, torch.device("cpu"))
    assert out["relative_symmetry_error"] < 1e-6
    assert out["x_F_x"] > 0.0
    assert out["y_F_y"] > 0.0
