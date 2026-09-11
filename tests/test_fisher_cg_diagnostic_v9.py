from __future__ import annotations

import torch

from awm.rcwa.fisher_cg_diagnostic_v9 import _cg_trace


def test_cg_trace_matches_spd_solution_and_true_residual() -> None:
    matrix=torch.tensor([[4.0,1.0],[1.0,3.0]],dtype=torch.float64)
    b=torch.tensor([1.0,2.0],dtype=torch.float64)
    def hvp(v:torch.Tensor)->torch.Tensor:
        return matrix @ v
    snaps,reason,last=_cg_trace(hvp,b,marks=(1,2))
    assert reason == "max_iterations"
    assert last == 2
    assert snaps[2]["true_relative_residual"] < 1e-12
    assert snaps[2]["recursive_relative_residual"] < 1e-12
    assert abs(snaps[2]["quadratic_ratio_Fd_over_gd"]-1.0) < 1e-12
