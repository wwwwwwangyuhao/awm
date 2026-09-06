from __future__ import annotations

import pytest

from awm.evaluation import (
    CheckpointValidation,
    EtaValidationMetrics,
    VALIDATION_YEARS,
    build_candidate_from_report,
    select_checkpoint,
    select_checkpoints_by_seed,
)


def metrics(lcvars, irrigations=(300.0, 340.0, 380.0)):
    return tuple(
        EtaValidationMetrics(
            eta=eta,
            validation_years=VALIDATION_YEARS,
            lcvar_retention=lcvar,
            mean_total_irrigation_mm=irrigation,
            minimum_retention=lcvar,
        )
        for eta, lcvar, irrigation in zip((0.90, 0.95, 0.98), lcvars, irrigations)
    )


def candidate(cid, step, lcvars, irrigations=(300.0, 340.0, 380.0), seed=21):
    return CheckpointValidation(
        checkpoint_id=cid,
        training_seed=seed,
        training_step=step,
        eta_metrics=metrics(lcvars, irrigations),
    )


def test_feasible_checkpoint_minimizes_irrigation_first():
    a = candidate("a", 100, (0.91, 0.96, 0.99), (320, 360, 400))
    b = candidate("b", 200, (0.905, 0.955, 0.985), (300, 340, 380))
    result = select_checkpoint([a, b])
    assert result.selected_checkpoint_id == "b"
    assert result.selection_status == "selected_feasible"
    assert result.jointly_feasible is True


def test_feasible_tie_prefers_larger_minimum_margin_then_earlier_step():
    a = candidate("a", 200, (0.91, 0.96, 0.99))
    b = candidate("b", 300, (0.92, 0.96, 0.99))
    assert select_checkpoint([a, b]).selected_checkpoint_id == "b"

    c = candidate("c", 100, (0.92, 0.96, 0.99))
    assert select_checkpoint([b, c]).selected_checkpoint_id == "c"


def test_no_feasible_checkpoint_minimizes_worst_shortfall():
    a = candidate("a", 100, (0.89, 0.95, 0.98))
    b = candidate("b", 200, (0.90, 0.94, 0.97))
    result = select_checkpoint([a, b])
    assert result.selected_checkpoint_id == "a"
    assert result.selection_status == "selected_infeasible"
    assert result.jointly_feasible is False


def test_fallback_uses_mean_shortfall_then_irrigation():
    a = candidate("a", 100, (0.89, 0.95, 0.98), (300, 300, 300))
    b = candidate("b", 200, (0.90, 0.94, 0.98), (200, 200, 200))
    assert select_checkpoint([a, b]).selected_checkpoint_id == "a"

    c = candidate("c", 300, (0.89, 0.95, 0.98), (280, 280, 280))
    assert select_checkpoint([a, c]).selected_checkpoint_id == "c"


def test_selection_rejects_mixed_seeds_and_duplicate_ids():
    with pytest.raises(ValueError, match="one training seed"):
        select_checkpoint([
            candidate("a", 100, (0.91, 0.96, 0.99), seed=1),
            candidate("b", 200, (0.91, 0.96, 0.99), seed=2),
        ])

    with pytest.raises(ValueError, match="unique"):
        select_checkpoint([
            candidate("same", 100, (0.91, 0.96, 0.99)),
            candidate("same", 200, (0.91, 0.96, 0.99)),
        ])


def test_candidate_requires_all_registered_eta_levels():
    bad = CheckpointValidation(
        checkpoint_id="x",
        training_seed=1,
        training_step=1,
        eta_metrics=metrics((0.91, 0.96, 0.99))[:2],
    )
    with pytest.raises(ValueError, match="registered levels"):
        select_checkpoint([bad])


def test_candidate_rejects_station_years():
    m = list(metrics((0.91, 0.96, 0.99)))
    m[0] = EtaValidationMetrics(
        eta=0.90,
        validation_years=(2018, 2019, 2020, 2021, 2023),
        lcvar_retention=0.91,
        mean_total_irrigation_mm=300,
        minimum_retention=0.91,
    )
    bad = CheckpointValidation("x", 1, 1, tuple(m))
    with pytest.raises(ValueError, match="final-test"):
        select_checkpoint([bad])


def test_report_parser_and_multi_seed_selection():
    def report(seed, cid, step, lcvars):
        return {
            "checkpoint_id": cid,
            "training_seed": seed,
            "training_step": step,
            "eta_metrics": [
                {
                    "eta": eta,
                    "validation_years": list(VALIDATION_YEARS),
                    "lcvar_retention": lcvar,
                    "mean_total_irrigation_mm": 300 + i * 40,
                    "minimum_retention": lcvar,
                }
                for i, (eta, lcvar) in enumerate(zip((0.90, 0.95, 0.98), lcvars))
            ],
        }

    parsed = build_candidate_from_report(report(1, "a", 100, (0.91, 0.96, 0.99)))
    assert parsed.training_seed == 1

    selected = select_checkpoints_by_seed([
        report(1, "a", 100, (0.91, 0.96, 0.99)),
        report(1, "b", 200, (0.905, 0.955, 0.985)),
        report(2, "c", 100, (0.91, 0.96, 0.99)),
    ])
    assert set(selected) == {1, 2}
    assert selected[1].selected_checkpoint_id == "a"
    assert selected[2].selected_checkpoint_id == "c"
