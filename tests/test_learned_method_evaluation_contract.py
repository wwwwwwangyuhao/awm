from __future__ import annotations

import json
from pathlib import Path

from awm.evaluation import REQUIRED_VALIDATION_CELL_COUNT, VALIDATION_YEARS
from awm.risk import REGISTERED_ETA_LEVELS


ROOT = Path(__file__).resolve().parents[1]


def test_machine_readable_evaluation_contract_matches_code() -> None:
    contract = json.loads(
        (ROOT / "configs" / "learned_method_evaluation_contract_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert contract["evaluation_contract_id"] == "awm-learned-method-evaluation-v1"
    assert tuple(contract["candidate_set"]["validation_years"]) == VALIDATION_YEARS
    assert tuple(contract["candidate_set"]["eta_levels"]) == REGISTERED_ETA_LEVELS
    assert contract["candidate_set"]["required_validation_cells_per_checkpoint"] == REQUIRED_VALIDATION_CELL_COUNT == 15
    assert contract["selection_scope"]["single_checkpoint_per_seed"] is True
    assert contract["selection_scope"]["single_checkpoint_shared_across_eta_levels"] is True
    assert contract["selection_scope"]["seed_is_not_a_selection_axis"] is True
    assert contract["candidate_set"]["validation_gradient_updates_forbidden"] is True
    assert contract["candidate_set"]["locked_final_test_station_years"] == [2023, 2024, 2025]


def test_selection_order_is_feasibility_then_water_and_fallback_is_not_success() -> None:
    contract = json.loads(
        (ROOT / "configs" / "learned_method_evaluation_contract_v1.json").read_text(
            encoding="utf-8"
        )
    )
    feasible_order = contract["selection_rule"]["feasible_checkpoint_order"]
    assert feasible_order[0].startswith("minimum mean validation total irrigation")
    fallback = contract["selection_rule"]["no_jointly_feasible_fallback_order"]
    assert fallback[0].startswith("minimum worst positive risk shortfall")
    assert contract["selection_rule"]["fallback_checkpoint_status"] == "selected_infeasible"
