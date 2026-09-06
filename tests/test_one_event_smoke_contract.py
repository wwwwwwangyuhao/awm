from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from awm.baselines.one_event_smoke import validate_one_event_acceptance
from awm.dssat.smoke_runtime import CANONICAL_DAILY_OUT_NAMES


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "engineering_one_event_smoke_v3_2000.json"


def _config() -> dict[str, object]:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _step(day: int) -> dict[str, object]:
    return {
        "policy_day": day,
        "decision_action_day": day + 1,
        "decision_action_yrdoy": f"00{120 + day:03d}",
        "requested_irrigation_event": False,
        "requested_irrigation_amount_fraction": 0.0,
        "canonical_irrigation_amount_fraction": 0.0,
        "applied_irrigation_mm": 0.0,
        "remaining_irrigation_before_mm": 10.0,
        "remaining_irrigation_after_mm": 10.0,
        "irrigation_event_applied": False,
        "irrigation_action_projected": False,
        "irrigation_execution_quantized": False,
        "irrigation_projection_reasons": [],
        "dssat_management_written": False,
        "dssat_rerun": False,
        "baseline_name": "local_conventional",
        "baseline_desired_mm": 0.0,
        "baseline_reason": f"conventional_dap_{day + 1}",
    }


def _accepted_fixture(workspace: Path) -> tuple[dict[str, object], dict[str, object]]:
    workspace.mkdir(parents=True, exist_ok=True)
    for name in CANONICAL_DAILY_OUT_NAMES:
        (workspace / name).write_text("data\n", encoding="utf-8")
    (workspace / "XJHX0001.COX").write_text(
        "@I IDATE IROP IRVAL\n 1 00129 IR005 10.00\n",
        encoding="utf-8",
    )

    steps = [_step(day) for day in range(125)]
    event = steps[9]
    event.update(
        {
            "decision_action_day": 10,
            "decision_action_yrdoy": "00129",
            "requested_irrigation_event": True,
            "applied_irrigation_mm": 10.0,
            "remaining_irrigation_after_mm": 0.0,
            "irrigation_event_applied": True,
            "dssat_management_written": True,
            "dssat_rerun": True,
            "baseline_desired_mm": 10.0,
        }
    )

    result = {
        "workspace": str(workspace),
        "runtime_id": "0123456789",
        "HWAM_kg_ha": 1000.0,
        "IRCM_mm": 10.0,
        "policy_irrigation_mm": 10.0,
        "IWP_kg_m3": 10.0,
        "irrigation_event_count": 1,
        "requested_event_count": 1,
        "projected_event_count": 0,
        "irrigation_accounting_passed": True,
    }
    audit = {"step_audits": steps}
    return result, audit


def test_one_event_config_is_engineering_only_and_exactly_10_mm_on_dap10():
    config = _config()
    assert config["status"] == "engineering_one_event_smoke_only_not_formal_protocol"
    assert config["seasonal_budget_mm"] == 10.0
    assert config["min_event_mm"] == 10.0
    assert config["max_event_mm"] == 10.0
    assert config["nonpolicy_irrigation_mm"] == 0.0
    assert config["horizon_days"] == 125
    assert config["baseline"]["schedule_mm_by_action_dap"] == {"10": 10.0}
    assert config["engineering_expectations"]["policy_day"] == 9
    assert config["engineering_expectations"]["action_day"] == 10
    assert config["engineering_expectations"]["action_yrdoy"] == "00129"


def test_acceptance_passes_only_for_single_exact_event():
    with tempfile.TemporaryDirectory(prefix="awmoe_") as name:
        result, audit = _accepted_fixture(Path(name))
        report = validate_one_event_acceptance(
            config=_config(),
            result=result,
            audit_payload=audit,
        )
        assert report["status"] == "passed"
        assert report["step_count"] == 125
        assert report["single_event_policy_day"] == 9
        assert report["single_event_action_day"] == 10
        assert report["single_event_action_yrdoy"] == "00129"
        assert report["single_event_applied_mm"] == 10.0
        assert report["IRCM_mm"] == 10.0
        assert report["dssat_management_write_count"] == 1
        assert report["irrigation_triggered_rerun_count"] == 1
        assert report["daily_output_file_count"] == 13


def test_acceptance_rejects_extra_rerun_on_noop_step():
    with tempfile.TemporaryDirectory(prefix="awmoe_") as name:
        result, audit = _accepted_fixture(Path(name))
        audit["step_audits"][20]["dssat_rerun"] = True
        with pytest.raises(AssertionError, match="exactly one irrigation-triggered DSSAT reruns"):
            validate_one_event_acceptance(
                config=_config(),
                result=result,
                audit_payload=audit,
            )


def test_acceptance_rejects_wrong_ircm_even_if_ledger_is_10_mm():
    with tempfile.TemporaryDirectory(prefix="awmoe_") as name:
        result, audit = _accepted_fixture(Path(name))
        result["IRCM_mm"] = 20.0
        with pytest.raises(AssertionError, match="DSSAT IRCM mismatch"):
            validate_one_event_acceptance(
                config=_config(),
                result=result,
                audit_payload=audit,
            )


def test_acceptance_rejects_wrong_action_date():
    with tempfile.TemporaryDirectory(prefix="awmoe_") as name:
        result, audit = _accepted_fixture(Path(name))
        audit["step_audits"][9]["decision_action_yrdoy"] = "00130"
        with pytest.raises(AssertionError, match="action_yrdoy mismatch"):
            validate_one_event_acceptance(
                config=_config(),
                result=result,
                audit_payload=audit,
            )
