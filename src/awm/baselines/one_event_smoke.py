"""Strict engineering acceptance for one real 10-mm DSSAT irrigation event."""
from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path
from typing import Any, Mapping

from awm.dssat.smoke_runtime import CANONICAL_DAILY_OUT_NAMES

from .smoke import run_real_baseline

_ENGINEERING_STATUS = "engineering_one_event_smoke_only_not_formal_protocol"


def _close(actual: float, expected: float, tolerance: float, *, label: str) -> None:
    if not math.isfinite(float(actual)):
        raise AssertionError(f"{label} is not finite: {actual!r}")
    if abs(float(actual) - float(expected)) > float(tolerance):
        raise AssertionError(
            f"{label} mismatch: actual={actual}, expected={expected}, tolerance={tolerance}"
        )


def validate_one_event_acceptance(
    *,
    config: Mapping[str, Any],
    result: Mapping[str, Any],
    audit_payload: Mapping[str, Any],
) -> dict[str, object]:
    """Enforce the exact one-event timing, execution and accounting contract."""
    if config.get("status") != _ENGINEERING_STATUS:
        raise ValueError(
            "one-event smoke requires engineering-only config status; formal protocols "
            "must not reuse this fixture"
        )
    expectations = config.get("engineering_expectations")
    if not isinstance(expectations, Mapping):
        raise TypeError("engineering_expectations must be a JSON object")

    expected_step_count = int(expectations["step_count"])
    expected_policy_day = int(expectations["policy_day"])
    expected_action_day = int(expectations["action_day"])
    expected_action_yrdoy = str(expectations["action_yrdoy"])
    expected_mm = float(expectations["applied_irrigation_mm"])
    expected_ircm = float(expectations["dssat_ircm_mm"])
    tolerance = float(expectations["tolerance_mm"])

    steps = audit_payload.get("step_audits")
    if not isinstance(steps, list):
        raise TypeError("audit payload missing step_audits array")
    if len(steps) != expected_step_count:
        raise AssertionError(
            f"step count mismatch: actual={len(steps)}, expected={expected_step_count}"
        )

    requested = [step for step in steps if bool(step.get("requested_irrigation_event"))]
    applied = [step for step in steps if bool(step.get("irrigation_event_applied"))]
    written = [step for step in steps if bool(step.get("dssat_management_written"))]
    rerun = [step for step in steps if bool(step.get("dssat_rerun"))]

    for label, values in (
        ("requested irrigation events", requested),
        ("applied irrigation events", applied),
        ("DSSAT management writes", written),
        ("irrigation-triggered DSSAT reruns", rerun),
    ):
        if len(values) != 1:
            raise AssertionError(f"expected exactly one {label}, got {len(values)}")

    event = applied[0]
    if requested[0] is not event or written[0] is not event or rerun[0] is not event:
        raise AssertionError(
            "requested/applied/written/rerun records must all be the same single step"
        )

    if int(event["policy_day"]) != expected_policy_day:
        raise AssertionError(
            f"policy_day mismatch: {event['policy_day']} != {expected_policy_day}"
        )
    if int(event["decision_action_day"]) != expected_action_day:
        raise AssertionError(
            f"action_day mismatch: {event['decision_action_day']} != {expected_action_day}"
        )
    if str(event["decision_action_yrdoy"]) != expected_action_yrdoy:
        raise AssertionError(
            "action_yrdoy mismatch: "
            f"{event['decision_action_yrdoy']} != {expected_action_yrdoy}"
        )
    _close(
        float(event["applied_irrigation_mm"]),
        expected_mm,
        tolerance,
        label="executed irrigation",
    )

    _close(
        float(result["policy_irrigation_mm"]),
        expected_mm,
        tolerance,
        label="policy irrigation ledger",
    )
    _close(
        float(result["IRCM_mm"]),
        expected_ircm,
        tolerance,
        label="DSSAT IRCM",
    )
    if int(result["irrigation_event_count"]) != 1:
        raise AssertionError("irrigation_event_count must equal 1")
    if int(result["requested_event_count"]) != 1:
        raise AssertionError("requested_event_count must equal 1")
    if not bool(result["irrigation_accounting_passed"]):
        raise AssertionError("terminal irrigation accounting did not pass")

    workspace = Path(str(result["workspace"])).resolve()
    rendered_name = str(config.get("rendered_cox_name", "AWM_SMOKE.COX"))
    rendered_cox = workspace / rendered_name
    if not rendered_cox.is_file():
        raise AssertionError(f"rendered COX missing: {rendered_cox}")
    cox_text = rendered_cox.read_text(encoding="utf-8")
    irrigation_rows = [
        line
        for line in cox_text.splitlines()
        if " IR005 " in f" {line.strip()} "
    ]
    if len(irrigation_rows) != 1:
        raise AssertionError(
            f"expected exactly one IR005 row in final COX, got {len(irrigation_rows)}"
        )
    irrigation_row = irrigation_rows[0]
    if expected_action_yrdoy not in irrigation_row:
        raise AssertionError(
            f"IR005 row does not contain expected YYDDD {expected_action_yrdoy}: {irrigation_row}"
        )
    if f"{expected_mm:.2f}" not in irrigation_row:
        raise AssertionError(
            f"IR005 row does not contain expected depth {expected_mm:.2f}: {irrigation_row}"
        )

    output_sizes: dict[str, int] = {}
    for name in CANONICAL_DAILY_OUT_NAMES:
        path = workspace / name
        if not path.is_file() or path.stat().st_size <= 0:
            raise AssertionError(f"canonical DSSAT output missing/empty after event: {path}")
        output_sizes[name] = int(path.stat().st_size)

    no_op_steps = [step for step in steps if step is not event]
    if any(bool(step.get("dssat_management_written")) for step in no_op_steps):
        raise AssertionError("a no-op step unexpectedly wrote DSSAT management")
    if any(bool(step.get("dssat_rerun")) for step in no_op_steps):
        raise AssertionError("a no-op step unexpectedly reran DSSAT")

    return {
        "status": "passed",
        "engineering_only": True,
        "step_count": len(steps),
        "single_event_policy_day": expected_policy_day,
        "single_event_action_day": expected_action_day,
        "single_event_action_yrdoy": expected_action_yrdoy,
        "single_event_applied_mm": float(event["applied_irrigation_mm"]),
        "IRCM_mm": float(result["IRCM_mm"]),
        "policy_irrigation_mm": float(result["policy_irrigation_mm"]),
        "irrigation_accounting_passed": True,
        "dssat_management_write_count": len(written),
        "irrigation_triggered_rerun_count": len(rerun),
        "final_ir005_row": irrigation_row,
        "daily_output_file_count": len(output_sizes),
        "daily_output_sizes_bytes": output_sizes,
        "workspace": str(workspace),
        "runtime_id": result.get("runtime_id"),
        "HWAM_kg_ha": result.get("HWAM_kg_ha"),
        "IWP_kg_m3": result.get("IWP_kg_m3"),
    }


def run_one_event_smoke(
    config_path: str,
    *,
    audit_output: str | None = None,
    project_root: str | None = None,
    runtime_base: str | None = None,
) -> dict[str, object]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))

    if audit_output is not None:
        destination = Path(audit_output).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        result = run_real_baseline(
            config_path,
            audit_output=str(destination),
            project_root=project_root,
            runtime_base=runtime_base,
        )
        audit_payload = json.loads(destination.read_text(encoding="utf-8"))
        acceptance = validate_one_event_acceptance(
            config=config,
            result=result,
            audit_payload=audit_payload,
        )
        acceptance["audit_output"] = str(destination)
        return acceptance

    with tempfile.TemporaryDirectory(prefix="awm_one_event_") as temporary:
        destination = Path(temporary) / "audit.json"
        result = run_real_baseline(
            config_path,
            audit_output=str(destination),
            project_root=project_root,
            runtime_base=runtime_base,
        )
        audit_payload = json.loads(destination.read_text(encoding="utf-8"))
        return validate_one_event_acceptance(
            config=config,
            result=result,
            audit_payload=audit_payload,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the strict one-event AWM engineering irrigation smoke"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--audit-output")
    parser.add_argument("--project-root")
    parser.add_argument("--runtime-base")
    args = parser.parse_args()

    print(
        json.dumps(
            run_one_event_smoke(
                args.config,
                audit_output=args.audit_output,
                project_root=args.project_root,
                runtime_base=args.runtime_base,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()


__all__ = ["run_one_event_smoke", "validate_one_event_acceptance"]
