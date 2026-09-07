import pytest

from awm.envs.ircm_reporting import accepted_summary_ircm_values, snap_to_execution_grid


def accepted(value: float) -> tuple[float, ...]:
    return accepted_summary_ircm_values(
        value,
        execution_resolution_mm=0.1,
        reporting_resolution_mm=1.0,
    )


def test_binary_drift_is_restored_to_execution_grid_before_reporting():
    assert float(snap_to_execution_grid(538.49999999999994, resolution_mm=0.1)) == pytest.approx(538.5)
    assert accepted(538.49999999999994) == (538.0, 539.0)
    assert accepted(494.49999999999994) == (494.0, 495.0)
    assert accepted(431.49999999999994) == (431.0, 432.0)


def test_non_half_reporting_bins_remain_strict():
    assert accepted(539.4) == (539.0,)
    assert accepted(539.6) == (540.0,)
    assert accepted(539.7) == (540.0,)


def test_exact_half_accepts_both_observed_dssat_neighbors():
    assert accepted(74.5) == (74.0, 75.0)
    assert accepted(494.5) == (494.0, 495.0)
    assert accepted(539.5) == (539.0, 540.0)


def test_invalid_resolution_is_rejected():
    with pytest.raises(ValueError):
        accepted_summary_ircm_values(10.0, execution_resolution_mm=0.0)
