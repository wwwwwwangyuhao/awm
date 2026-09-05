from types import SimpleNamespace

import pytest

from awm.baselines import (
    ConventionalScheduleBaseline,
    PotentialETWaterBalanceBaseline,
    RootZoneREWThresholdBaseline,
)


class Obs:
    def __init__(self, values):
        self.feature_names = tuple(values)
        self._values = tuple(values.values())

    def flat(self):
        return self._values


class Controller:
    def __init__(self, min_event=5.0, feasible=30.0):
        self.spec = SimpleNamespace(min_event_mm=min_event)
        self.feasible = feasible

    def feasible_max_mm(self, *, day):
        del day
        return self.feasible


def test_conventional_schedule_uses_biological_action_day():
    p = ConventionalScheduleBaseline({1: 10.0, 3: 20.0})
    c = Controller()
    r0 = p.act(policy_day=0, observation=Obs({"x": 0.0}), controller=c)
    r1 = p.act(policy_day=1, observation=Obs({"x": 0.0}), controller=c)
    r2 = p.act(policy_day=2, observation=Obs({"x": 0.0}), controller=c)
    assert r0.irrigate and r0.desired_mm == 10.0
    assert not r1.irrigate
    assert r2.irrigate and r2.desired_mm == 20.0


def test_conventional_schedule_respects_controller_infeasible_day():
    p = ConventionalScheduleBaseline({1: 10.0})
    r = p.act(
        policy_day=0,
        observation=Obs({"x": 0.0}),
        controller=Controller(feasible=0.0),
    )
    assert not r.irrigate
    assert r.desired_mm == 10.0
    assert r.reason.endswith(":infeasible")


def test_et_baseline_is_causal_and_execution_feedback_reduces_deficit():
    p = PotentialETWaterBalanceBaseline(
        trigger_deficit_mm=5.0,
        irrigation_efficiency=0.8,
        effective_rain_fraction=1.0,
        refill_fraction=1.0,
    )
    c = Controller(min_event=5.0, feasible=30.0)
    r0 = p.act(
        policy_day=0,
        observation=Obs({"EOAA": 3.0, "PRED": 0.0}),
        controller=c,
    )
    assert not r0.irrigate
    r1 = p.act(
        policy_day=1,
        observation=Obs({"EOAA": 3.0, "PRED": 0.0}),
        controller=c,
    )
    assert r1.irrigate
    assert r1.desired_mm == pytest.approx(7.5)
    p.observe_execution(applied_mm=7.5)
    assert p.deficit_mm == pytest.approx(0.0)


def test_et_rain_offsets_observed_potential_et():
    p = PotentialETWaterBalanceBaseline(
        trigger_deficit_mm=4.0,
        irrigation_efficiency=1.0,
        effective_rain_fraction=0.5,
        refill_fraction=1.0,
    )
    r = p.act(
        policy_day=0,
        observation=Obs({"EOAA": 5.0, "PRED": 4.0}),
        controller=Controller(),
    )
    assert not r.irrigate
    assert p.deficit_mm == pytest.approx(3.0)


def test_root_zone_rew_uses_root_length_weights():
    p = RootZoneREWThresholdBaseline(
        trigger_rew=0.5,
        event_depth_mm=15.0,
        fallback_rew_layers=(1, 2),
    )
    values = {}
    for i in range(1, 11):
        values[f"REW{i}"] = 0.8
        values[f"RL{i}D"] = 0.0
    values["REW1"] = 0.2
    values["REW2"] = 0.4
    values["RL1D"] = 1.0
    values["RL2D"] = 3.0
    r = p.act(
        policy_day=0,
        observation=Obs(values),
        controller=Controller(),
    )
    assert p.last_root_zone_rew == pytest.approx(0.35)
    assert r.irrigate


def test_root_zone_rew_fallback_is_explicit():
    p = RootZoneREWThresholdBaseline(
        trigger_rew=0.5,
        event_depth_mm=15.0,
        fallback_rew_layers=(1, 2),
    )
    values = {}
    for i in range(1, 11):
        values[f"REW{i}"] = 0.8
        values[f"RL{i}D"] = 0.0
    values["REW1"] = 0.2
    values["REW2"] = 0.4
    r = p.act(
        policy_day=0,
        observation=Obs(values),
        controller=Controller(),
    )
    assert p.last_root_zone_rew == pytest.approx(0.3)
    assert r.irrigate


def test_baselines_have_no_hidden_agronomic_defaults():
    with pytest.raises(TypeError):
        PotentialETWaterBalanceBaseline()
    with pytest.raises(TypeError):
        RootZoneREWThresholdBaseline()
