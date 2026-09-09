import json
from pathlib import Path

import pytest

from awm.rcwa.agent import RCWAAgent
from awm.rcwa.agent_v3 import RCWAV3Agent
from awm.rcwa.real_smoke import _smoke_protocol_spec


def test_rcwa_real_smoke_dispatches_v2_and_v3_without_aliasing():
    assert _smoke_protocol_spec("awm-rcwa-rl-v2") == (RCWAAgent, "rcwa_rl_v2")
    assert _smoke_protocol_spec("awm-rcwa-rl-v3") == (RCWAV3Agent, "rcwa_rl_v3")
    with pytest.raises(ValueError, match="protocol id mismatch"):
        _smoke_protocol_spec("awm-rcwa-rl-v999")


def test_v3_real_smoke_config_is_explicitly_versioned():
    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root / "configs" / "formal_rcwa_smoke_2000_v3.json").read_text())
    assert payload["rcwa_protocol_id"] == "awm-rcwa-rl-v3"
    assert payload["weather_year"] == 2000
    assert payload["weather_split"] == "train"
    assert payload["eta"] == 0.95
    assert payload["seed"] == 21
    assert payload["state_normalization"] is False
