from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "legacy_v3"


def test_lrmb_v3_snapshot_is_present_and_algorithm_free():
    assert (LEGACY / "MIGRATION.md").is_file()
    assert (LEGACY / "dssat" / "runner.py").is_file()
    assert (LEGACY / "dssat" / "file_handler.py").is_file()
    assert (LEGACY / "envs" / "cotton" / "env.py").is_file()
    assert (LEGACY / "envs" / "cotton" / "state_schema.py").is_file()

    forbidden = (
        "algorithms",
        "models",
        "pipeline",
        "toy_v2",
    )
    for name in forbidden:
        assert not (LEGACY / name).exists(), f"algorithm/training tree leaked into legacy_v3: {name}"

    forbidden_files = (
        "launch.py",
        "run.py",
        "train.py",
    )
    for name in forbidden_files:
        assert not (LEGACY / name).exists(), f"training entrypoint leaked into legacy_v3: {name}"


def test_lrmb_v3_snapshot_records_exact_source_commit():
    text = (LEGACY / "MIGRATION.md").read_text(encoding="utf-8")
    assert "d56336e09fdb9a9aea60ae61eaa892833314ab33" in text
    assert "exp/v3-recoverable-policy-manifold-train10000" in text
    assert "src/awm" in text
