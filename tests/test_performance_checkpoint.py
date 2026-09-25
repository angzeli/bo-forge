"""Checkpoint failures preserve the original interruption, not fictional success."""

import pytest

from benchmarks.performance import runner


def test_checkpoint_failure_preserves_interruption(monkeypatch, tmp_path):
    def fail(*args):
        raise OSError("disk full")

    monkeypatch.setattr(runner, "write_json", fail)
    original = KeyboardInterrupt("cancelled")
    with pytest.raises(KeyboardInterrupt) as caught:
        try:
            raise original
        finally:
            runner._checkpoint(tmp_path / "status.json", {"status": "interrupted"})
    assert caught.value is original
    assert original.__notes__ == ["Evidence checkpoint failed: OSError: disk full"]


def test_checkpoint_failure_is_not_suppressed_without_original(monkeypatch, tmp_path):
    def fail(*args):
        raise OSError("disk full")

    monkeypatch.setattr(runner, "write_json", fail)
    with pytest.raises(OSError, match="disk full"):
        runner._checkpoint(tmp_path / "status.json", {"status": "complete"})
