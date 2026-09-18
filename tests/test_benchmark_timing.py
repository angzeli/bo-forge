"""Monotonic trial timing includes startup and cancellation cleanup."""

import json
import subprocess

import pytest

from benchmarks import runner
from benchmarks.storage import write_json
from tests._benchmark_support import prepare_trial, smoke_spec, spec_file


@pytest.mark.parametrize("error", [KeyboardInterrupt, SystemExit])
def test_cancellation_finalizes_elapsed_after_worker_shutdown(tmp_path, monkeypatch, error):
    directory, _ = prepare_trial(tmp_path)
    interruption = error("original cancellation")
    clock = [100.0]
    stopped = []

    class Process:
        def __init__(self, *args, **kwargs):
            clock[0] += 2.0

        def wait(self, *, timeout):
            clock[0] += 3.0
            raise interruption

    def stop(process):
        clock[0] += 5.0
        stopped.append(process)

    monkeypatch.setattr(runner.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(runner.subprocess, "Popen", Process)
    monkeypatch.setattr(runner, "_stop", stop)
    with pytest.raises(error) as caught:
        runner._execute_trial(directory, 60)
    assert caught.value is interruption
    assert len(stopped) == 1
    status = json.loads((directory / "status.json").read_text())
    assert status.get("wall_seconds") == 10.0
    assert status["timing_status"] == "measured"
    assert status["status"] == "interrupted"


@pytest.mark.parametrize("failure", [OSError, KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("persist", [True, False])
def test_cleanup_failures_attach_notes_and_still_attempt_timing(
    tmp_path, monkeypatch, failure, persist,
):
    directory, _ = prepare_trial(tmp_path)
    interruption = KeyboardInterrupt("original cancellation")
    clock = [100.0]
    finalizations = []
    original_finalize = runner._finalize_status

    class Process:
        def __init__(self, *args, **kwargs):
            clock[0] += 2.0

        def wait(self, *, timeout):
            clock[0] += 3.0
            raise interruption

    def stop(process):
        clock[0] += 5.0
        raise failure("shutdown failed")

    def finalize(directory, outcome, message, seconds):
        finalizations.append((outcome, seconds))
        if not persist:
            raise failure("status persistence failed")
        original_finalize(directory, outcome, message, seconds)

    monkeypatch.setattr(runner.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(runner.subprocess, "Popen", Process)
    monkeypatch.setattr(runner, "_stop", stop)
    monkeypatch.setattr(runner, "_finalize_status", finalize)
    with pytest.raises(KeyboardInterrupt) as caught:
        runner._execute_trial(directory, 60)
    assert caught.value is interruption
    assert finalizations == [("interrupted", 10.0)]
    assert any("shutdown failed" in note for note in interruption.__notes__)
    status = json.loads((directory / "status.json").read_text())
    if persist:
        assert status["wall_seconds"] == 10.0
        assert status["timing_status"] == "measured"
    else:
        assert any("status persistence failed" in note for note in interruption.__notes__)
        assert status.get("wall_seconds") is None


@pytest.mark.parametrize("outcome, seconds", [
    ("complete", 5.0), ("failed", 5.0), ("timeout", 10.0), ("startup", 2.0),
])
def test_normal_terminal_timing_includes_startup_and_shutdown(
    tmp_path, monkeypatch, outcome, seconds,
):
    directory, trial = prepare_trial(tmp_path)
    clock = [100.0]
    stopped = []

    class Process:
        def __init__(self, *args, **kwargs):
            clock[0] += 2.0
            if outcome == "startup":
                raise OSError("controlled startup failure")

        def wait(self, *, timeout):
            clock[0] += 3.0
            if outcome == "timeout":
                raise subprocess.TimeoutExpired("worker", timeout)
            if outcome == "complete":
                write_json(directory / "status.json", {"status": "complete"})
                (directory / "trace.jsonl").write_text('{}\n' * trial["evaluations"])
            return 0 if outcome == "complete" else 7

    def stop(process):
        clock[0] += 5.0
        stopped.append(process)

    monkeypatch.setattr(runner.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(runner.subprocess, "Popen", Process)
    monkeypatch.setattr(runner, "_stop", stop)
    runner._execute_trial(directory, 60)
    status = json.loads((directory / "status.json").read_text())
    assert status["status"] == ("failed" if outcome == "startup" else outcome)
    assert status["wall_seconds"] == seconds
    assert status["timing_status"] == "measured"
    assert len(stopped) == (1 if outcome == "timeout" else 0)


@pytest.mark.parametrize("worker_status", ["pending", "running", "failed", "complete", "corrupt"])
def test_scheduler_tracks_started_independently_of_worker_status(
    tmp_path, monkeypatch, worker_status,
):
    interruption = KeyboardInterrupt("original cancellation")
    calls = []
    persisted = []
    original_write = runner.write_json

    def write(path, value):
        if path.name == "status.json":
            persisted.append(path.parent)
        original_write(path, value)

    def execute(directory, timeout):
        calls.append(directory)
        if len(calls) == 1:
            runner._finalize_status(directory, "failed", "controlled failure", 1.25)
            return
        if worker_status == "corrupt":
            (directory / "status.json").write_text("invalid")
        else:
            write_json(directory / "status.json", {"status": worker_status})
        raise interruption

    monkeypatch.setattr(runner, "write_json", write)
    monkeypatch.setattr(runner, "_execute_trial", execute)
    output = tmp_path / "run"
    with pytest.raises(KeyboardInterrupt) as caught:
        runner.run_suite(spec_file(tmp_path), output)
    assert caught.value is interruption
    metadata = json.loads((output / "run.json").read_text())
    statuses = [json.loads((output / "trials" / trial["trial_id"] / "status.json").read_text())
                for trial in metadata["trials"]]
    assert len(calls) == 2
    assert statuses[0]["timing_status"] == "measured"
    assert statuses[0]["wall_seconds"] == 1.25
    assert persisted.count(calls[0]) == 2  # Initialization and finalization only.
    assert statuses[1]["timing_status"] == "unknown"
    assert statuses[1]["wall_seconds"] is None
    for status in statuses[2:]:
        assert status["timing_status"] == "not_started"
        assert status["wall_seconds"] == 0.0


def test_failed_active_timing_persistence_recovers_as_unknown(tmp_path, monkeypatch):
    interruption = KeyboardInterrupt("original cancellation")

    class Process:
        def __init__(self, *args, **kwargs):
            pass

        def wait(self, *, timeout):
            raise interruption

    def finalize(*args):
        raise OSError("active timing persistence unavailable")

    monkeypatch.setattr(runner, "environment", lambda: {})
    monkeypatch.setattr(runner.subprocess, "Popen", Process)
    monkeypatch.setattr(runner, "_stop", lambda process: None)
    monkeypatch.setattr(runner, "_finalize_status", finalize)
    output = tmp_path / "run"
    with pytest.raises(KeyboardInterrupt) as caught:
        runner.run_suite(spec_file(tmp_path), output)
    assert caught.value is interruption
    assert any("active timing persistence unavailable" in note for note in interruption.__notes__)
    metadata = json.loads((output / "run.json").read_text())
    active = output / "trials" / metadata["trials"][0]["trial_id"] / "status.json"
    status = json.loads(active.read_text())
    assert status["status"] == "interrupted"
    assert status["timing_status"] == "unknown"
    assert status["wall_seconds"] is None


@pytest.mark.parametrize("version", [1, 2])
def test_run_metadata_preserves_spec_schema_version(tmp_path, monkeypatch, version):
    spec = smoke_spec()
    trials = runner.schedule(spec)
    spec["schema_version"] = version
    # Isolate runner metadata from the separately owned v2 spec/route validation.
    monkeypatch.setattr(runner, "_parse_spec", lambda payload: spec)
    monkeypatch.setattr(runner, "schedule", lambda spec: trials)
    monkeypatch.setattr(runner, "_execute_trial", lambda directory, timeout:
                        runner._finalize_status(directory, "failed", "controlled failure", .1))
    output = runner.run_suite(spec_file(tmp_path, spec), tmp_path / "run")
    metadata = json.loads((output / "run.json").read_text())
    assert metadata["schema_version"] == metadata["spec"]["schema_version"] == version
