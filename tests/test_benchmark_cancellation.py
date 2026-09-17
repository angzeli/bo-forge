"""Owned-process cancellation and corruption-tolerant terminal accounting."""

import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from benchmarks import runner
from benchmarks.storage import write_json
from tests._benchmark_support import prepare_trial, spec_file


@pytest.mark.parametrize("error", [None, KeyboardInterrupt, RuntimeError, SystemExit])
def test_sigterm_handler_is_scoped_to_run(tmp_path, monkeypatch, error):
    def previous_handler(signum, frame):
        raise AssertionError("Previous handler must not run during the suite")

    def execute(directory, timeout):
        assert signal.getsignal(signal.SIGTERM) is not previous_handler
        if error is not None:
            raise error("controlled interruption")
        runner._finalize_status(directory, "failed", "controlled worker failure", .1)

    monkeypatch.setattr(runner, "_execute_trial", execute)
    previous = signal.signal(signal.SIGTERM, previous_handler)
    try:
        if error is None:
            runner.run_suite(spec_file(tmp_path), tmp_path / "run")
        else:
            with pytest.raises(error, match="controlled interruption"):
                runner.run_suite(spec_file(tmp_path), tmp_path / "run")
        assert signal.getsignal(signal.SIGTERM) is previous_handler
    finally:
        signal.signal(signal.SIGTERM, previous)


def test_run_from_thread_does_not_change_signal_handler(tmp_path, monkeypatch):
    previous = signal.getsignal(signal.SIGTERM)
    monkeypatch.setattr(runner, "_execute_trial", lambda directory, timeout:
                        runner._finalize_status(directory, "failed", "controlled failure", .1))
    with ThreadPoolExecutor(max_workers=1) as pool:
        output = pool.submit(runner.run_suite, spec_file(tmp_path), tmp_path / "run").result()
    assert signal.getsignal(signal.SIGTERM) is previous
    assert json.loads((output / "run.json").read_text())["status"] == "incomplete"


@pytest.mark.parametrize("phase", ["running", "incomplete"])
def test_interrupted_run_status_write_still_records_terminal_state(tmp_path, monkeypatch, phase):
    original_write = runner.write_json
    interruption = KeyboardInterrupt("metadata write interrupted")
    interrupted = False

    def write(path, value):
        nonlocal interrupted
        if path.name == "run.json" and value["status"] == phase and not interrupted:
            interrupted = True
            raise interruption
        original_write(path, value)

    monkeypatch.setattr(runner, "write_json", write)
    monkeypatch.setattr(runner, "_execute_trial", lambda directory, timeout:
                        runner._finalize_status(directory, "failed", "controlled failure", .1))
    output = tmp_path / "run"
    with pytest.raises(KeyboardInterrupt) as caught:
        runner.run_suite(spec_file(tmp_path), output)
    assert caught.value is interruption
    metadata = json.loads((output / "run.json").read_text())
    assert metadata["status"] == "interrupted"
    for trial in metadata["trials"]:
        status = json.loads((output / "trials" / trial["trial_id"] / "status.json").read_text())
        assert status["status"] == ("interrupted" if phase == "running" else "failed")


@pytest.mark.parametrize("status", [[], {"status": []}, {"status": {}},
                                     {"status": "pending", "message": {}},
                                     {"status": "pending", "message": None}])
def test_nested_malformed_status_is_terminalized(tmp_path, status):
    directory, _ = prepare_trial(tmp_path)
    write_json(directory / "status.json", status)
    runner._finalize_status(directory, "failed", "Original worker failure", .1)
    result = json.loads((directory / "status.json").read_text())
    assert result["status"] == "failed"
    assert "Original worker failure" in result["message"]
    assert result["status_warning"]
    assert result["completed_evaluations"] == 0


@pytest.mark.parametrize("status", [{"status": []}, {"status": {}},
                                     {"status": "pending", "message": []}])
@pytest.mark.parametrize("trace, warning", [
    ('{"evaluation": 1}\ninvalid\n{"evaluation": 3}\n', "Malformed evaluation trace"),
    ('[]\n', "Trace rows must be objects"),
    ('{"evaluation": 1}\n{"evaluation":', "Interrupted final trace record"),
])
def test_corrupt_cleanup_preserves_interruption_and_accounts_for_pending_trials(
    tmp_path, monkeypatch, status, trace, warning,
):
    interruption = KeyboardInterrupt("original cancellation")
    damaged = []

    def execute(directory, timeout):
        damaged.append(directory)
        write_json(directory / "status.json", status)
        (directory / "trace.jsonl").write_text(trace)
        raise interruption

    monkeypatch.setattr(runner, "_execute_trial", execute)
    output = tmp_path / "run"
    with pytest.raises(KeyboardInterrupt) as caught:
        runner.run_suite(spec_file(tmp_path), output)
    assert caught.value is interruption
    metadata = json.loads((output / "run.json").read_text())
    assert metadata["status"] == "interrupted" and metadata["finished_at"]
    assert len(damaged) == 1
    for trial in metadata["trials"]:
        directory = output / "trials" / trial["trial_id"]
        result = json.loads((directory / "status.json").read_text())
        assert result["status"] == "interrupted" and result["finished_at"]
        if directory == damaged[0]:
            assert warning in result["trace_warning"]
            assert result["status_warning"]
            assert result["completed_evaluations"] == (1 if "final" in warning else 0)
            assert (directory / "trace.jsonl").read_text() == trace
        else:
            assert result["completed_evaluations"] == 0


def test_trace_corruption_does_not_replace_explicit_interruption(tmp_path):
    directory, _ = prepare_trial(tmp_path)
    (directory / "trace.jsonl").write_text("invalid\n")
    runner._finalize_status(directory, "interrupted", "Original interruption", .1)
    status = json.loads((directory / "status.json").read_text())
    assert status["status"] == "interrupted"
    assert "Original interruption" in status["message"]
    assert "Malformed evaluation trace" in status["trace_warning"]


def test_cleanup_error_does_not_mask_original_or_skip_run_terminal_update(tmp_path, monkeypatch):
    interruption = KeyboardInterrupt("original cancellation")

    def execute(directory, timeout):
        raise interruption

    def cleanup(output, trials):
        raise OSError("controlled cleanup failure")

    monkeypatch.setattr(runner, "_execute_trial", execute)
    monkeypatch.setattr(runner, "_terminalize_remaining", cleanup)
    output = tmp_path / "run"
    with pytest.raises(KeyboardInterrupt) as caught:
        runner.run_suite(spec_file(tmp_path), output)
    assert caught.value is interruption
    assert any("controlled cleanup failure" in note for note in interruption.__notes__)
    assert json.loads((output / "run.json").read_text())["status"] == "interrupted"


@pytest.mark.skipif(os.name != "posix", reason="Controlled POSIX SIGTERM process test")
def test_sigterm_reaps_owned_worker_and_terminalizes_every_trial(tmp_path):
    spec = spec_file(tmp_path)
    output = tmp_path / "run"
    ready, stopped = tmp_path / "worker-ready", tmp_path / "worker-stopped"
    worker = f"""
import os
import signal
import time
from pathlib import Path

def stop(signum, frame):
    Path({str(stopped)!r}).write_text('stopped')
    raise SystemExit(0)

signal.signal(signal.SIGTERM, stop)
ready = Path({str(ready)!r})
temporary = ready.with_suffix('.tmp')
temporary.write_text(str(os.getpid()))
temporary.replace(ready)
time.sleep(60)
"""
    script = f"""
import sys
from benchmarks import runner

execute = runner._execute_trial
def controlled_worker(directory, timeout):
    execute(directory, 60, command=[sys.executable, '-c', {worker!r}])

runner._execute_trial = controlled_worker
runner.run_suite({str(spec)!r}, {str(output)!r})
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[1],
        env=runner._worker_environment(), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 30
        while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(.02)
        assert ready.exists(), "Controlled worker did not become ready"
        worker_pid = int(ready.read_text())
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=15)
        assert process.returncode == 128 + signal.SIGTERM, (stdout, stderr)
        assert stopped.read_text() == "stopped"
        with pytest.raises(ProcessLookupError):
            os.kill(worker_pid, 0)
        metadata = json.loads((output / "run.json").read_text())
        assert metadata["status"] == "interrupted" and metadata["finished_at"]
        assert len(metadata["trials"]) == 6
        for trial in metadata["trials"]:
            status = json.loads(
                (output / "trials" / trial["trial_id"] / "status.json").read_text(),
            )
            assert status["status"] == "interrupted"
            assert status["completed_evaluations"] == 0 and status["finished_at"]
    finally:
        # This session contains only this test's runner and its controlled worker.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate(timeout=10)
