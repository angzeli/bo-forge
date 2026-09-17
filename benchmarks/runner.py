"""Sequential isolated trials with bounded time and explicit terminal records."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from benchmarks.spec import load_spec, schedule
from benchmarks.storage import TERMINAL, environment, read_trace, sha256, utc_now, write_json


@contextmanager
def _cancellation_handler():
    # Python only permits signal handlers in the main thread.
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def interrupt(signum, frame):
        # Let owned-worker cleanup finish even if SIGTERM is sent again.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise SystemExit(128 + signum)

    previous = signal.signal(signal.SIGTERM, interrupt)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def _read_status(path):
    try:
        status = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        return {"status_warning": f"Unreadable worker status: {exc}"}
    if not isinstance(status, dict):
        return {"status_warning": "Worker status must be an object."}
    malformed = []
    for field in ("status", "message"):
        if field in status and not isinstance(status[field], str):
            status.pop(field)
            malformed.append(field)
    if malformed:
        status["status_warning"] = "Malformed worker status fields: " + ", ".join(malformed)
    return status


def _worker_environment():
    env = dict(os.environ)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"
    env.update(CUDA_VISIBLE_DEVICES="", MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1")
    return env


def _stop(process):
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _execute_trial(directory, timeout, *, command=None):
    command = command or [sys.executable, "-m", "benchmarks.worker", str(directory)]
    started = time.monotonic()
    outcome, message = None, ""
    with (directory / "worker.log").open("xb") as output:
        try:
            process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[1],
                                       env=_worker_environment(), stdout=output, stderr=output)
            try:
                code = process.wait(timeout=timeout)
                if code != 0:
                    outcome, message = "failed", f"Worker exited with code {code}; see worker.log."
            except subprocess.TimeoutExpired:
                _stop(process)
                outcome, message = "timeout", f"Trial exceeded {timeout:g} seconds; no retry."
            except BaseException:
                # Cancellation owns this worker; stop it before recording remaining trials.
                _stop(process)
                raise
        except OSError as exc:
            outcome, message = "failed", f"Worker startup failed: {exc}"
    _finalize_status(directory, outcome, message, time.monotonic() - started)


def _finalize_status(directory, outcome, message, seconds):
    status_path = directory / "status.json"
    status = _read_status(status_path)
    try:
        rows, warning = read_trace(directory / "trace.jsonl")
    except (OSError, ValueError) as exc:
        rows, warning = [], str(exc)
        if outcome is None:
            outcome, message = "failed", str(exc)
    if outcome is not None:
        status.update(status=outcome, message=message + " " + status.get("message", ""))
    if status.get("status") not in TERMINAL:
        status.update(status="failed", message="Worker returned without a terminal status.")
    trial = json.loads((directory / "trial.json").read_text())
    if status["status"] == "complete" and len(rows) != trial["evaluations"]:
        status.update(status="failed", message="Complete status has missing evaluation evidence.")
    status.update(completed_evaluations=len(rows), wall_seconds=seconds,
                  trace_warning=warning, finished_at=utc_now())
    write_json(status_path, status)


def run_suite(spec_path, output):
    """Run trials; main-thread SIGTERM cancels owned work and exits with code 143."""
    with _cancellation_handler():
        return _run_suite(spec_path, output)


def _run_suite(spec_path, output):
    spec_path, output = Path(spec_path), Path(output).expanduser().absolute()
    spec = load_spec(spec_path)
    trials = schedule(spec)
    print(f"{spec['name']}: {len(trials)} trials; {spec['evaluations']} evaluations/trial "
          f"({len(trials) * spec['evaluations']} total); "
          f"{spec['timeout_seconds']:g}s/trial; sequential CPU, 1 thread.", flush=True)
    output.mkdir(parents=True, exist_ok=False)
    (output / "spec.yaml").write_bytes(spec_path.read_bytes())
    metadata = {"schema_version": 1, "created_at": utc_now(), "status": "running",
                "spec": spec, "spec_sha256": sha256((output / "spec.yaml").read_bytes()),
                "environment": environment(), "trials": trials}
    started = time.monotonic()
    try:
        write_json(output / "run.json", metadata)
        for trial in trials:
            directory = output / "trials" / trial["trial_id"]
            directory.mkdir(parents=True)
            write_json(directory / "trial.json", trial)
            write_json(directory / "status.json", {"status": "pending", "completed_evaluations": 0})
        for number, trial in enumerate(trials, 1):
            directory = output / "trials" / trial["trial_id"]
            _execute_trial(directory, spec["timeout_seconds"])
            status = json.loads((directory / "status.json").read_text())["status"]
            print(f"[{number}/{len(trials)}] {trial['trial_id']}: {status}", flush=True)
        statuses = [json.loads((output / "trials" / trial["trial_id"] / "status.json").read_text())
                    for trial in trials]
        metadata.update(
            status="complete" if all(s["status"] == "complete" for s in statuses) else "incomplete",
            finished_at=utc_now(), wall_seconds=time.monotonic() - started,
        )
        write_json(output / "run.json", metadata)
    except BaseException as exc:
        # Scheduler boundary: preserve completed trials and explicitly cancel all unfinished work.
        try:
            _terminalize_remaining(output, trials)
        except Exception as cleanup_error:
            exc.add_note(f"Trial cleanup failed: {cleanup_error}")
        metadata.update(status="interrupted", finished_at=utc_now(),
                        wall_seconds=time.monotonic() - started)
        try:
            write_json(output / "run.json", metadata)
        except Exception as cleanup_error:
            exc.add_note(f"Run cleanup failed: {cleanup_error}")
        raise
    return output


def _terminalize_remaining(output, trials):
    for trial in trials:
        directory = output / "trials" / trial["trial_id"]
        directory.mkdir(parents=True, exist_ok=True)
        if not (directory / "trial.json").exists():
            write_json(directory / "trial.json", trial)
        status_path = directory / "status.json"
        status = _read_status(status_path)
        if status.get("status") not in TERMINAL:
            try:
                rows, warning = read_trace(directory / "trace.jsonl")
            except (OSError, ValueError) as exc:
                rows, warning = [], str(exc)
            status.update(status="interrupted", completed_evaluations=len(rows),
                          message="Run interrupted; no automatic resume.",
                          trace_warning=warning, finished_at=utc_now())
            write_json(status_path, status)
