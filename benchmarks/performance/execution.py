"""Fresh-process measurement with bounded, ownership-scoped cleanup."""

import subprocess
import time
from pathlib import Path

import psutil

from benchmarks.runner import _worker_environment
from notebook_assurance.processes import OwnedProcesses


def environment():
    env = _worker_environment()
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        env.pop(key, None)
    env.update(PYTHONNOUSERSITE="1", PIP_DISABLE_PIP_VERSION_CHECK="1")
    return env


def execute(command, cwd, log, timeout, *, clock=time.monotonic):
    """Elapsed seconds include startup and owned-worker shutdown, not preparation."""
    start = clock()
    result = {"status": "not_run", "process_seconds": None, "returncode": None}
    process = tracker = None
    try:
        if timeout <= 0:
            result.update(status="not_run", message="Overall deadline exhausted.")
            return result
        with Path(log).open("w", encoding="utf-8") as output:
            process = subprocess.Popen(command, cwd=cwd, env=environment(), stdout=output,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                tracker = OwnedProcesses(process.pid)
            except psutil.NoSuchProcess:
                # A lightweight command can exit before ownership inspection.
                process.wait()
            while process.poll() is None:
                if tracker is not None:
                    tracker.refresh()
                remaining = timeout - (clock() - start)
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout)
                try:
                    process.wait(timeout=min(0.2, remaining))
                except subprocess.TimeoutExpired:
                    continue
            result.update(status="complete" if process.returncode == 0 else "failed",
                          returncode=process.returncode)
    except subprocess.TimeoutExpired:
        result.update(status="timeout", message="Subprocess deadline exceeded.")
    except BaseException as exc:
        # Cancellation boundary: preserve interruption and attach elapsed/cleanup evidence.
        result.update(status="interrupted" if not isinstance(exc, Exception) else "failed",
                      message=f"{type(exc).__name__}: {exc}")
        exc.performance_result = result
        raise
    finally:
        _cleanup(process, tracker, result)
        if process is not None:
            result["process_seconds"] = clock() - start
    return result


def _cleanup(process, tracker, result):
    import sys

    original = sys.exception()
    actions = [tracker.stop] if tracker is not None else []
    if process is not None:
        actions.extend([lambda: process.kill() if process.poll() is None else None,
                        lambda: process.wait(timeout=5)])
    errors = []
    for action in actions:
        try:
            action()
        except BaseException as exc:
            # Cleanup boundary: attempt every owned cleanup and retain the original failure.
            errors.append(f"{type(exc).__name__}: {exc}")
    if errors:
        result["cleanup_error"] = "; ".join(errors)
        if original is not None:
            original.add_note(result["cleanup_error"])
        else:
            result["status"] = "failed"


def thread_settings():
    return {key: value for key, value in environment().items()
            if key.endswith("NUM_THREADS") or key in ("CUDA_VISIBLE_DEVICES", "MPLBACKEND")}
