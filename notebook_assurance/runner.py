"""Sequential fresh-source execution; failed attempts are retained, never retried."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from notebook_assurance.archive import (
    check_protected,
    digest,
    extract,
    protected_inputs,
    write_json,
)
from notebook_assurance.contracts import CONTRACTS, select
from notebook_assurance.processes import OwnedProcesses


@contextmanager
def cancellation():
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def interrupt(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise SystemExit(128 + signum)

    previous = signal.signal(signal.SIGTERM, interrupt)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def prepare(directory, archive, name):
    root = extract(archive, directory / "source")
    runtime = directory / "runtime"
    for child in ("tmp", "jupyter", "ipython", "cache", "kernels/assurance", "guard"):
        (runtime / child).mkdir(parents=True, exist_ok=True)
    (directory / "captures").mkdir()
    identities = protected_inputs(root)
    write_json(runtime / "protected.json", identities)
    for path in identities:
        Path(path).chmod(0o444)
    (runtime / "guard/sitecustomize.py").write_text(
        "from notebook_assurance.guard import install\ninstall()\n", encoding="utf-8")
    write_json(runtime / "kernels/assurance/kernel.json", {
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "BO Forge isolated acceptance", "language": "python",
    })
    settings = {"notebook": name, "directory": str(directory), "root": str(root),
                **CONTRACTS[name]}
    write_json(directory / "settings.json", settings)
    env = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    env.update(PYTHONPATH=os.pathsep.join([str(runtime / "guard"), str(root)]),
               PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", CUDA_VISIBLE_DEVICES="",
               MPLBACKEND="Agg", MPLCONFIGDIR=str(runtime / "cache/matplotlib"),
               XDG_CACHE_HOME=str(runtime / "cache"), IPYTHONDIR=str(runtime / "ipython"),
               JUPYTER_DATA_DIR=str(runtime / "jupyter"),
               JUPYTER_RUNTIME_DIR=str(runtime / "jupyter"),
               JUPYTER_CONFIG_DIR=str(runtime / "jupyter"),
               TMPDIR=str(runtime / "tmp"), TMP=str(runtime / "tmp"), TEMP=str(runtime / "tmp"),
               NB_PROTECTED=str(runtime / "protected.json"))
    return settings, env


def read_status(directory, name):
    try:
        return json.loads((directory / "status.json").read_text())
    except (OSError, ValueError):
        return {"notebook": name, "status": "failed", "completed_cells": 0,
                "message": "Worker did not leave readable status evidence."}


def cleanup(process, owned):
    warnings = []
    actions = []
    if owned is not None:
        actions.append(owned.stop)
    if process is not None:
        actions.extend([lambda: process.terminate() if process.poll() is None else None,
                        lambda: process.wait(timeout=10)])
    for action in actions:
        try:
            action()
        except BaseException as exc:
            # Cleanup boundary: continue even if process inspection is unavailable.
            warnings.append(f"{type(exc).__name__}: {exc}")
    return warnings


def execute_one(directory, settings, env):
    started = time.monotonic()
    process, owned = None, None
    status = {"notebook": settings["notebook"], "status": "failed", "completed_cells": 0}
    try:
        with (directory / "worker.log").open("xb") as log:
            command = [sys.executable, "-m", "notebook_assurance.worker",
                       str(directory / "settings.json")]
            process = subprocess.Popen(
                command,
                cwd=settings["root"], env=env, stdout=log, stderr=log, start_new_session=True)
            owned = OwnedProcesses(process.pid)
            while process.poll() is None:
                owned.refresh()
                if time.monotonic() - started > settings["notebook_timeout"]:
                    status = read_status(directory, settings["notebook"])
                    status.update(status="timeout", message="Notebook wall-time limit exceeded.")
                    break
                time.sleep(0.2)
            else:
                status = read_status(directory, settings["notebook"])
                if (status.get("status") not in {"passed", "failed", "timeout"}
                        or (process.returncode != 0 and status.get("status") == "passed")):
                    status["status"] = "failed"
    except BaseException as exc:
        # Scheduler boundary preserves cancellation and its partial evidence.
        status = read_status(directory, settings["notebook"])
        status.update(status="interrupted", message=str(exc), exception_type=type(exc).__name__)
        raise
    finally:
        original = sys.exception()
        warnings = cleanup(process, owned)
        if warnings:
            status["cleanup_warning"] = "; ".join(warnings)
            if original is not None:
                original.add_note(status["cleanup_warning"])
            if status["status"] == "passed":
                status["status"] = "failed"
        status["elapsed_seconds"] = time.monotonic() - started
        try:
            identities = json.loads((directory / "runtime/protected.json").read_text())
            check_protected(identities)
        except (OSError, ValueError) as exc:
            status["protected_input_error"] = str(exc)
            if status["status"] == "passed":
                status["status"] = "failed"
        try:
            write_json(directory / "status.json", status)
        except OSError as exc:
            if original is None:
                raise
            original.add_note(f"Status persistence failed: {exc}")
    return status


def run(sdist, profile, output, selection=None):
    output, sdist = Path(output).absolute(), Path(sdist).resolve()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(sdist, output / "source.tar.gz")
    archive = output / "source.tar.gz"
    root = extract(archive, output / "discovery")
    scheduled = select(root, profile)
    selected = select(root, profile, selection)
    result = {"schema_version": 1, "archive_sha256": digest(archive), "profile": profile,
              "scheduled": scheduled, "selected": selected, "notebooks": [], "status": "running"}
    result["notebooks"] = [{"notebook": name, "status": "not_run", "elapsed_seconds": None}
                           for name in selected]
    write_json(output / "result.json", result)
    with cancellation():
        try:
            for index, name in enumerate(selected):
                directory = output / Path(name).stem
                directory.mkdir()
                print(f"[{index + 1}/{len(selected)}] {name}", flush=True)
                settings, env = prepare(directory, archive, name)
                status = execute_one(directory, settings, env)
                status["directory"] = directory.name
                result["notebooks"][index] = status
                write_json(output / "result.json", result)
                print(f"  {status['status']} ({status['elapsed_seconds']:.1f}s)", flush=True)
            result["status"] = (
                "passed" if all(s["status"] == "passed" for s in result["notebooks"]) else "failed")
        except BaseException:
            # Run boundary: keep all scheduled records, including work never started.
            result["status"] = "interrupted"
            for item in result["notebooks"]:
                status_path = output / Path(item["notebook"]).stem / "status.json"
                if status_path.exists():
                    item.update(json.loads(status_path.read_text()))
            raise
        finally:
            write_json(output / "result.json", result)
    return result
