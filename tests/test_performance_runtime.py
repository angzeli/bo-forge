"""Bounded runtime contracts without fitting models or installing environments."""

import copy
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import psutil
import pytest

from benchmarks.performance import execution, preparation, probe, runner, schedule


def test_schedule_has_alternating_pairs_and_per_case_warmup_plus_five():
    rows = schedule()
    assert rows == schedule()
    assert len(rows) == len({row["sample_id"] for row in rows}) == 156
    expected_cases = {
        "import", "version", "help", "validate",
        "log_ei-q1", "log_ei-q2", "log_ei-q4",
        "qlog_ehvi-q1", "qlog_ehvi-q2", "qlog_ehvi-q4",
        "qmf_kg-q1", "qmf_kg-q2", "qmf_kg-q4",
    }
    assert {row["case_id"] for row in rows} == expected_cases
    assert sum(row["warmup"] for row in rows) == 26
    for case_id in expected_cases:
        samples = [row for row in rows if row["case_id"] == case_id]
        assert len(samples) == 12
        assert [row["warmup"] for row in samples] == [True, True] + [False] * 10
        assert Counter(row["version"] for row in samples) == {"baseline": 6, "candidate": 6}
        pairs = [samples[index:index + 2] for index in range(0, 12, 2)]
        assert [tuple(row["version"] for row in pair) for pair in pairs] == [
            ("baseline", "candidate"), ("candidate", "baseline"),
            ("baseline", "candidate"), ("candidate", "baseline"),
            ("baseline", "candidate"), ("candidate", "baseline"),
        ]
        assert [tuple(row["repetition"] for row in pair) for pair in pairs] == [
            (0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5),
        ]
        for row in samples:
            if row["kind"] == "suggest":
                assert f"{row['route']}-q{row['batch_size']}" == case_id
            else:
                assert row["kind"] == case_id


@pytest.fixture
def identities():
    baseline = {"python": "3.12.8", "requirements": ["numpy>=2"],
                "packages": {"bo-forge": "3.3.3", "numpy": "2.1.0"}}
    candidate = copy.deepcopy(baseline)
    candidate["packages"]["bo-forge"] = "3.3.4"
    return {"baseline": baseline, "candidate": candidate}


def test_match_environments_allows_only_package_under_test_to_differ(identities):
    original = copy.deepcopy(identities)
    preparation.match_environments(identities)
    assert identities == original


@pytest.mark.parametrize("mismatch", ["python", "requirements", "version", "missing", "extra"])
def test_match_environments_rejects_python_and_dependency_mismatches(identities, mismatch):
    candidate = identities["candidate"]
    if mismatch == "python":
        candidate["python"] = "3.13.0"
    elif mismatch == "requirements":
        candidate["requirements"] = ["numpy>=2.1"]
    elif mismatch == "version":
        candidate["packages"]["numpy"] = "2.2.0"
    elif mismatch == "missing":
        candidate["packages"].pop("numpy")
    else:
        candidate["packages"]["unexpected"] = "1.0"

    with pytest.raises(ValueError, match="Environment mismatch"):
        preparation.match_environments(identities)


def test_worker_environment_removes_import_overrides_and_bounds_threads(monkeypatch):
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        monkeypatch.setenv(key, "/untrusted/checkout")
    monkeypatch.setenv("OMP_NUM_THREADS", "64")
    env = execution.environment()
    assert not {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"} & env.keys()
    assert env["PYTHONNOUSERSITE"] == "1"
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        assert env[key] == "1"
    assert os.environ["OMP_NUM_THREADS"] == "64"


@pytest.fixture
def installed_probe(tmp_path, monkeypatch):
    prefix = tmp_path / "venv"
    package = prefix / "lib/site-packages/bo_forge/__init__.py"
    package.parent.mkdir(parents=True)
    package.write_text('__version__ = "3.3.4"\n')
    module = SimpleNamespace(__file__=str(package), __version__="3.3.4")
    entry = package.relative_to(prefix)
    distribution = SimpleNamespace(
        version="3.3.4", metadata={"Name": "bo_forge"}, files=[entry],
        locate_file=lambda item: prefix / item,
    )
    monkeypatch.setitem(sys.modules, "bo_forge", module)
    monkeypatch.setattr(sys, "prefix", str(prefix))
    monkeypatch.setattr(probe.metadata, "distribution", lambda name: distribution)
    monkeypatch.setattr(probe.metadata, "distributions", lambda: [distribution])
    return SimpleNamespace(prefix=prefix, package=package, module=module,
                           distribution=distribution)


def test_probe_accepts_installed_package_and_records_identity(installed_probe):
    result = probe.identity()
    assert result["prefix"] == str(installed_probe.prefix)
    assert result["import_path"] == str(installed_probe.package.resolve())
    assert result["version"] == "3.3.4"
    assert result["packages"] == {"bo-forge": "3.3.4"}
    entry = installed_probe.package.relative_to(installed_probe.prefix)
    assert list(result["files"]) == [str(entry)]


@pytest.mark.parametrize("location", ["checkout", "prefix-lookalike", "symlink"])
def test_probe_rejects_import_paths_outside_installation(installed_probe, tmp_path, location):
    outside = tmp_path / ("venv-other" if location == "prefix-lookalike" else "checkout")
    outside.mkdir()
    escaped = outside / "__init__.py"
    escaped.write_text("# external source\n")
    if location == "symlink":
        link = installed_probe.prefix / "linked.py"
        link.symlink_to(escaped)
        escaped = link
    installed_probe.module.__file__ = str(escaped)

    with pytest.raises(ValueError, match="escaped installation"):
        probe.identity()


def test_probe_rejects_imported_and_installed_version_mismatch(installed_probe):
    installed_probe.module.__version__ = "0.0.0"
    with pytest.raises(ValueError, match="version differs"):
        probe.identity()


def test_preparation_probe_uses_isolated_python_without_installing(tmp_path, monkeypatch):
    expected = {"version": "3.3.4", "packages": {"bo-forge": "3.3.4"}}

    def checked(command, cwd, log, deadline):
        assert command[:2] == ["/fake/venv/bin/python", "-I"]
        assert Path(command[2]) == Path(probe.__file__)
        assert cwd == tmp_path and log == tmp_path / "verification.log"
        assert deadline == 123.0
        Path(command[3]).write_text(json.dumps(expected))
    command = Mock(side_effect=checked)
    monkeypatch.setattr(preparation, "checked", command)

    assert preparation.probe(Path("/fake/venv/bin/python"), tmp_path, 123.0,
                             name="verification") == expected
    command.assert_called_once()


@pytest.mark.parametrize("exitcode,status", [(0, "complete"), (7, "failed")])
def test_execute_real_tiny_process(tmp_path, exitcode, status):
    log = tmp_path / "process.log"
    result = execution.execute(
        [sys.executable, "-I", "-c", f"print('tiny process'); raise SystemExit({exitcode})"],
        tmp_path, log, 10,
    )
    assert result["status"] == status
    assert result["returncode"] == exitcode
    assert result["process_seconds"] > 0
    assert "cleanup_error" not in result
    assert "tiny process" in log.read_text()


def test_execute_real_tiny_process_timeout(tmp_path):
    result = execution.execute(
        [sys.executable, "-I", "-c", "import time; time.sleep(30)"],
        tmp_path, tmp_path / "timeout.log", 0.25,
    )
    assert result["status"] == "timeout"
    assert result["process_seconds"] >= 0.25
    assert "cleanup_error" not in result


def test_exhausted_deadline_does_not_launch_process(tmp_path, monkeypatch):
    launch = Mock(side_effect=AssertionError("must not launch"))
    monkeypatch.setattr(execution.subprocess, "Popen", launch)
    result = execution.execute([sys.executable], tmp_path, tmp_path / "unused.log", 0)
    assert result["status"] == "not_run"
    assert result["process_seconds"] is None
    assert not (tmp_path / "unused.log").exists()
    launch.assert_not_called()


@pytest.mark.parametrize("original", [KeyboardInterrupt("cancel"), SystemExit(143),
                                      PermissionError("inspection failed")])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_execute_preserves_original_error_elapsed_and_cleanup_diagnostics(
    tmp_path, monkeypatch, original, cleanup_fails,
):
    process = Mock(pid=123, returncode=None)
    process.poll.return_value = None
    tracker = Mock()
    tracker.refresh.side_effect = original
    if cleanup_fails:
        tracker.stop.side_effect = PermissionError("cleanup denied")
    monkeypatch.setattr(execution.subprocess, "Popen", Mock(return_value=process))
    monkeypatch.setattr(execution, "OwnedProcesses", Mock(return_value=tracker))

    with pytest.raises(type(original)) as caught:
        execution.execute(["fake-worker"], tmp_path, tmp_path / "failure.log", 20,
                          clock=Mock(side_effect=[10.0, 14.5]))

    assert caught.value is original
    result = original.performance_result
    assert result["process_seconds"] == 4.5
    assert result["status"] == ("failed" if isinstance(original, Exception) else "interrupted")
    assert result["message"] == f"{type(original).__name__}: {original}"
    tracker.stop.assert_called_once_with()
    if cleanup_fails:
        assert "cleanup denied" in result["cleanup_error"]
        assert result["cleanup_error"] in original.__notes__
    else:
        process.wait.assert_called_once_with(timeout=5)
        assert "cleanup_error" not in result


def test_tracker_cleanup_failure_still_reaps_direct_process():
    process = Mock()
    process.poll.return_value = None
    tracker = Mock()
    tracker.stop.side_effect = PermissionError("descendant inspection denied")
    result = {"status": "complete"}

    execution._cleanup(process, tracker, result)

    assert result["status"] == "failed"
    assert "descendant inspection denied" in result["cleanup_error"]
    process.wait.assert_called_once_with(timeout=5)


def test_execute_timing_starts_before_log_open_and_includes_cleanup(tmp_path, monkeypatch):
    elapsed, events = [0.0], []
    process = Mock(pid=123, returncode=0)
    process.poll.return_value = 0
    tracker = Mock()
    real_open = Path.open

    def clock():
        events.append("clock")
        return elapsed[0]

    def open_log(path, *args, **kwargs):
        events.append("open")
        elapsed[0] += 2
        return real_open(path, *args, **kwargs)

    def launch(*args, **kwargs):
        events.append("launch")
        elapsed[0] += 3
        return process

    def stop():
        events.append("stop")
        elapsed[0] += 5

    def wait(timeout):
        events.append("wait")
        elapsed[0] += 7
        return 0

    tracker.stop.side_effect = stop
    process.wait.side_effect = wait
    monkeypatch.setattr(Path, "open", open_log)
    monkeypatch.setattr(execution.subprocess, "Popen", launch)
    monkeypatch.setattr(execution, "OwnedProcesses", Mock(return_value=tracker))

    result = execution.execute(["fake-worker"], tmp_path, tmp_path / "timed.log", 30,
                               clock=clock)

    assert result["status"] == "complete"
    assert result["process_seconds"] == 17.0
    assert events == ["clock", "open", "launch", "stop", "wait", "clock"]


@pytest.fixture
def sample_run(tmp_path, monkeypatch):
    root = tmp_path / "performance"
    inputs = root / "inputs"
    inputs.mkdir(parents=True)
    content = {"config": b"seed: 42\n", "log": b"x,y\n1,2\n"}
    paths = {"config": "config.yaml", "log": "observations.csv"}
    for key, data in content.items():
        (inputs / paths[key]).write_bytes(data)
    items = [item for item in schedule()
             if item["case_id"] == "log_ei-q2" and item["repetition"] == 1]
    metadata = {
        "schedule": items,
        "installations": {
            version: {"prefix": str(root / version / "venv"),
                      "executable": str(root / version / "venv/bin/python")}
            for version in ("baseline", "candidate")
        },
        "inputs": {"log_ei": {"expected_source": "log_ei",
                              "config_path": paths["config"], "log_path": paths["log"]}},
    }
    runner._retain_harness(root, metadata)
    runner._initialize_samples(root, metadata)
    monkeypatch.setattr(runner, "time", SimpleNamespace(monotonic=lambda: 100.0))
    return SimpleNamespace(root=root, metadata=metadata, items=items, content=content, paths=paths)


@pytest.mark.parametrize("deadline,budget", [(107.0, 7.0), (1000.0, 600)])
def test_sample_stores_timeout_result_and_bounds_process_budget(
    sample_run, monkeypatch, deadline, budget,
):
    item = sample_run.items[0]
    result = {"status": "timeout", "process_seconds": 7.25, "returncode": None,
              "message": "Subprocess deadline exceeded."}
    execute = Mock(return_value=result)
    monkeypatch.setattr(runner, "execute", execute)

    runner._sample(sample_run.root, item, sample_run.metadata, deadline)

    directory = sample_run.root / "samples" / item["sample_id"]
    status = json.loads((directory / "status.json").read_text())
    assert {key: status[key] for key in result} == result
    assert status["sample_id"] == item["sample_id"]
    assert status["call_seconds"] is None and status["peak_rss_bytes"] is None
    assert execute.call_args.args[1:] == (directory, directory / "process.log", budget)


@pytest.mark.parametrize("original", [KeyboardInterrupt("cancel sample"), SystemExit(143)])
def test_sample_preserves_interruption_and_saves_elapsed_diagnostics(
    sample_run, monkeypatch, original,
):
    original.performance_result = {
        "status": "interrupted", "process_seconds": 4.75, "returncode": None,
        "cleanup_error": "PermissionError: cleanup denied",
    }
    original.add_note("retained cancellation diagnostic")
    monkeypatch.setattr(runner, "execute", Mock(side_effect=original))
    item = sample_run.items[0]

    with pytest.raises(type(original)) as caught:
        runner._sample(sample_run.root, item, sample_run.metadata, 120.0)

    assert caught.value is original
    assert "retained cancellation diagnostic" in original.__notes__
    status_path = sample_run.root / "samples" / item["sample_id"] / "status.json"
    status = json.loads(status_path.read_text())
    assert {key: status[key] for key in original.performance_result} == original.performance_result
    assert status["message"] == f"{type(original).__name__}: {original}"
    assert status["sample_id"] == item["sample_id"]


def test_sample_not_run_keeps_unknown_timing_without_launching(sample_run, monkeypatch):
    launch = Mock(side_effect=AssertionError("exhausted sample must not launch"))
    monkeypatch.setattr(execution.subprocess, "Popen", launch)
    item = sample_run.items[0]

    runner._sample(sample_run.root, item, sample_run.metadata, 99.0)

    status_path = sample_run.root / "samples" / item["sample_id"] / "status.json"
    status = json.loads(status_path.read_text())
    assert status["status"] == "not_run"
    for key in ("process_seconds", "call_seconds", "peak_rss_bytes", "returncode"):
        assert status[key] is None
    launch.assert_not_called()


def test_sample_uses_frozen_isolated_worker_and_identical_copied_inputs(sample_run, monkeypatch):
    root = sample_run.root
    checkout = Path(runner.__file__).resolve().parents[2]
    calls = []

    def execute(command, cwd, log, timeout):
        request = json.loads((cwd / "request.json").read_text())
        version = request["version"]
        assert command == [sample_run.metadata["installations"][version]["executable"],
                           "-I", str(root / "harness/worker.py"), str(cwd / "request.json")]
        assert Path(command[2]).is_file()
        assert not cwd.resolve().is_relative_to(checkout)
        assert not Path(command[2]).resolve().is_relative_to(checkout)
        assert log == cwd / "process.log" and timeout == 20.0
        for key, data in sample_run.content.items():
            path = Path(request[f"{key}_path"])
            assert path.parent == cwd
            assert path.read_bytes() == data
            assert not path.samefile(root / "inputs" / sample_run.paths[key])
        worker_result = {"call_seconds": 0.125, "peak_rss_bytes": 4096}
        Path(request["result_path"]).write_text(json.dumps(worker_result))
        calls.append(request)
        return {"status": "complete", "process_seconds": 1.5, "returncode": 0}

    monkeypatch.setattr(runner, "execute", execute)
    for item in sample_run.items:
        runner._sample(root, item, sample_run.metadata, 120.0)
        status = json.loads((root / "samples" / item["sample_id"] / "status.json").read_text())
        assert status["status"] == "complete" and status["process_seconds"] == 1.5
        assert status["call_seconds"] == 0.125 and status["peak_rss_bytes"] == 4096
    assert {request["version"] for request in calls} == {"baseline", "candidate"}
    for key, data in sample_run.content.items():
        assert calls[0][f"{key}_sha256"] == calls[1][f"{key}_sha256"]
        assert calls[0][f"{key}_path"] != calls[1][f"{key}_path"]
        assert (root / "inputs" / sample_run.paths[key]).read_bytes() == data


@pytest.mark.skipif(os.name != "posix", reason="Detached POSIX descendant cleanup")
def test_cancellation_cleans_real_owned_child_in_separate_session(tmp_path, monkeypatch):
    ready = tmp_path / "child.json"
    child = (
        "import json,os,time; from pathlib import Path; "
        f"p=Path({str(ready)!r}); q=p.with_suffix('.tmp'); "
        "q.write_text(json.dumps({'pid':os.getpid()})); q.replace(p); time.sleep(30)"
    )
    worker = (
        "import subprocess,sys,time; "
        f"subprocess.Popen([sys.executable,'-I','-c',{child!r}],start_new_session=True); "
        "time.sleep(30)"
    )
    original = KeyboardInterrupt("controlled cancellation")
    launched, owned = [], []
    popen = execution.subprocess.Popen

    def launch(*args, **kwargs):
        process = popen(*args, **kwargs)
        launched.append(process)
        owned.append(psutil.Process(process.pid))
        return process

    def clock():
        if ready.exists() and len(owned) == 1:
            owned.append(psutil.Process(json.loads(ready.read_text())["pid"]))
            raise original
        return time.monotonic()

    monkeypatch.setattr(execution.subprocess, "Popen", launch)
    try:
        with pytest.raises(KeyboardInterrupt) as caught:
            execution.execute([sys.executable, "-I", "-c", worker], tmp_path,
                              tmp_path / "cancel.log", 10, clock=clock)
        assert caught.value is original
        assert len(owned) == 2
        assert original.performance_result["status"] == "interrupted"
        assert original.performance_result["process_seconds"] > 0
        assert "cleanup_error" not in original.performance_result
        for process in owned:
            assert not process.is_running() or process.status() == psutil.STATUS_ZOMBIE
    finally:
        # Only handles captured from this test's worker and its reported child are eligible.
        for process in owned:
            try:
                process.kill()
            except psutil.NoSuchProcess:
                pass
        for process in launched:
            process.wait(timeout=5)
