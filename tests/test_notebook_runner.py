"""Small real-kernel fixtures for source-only notebook assurance, not tutorial fits."""

import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from types import SimpleNamespace

import nbformat
import pytest

from notebook_assurance.aggregate import aggregate
from notebook_assurance.archive import extract
from notebook_assurance.runner import run
from notebook_assurance.worker import check_cells

NAME = "01_synthetic.ipynb"
ROOT = Path(__file__).resolve().parents[1]


def archive(tmp_path, cells, *, cell_timeout=20, notebook_timeout=45, omit_core=False):
    root = tmp_path / "fixture"
    root.mkdir()
    shutil.copytree(ROOT / "notebook_assurance", root / "notebook_assurance")
    (root / "pyproject.toml").write_text('[project]\nname="bo-forge"\nversion="3.3.3"\n')
    for package in ("bo_forge", "benchmarks"):
        if omit_core and package == "bo_forge":
            continue
        (root / package).mkdir()
        (root / package / "__init__.py").write_text('__version__ = "3.3.3"\n')
    (root / "notebooks").mkdir()
    (root / "configs").mkdir()
    (root / "configs/seed.yaml").write_text("untouched: true\n")
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(s) for s in cells])
    nbformat.write(notebook, root / "notebooks" / NAME)
    (root / "notebook_assurance/contracts.py").write_text(
        f"CONTRACTS = {{{NAME!r}: {{'cell_timeout': {cell_timeout}, "
        f"'notebook_timeout': {notebook_timeout}}}}}\n"
        f"def select(*args): return [{NAME!r}]\n"
        "def validate(root, captures, name):\n"
        "    assert (root / 'done').exists(), 'Declared outcome not reached'\n"
        "    assert (root / 'done').read_text() == 'yes', 'Declared outcome not reached'\n"
        "    return {'outcome': True}\n"
    )
    path = tmp_path / "source.tar.gz"
    with tarfile.open(path, "w:gz") as output:
        output.add(root, arcname="bo_forge-3.3.3")
    return path


@pytest.fixture
def synthetic_registry(monkeypatch):
    monkeypatch.setattr("notebook_assurance.runner.CONTRACTS", {
        NAME: {"cell_timeout": 20, "notebook_timeout": 45}})
    monkeypatch.setattr("notebook_assurance.runner.select", lambda *args: [NAME])
    monkeypatch.setattr("notebook_assurance.aggregate.select", lambda *args: [NAME])
    def validate(root, captures, name):
        assert (root / 'done').read_text() == 'yes'
        return {"outcome": True}
    monkeypatch.setattr("notebook_assurance.aggregate.validate", validate)


def test_real_kernel_fresh_sources_identity_and_aggregation(tmp_path, synthetic_registry):
    source = archive(tmp_path, [
        "from pathlib import Path\nimport sys\nassert not Path('done').exists()\n"
        "Path('done').write_text('yes')\nprint(sys.executable)"
    ])
    for name in ("first", "second"):
        result = run(source, "full", tmp_path / name)
        assert result["status"] == "passed"
        directory = tmp_path / name / Path(NAME).stem
        environment = json.loads((directory / "environment.json").read_text())
        assert "/source/" in environment["imports"]["bo_forge"]
        assert str(ROOT / "bo_forge") not in environment["imports"]["bo_forge"]
        assert result["notebooks"][0]["completed_cells"] == 1
    combined = aggregate(source, "full", tmp_path / "first", tmp_path / "combined")
    assert combined["status"] == "passed"
    with pytest.raises(FileExistsError):
        run(source, "full", tmp_path / "first")


@pytest.mark.parametrize("code,message", [
    ("raise RuntimeError('visible failure')", "visible failure"),
    ("print('no campaign outcome')", "Declared outcome not reached"),
    ("from pathlib import Path\nPath('configs/seed.yaml').write_text('bad')", "read-only"),
])
def test_failures_retain_evidence_and_protect_inputs(tmp_path, synthetic_registry, code, message):
    source = archive(tmp_path, ["print('checkpoint')", code])
    output = tmp_path / "run"
    result = run(source, "full", output)
    assert result["status"] == "failed"
    state = result["notebooks"][0]
    assert message in state["message"]
    assert state["completed_cells"] >= 1
    directory = output / Path(NAME).stem
    executed = nbformat.read(directory / "executed.ipynb", as_version=4)
    assert executed.cells[0].execution_count is not None
    seed = directory / "source/bo_forge-3.3.3/configs/seed.yaml"
    assert seed.read_text() == "untouched: true\n"


def test_timeout_terminates_owned_child_and_keeps_checkpoint(
    tmp_path, synthetic_registry, monkeypatch,
):
    import psutil

    monkeypatch.setattr("notebook_assurance.runner.CONTRACTS", {
        NAME: {"cell_timeout": 60, "notebook_timeout": 20}})
    source = archive(tmp_path, ["print('first completed')", (
        "import subprocess, sys, time\nfrom pathlib import Path\n"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
        "Path('child.pid').write_text(str(p.pid))\ntime.sleep(120)"
    )])
    child_file = (tmp_path / "run" / Path(NAME).stem
                  / "source/bo_forge-3.3.3/child.pid")
    started = time.monotonic()

    def clock():
        # Test the active-cell deadline, independent of cold font-cache startup latency.
        return 21.0 if child_file.exists() or time.monotonic() - started > 45 else 0.0

    monkeypatch.setattr("notebook_assurance.runner.time", SimpleNamespace(
        monotonic=clock, sleep=time.sleep))
    result = run(source, "full", tmp_path / "run")
    record = result["notebooks"][0]
    assert record["status"] == "timeout"
    assert record["completed_cells"] == 1
    child = int(child_file.read_text())
    assert not psutil.pid_exists(child) or psutil.Process(child).status() == psutil.STATUS_ZOMBIE
    assert record["elapsed_seconds"] >= 20


def test_capture_precedes_unmodified_cleanup(tmp_path, synthetic_registry):
    source = archive(tmp_path, [
        "from tempfile import TemporaryDirectory\nfrom pathlib import Path\n"
        "workspace=TemporaryDirectory(prefix='bo-forge-predictive-')\n"
        "p=Path(workspace.name)\n(p/'evidence.txt').write_text('retained')",
        "workspace.cleanup()\nassert not p.exists()\nPath('done').write_text('yes')",
    ])
    output = tmp_path / "run"
    assert run(source, "full", output)["status"] == "passed"
    captured = list((output / Path(NAME).stem / "captures").glob("*/evidence.txt"))
    assert len(captured) == 1 and captured[0].read_text() == "retained"


def test_guard_allows_atomic_working_file_write_beside_seed(tmp_path, synthetic_registry):
    source = archive(tmp_path, [
        "from tempfile import NamedTemporaryFile\nfrom pathlib import Path\n"
        "with NamedTemporaryFile(mode='w', dir='configs', delete=False) as handle:\n"
        "    handle.write('working')\n"
        "Path(handle.name).replace('configs/working.yaml')\nPath('done').write_text('yes')",
    ])
    assert run(source, "full", tmp_path / "run")["status"] == "passed"


@pytest.mark.parametrize("tag", ["skip-execution", "raises-exception", "allow-errors"])
def test_execution_escape_tags_fail(tag):
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("1")])
    notebook.cells[0].metadata.tags = [tag]
    with pytest.raises(ValueError, match="tag"):
        check_cells(notebook)


@pytest.mark.parametrize("name,kind", [("../escape", "file"), ("/escape", "file"),
                                        ("root/link", "link"), ("root/fifo", "fifo")])
def test_archive_rejects_unsafe_content(tmp_path, name, kind):
    source = tmp_path / "bad.tar.gz"
    member = tarfile.TarInfo(name)
    if kind == "link":
        member.type, member.linkname = tarfile.SYMTYPE, "/tmp"
    elif kind == "fifo":
        member.type = tarfile.FIFOTYPE
    with tarfile.open(source, "w:gz") as output:
        output.addfile(member, io.BytesIO())
    with pytest.raises(ValueError, match="Unsafe"):
        extract(source, tmp_path / "extracted")
    assert not (tmp_path / "extracted").exists()


def test_aggregate_missing_jobs_unknown_time_and_mismatched_archive(tmp_path, synthetic_registry):
    source = archive(tmp_path, ["pass"])
    evidence = tmp_path / "empty"
    evidence.mkdir()
    result = aggregate(source, "full", evidence, tmp_path / "missing")
    assert result["status"] == "failed"
    assert result["notebooks"][0]["elapsed_seconds"] is None
    (evidence / "result.json").write_text(json.dumps({"archive_sha256": "wrong"}))
    with pytest.raises(ValueError, match="Mismatched"):
        aggregate(source, "full", evidence, tmp_path / "mismatch")


def test_worker_limits_keep_parent_environment_unmodified(tmp_path, synthetic_registry):
    from notebook_assurance.runner import prepare

    source = archive(tmp_path, ["pass"])
    destination = tmp_path / "one"
    destination.mkdir()
    before = dict(os.environ)
    _, env = prepare(destination, source, NAME)
    assert env["OMP_NUM_THREADS"] == env["OPENBLAS_NUM_THREADS"] == "1"
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert str(destination) in env["TMPDIR"]
    assert os.environ == before


def test_sigterm_preserves_partial_execution_and_stops_children(tmp_path):
    import psutil

    source = archive(tmp_path, [
        "print('checkpoint')",
        "import subprocess,sys,time\nfrom pathlib import Path\n"
        "p=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
        "Path('child.pid').write_text(str(p.pid))\ntime.sleep(120)",
    ], cell_timeout=60, notebook_timeout=90)
    output = tmp_path / "cancelled"
    env = dict(os.environ, PYTHONPATH=str(tmp_path / "fixture"))
    process = subprocess.Popen([
        sys.executable, "-m", "notebook_assurance", "run", "--sdist", str(source),
        "--profile", "full", "--output", str(output),
    ], cwd=tmp_path / "fixture", env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    child_file = output / Path(NAME).stem / "source/bo_forge-3.3.3/child.pid"
    try:
        # A fresh inline backend may build its font cache before the test cell starts.
        deadline = time.monotonic() + 60
        while not child_file.exists() and time.monotonic() < deadline:
            time.sleep(0.1)
        assert child_file.exists()
        time.sleep(0.5)  # Permit one descendant-tracking poll after the child starts.
        process.send_signal(signal.SIGTERM)
        _, stderr = process.communicate(timeout=20)
        assert process.returncode == 143, stderr.decode()
        result = json.loads((output / "result.json").read_text())
        state = result["notebooks"][0]
        assert result["status"] == state["status"] == "interrupted"
        assert state["completed_cells"] == 1 and state["elapsed_seconds"] > 0
        child = int(child_file.read_text())
        assert (not psutil.pid_exists(child)
                or psutil.Process(child).status() == psutil.STATUS_ZOMBIE)
    finally:
        if process.poll() is None:
            process.terminate()
            process.communicate(timeout=20)


def test_aggregation_rejects_missing_uploaded_execution_artifact(tmp_path, synthetic_registry):
    source = archive(tmp_path, ["from pathlib import Path\nPath('done').write_text('yes')"])
    output = tmp_path / "run"
    assert run(source, "full", output)["status"] == "passed"
    (output / Path(NAME).stem / "executed.ipynb").unlink()
    with pytest.raises(FileNotFoundError):
        aggregate(source, "full", output, tmp_path / "aggregate")


def test_cleanup_keeps_attempting_after_inspection_failure():
    from notebook_assurance.runner import cleanup

    class Owned:
        def stop(self):
            raise PermissionError("inspection unavailable")

    class Process:
        pid = -99999999
        waited = False

        def poll(self):
            return 0

        def wait(self, timeout):
            self.waited = True

    process = Process()
    warnings = cleanup(process, Owned())
    assert process.waited and "inspection unavailable" in warnings[0]


@pytest.mark.parametrize("fault", ["outcome", "status", "environment-final.json", "source"])
def test_aggregate_reconciles_retained_outcomes(tmp_path, synthetic_registry, fault):
    source = archive(tmp_path, ["from pathlib import Path\nPath('done').write_text('yes')"])
    output = tmp_path / "run"
    assert run(source, "full", output)["status"] == "passed"
    directory = output / Path(NAME).stem
    if fault == "outcome":
        (directory / "source/bo_forge-3.3.3/done").unlink()
    elif fault == "status":
        status = json.loads((directory / "status.json").read_text())
        status["status"] = "failed"
        (directory / "status.json").write_text(json.dumps(status))
    elif fault == "source":
        seed = directory / "source/bo_forge-3.3.3/configs/seed.yaml"
        seed.chmod(0o600)
        seed.write_text("modified: true\n")
    else:
        (directory / fault).unlink()
    with pytest.raises((ValueError, OSError)):
        aggregate(source, "full", output, tmp_path / "aggregate")


def test_benchmark_workload_is_protected(tmp_path):
    from notebook_assurance.archive import check_protected, protected_inputs

    spec = tmp_path / "benchmarks/specs/smoke.yaml"
    spec.parent.mkdir(parents=True)
    spec.write_text("evaluations: 6\n")
    identities = protected_inputs(tmp_path)
    assert str(spec.resolve()) in identities
    spec.write_text("evaluations: 4\n")
    with pytest.raises(ValueError, match="input changed"):
        check_protected(identities)


def test_kernel_rejects_import_from_original_checkout(tmp_path, synthetic_registry):
    source = archive(tmp_path, ["raise AssertionError('must not execute tutorial')"],
                     omit_core=True)
    result = run(source, "full", tmp_path / "run")
    assert result["status"] == "failed"
    state = result["notebooks"][0]
    assert state["completed_cells"] == 0
    assert state["exception_type"] == "CellExecutionError"
    # Either the developer editable install is rejected or an independent environment
    # correctly cannot import the deliberately absent core at all.
    assert "bo_forge" in state["message"]


def test_per_cell_timeout_keeps_prior_cell_evidence(tmp_path, synthetic_registry, monkeypatch):
    monkeypatch.setattr("notebook_assurance.runner.CONTRACTS", {
        NAME: {"cell_timeout": 1, "notebook_timeout": 45}})
    source = archive(tmp_path, ["print('complete')", "import time; time.sleep(120)"])
    result = run(source, "full", tmp_path / "run")
    state = result["notebooks"][0]
    assert state["status"] == "timeout"
    assert state["exception_type"] == "CellTimeoutError"
    assert state["failing_cell_index"] == 1
    assert state["completed_cells"] == 1


@pytest.mark.skipif(os.name != "posix", reason="POSIX kernel process groups")
def test_frozen_groups_are_killed_when_final_inspection_fails(monkeypatch):
    from notebook_assurance.processes import OwnedProcesses

    owned = OwnedProcesses.__new__(OwnedProcesses)
    owned.members, owned.groups, owned.frozen = {}, {77}, set()
    calls = []

    def refresh():
        if owned.frozen:
            raise PermissionError("process inspection failed after freeze")

    def freeze(signum):
        assert signum == signal.SIGSTOP
        owned.frozen.add(77)

    monkeypatch.setattr(owned, "refresh", refresh)
    monkeypatch.setattr(owned, "signal_groups", freeze)
    monkeypatch.setattr(os, "killpg", lambda group, signum: calls.append((group, signum)))
    with pytest.raises(PermissionError, match="after freeze"):
        owned.stop()
    assert calls == [(77, signal.SIGKILL)]


def test_kernel_renders_display_only_figures(tmp_path, synthetic_registry):
    source = archive(tmp_path, [
        "import matplotlib.pyplot as plt\nplt.plot([0,1], [1,2]);",
        "from pathlib import Path\nPath('done').write_text('yes')",
    ])
    output = tmp_path / "run"
    assert run(source, "full", output)["status"] == "passed"
    notebook = nbformat.read(output / Path(NAME).stem / "executed.ipynb", as_version=4)
    assert any("image/png" in item.get("data", {}) for item in notebook.cells[0].outputs)


def test_interrupted_checkpoint_preserves_previous_notebook(tmp_path, monkeypatch):
    from notebook_assurance.worker import checkpoint_notebook

    path = tmp_path / "executed.ipynb"
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("print('done')")])
    checkpoint_notebook(notebook, path)
    previous = path.read_bytes()

    def broken_write(notebook, temporary):
        temporary.write_text("{partial")
        raise OSError("interrupted write")

    monkeypatch.setattr(nbformat, "write", broken_write)
    with pytest.raises(OSError, match="interrupted write"):
        checkpoint_notebook(notebook, path)
    assert path.read_bytes() == previous
