"""Aggregation failure reports and concise CLI diagnostics without tutorial runs."""

import json
import shutil
import tarfile
from pathlib import Path

import nbformat
import pytest

from notebook_assurance.__main__ import main
from notebook_assurance.aggregate import aggregate
from notebook_assurance.archive import digest, write_json

NAME = "01_fixture.ipynb"


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    root = tmp_path / "package"
    (root / "notebooks").mkdir(parents=True)
    (root / "pyproject.toml").write_text('[project]\nname="fixture"\n')
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("pass")])
    nbformat.write(notebook, root / "notebooks" / NAME)
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        stream.add(root, arcname=root.name)
    bundle = tmp_path / "evidence"
    directory = bundle / Path(NAME).stem
    shutil.copytree(root, directory / "source" / root.name)
    (directory / "captures").mkdir()
    notebook.cells[0].execution_count = 1
    nbformat.write(notebook, directory / "executed.ipynb")
    environment = {"executable": "/python", "imports": {
        name: str(directory / "source" / root.name / name / "__init__.py")
        for name in ("bo_forge", "benchmarks")}}
    for name in ("environment.json", "environment-final.json"):
        write_json(directory / name, environment)
    (directory / "worker.log").write_text("")
    record = {"notebook": NAME, "status": "passed", "completed_cells": 1,
              "completion_checks": {"done": True}, "rendered_figures": 0,
              "notebook_sha256": digest(root / "notebooks" / NAME),
              "failing_cell_index": None, "failing_cell_id": None}
    write_json(directory / "status.json", record)
    write_json(bundle / "result.json", {
        "schema_version": 1, "archive_sha256": digest(archive), "profile": "full",
        "scheduled": [NAME], "selected": [NAME], "status": "passed",
        "notebooks": [{**record, "directory": directory.name}]})
    monkeypatch.setattr("notebook_assurance.aggregate.select", lambda *args: [NAME])
    monkeypatch.setattr("notebook_assurance.aggregate.validate", lambda *args: {"done": True})
    return archive, bundle, directory


def retained_failure(evidence, tmp_path, exception=ValueError):
    archive, bundle, _ = evidence
    output = tmp_path / "aggregate"
    with pytest.raises(exception):
        aggregate(archive, "full", bundle, output)
    result = json.loads((output / "aggregate.json").read_text())
    assert result["status"] == "failed"
    assert result["scheduled"] == [NAME]
    assert len(result["errors"]) == 1
    assert result["errors"][0]["evidence_path"]
    assert result["errors"][0]["exception_type"]
    return result


@pytest.mark.parametrize("filename", ["result.json", "status.json", "environment.json",
                                      "executed.ipynb"])
def test_corrupt_json_is_durable(evidence, tmp_path, filename):
    _, bundle, directory = evidence
    target = (bundle if filename == "result.json" else directory) / filename
    target.write_text("{broken")
    result = retained_failure(evidence, tmp_path)
    assert result["notebooks"][0]["status"] != "passed"


@pytest.mark.parametrize("field,value", [("archive_sha256", "wrong"), ("profile", "pr"),
                                       ("scheduled", [])])
def test_mismatch_records_before_raising(evidence, tmp_path, field, value):
    path = evidence[1] / "result.json"
    payload = json.loads(path.read_text())
    payload[field] = value
    write_json(path, payload)
    result = retained_failure(evidence, tmp_path)
    assert "Mismatched" in result["errors"][0]["message"]


@pytest.mark.parametrize("filename", ["executed.ipynb", "status.json", "environment.json",
                                      "environment-final.json", "worker.log", "source"])
def test_missing_evidence_records_before_raising(evidence, tmp_path, filename):
    target = evidence[2] / filename
    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()
    result = retained_failure(evidence, tmp_path, (ValueError, FileNotFoundError))
    assert result["notebooks"][0]["status"] == "failed"
    assert result["errors"][0]["notebook"] == NAME
    if filename == "executed.ipynb":
        assert result["errors"][0]["exception_type"] == "FileNotFoundError"


@pytest.mark.parametrize("payload", [[], None, {"notebooks": []}])
def test_invalid_summary_shape_is_durable(evidence, tmp_path, payload):
    write_json(evidence[1] / "result.json", payload)
    retained_failure(evidence, tmp_path)


@pytest.mark.parametrize("fault", [
    "record", "later_record", "status", "duplicate", "run_status", "nan",
])
def test_invalid_records_cannot_pass(evidence, tmp_path, fault):
    path = evidence[1] / "result.json"
    payload = json.loads(path.read_text())
    if fault == "record":
        payload["notebooks"] = [None]
    elif fault == "later_record":
        payload["notebooks"].append(None)
    elif fault == "status":
        payload["notebooks"][0]["status"] = "unknown"
    elif fault == "duplicate":
        payload["notebooks"] *= 2
    elif fault == "run_status":
        payload["status"] = "failed"
    else:
        payload["notebooks"][0]["elapsed_seconds"] = float("nan")
    path.write_text(json.dumps(payload))
    result = retained_failure(evidence, tmp_path)
    if fault == "later_record":
        assert result["errors"][0]["notebook"] is None
        assert result["notebooks"][0]["status"] == "not_run"


def test_corruption_after_valid_bundle_cannot_pass(evidence, tmp_path):
    extra = evidence[1] / "zzz"
    extra.mkdir()
    (extra / "result.json").write_text("{")
    result = retained_failure(evidence, tmp_path)
    assert result["notebooks"][0]["status"] == "passed"
    assert result["status"] == "failed"


def test_missing_bundle_and_empty_schedule_fail_closed(evidence, tmp_path, monkeypatch):
    archive, bundle, _ = evidence
    (bundle / "result.json").unlink()
    result = aggregate(archive, "full", bundle, tmp_path / "empty")
    assert result["status"] == "failed"
    assert result["notebooks"][0]["status"] == "not_run"
    monkeypatch.setattr("notebook_assurance.aggregate.select", lambda *args: [])
    with pytest.raises(ValueError, match="Empty"):
        aggregate(archive, "full", bundle, tmp_path / "no-schedule")
    assert json.loads((tmp_path / "no-schedule/aggregate.json").read_text())["status"] == "failed"


def test_bad_archive_retains_failure(evidence, tmp_path):
    evidence[0].write_bytes(b"not a tar archive")
    with pytest.raises(tarfile.ReadError):
        aggregate(evidence[0], "full", evidence[1], tmp_path / "bad-archive")
    result = json.loads((tmp_path / "bad-archive/aggregate.json").read_text())
    assert result["status"] == "failed" and result["errors"]


def test_selection_failure_retains_diagnostics(evidence, tmp_path, monkeypatch):
    def fail(*args):
        raise ValueError("Notebook registry mismatch")
    monkeypatch.setattr("notebook_assurance.aggregate.select", fail)
    output = tmp_path / "bad-selection"
    with pytest.raises(ValueError, match="registry mismatch"):
        aggregate(evidence[0], "full", evidence[1], output)
    result = json.loads((output / "aggregate.json").read_text())
    assert result["status"] == "failed"
    assert result["notebooks"] == []
    assert result["errors"][0]["message"] == "Notebook registry mismatch"
    assert result["errors"][0]["evidence_path"] == str(evidence[0])


@pytest.mark.parametrize("original_failure", [True, False])
def test_report_write_failure_preserves_original(evidence, tmp_path, monkeypatch, original_failure):
    if original_failure:
        (evidence[1] / "result.json").write_text("{")

    def fail(*args):
        raise OSError("report disk unavailable")

    monkeypatch.setattr("notebook_assurance.aggregate.write_json", fail)
    expected = json.JSONDecodeError if original_failure else OSError
    with pytest.raises(expected) as caught:
        aggregate(evidence[0], "full", evidence[1], tmp_path / "write-failure")
    if original_failure:
        assert "report disk unavailable" in caught.value.__notes__[0]
    else:
        assert str(caught.value) == "report disk unavailable"


def arguments(evidence, output, command="aggregate"):
    args = [command, "--sdist", str(evidence[0]), "--profile", "full", "--output", str(output)]
    return args + (["--evidence", str(evidence[1])] if command == "aggregate" else [])


def test_cli_corruption_is_concise_and_reports_retained_path(evidence, tmp_path, capsys):
    (evidence[1] / "result.json").write_text("{")
    output = tmp_path / "aggregate"
    assert main(arguments(evidence, output)) == 1
    captured = capsys.readouterr()
    assert "failed=0, not_run=1, errors=1" in captured.out
    assert str(output / "aggregate.json") in captured.out
    assert NAME in captured.out and "cell_index=None" in captured.out
    assert f"cell_id=None; evidence: {output / 'aggregate.json'}" in captured.out
    assert "new --output" in captured.err
    assert "Traceback" not in captured.out + captured.err


@pytest.mark.parametrize("command", ["run", "aggregate"])
def test_cli_failed_cell_summary(evidence, tmp_path, monkeypatch, capsys, command):
    result = {"status": "failed", "notebooks": [{
        "notebook": NAME, "status": "failed", "directory": Path(NAME).stem,
        "failing_cell_index": 0, "failing_cell_id": "cell-zero",
        "message": "failure\n" * 1000, "traceback": "DO NOT PRINT"}]}
    monkeypatch.setattr(f"notebook_assurance.__main__.{command}", lambda *args: result)
    output = tmp_path / "attempt"
    assert main(arguments(evidence, output, command)) == 1
    captured = capsys.readouterr()
    assert "failed=1" in captured.out
    assert "cell_index=0, cell_id=cell-zero" in captured.out
    assert str(output / Path(NAME).stem) in captured.out
    assert "DO NOT PRINT" not in captured.out and len(captured.out) < 1500
    assert "fresh output directory" in captured.err


def test_cli_unexpected_error_has_no_traceback(evidence, tmp_path, monkeypatch, capsys):
    def fail(*args):
        raise RuntimeError("details\n" * 1000)
    monkeypatch.setattr("notebook_assurance.__main__.run", fail)
    assert main(arguments(evidence, tmp_path / "attempt", "run")) == 1
    captured = capsys.readouterr()
    assert "RuntimeError" in captured.err and "Traceback" not in captured.err
    assert len(captured.err) < 1000


def test_cli_cell_error_shows_cause_and_preserves_full_message(evidence, tmp_path, capsys):
    path = evidence[1] / "result.json"
    payload = json.loads(path.read_text())
    message = ("An error occurred while executing the following cell:\n"
               + "echoed_source = 'not the cause'\n" * 30
               + "\x1b[31mRuntimeError\x1b[0m: genuine failure\n  \n")
    payload["status"] = "failed"
    payload["notebooks"][0].update(
        status="failed", exception_type="CellExecutionError", message=message,
        failing_cell_index=0, failing_cell_id="cell-zero")
    write_json(path, payload)
    output = tmp_path / "failed-cell"
    assert main(arguments(evidence, output)) == 1
    captured = capsys.readouterr()
    assert "CellExecutionError: RuntimeError: genuine failure" in captured.out
    assert "echoed_source" not in captured.out and "\x1b" not in captured.out
    retained = json.loads((output / "aggregate.json").read_text())
    assert retained["notebooks"][0]["message"] == message


@pytest.mark.parametrize("command", ["run", "aggregate"])
def test_existing_destination_is_untouched(evidence, tmp_path, capsys, command):
    output = tmp_path / "existing"
    output.mkdir()
    report = output / "aggregate.json"
    report.write_bytes(b"prior evidence\n")
    assert main(arguments(evidence, output, command)) == 1
    captured = capsys.readouterr()
    assert "Nothing overwritten" in captured.err and "new --output" in captured.err
    assert report.read_bytes() == b"prior evidence\n"
    assert list(output.iterdir()) == [report]
    with pytest.raises(FileExistsError):
        aggregate(evidence[0], "full", evidence[1], output)
    assert report.read_bytes() == b"prior evidence\n"


def test_cli_success_and_new_destination_retry(evidence, tmp_path, capsys):
    output = tmp_path / "aggregate"
    assert main(arguments(evidence, output)) == 0
    assert "passed=1, failed=0, not_run=0, errors=0" in capsys.readouterr().out
    original = (output / "aggregate.json").read_bytes()
    assert main(arguments(evidence, output)) == 1
    assert main(arguments(evidence, tmp_path / "retry")) == 0
    assert (output / "aggregate.json").read_bytes() == original
