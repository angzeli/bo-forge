"""Missing artifacts cannot excuse contradictions in retained benchmark evidence."""

import json
import shutil
from unittest.mock import patch

import pytest

from benchmarks import runner
from benchmarks.designs import Proposals
from benchmarks.report import generate_report, load_evidence
from benchmarks.storage import append_trace, write_json
from benchmarks.trial import _external_row
from bo_forge import CampaignSession
from tests._benchmark_support import controlled_execute, spec_file
from tests.test_benchmark_multi_scores import multi_spec


@pytest.fixture(scope="module")
def seed_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("partial-integrity")
    spec = multi_spec("multi_fidelity")
    spec["strategies"] = ["bo"]
    with patch.object(runner, "_execute_trial", controlled_execute):
        return runner.run_suite(spec_file(root, spec), root / "run")


@pytest.fixture
def run(seed_run, tmp_path, monkeypatch):
    import benchmarks.figures

    monkeypatch.setattr(benchmarks.figures, "render_figures", lambda *args: None)
    return shutil.copytree(seed_run, tmp_path / "run")


def _trial(run):
    return next((run / "trials").iterdir())


def _partial(directory, count):
    write_json(directory / "status.json", {
        "status": "interrupted", "completed_evaluations": count,
        "wall_seconds": 1., "message": "Controlled interruption",
    })


def _reject_without_changes(run, output, match):
    before = {p: p.read_bytes() for p in run.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match=match):
        generate_report(run, output)
    assert not output.exists()
    assert all(p.read_bytes() == data for p, data in before.items())


@pytest.mark.parametrize("changed", [False, True])
def test_missing_config_still_checks_manifest_log_identity(run, tmp_path, changed):
    directory = _trial(run)
    (directory / "campaign.yaml").unlink()
    _partial(directory, 6)
    if changed:
        log = directory / "campaign.csv"
        log.write_bytes(log.read_bytes() + b"\n")
        _reject_without_changes(run, tmp_path / "report", "log_hash_changed")
    else:
        _, _, statuses = load_evidence(run)
        assert "campaign.yaml" in statuses.iloc[0].evidence_warning


@pytest.mark.parametrize("changed", [False, True])
def test_missing_inputs_still_checks_unscored_initialization(run, tmp_path, changed):
    directory = _trial(run)
    trial = json.loads((directory / "trial.json").read_text())
    for name in ("campaign.csv", "campaign.csv.manifest.json", "inputs.json",
                 "trace.jsonl", "events.jsonl"):
        (directory / name).unlink()
    campaign = CampaignSession.initialize(directory / "campaign.yaml", directory / "campaign.csv")
    proposals, initial = Proposals(trial), []
    for index in range(2):
        point = proposals.draw("sobol", initial)
        initial.append(point.copy())
        if changed:
            point[-1] = .5
        row = _external_row(campaign, point, index, "sobol")
        row_id = str(row.iloc[0].row_id)
        campaign.append_suggestions(row)
        append_trace(directory / "events.jsonl", {
            "operation": "submit", "row_id": row_id,
            "submission": index + 1, "pending_row_ids": [],
        })
        campaign.mark_observed(row_id, 10. + index)
        append_trace(directory / "events.jsonl", {
            "operation": "observe", "row_id": row_id,
            "submission": index + 1, "pending_row_ids": [],
        })
    _partial(directory, 0)
    if changed:
        _reject_without_changes(run, tmp_path / "report", "initial row 1.x3.*expected.*actual")
    else:
        _, _, statuses = load_evidence(run)
        assert "inputs.json absent" in statuses.iloc[0].evidence_warning


@pytest.mark.parametrize("payload", ["not JSON\n", '{"operation":"invented"}\n'])
def test_missing_csv_does_not_skip_present_event_validation(run, tmp_path, payload):
    directory = _trial(run)
    (directory / "campaign.csv").unlink()
    (directory / "trace.jsonl").unlink()
    (directory / "events.jsonl").write_text(payload)
    _partial(directory, 0)
    _reject_without_changes(run, tmp_path / "report", "(Malformed|workflow events)")


def test_missing_csv_retains_event_counts_and_reconciliation_warning(run):
    directory = _trial(run)
    (directory / "campaign.csv").unlink()
    (directory / "trace.jsonl").unlink()
    _partial(directory, 0)
    _, _, statuses = load_evidence(run)
    assert statuses.iloc[0].workflow_event_count == 12
    assert "cannot be reconciled" in statuses.iloc[0].evidence_warning


def test_missing_csv_rejects_events_beyond_scheduled_budget(run, tmp_path):
    directory = _trial(run)
    (directory / "campaign.csv").unlink()
    (directory / "trace.jsonl").unlink()
    for operation in ("submit", "observe"):
        append_trace(directory / "events.jsonl", {
            "operation": operation, "row_id": "extra-row", "submission": 7,
            "pending_row_ids": [],
        })
    _partial(directory, 0)
    _reject_without_changes(run, tmp_path / "report", "exceed the evaluation budget")


@pytest.mark.parametrize("alias", [False, True])
def test_report_rejects_resolved_trial_destination(run, alias):
    directory = _trial(run)
    relocated = run / "relocated-trial"
    directory.rename(relocated)
    directory.symlink_to(relocated, target_is_directory=True)
    output = (directory if alias else relocated) / "nested-report"
    _reject_without_changes(run, output, "separate from trial")


def test_truncated_trace_warning_is_in_markdown_and_trial_table(run, tmp_path):
    directory = _trial(run)
    trace = directory / "trace.jsonl"
    lines = trace.read_bytes().splitlines(keepends=True)
    trace.write_bytes(b"".join(lines[:5]) + b'{"evaluation":')
    _partial(directory, 5)
    before = {p: p.read_bytes() for p in directory.iterdir() if p.is_file()}
    output = generate_report(run, tmp_path / "report")
    warning = "Interrupted final trace record; partial evidence retained."
    assert warning in (output / "report.md").read_text()
    assert warning in (output / "trials.csv").read_text()
    assert all(p.read_bytes() == data for p, data in before.items())
