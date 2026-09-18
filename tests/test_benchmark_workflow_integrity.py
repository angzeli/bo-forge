"""Reject internally plausible but contradictory typed/workflow artifacts."""

import json
import shutil
from unittest.mock import patch

import pandas as pd
import pytest

from benchmarks import runner
from benchmarks.problems import score
from benchmarks.report import generate_report, load_evidence
from benchmarks.storage import read_trace, write_json
from tests._benchmark_support import spec_file
from tests.test_benchmark_extensions import execute, extended_spec


@pytest.fixture(scope="module")
def seed_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("pending-evidence")
    with patch.object(runner, "_execute_trial", execute):
        return runner.run_suite(spec_file(root, extended_spec("pending_noisy")), root / "run")


@pytest.fixture
def run(seed_run, tmp_path):
    return shutil.copytree(seed_run, tmp_path / "run")


@pytest.mark.parametrize("field,value", [("variables", []), ("dtype", "object"),
                                       ("initial_design_sha256", "0" * 64)])
@pytest.mark.parametrize("empty_trace", [False, True])
def test_typed_inputs_are_checked_even_without_scored_rows(run, field, value, empty_trace):
    directory = next((run / "trials").iterdir())
    inputs = json.loads((directory / "inputs.json").read_text())
    inputs[field] = value
    write_json(directory / "inputs.json", inputs)
    if empty_trace:
        (directory / "trace.jsonl").unlink()
        write_json(directory / "status.json", {"status": "failed", "completed_evaluations": 0})
    before = {p: p.read_bytes() for p in directory.iterdir() if p.is_file()}
    with pytest.raises(ValueError, match=f"inputs.{field}.*expected.*actual"):
        load_evidence(run)
    assert all(p.read_bytes() == b for p, b in before.items())


def test_trace_cannot_reorder_a_pending_pair_even_with_consistent_scores(run):
    directory = next((run / "trials").iterdir())
    rows, _ = read_trace(directory / "trace.jsonl")
    rows[4], rows[5] = rows[5], rows[4]
    inputs = json.loads((directory / "inputs.json").read_text())
    updated = []
    for index, row in enumerate(rows):
        row["evaluation"] = index + 1
        row.update(score(updated, row["latent"], row["observed"], inputs["optimum"],
                         inputs["optimum_tolerance"]))
        updated.append(row)
    (directory / "trace.jsonl").write_text("".join(json.dumps(r) + "\n" for r in updated))
    with pytest.raises(ValueError, match="trace observation order.*expected.*actual"):
        load_evidence(run)


def test_complete_baseline_cannot_claim_zero_accepted_proposals(run):
    directory = next((run / "trials").glob("*-random"))
    write_json(directory / "feasibility.json", {
        "draw_limit": 10000, "draws": 0, "infeasible": 0, "duplicate": 0,
    })
    with pytest.raises(ValueError, match="accepted proposal draws: expected 6..6; actual 0"):
        load_evidence(run)


def test_partial_truncated_workflow_tail_is_disclosed(run):
    directory = next((run / "trials").iterdir())
    with (directory / "events.jsonl").open("a") as stream:
        stream.write('{"operation":')
    path = directory / "status.json"
    status = json.loads(path.read_text())
    status["status"] = "failed"
    write_json(path, status)
    _, _, statuses = load_evidence(run)
    row = statuses.loc[statuses.trial_id.eq(directory.name)].iloc[0]
    assert "Interrupted final trace record" in row.evidence_warning


@pytest.mark.parametrize("operation,contradiction", [
    ("submit", False), ("accept", False), ("observe", False),
    ("submit", True), ("observe", True),
])
def test_persisted_mutation_can_lead_events_but_events_cannot_lead_csv(
    tmp_path, monkeypatch, operation, contradiction,
):
    from benchmarks import workflows
    from benchmarks.storage import append_trace

    original = workflows._event

    def interrupt(directory, action, row_id, index, pending):
        if action == operation and index == 4:
            raise OSError("interrupted after persistence, before event")
        return original(directory, action, row_id, index, pending)

    spec = extended_spec("pending_noisy")
    spec["strategies"] = ["bo"]
    monkeypatch.setattr(runner, "_execute_trial", execute)
    monkeypatch.setattr(workflows, "_event", interrupt)
    run = runner.run_suite(spec_file(tmp_path, spec), tmp_path / "run")
    directory = next((run / "trials").iterdir())
    frame = pd.read_csv(directory / "campaign.csv", keep_default_na=False)
    ids = frame.row_id.astype(str).tolist()
    if contradiction:
        # Add the persisted event, then falsely claim the next mutation completed.
        tail = {
            "submit": [("submit", 4, []), ("accept", 4, [ids[4]])],
            "observe": [("observe", 4, ids[5:]), ("observe", 5, [])],
        }[operation]
        for action, index, pending in tail:
            append_trace(directory / "events.jsonl", {
                "operation": action, "row_id": ids[index],
                "submission": index + 1, "pending_row_ids": pending,
            })
        before = {p: p.read_bytes() for p in directory.iterdir() if p.is_file()}
        with pytest.raises(ValueError, match="(status|review_status)"):
            generate_report(run, tmp_path / "report")
        assert all(p.read_bytes() == data for p, data in before.items())
        assert not (tmp_path / "report").exists()
    else:
        before = {p: p.read_bytes() for p in directory.iterdir() if p.is_file()}
        _, _, statuses = load_evidence(run)
        assert "persisted mutation" in statuses.iloc[0].evidence_warning
        assert all(p.read_bytes() == data for p, data in before.items())
