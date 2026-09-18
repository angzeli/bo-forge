"""Benchmark evidence remains attributable, coherent, and read-only on failure."""

import json
import shutil
from unittest.mock import patch

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from benchmarks import runner
from benchmarks.__main__ import main
from benchmarks.figures import COLORS, _seed_lines
from benchmarks.report import _markdown, generate_report, load_evidence, tables
from benchmarks.storage import read_trace, sha256, write_json
from bo_forge import CampaignSession
from tests._benchmark_support import controlled_execute, controlled_suggest, spec_file


@pytest.fixture(scope="module")
def seed_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("benchmark-evidence")
    with patch.object(runner, "_execute_trial", controlled_execute):
        return runner.run_suite(spec_file(root), root / "run")


@pytest.fixture
def run(seed_run, tmp_path):
    return shutil.copytree(seed_run, tmp_path / "run")


def snapshot(run):
    return {p: p.read_bytes() for p in run.rglob("*") if p.is_file()}


@pytest.mark.parametrize("fault", ["missing", "malformed", "changed_log", "unscored", "pending"])
def test_complete_trials_require_coherent_campaign_evidence(run, tmp_path, fault):
    directory = next((run / "trials").iterdir())
    config, log = directory / "campaign.yaml", directory / "campaign.csv"
    manifest = directory / "campaign.csv.manifest.json"
    if fault == "missing":
        manifest.unlink()
    elif fault == "malformed":
        manifest.write_text("[]")
    elif fault == "changed_log":
        log.write_bytes(log.read_bytes() + b"\n")
    elif fault == "unscored":
        session = CampaignSession.from_files(config, log)
        suggestion = controlled_suggest(session)
        trial = json.loads((directory / "trial.json").read_text())
        if trial["strategy"] != "bo":
            suggestion["source"] = trial["strategy"]
        session.append_suggestions(suggestion)
        session.mark_observed(str(suggestion.iloc[0].row_id), 42.0)
    else:
        from bo_forge._campaign.provenance import _manifest_with_pending_transaction

        pending = _manifest_with_pending_transaction(
            json.loads(manifest.read_text()), config_file=config,
            operation="append_suggestions", affected_row_ids=["next_row"],
            metadata={"appended_row_count": 1}, resulting_hash="1" * 64, resulting_row_count=7,
        )
        write_json(manifest, pending)
    before = snapshot(run)
    with pytest.raises(ValueError, match="Incomplete campaign evidence"):
        generate_report(run, tmp_path / "report")
    assert before == snapshot(run)
    assert not (tmp_path / "report").exists()


def test_interrupted_scoring_exposes_persisted_but_unscored_observation(tmp_path, monkeypatch):
    import benchmarks.trial as trial_module

    original = trial_module.append_trace

    def interrupt(path, row):
        if row.get("evaluation") == 3:
            raise OSError("controlled trace-write failure after observation")
        original(path, row)

    monkeypatch.setattr(trial_module, "append_trace", interrupt)
    monkeypatch.setattr(runner, "_execute_trial", controlled_execute)
    run = runner.run_suite(spec_file(tmp_path), tmp_path / "run")
    before = snapshot(run)
    meta, traces, statuses = load_evidence(run)
    summary, _ = tables(meta, traces, statuses)
    assert statuses.status.eq("failed").all()
    assert statuses.completed_evaluations.eq(2).all()
    assert statuses.evidence_warning.str.contains("1 observed CSV row").all()
    assert summary.complete.eq(0).all() and summary.final_regret_median.isna().all()
    assert "Evidence warnings" in _markdown(meta, summary, statuses)
    assert before == snapshot(run)


def test_partial_missing_manifest_is_disclosed_without_promoting_success(run):
    directory = next((run / "trials").iterdir())
    (directory / "campaign.csv.manifest.json").unlink()
    status = json.loads((directory / "status.json").read_text())
    status["status"] = "failed"
    write_json(directory / "status.json", status)
    meta, traces, statuses = load_evidence(run)
    summary, _ = tables(meta, traces, statuses)
    assert statuses.loc[statuses.status.eq("failed"), "evidence_warning"].str.contains(
        "manifest is required"
    ).all()
    assert summary.complete.sum() == 5


@pytest.mark.parametrize("status_name", ["failed", "interrupted", "timeout"])
@pytest.mark.parametrize("fault", ["malformed", "changed_log", "wrong_path"])
def test_partial_contradictory_provenance_rejects_report(run, tmp_path, status_name, fault):
    directory = next((run / "trials").iterdir())
    status = json.loads((directory / "status.json").read_text())
    status["status"] = status_name
    write_json(directory / "status.json", status)
    manifest = directory / "campaign.csv.manifest.json"
    if fault == "malformed":
        manifest.write_text("[]")
    elif fault == "changed_log":
        log = directory / "campaign.csv"
        log.write_bytes(log.read_bytes() + b"\n")
    else:
        value = json.loads(manifest.read_text())
        value["paths"]["config"] = "different.yaml"
        write_json(manifest, value)
    before = snapshot(run)
    with pytest.raises(ValueError, match=directory.name):
        generate_report(run, tmp_path / "report")
    assert before == snapshot(run)
    assert not (tmp_path / "report").exists()


@pytest.mark.parametrize("state", ["previous", "resulting", "unknown"])
def test_partial_pending_provenance_is_disclosed_only_for_known_states(run, tmp_path, state):
    from bo_forge._campaign.provenance import _manifest_with_pending_transaction

    directory = next((run / "trials").iterdir())
    log, manifest = directory / "campaign.csv", directory / "campaign.csv.manifest.json"
    previous = log.read_bytes()
    resulting = previous + b"\n"
    payload = json.loads(manifest.read_text())
    pending = _manifest_with_pending_transaction(
        payload, config_file=directory / "campaign.yaml", operation="mark_observed",
        affected_row_ids=[pd.read_csv(log).row_id.iloc[-1]], metadata={},
        resulting_hash=sha256(resulting), resulting_row_count=payload["log"]["row_count"],
    )
    write_json(manifest, pending)
    if state != "previous":
        log.write_bytes(resulting if state == "resulting" else resulting + b"\n")
    status = json.loads((directory / "status.json").read_text())
    status["status"] = "interrupted"
    write_json(directory / "status.json", status)
    before = snapshot(run)
    if state == "unknown":
        with pytest.raises(ValueError, match="pending_unknown_state"):
            generate_report(run, tmp_path / "report")
        assert not (tmp_path / "report").exists()
    else:
        _, _, statuses = load_evidence(run)
        row = statuses.loc[statuses.trial_id.eq(directory.name)].iloc[0]
        assert f"pending_{state}_state" in row.evidence_warning
    assert before == snapshot(run)


@pytest.mark.parametrize("missing", ["campaign.yaml", "campaign.csv"])
def test_partial_absent_sources_remain_disclosed(run, missing):
    directory = next((run / "trials").iterdir())
    (directory / missing).unlink()
    (directory / "trace.jsonl").unlink()
    write_json(directory / "status.json", {"status": "interrupted", "completed_evaluations": 0})
    before = snapshot(run)
    _, _, statuses = load_evidence(run)
    assert statuses.loc[statuses.trial_id.eq(directory.name), "evidence_warning"].iloc[0]
    assert before == snapshot(run)


@pytest.mark.parametrize("field", ["trial_id", "problem", "mode", "strategy", "seed", "status"])
def test_trace_cannot_override_scheduled_identity(run, field):
    path = next((run / "trials").glob("*/trace.jsonl"))
    rows, _ = read_trace(path)
    rows[0][field] = "incorrect"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="reserved trial identity"):
        load_evidence(run)


@pytest.mark.parametrize("file", ["run.json", "trial.json", "status.json", "inputs.json"])
def test_non_object_metadata_has_controlled_cli_error(run, tmp_path, capsys, file):
    path = run / file if file == "run.json" else next((run / "trials").iterdir()) / file
    path.write_text("[]")
    before = snapshot(run)
    assert main(["report", "--run", str(run), "--output", str(tmp_path / "out")]) == 1
    assert "must be a JSON object" in capsys.readouterr().err
    assert before == snapshot(run)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("file,key,value", [
    ("status.json", "status", []), ("status.json", "message", {}),
    ("status.json", "wall_seconds", "slow"), ("status.json", "completed_evaluations", True),
    ("inputs.json", "bounds", None), ("inputs.json", "optimum", {}),
    ("trace.jsonl", "x", None), ("trace.jsonl", "row_id", []),
    ("trace.jsonl", "observed", {}), ("trace.jsonl", "fit_evidence", []),
    ("trace.jsonl", "source", "incorrect"),
])
def test_bad_field_shapes_have_controlled_cli_errors(run, tmp_path, capsys, file, key, value):
    path = next((run / "trials").iterdir()) / file
    if file == "trace.jsonl":
        rows, _ = read_trace(path)
        rows[0][key] = value
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    else:
        data = json.loads(path.read_text())
        data[key] = value
        write_json(path, data)
    assert main(["report", "--run", str(run), "--output", str(tmp_path / "out")]) == 1
    assert "Benchmark error:" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_report_seed_wording_uses_specification(run):
    meta, traces, statuses = load_evidence(run)
    summary, _ = tables(meta, traces, statuses)
    assert "Scheduled seed count: 1" in _markdown(meta, summary, statuses)
    meta["spec"]["seeds"] = [0, 1, 2, 3, 4]
    assert "Scheduled seed count: 5" in _markdown(meta, summary, statuses)
    assert "Five seeds" not in _markdown(meta, summary, statuses)


def test_seed_styles_are_distinct_and_partial_markers_are_hollow():
    rows = pd.DataFrame([
        {"trial_id": str(seed), "strategy": "bo", "seed": seed, "evaluation": 1,
         "simple_regret": 1.0, "status": "complete" if seed < 4 else "failed"}
        for seed in range(5)
    ])
    fig, ax = plt.subplots()
    try:
        with patch.object(ax, "plot", wraps=ax.plot) as plot:
            _seed_lines(ax, rows, list(range(5)))
        assert len({repr(call.kwargs["linestyle"]) for call in plot.call_args_list}) == 5
        assert all(line.get_color() == COLORS["bo"] for line in ax.lines)
        assert ax.lines[-1].get_markerfacecolor() == "white"
        assert all(line.get_marker() == "o" for line in ax.lines)
        assert [line.get_label() for line in ax.lines] == [
            f"bo, seed {seed}" + (" (partial)" if seed == 4 else "") for seed in range(5)
        ]
    finally:
        plt.close(fig)
