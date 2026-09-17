"""Read-only benchmark reporting, denominators, partial evidence, and CLI errors."""

import json

import pandas as pd
import pytest

from benchmarks import runner
from benchmarks.__main__ import main
from benchmarks.report import generate_report, load_evidence, tables
from benchmarks.storage import read_trace, write_json
from tests._benchmark_support import controlled_execute, spec_file


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "_execute_trial", controlled_execute)
    return runner.run_suite(spec_file(tmp_path), tmp_path / "run")


def test_complete_report_is_read_only_and_never_evaluates_or_fits(run, tmp_path, monkeypatch):
    import benchmarks.problems as problems
    import bo_forge.models as models

    before = {p: p.read_bytes() for p in run.rglob("*") if p.is_file()}

    def forbidden(*args, **kwargs):
        pytest.fail("Reporting must not evaluate objectives or fit models.")

    monkeypatch.setattr(problems, "true_value", forbidden)
    monkeypatch.setattr(problems, "problem_for", forbidden)
    monkeypatch.setattr(models, "fit_gp_model", forbidden)
    output = generate_report(run, tmp_path / "report")
    summary = pd.read_csv(output / "summary.csv")
    trajectories = pd.read_csv(output / "trajectories.csv")
    assert len(summary) == 6
    assert summary.complete.eq(1).all() and summary.scheduled.eq(1).all()
    assert summary.failed.eq(0).all()
    assert trajectories.contributing_complete.eq(1).all()
    assert trajectories.contributing_partial.eq(0).all()
    assert len(list((output / "figures").glob("*.png"))) == 2
    assert "conditioned" in (output / "report.md").read_text()
    assert {p: p.read_bytes() for p in before} == before
    with pytest.raises(FileExistsError):
        generate_report(run, output)


def test_partial_trials_remain_in_denominators_without_extrapolation(run, tmp_path):
    directory = next((run / "trials").glob("*-bo"))
    trace_path = directory / "trace.jsonl"
    lines = trace_path.read_text().splitlines()
    trace_path.write_text("\n".join(lines[:4]) + "\n")
    write_json(directory / "status.json", {
        "status": "timeout", "completed_evaluations": 4, "wall_seconds": 600,
        "message": "controlled timeout",
    })
    metadata, traces, statuses = load_evidence(run)
    summary, trajectories = tables(metadata, traces, statuses)
    row = summary.loc[summary.timeout.eq(1)].iloc[0]
    assert row.scheduled == 1 and row.complete == 0
    assert pd.isna(row.final_regret_median)
    curve = trajectories.loc[trajectories.problem.eq(row.problem)
                             & trajectories["mode"].eq(row["mode"])
                             & trajectories.strategy.eq(row.strategy)]
    assert curve.scheduled.eq(1).all() and curve.contributing_complete.eq(0).all()
    assert curve.contributing_partial.tolist() == [1, 1, 1, 1, 0, 0]
    assert curve["median"].isna().all()
    partial = traces.loc[traces.trial_id.eq(directory.name)]
    assert partial.evaluation.tolist() == [1, 2, 3, 4]
    report = generate_report(run, tmp_path / "partial-report")
    assert "controlled timeout" in (report / "report.md").read_text()


def test_all_failed_report_has_explicit_empty_figures_and_missing_metrics(run, tmp_path):
    for directory in (run / "trials").iterdir():
        (directory / "trace.jsonl").unlink()
        write_json(directory / "status.json", {"status": "failed", "completed_evaluations": 0,
                                               "message": "controlled failure", "wall_seconds": 1})
    output = generate_report(run, tmp_path / "all-failed")
    summary = pd.read_csv(output / "summary.csv")
    assert summary.failed.eq(1).all() and summary.final_regret_median.isna().all()
    assert len(list((output / "figures").glob("*.png"))) == 2


@pytest.mark.parametrize("fault", ["score", "source", "count", "status", "spec", "trial"])
def test_inconsistent_evidence_is_not_silently_reported_as_valid(run, tmp_path, fault):
    directory = next((run / "trials").iterdir())
    if fault == "score":
        rows, _ = read_trace(directory / "trace.jsonl")
        rows[-1]["simple_regret"] = -200
        (directory / "trace.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    elif fault == "source":
        path = directory / "campaign.csv"
        data = pd.read_csv(path)
        data.loc[0, "outcome"] += 5
        data.to_csv(path, index=False)
    elif fault == "count":
        write_json(directory / "status.json", {"status": "complete", "completed_evaluations": 5})
    elif fault == "status":
        write_json(directory / "status.json", {"status": "running", "completed_evaluations": 6})
    elif fault == "spec":
        (run / "spec.yaml").write_text("invalid identity")
    else:
        write_json(directory / "trial.json", {"trial_id": "wrong"})
    output = tmp_path / "report"
    with pytest.raises(ValueError):
        generate_report(run, output)
    assert not output.exists()


def test_report_destination_must_be_separate_from_trial_sources(run):
    directory = next((run / "trials").iterdir())
    before = {p: p.read_bytes() for p in directory.iterdir() if p.is_file()}
    with pytest.raises(ValueError, match="separate"):
        generate_report(run, directory / "new-report")
    alias = run / "alias.png"
    alias.symlink_to(directory / "campaign.csv")
    with pytest.raises(FileExistsError):
        generate_report(run, alias)
    assert {p: p.read_bytes() for p in before} == before


def test_cli_run_report_and_conflict_contract(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(runner, "_execute_trial", controlled_execute)
    spec = spec_file(tmp_path)
    output = tmp_path / "run"
    assert main(["run", "--spec", str(spec), "--output", str(output)]) == 0
    assert "6 trials" in capsys.readouterr().out
    assert (output / "report/report.md").is_file()
    assert main(["run", "--spec", str(spec), "--output", str(output)]) == 1
    assert "Benchmark error" in capsys.readouterr().err
    assert main(["report", "--run", str(output), "--output", str(tmp_path / "re-report")]) == 0


def test_report_failure_does_not_publish_partial_bundle(run, tmp_path, monkeypatch):
    import benchmarks.figures as figures

    def fail(*args):
        raise OSError("controlled rendering failure")

    monkeypatch.setattr(figures, "render_figures", fail)
    output = tmp_path / "report"
    with pytest.raises(OSError, match="controlled"):
        generate_report(run, output)
    assert not output.exists()
    assert list(tmp_path.glob(".benchmark-report-*")) == []
