"""Actual observations survive failures in scoring-only oracle diagnostics."""

import json
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from benchmarks import multi, runner
from benchmarks.definitions import campaign_config
from benchmarks.designs import Proposals
from benchmarks.report import _markdown, load_evidence, tables
from benchmarks.spec import schedule
from benchmarks.storage import read_trace
from benchmarks.trial import run_trial
from bo_forge import CampaignSession
from tests._benchmark_support import controlled_suggest, spec_file
from tests.test_benchmark_multi_scores import multi_spec


@pytest.mark.parametrize("failure", [RuntimeError("oracle failed"), float("nan"), float("inf"),
                                    KeyboardInterrupt("oracle cancelled"),
                                    SystemExit("oracle cancelled")])
def test_oracle_failure_retains_actual_observation_and_report_warning(
    tmp_path, monkeypatch, failure,
):
    retained = {}

    def execute(directory, timeout):
        calls = 0

        def objective(problem, point):
            nonlocal calls
            calls += 1
            if calls == 5:
                retained[directory.name] = multi.evaluate(problem, point)
                return retained[directory.name]
            if calls == 6:
                # The third initial observation is low-fidelity, unlike its projection.
                frame = pd.read_csv(directory / "campaign.csv")
                assert frame.status.tolist() == ["observed"] * 3
                assert frame.iloc[-1].x3 < point[-1] == 1.
                assert frame.iloc[-1].outcome == pytest.approx(retained[directory.name])
                events, warning = read_trace(directory / "events.jsonl")
                assert not warning and events[-1]["operation"] == "observe"
                assert events[-1]["row_id"] == frame.iloc[-1].row_id
                if isinstance(failure, BaseException):
                    raise failure
                return failure
            return multi.evaluate(problem, point)

        try:
            run_trial(directory, suggest=controlled_suggest, objective=objective)
        except (KeyboardInterrupt, SystemExit) as exc:
            runner._finalize_status(directory, "interrupted", str(exc), 5.)
            raise
        except Exception as exc:
            runner._finalize_status(directory, "failed", str(exc), 5.)

    monkeypatch.setattr(runner, "_execute_trial", execute)
    output = tmp_path / "run"
    spec = spec_file(tmp_path, multi_spec("multi_fidelity"))
    cancelled = isinstance(failure, (KeyboardInterrupt, SystemExit))
    if cancelled:
        with pytest.raises(type(failure)) as caught:
            runner.run_suite(spec, output)
        assert caught.value is failure
    else:
        runner.run_suite(spec, output)

    for trial_id, actual in retained.items():
        directory = output / "trials" / trial_id
        campaign = CampaignSession.from_files(directory / "campaign.yaml",
                                              directory / "campaign.csv")
        campaign.validate()
        assert len(campaign.observed_data()) == 3
        assert campaign.pending_suggestions().empty
        assert float(campaign.observed_data().iloc[-1].outcome) == pytest.approx(actual)
        rows, warning = read_trace(directory / "trace.jsonl")
        assert not warning and len(rows) == 2
        events, warning = read_trace(directory / "events.jsonl")
        assert not warning
        assert [event["operation"] for event in events] == ["submit", "observe"] * 3

    metadata, traces, statuses = load_evidence(output)
    active = statuses.loc[statuses.trial_id.isin(retained)]
    assert len(active) == (1 if cancelled else 3)
    assert active.status.eq("interrupted" if cancelled else "failed").all()
    assert active.completed_evaluations.eq(2).all()
    message = str(failure) if isinstance(failure, BaseException) else "must be finite"
    assert active.message.str.contains(message, regex=False).all()
    warning = "1 observed CSV row(s) have no scoring trace."
    assert active.evidence_warning.str.contains(warning, regex=False).all()
    summary, curves = tables(metadata, traces, statuses)
    assert summary.complete.eq(0).all() and curves["median"].isna().all()
    report = _markdown(metadata, summary, statuses)
    assert warning in report and message in report


@pytest.mark.parametrize("route", ["multi_objective", "multi_fidelity"])
def test_success_preserves_scoring_and_separate_timing(tmp_path, monkeypatch, route):
    trial = schedule(multi_spec(route))[0]
    config = tmp_path / "campaign.yaml"
    config.write_text(yaml.safe_dump(campaign_config(trial)))
    campaign = CampaignSession.initialize(config, tmp_path / "campaign.csv")
    proposals = Proposals(trial)
    initial = [proposals.draw("sobol", [])]
    submission = multi._submit(tmp_path, campaign, trial, proposals, initial, 0, None)
    submission["submission_seconds"] = 5.
    ticks = iter([10., 12., 20., 27., 30., 41.])
    monkeypatch.setattr(multi, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
    calls = []
    problem = multi.problem_for_route(route)

    def objective(problem, point):
        calls.append(point)
        if len(calls) == 2:
            assert len(campaign.observed_data()) == 1
            events, _ = read_trace(tmp_path / "events.jsonl")
            assert events[-1]["operation"] == "observe"
        return multi.evaluate(problem, point)

    rows = []
    multi._observe(tmp_path, campaign, trial, problem, rows, submission, objective)
    stored, warning = read_trace(tmp_path / "trace.jsonl")
    assert not warning and stored == rows and len(rows) == 1
    row = rows[0]
    assert row["objective_seconds"] == 2.
    assert row["mutation_seconds"] == 12.
    assert row["suggestion_seconds"] == submission["suggestion_seconds"]
    assert len(campaign.observed_data()) == 1 and campaign.pending_suggestions().empty
    actual = multi.evaluate(problem, submission["point"])
    if route == "multi_fidelity":
        projected = multi.evaluate(problem, [*submission["point"][:-1], 1.])
        assert len(calls) == 2 and row["oracle_seconds"] == 11.
        assert row["observed"] == actual and row["oracle_target_value"] == projected
        expected = multi.mf_score([], actual, projected, submission["point"], trial)
    else:
        assert len(calls) == 1 and row["oracle_seconds"] == 0.
        expected = multi.mo_score([], dict(zip(multi.OBJECTIVES, actual, strict=True)),
                                  trial, submission["row_id"])
    assert {key: row[key] for key in expected} == expected


@pytest.mark.parametrize("values", [{"branin": 1.}, {"branin": 1., "currin": float("nan")}])
def test_invalid_mo_observation_still_precedes_writes(tmp_path, values):
    trial = schedule(multi_spec("multi_objective"))[0]
    (tmp_path / "trial.json").write_text(json.dumps(trial))
    with pytest.raises(ValueError):
        run_trial(tmp_path, objective=lambda *_: values)
    frame = pd.read_csv(tmp_path / "campaign.csv", keep_default_na=False)
    assert frame.status.tolist() == ["suggested"]
    assert frame.branin.tolist() == frame.currin.tolist() == [""]
    events, warning = read_trace(tmp_path / "events.jsonl")
    assert not warning and [event["operation"] for event in events] == ["submit"]
