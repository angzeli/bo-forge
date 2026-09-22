"""Cross-surface execution and adversarial evidence checks for schema-v3 routes."""

import json
import shutil
from unittest.mock import patch

import pandas as pd
import pytest
import yaml

from benchmarks import runner
from benchmarks.report import _markdown, generate_report, load_evidence, tables
from benchmarks.storage import append_trace, read_trace, write_json
from benchmarks.trial import run_trial
from bo_forge import CampaignSession
from tests._benchmark_support import controlled_execute, controlled_suggest, spec_file
from tests.test_benchmark_multi_scores import multi_spec


@pytest.fixture(scope="module", params=["multi_objective", "multi_fidelity"])
def seed_run(tmp_path_factory, request):
    root = tmp_path_factory.mktemp(request.param)
    with patch.object(runner, "_execute_trial", controlled_execute):
        return runner.run_suite(spec_file(root, multi_spec(request.param)), root / "run")


@pytest.fixture
def run(seed_run, tmp_path):
    return shutil.copytree(seed_run, tmp_path / "run")


def test_complete_roundtrip_preserves_campaigns_and_reports_without_objectives(
    run, tmp_path, monkeypatch,
):
    import benchmarks.multi as multi

    meta, traces, statuses = load_evidence(run)
    assert len(traces) == 18 and statuses.status.eq("complete").all(), statuses.to_dict("records")
    assert "simple_regret" not in traces.columns
    initials = []
    for path in (run / "trials").iterdir():
        session = CampaignSession.from_files(path / "campaign.yaml", path / "campaign.csv")
        session.validate()
        assert len(session.observed_data()) == 6 and session.pending_suggestions().empty
        rows, _ = read_trace(path / "trace.jsonl")
        initials.append([r["x"] for r in rows[:4]])
        forbidden = {"oracle_target_value", "oracle_target_regret",
                     "latent_objectives", "hypervolume"}
        assert not forbidden.intersection(session.df.columns)
    assert initials[0] == initials[1] == initials[2]
    summary, curves = tables(meta, traces, statuses)
    assert summary.complete.eq(1).all() and curves.contributing_partial.eq(0).all()
    before = {p: p.read_bytes() for p in run.rglob("*") if p.is_file()}

    def forbidden(*args, **kwargs):
        pytest.fail("Report must not evaluate objectives or fit models")

    monkeypatch.setattr(multi, "evaluate", forbidden)
    monkeypatch.setattr(multi, "problem_for_route", forbidden)
    monkeypatch.setattr(CampaignSession, "suggest_next", forbidden)
    output = generate_report(run, tmp_path / "report")
    assert (output / "report.md").exists() and list((output / "figures").glob("*.png"))
    assert all(p.read_bytes() == data for p, data in before.items())
    csv = pd.read_csv(output / "traces.csv")
    assert list(csv.columns) == list(traces.columns)
    text = (output / "report.md").read_text()
    assert "conditioned on successfully completed" in text
    if meta["spec"]["route"] == "multi_fidelity":
        assert "not equal-cost" in text and "Oracle" in text
        assert traces.oracle_seconds.ge(0).all()
        baseline = traces.loc[traces.strategy.ne("bo") & traces.evaluation.gt(4)]
        assert baseline.fidelity.eq(1).all()


@pytest.mark.parametrize("strategy", ["random", "sobol"])
def test_swapped_artifact_bundles_are_rejected(run, tmp_path, strategy):
    target = next((run / "trials").glob("*-bo"))
    source = next((run / "trials").glob(f"*-{strategy}"))
    for path in source.iterdir():
        if path.is_file() and path.name != "trial.json":
            shutil.copyfile(path, target / path.name)
    _reject_unchanged(run, tmp_path, "campaign_name.*expected.*actual")


def _reject_unchanged(run, tmp_path, match):
    before = {p: p.read_bytes() for p in run.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match=match):
        generate_report(run, tmp_path / "rejected-report")
    assert not (tmp_path / "rejected-report").exists()
    assert all(p.read_bytes() == data for p, data in before.items())


@pytest.mark.parametrize("change", ["definition", "reference_cost", "optimizer", "seed", "source"])
@pytest.mark.parametrize("partial", [False, True])
def test_changed_bindings_rejected_before_reporting(run, tmp_path, change, partial):
    directory = next((run / "trials").glob("*-bo"))
    path = directory / "campaign.yaml"
    config = yaml.safe_load(path.read_text())
    if change == "definition":
        if "objectives" in config:
            config["objectives"].reverse()
        else:
            config["variables"][-1]["lower"] = -.1
    elif change == "reference_cost":
        if "objectives" in config:
            config["objectives"][0]["reference_point"] = 19
        else:
            config["fidelity"]["fixed_cost"] = .5
    elif change == "optimizer":
        config["bo"]["raw_samples"] += 1
    elif change == "seed":
        config["bo"]["random_seed"] += 1
    else:
        frame = pd.read_csv(directory / "campaign.csv")
        frame.loc[4, "source"] = "random"
        frame.to_csv(directory / "campaign.csv", index=False)
    if change != "source":
        path.write_text(yaml.safe_dump(config))
    if partial:
        (directory / "trace.jsonl").unlink()
        write_json(directory / "status.json", {"status": "failed", "completed_evaluations": 0})
    _reject_unchanged(run, tmp_path, "expected.*actual")


@pytest.mark.parametrize("field", ["baseline_policy", "initialization_policy", "seed_mapping"])
def test_input_policy_tampering_rejected(run, tmp_path, field):
    path = next((run / "trials").iterdir()) / "inputs.json"
    inputs = json.loads(path.read_text())
    inputs[field] = "changed"
    write_json(path, inputs)
    _reject_unchanged(run, tmp_path, f"inputs.{field}.*expected.*actual")


def test_score_tampering_rejected(run, tmp_path):
    directory = next((run / "trials").iterdir())
    rows, _ = read_trace(directory / "trace.jsonl")
    key = "hypervolume" if "hypervolume" in rows[-1] else "cumulative_modeled_cost"
    rows[-1][key] += 1
    (directory / "trace.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    _reject_unchanged(run, tmp_path, key)


def test_unknown_timing_and_partial_metrics_remain_visible(run):
    directory = next((run / "trials").glob("*-bo"))
    path = directory / "status.json"
    status = json.loads(path.read_text())
    status.pop("wall_seconds")
    status["status"] = "failed"
    write_json(path, status)
    meta, traces, statuses = load_evidence(run)
    summary, curves = tables(meta, traces, statuses)
    bo = summary.loc[summary.strategy.eq("bo")].iloc[0]
    assert pd.isna(bo.wall_seconds) and bo.unknown_timing_trials == 1
    assert bo.complete == 0 and bo.failed == 1
    assert curves.loc[curves.strategy.eq("bo"), "median"].isna().all()
    assert curves.loc[curves.strategy.eq("bo"), "contributing_partial"].eq(1).all()
    assert "not available" in _markdown(meta, summary, statuses)


@pytest.mark.parametrize("route", ["multi_objective", "multi_fidelity"])
@pytest.mark.parametrize("failure", ["duplicate", "bounds", "objective", "partial", "source"])
def test_failed_execution_preserves_partial_evidence(tmp_path, monkeypatch, route, failure):
    spec = multi_spec(route)
    spec["strategies"] = ["bo"]

    def suggest(campaign):
        row = controlled_suggest(campaign)
        names = campaign.config.variable_names
        if failure == "duplicate":
            row.loc[0, names] = campaign.df.iloc[0][names].values
        elif failure == "bounds":
            row.loc[0, names[0]] = 1000
        elif failure == "source":
            row.loc[0, "source"] = "random"
        elif failure == "partial":
            return row.iloc[:0]
        return row

    def bad_objective(*_):
        raise ValueError("controlled objective failure")

    def execute(directory, timeout):
        try:
            run_trial(directory, suggest=suggest,
                      objective=bad_objective if failure == "objective" else None)
        except ValueError as exc:
            runner._finalize_status(directory, "failed", str(exc), 1.)

    monkeypatch.setattr(runner, "_execute_trial", execute)
    output = runner.run_suite(spec_file(tmp_path, spec), tmp_path / "run")
    _, traces, statuses = load_evidence(output)
    assert statuses.status.tolist() == ["failed"]
    assert len(traces) == (0 if failure == "objective" else 4)


@pytest.mark.parametrize("route", ["multi_objective", "multi_fidelity"])
def test_partial_event_cannot_claim_unobserved_csv_row(tmp_path, monkeypatch, route):
    from benchmarks import multi

    def stop_before_observation(*_):
        raise ValueError("stopped with a suggested row")

    monkeypatch.setattr(multi, "_observe", stop_before_observation)
    monkeypatch.setattr(runner, "_execute_trial", controlled_execute)
    output = runner.run_suite(spec_file(tmp_path, multi_spec(route)), tmp_path / "run")
    load_evidence(output)
    directory = next((output / "trials").iterdir())
    frame = pd.read_csv(directory / "campaign.csv")
    append_trace(directory / "events.jsonl", {"operation": "observe",
                                             "row_id": frame.row_id.iloc[0],
                                             "submission": 1, "pending_row_ids": []})
    _reject_unchanged(output, tmp_path, "status.*observed.*suggested")
