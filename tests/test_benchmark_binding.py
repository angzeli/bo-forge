"""Scheduled identity is authoritative even for internally coherent copied evidence."""

import json
import shutil
from unittest.mock import patch

import pandas as pd
import pytest
import yaml

from benchmarks import runner
from benchmarks.binding import verify_binding
from benchmarks.report import _markdown, generate_report, load_evidence, tables
from benchmarks.storage import write_json
from benchmarks.trial import run_trial
from tests._benchmark_support import controlled_execute, spec_file


@pytest.fixture(scope="module")
def seed_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("binding-seed")
    with patch.object(runner, "_execute_trial", controlled_execute):
        return runner.run_suite(spec_file(root), root / "run")


@pytest.fixture
def run(seed_run, tmp_path):
    return shutil.copytree(seed_run, tmp_path / "run")


@pytest.mark.parametrize("strategy", ["random", "sobol"])
def test_copied_baseline_bundle_cannot_be_reported_as_bo(run, tmp_path, strategy):
    target = next((run / "trials").glob("*-deterministic-*-bo"))
    source = next((run / "trials").glob(f"*-deterministic-*-{strategy}"))
    for path in source.iterdir():
        if path.is_file() and path.name != "trial.json":
            shutil.copyfile(path, target / path.name)
    before = {p: p.read_bytes() for p in target.iterdir() if p.is_file()}
    output = tmp_path / "report"
    with pytest.raises(ValueError, match=f"{target.name}.*campaign_name.*expected.*actual"):
        generate_report(run, output)
    assert not output.exists()
    assert all(p.read_bytes() == value for p, value in before.items())


@pytest.mark.parametrize("field,value", [
    ("campaign_name", "another"), ("bo.acquisition", "qlog_nei"),
    ("bo.random_seed", 123), ("bo.raw_samples", 4), ("bo.num_restarts", 2),
    ("bo.initial_design_method", "random"), ("model.profile", "smooth"),
    ("objective.direction", "maximize"), ("variables", [
        {"name": "different", "type": "continuous", "lower": 0, "upper": 1}]),
])
def test_changed_definition_is_rejected_even_without_scored_rows(run, field, value):
    directory = next((run / "trials").glob("*-deterministic-*-bo"))
    config = directory / "campaign.yaml"
    data = yaml.safe_load(config.read_text())
    keys = field.split(".")
    target = data if len(keys) == 1 else data[keys[0]]
    target[keys[-1]] = value
    config.write_text(yaml.safe_dump(data))
    (directory / "trace.jsonl").unlink()
    write_json(directory / "status.json", {"status": "failed", "completed_evaluations": 0})
    with pytest.raises(ValueError, match=f"{directory.name}.*expected.*actual"):
        load_evidence(run)


@pytest.mark.parametrize("key,value", [("seed_mapping", {}), ("bounds", [[0, 0], [1, 1]]),
                                      ("direction", "maximize"), ("optimum", 0)])
def test_input_metadata_cannot_override_schedule(run, key, value):
    directory = next((run / "trials").iterdir())
    path = directory / "inputs.json"
    data = json.loads(path.read_text())
    data[key] = value
    write_json(path, data)
    with pytest.raises(ValueError, match=f"{directory.name}.*inputs.{key}.*expected.*actual"):
        load_evidence(run)


def test_coherent_bo_campaign_with_baseline_sources_is_rejected(tmp_path, monkeypatch):
    from tests._benchmark_support import controlled_suggest

    def execute(directory, timeout):
        def wrong_source(campaign):
            row = controlled_suggest(campaign)
            row["source"] = "random"
            return row
        run_trial(directory, suggest=wrong_source)
        runner._finalize_status(directory, None, "", 1.0)

    monkeypatch.setattr(runner, "_execute_trial", execute)
    run = runner.run_suite(spec_file(tmp_path), tmp_path / "run")
    with pytest.raises(ValueError, match="source: expected 'log_ei'; actual 'random'"):
        load_evidence(run)


def test_unknown_timing_is_not_reported_as_zero_or_complete_total(run):
    for directory in (run / "trials").glob("*-bo"):
        status = json.loads((directory / "status.json").read_text())
        status.pop("wall_seconds")
        write_json(directory / "status.json", status)
    metadata, traces, statuses = load_evidence(run)
    summary, _ = tables(metadata, traces, statuses)
    bo = summary.loc[summary.strategy.eq("bo")]
    assert bo.wall_seconds.isna().all() and bo.unknown_timing_trials.eq(1).all()
    assert bo.known_wall_seconds.eq(0).all() and bo.known_timing_trials.eq(0).all()
    assert "not available" in _markdown(metadata, summary, statuses)
    paired = pd.concat([statuses, statuses.assign(seed=1, wall_seconds=12.0)])
    mixed, _ = tables(metadata, traces, paired)
    bo = mixed.loc[mixed.strategy.eq("bo")]
    assert bo.wall_seconds.isna().all() and bo.known_wall_seconds.eq(12).all()
    assert bo.known_timing_trials.eq(1).all() and bo.unknown_timing_trials.eq(1).all()


def test_initialization_source_and_schedule_seeds_are_verified(run):
    directory = next((run / "trials").iterdir())
    trial = json.loads((directory / "trial.json").read_text())
    trial["seeds"]["fitting"] += 1
    with pytest.raises(ValueError, match="seeds.*expected.*actual"):
        verify_binding(directory, trial)
    frame = pd.read_csv(directory / "campaign.csv")
    frame.loc[0, "source"] = "random"
    frame.to_csv(directory / "campaign.csv", index=False)
    with pytest.raises(ValueError, match="source: expected 'sobol'; actual 'random'"):
        load_evidence(run)
