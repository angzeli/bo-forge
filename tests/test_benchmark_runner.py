"""Managed histories, oracle exclusion, failure evidence, and process deadlines."""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from benchmarks import runner
from benchmarks.storage import read_trace
from benchmarks.trial import run_trial
from bo_forge import CampaignSession
from tests._benchmark_support import (
    controlled_execute,
    controlled_suggest,
    prepare_trial,
    spec_file,
)


def test_runner_pairs_strategies_and_reloads_complete_managed_histories(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "_execute_trial", controlled_execute)
    output = runner.run_suite(spec_file(tmp_path), tmp_path / "run")
    metadata = json.loads((output / "run.json").read_text())
    assert metadata["status"] == "complete"
    for mode in ("deterministic", "noisy"):
        histories, traces = [], []
        for strategy in ("bo", "random", "sobol"):
            directory = output / "trials" / f"branin-{mode}-0000000000-{strategy}"
            campaign = CampaignSession.from_files(directory / "campaign.yaml",
                                                   directory / "campaign.csv")
            campaign.validate()
            assert len(campaign.observed_data()) == 6
            assert not len(campaign.pending_suggestions())
            manifest = json.loads((directory / "campaign.csv.manifest.json").read_text())
            assert len(manifest["events"]) == 13
            assert [event["operation"] for event in manifest["events"]] == [
                "initialize", *[operation for _ in range(6)
                                for operation in ("append_suggestions", "mark_observed")],
            ]
            for filename in ("campaign.yaml", "campaign.csv", "campaign.csv.manifest.json"):
                content = (directory / filename).read_text()
                assert all(word not in content for word in ("latent", "optimum", "noise_std"))
            histories.append(campaign.df)
            traces.append(read_trace(directory / "trace.jsonl")[0])
        for history in histories[1:]:
            pd.testing.assert_frame_equal(histories[0].iloc[:4], history.iloc[:4])
        for trace in traces[1:]:
            assert [r["noise"] for r in trace] == pytest.approx([r["noise"] for r in traces[0]])
        trial = next(t for t in metadata["trials"] if t["mode"] == mode)
        unit = torch.quasirandom.SobolEngine(
            2, scramble=True, seed=trial["seeds"]["initialization"],
        ).draw(6, dtype=torch.double).numpy()
        expected = unit * 15 + np.array([-5, 0])
        np.testing.assert_allclose(histories[2][["x1", "x2"]], expected)
        assert histories[0].source.iloc[4:].tolist() == [
            "log_ei" if mode == "deterministic" else "qlog_nei",
        ] * 2
    before = (output / "run.json").read_bytes()
    with pytest.raises(FileExistsError):
        runner.run_suite(spec_file(tmp_path), output)
    assert (output / "run.json").read_bytes() == before


@pytest.mark.parametrize("fault", ["duplicate", "bounds", "partial", "nan", "objective"])
def test_invalid_step_ends_visibly_without_fallback(tmp_path, fault):
    directory, _ = prepare_trial(tmp_path)
    calls = []

    def suggest(campaign):
        calls.append(1)
        result = controlled_suggest(campaign)
        if fault == "duplicate":
            result.loc[0, ["x1", "x2"]] = campaign.df.loc[0, ["x1", "x2"]].values
        if fault == "bounds":
            result.loc[0, "x1"] = 100
        if fault == "partial":
            return result.iloc[:0]
        if fault == "nan":
            result.loc[0, "x1"] = float("nan")
        return result

    count = 0

    def objective(problem, point):
        nonlocal count
        count += 1
        if fault == "objective" and count == 5:
            raise RuntimeError("controlled simulator failure")
        return float(problem.evaluate_true(torch.tensor(point, dtype=torch.double)).item())

    with pytest.raises((ValueError, RuntimeError)):
        run_trial(directory, suggest=suggest, objective=objective)
    runner._finalize_status(directory, "failed", "controlled failure", .1)
    rows, _ = read_trace(directory / "trace.jsonl")
    assert len(rows) == 4 and len(calls) == 1
    status = json.loads((directory / "status.json").read_text())
    assert status["status"] == "failed" and status["completed_evaluations"] == 4
    campaign = CampaignSession.from_files(directory / "campaign.yaml", directory / "campaign.csv")
    assert len(campaign.observed_data()) == 4
    assert len(campaign.df) == (5 if fault == "objective" else 4)


@pytest.mark.parametrize("fault", ["exit", "timeout", "missing-status", "malformed-status"])
def test_worker_failures_and_timeouts_have_terminal_records(tmp_path, fault):
    directory, _ = prepare_trial(tmp_path)
    commands = {
        "exit": "raise SystemExit(7)", "timeout": "import time; time.sleep(30)",
        "missing-status": "pass",
        "malformed-status": f"from pathlib import Path; Path({str(directory / 'status.json')!r})"
                            ".write_text('[]')",
    }
    started = time.monotonic()
    runner._execute_trial(directory, .1 if fault == "timeout" else 10,
                          command=[sys.executable, "-c", commands[fault]])
    assert time.monotonic() - started < 15
    status = json.loads((directory / "status.json").read_text())
    assert status["status"] == ("timeout" if fault == "timeout" else "failed")
    assert status["message"] and status["completed_evaluations"] == 0


def test_scheduler_records_every_trial_even_when_interrupted(tmp_path, monkeypatch):
    calls = []

    def execute(directory, timeout):
        calls.append(directory)
        if len(calls) == 2:
            raise KeyboardInterrupt
        controlled_execute(directory, timeout)

    monkeypatch.setattr(runner, "_execute_trial", execute)
    output = tmp_path / "run"
    with pytest.raises(KeyboardInterrupt):
        runner.run_suite(spec_file(tmp_path), output)
    states = [json.loads(path.read_text())["status"] for path in
              sorted((output / "trials").glob("*/status.json"))]
    assert len(states) == 6
    assert states.count("complete") == 1 and states.count("interrupted") == 5
    assert json.loads((output / "run.json").read_text())["status"] == "interrupted"


def test_setup_interruption_terminalizes_not_yet_created_trials(tmp_path, monkeypatch):
    original = Path.mkdir
    interrupted = False

    def interrupt(path, *args, **kwargs):
        nonlocal interrupted
        if path.name.endswith("-random") and path.parent.name == "trials" and not interrupted:
            interrupted = True
            raise KeyboardInterrupt
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", interrupt)
    output = tmp_path / "run"
    with pytest.raises(KeyboardInterrupt):
        runner.run_suite(spec_file(tmp_path), output)
    metadata = json.loads((output / "run.json").read_text())
    assert metadata["status"] == "interrupted"
    for trial in metadata["trials"]:
        directory = output / "trials" / trial["trial_id"]
        assert json.loads((directory / "trial.json").read_text()) == trial
        status = json.loads((directory / "status.json").read_text())
        assert status["status"] == "interrupted" and status["completed_evaluations"] == 0


def test_runner_failure_does_not_skip_later_trials(tmp_path, monkeypatch):
    def execute(directory, timeout):
        if directory.name.endswith("-bo"):
            runner._finalize_status(directory, "failed", "controlled BO failure", .1)
        else:
            controlled_execute(directory, timeout)

    monkeypatch.setattr(runner, "_execute_trial", execute)
    output = runner.run_suite(spec_file(tmp_path), tmp_path / "run")
    states = [json.loads(p.read_text())["status"]
              for p in (output / "trials").glob("*/status.json")]
    assert states.count("failed") == 2 and states.count("complete") == 4
    assert json.loads((output / "run.json").read_text())["status"] == "incomplete"


def test_warning_evidence_survives_and_cpu_limits_are_explicit(tmp_path):
    import warnings

    directory, _ = prepare_trial(tmp_path)

    def suggest(campaign):
        warnings.warn("controlled warning", RuntimeWarning, stacklevel=2)
        return controlled_suggest(campaign)

    run_trial(directory, suggest=suggest)
    evidence, _ = read_trace(directory / "warnings.jsonl")
    assert [row["message"] for row in evidence] == ["controlled warning"] * 2
    env = runner._worker_environment()
    assert env["OMP_NUM_THREADS"] == env["MKL_NUM_THREADS"] == env["OPENBLAS_NUM_THREADS"] == "1"
    assert env["CUDA_VISIBLE_DEVICES"] == ""


def test_trial_cannot_overwrite_existing_campaign(tmp_path):
    directory, _ = prepare_trial(tmp_path)
    run_trial(directory, suggest=controlled_suggest)
    before = {path: path.read_bytes() for path in directory.glob("campaign*")}
    with pytest.raises(FileExistsError):
        run_trial(directory, suggest=controlled_suggest)
    assert before == {path: path.read_bytes() for path in before}
