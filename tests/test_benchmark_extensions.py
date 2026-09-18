"""Independent mixed optima and paired typed/delayed workflow acceptance."""

import copy
import hashlib
import itertools
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from benchmarks import runner
from benchmarks.definitions import definition
from benchmarks.designs import Proposals, decode, feasible, mixed_value, normalize, typed_digest
from benchmarks.report import load_evidence, tables
from benchmarks.spec import ROUTES, load_spec, schedule, seed_mapping, validate_spec
from benchmarks.storage import read_trace, write_json
from benchmarks.trial import _external_row, run_trial
from bo_forge import CampaignSession
from tests._benchmark_support import ROOT, controlled_suggest, spec_file


def extended_spec(route, size="smoke"):
    return load_spec(ROOT / f"benchmarks/specs/{route}_{size}.yaml")


def extended_suggest(campaign):
    if campaign.config.variables[-1].type != "categorical":
        return controlled_suggest(campaign)
    point = [-.9 + .1 * len(campaign.df), 3, .5, "B"]
    return _external_row(campaign, point, len(campaign.df), "log_ei")


def execute(directory, timeout):
    try:
        run_trial(directory, suggest=extended_suggest)
    except Exception as exc:
        # Controlled worker transport: retain failures like the isolated process boundary.
        runner._finalize_status(directory, "failed", str(exc), 1.0)
    else:
        runner._finalize_status(directory, None, "", 1.0)


def test_v2_budgets_and_v1_seed_identity():
    smoke = [trial for route in ROUTES for trial in schedule(extended_spec(route))]
    standard = [trial for route in ROUTES for trial in schedule(extended_spec(route, "standard"))]
    assert len(smoke) == 9 and sum(t["evaluations"] for t in smoke) == 54
    assert len(standard) == 45 and sum(t["evaluations"] for t in standard) == 1080
    assert all(t["timeout_seconds"] == 600 for t in smoke + standard)
    for stream, value in seed_mapping("branin", 0, "noisy").items():
        payload = json.dumps([1, "branin", 0, "noisy", stream], separators=(",", ":")).encode()
        assert value == int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")
    assert len({t["trial_id"] for t in standard}) == 45


@pytest.mark.parametrize("change", ["route", "mode", "problem", "noise", "odd", "schema"])
def test_invalid_extended_combinations(change):
    spec = extended_spec("pending_noisy")
    if change == "route":
        spec["route"] = "asynchronous"
    elif change == "mode":
        spec["modes"] = ["deterministic"]
    elif change == "problem":
        spec["problems"][0]["name"] = "hartmann3"
    elif change == "noise":
        spec["problems"][0]["noise_std"] = .5
    elif change == "odd":
        spec["evaluations"] = 7
    else:
        spec["schema_version"] = 1
    with pytest.raises(ValueError):
        validate_spec(spec)


@pytest.mark.parametrize("route,expected", [("mixed", 0.0), ("constrained_mixed", .125)])
def test_reference_minimum_by_independent_finite_enumeration(route, expected):
    trial = schedule(extended_spec(route))[0]
    candidates = []
    for k, z, c in itertools.product(range(5), [0, .5, 1], ["A", "B", "C"]):
        if route == "constrained_mixed" and c == "B" and k < 3:
            continue
        upper = min(1.0, .625 - .125 * k) if route == "constrained_mixed" else 1.0
        x = max(-1.0, min(.25, upper))
        penalty = .5 if c == "A" else .25 if c == "C" else 0
        value = (x - .25) * (x - .25) + (k - 2)**2 / 8 + (z - .5)**2 / 4 + penalty
        candidates.append(value)
        assert mixed_value([x, k, z, c]) == value
    assert min(candidates) == expected == definition(trial)["optimum"]


def test_typed_design_roundtrip_and_uniform_bins():
    trial = schedule(extended_spec("mixed"))[0]
    point = decode([.625, .4, .4, .4], trial)
    assert point == [.25, 2, .5, "B"] and type(point[1]) is int
    assert normalize(json.loads(json.dumps(point)), trial) == point
    assert typed_digest(point) == typed_digest(copy.deepcopy(point))
    assert typed_digest(point) != typed_digest([.25, 2.0, .5, "B"])
    assert [decode([.5, (i + .5) / 5, .5, .5], trial)[1] for i in range(5)] == list(range(5))
    constrained = schedule(extended_spec("constrained_mixed"))[0]
    for invalid in ([.25, 2, .5, "B"], [1, 3, .5, "B"]):
        assert not feasible(invalid, constrained)
        with pytest.raises(ValueError, match="feasibility"):
            normalize(invalid, constrained)


def test_bounded_proposals_reject_infeasible_and_duplicates(monkeypatch):
    import benchmarks.designs as designs

    trial = schedule(extended_spec("constrained_mixed"))[0]
    monkeypatch.setattr(designs, "DRAW_LIMIT", 3)
    sampler = Proposals(trial)
    monkeypatch.setattr(designs, "decode", lambda *_: [.25, 2, .5, "B"])
    with pytest.raises(ValueError, match="after 3 draws"):
        sampler.draw("sobol", [])
    assert sampler.counts == {"draws": 3, "infeasible": 3, "duplicate": 0}
    sampler = Proposals(trial)
    monkeypatch.setattr(designs, "decode", lambda *_: [0.0, 3, .5, "B"])
    with pytest.raises(ValueError, match="after 3 draws"):
        sampler.draw("random", [[0.0, 3, .5, "B"]])
    assert sampler.counts["duplicate"] == 3


@pytest.mark.parametrize("route", ROUTES)
def test_paired_workflows_roundtrip_reports_and_event_order(tmp_path, monkeypatch, route):
    monkeypatch.setattr(runner, "_execute_trial", execute)
    output = runner.run_suite(spec_file(tmp_path, extended_spec(route)), tmp_path / "run")
    metadata, traces, statuses = load_evidence(output)
    assert statuses.status.eq("complete").all()
    assert len(traces) == 18 and traces.feasible.all()
    assert set(traces.route) == {route}
    initials, hashes, noises = [], [], []
    for directory in (output / "trials").iterdir():
        frame = CampaignSession.from_files(directory / "campaign.yaml", directory / "campaign.csv")
        assert len(frame.observed_data()) == 6 and frame.pending_suggestions().empty
        rows, _ = read_trace(directory / "trace.jsonl")
        initials.append([r["x"] for r in rows[:4]])
        noises.append([r["noise"] for r in rows])
        hashes.append(json.loads((directory / "inputs.json").read_text())["initial_design_sha256"])
        if route == "pending_noisy":
            assert rows[4]["pending_row_ids"] == []
            assert rows[5]["pending_row_ids"] == [rows[4]["row_id"]]
            events, _ = read_trace(directory / "events.jsonl")
            assert [e["operation"] for e in events[-6:]] == [
                "submit", "accept", "submit", "accept", "observe", "observe"]
        else:
            assert type(rows[0]["x"][1]) is int and isinstance(rows[0]["x"][3], str)
    assert initials[0] == initials[1] == initials[2] and len(set(hashes)) == 1
    np.testing.assert_allclose(noises[0], noises[1], atol=1e-12)
    np.testing.assert_allclose(noises[0], noises[2], atol=1e-12)
    summary, _ = tables(metadata, traces, statuses)
    assert summary.complete.eq(1).all() and summary.unknown_timing_trials.eq(0).all()


def test_pending_second_suggestion_passes_x_pending_without_outcomes(tmp_path, monkeypatch):
    from bo_forge._optimization import single_objective
    from bo_forge.transforms import values_to_unit_cube

    trial = schedule(extended_spec("pending_noisy"))[0]
    directory = tmp_path / trial["trial_id"]
    directory.mkdir()
    write_json(directory / "trial.json", trial)
    calls, training = [], []

    def fit(config, observed):
        training.append(observed.copy(deep=True))
        return SimpleNamespace(posterior=lambda x: SimpleNamespace(
            mean=torch.zeros((len(x), 1), dtype=torch.double),
            variance=torch.ones((len(x), 1), dtype=torch.double)))

    def optimize(**kwargs):
        config = kwargs["config"]
        pending = kwargs["x_pending"]
        calls.append(None if pending is None else pending.clone())
        assert kwargs["x_baseline"].shape == (4, 2)
        point = [-2.0 + len(calls), 1.0]
        return values_to_unit_cube(config, [point]), torch.tensor(.1), "qlog_nei"

    monkeypatch.setattr(single_objective, "fit_gp_model", fit)
    monkeypatch.setattr(single_objective, "optimize_qlog_nei", optimize)
    run_trial(directory)
    assert len(calls) == 2 and calls[0] is None
    torch.testing.assert_close(calls[1], torch.tensor([[4 / 15, 1 / 15]], dtype=torch.double))
    for frame in training:
        frame["outcome"] = pd.to_numeric(frame["outcome"])
    pd.testing.assert_frame_equal(training[0], training[1])
    assert training[0].status.eq("observed").all()
    events, _ = read_trace(directory / "events.jsonl")
    assert [e["operation"] for e in events[-6:]] == [
        "submit", "accept", "submit", "accept", "observe", "observe"]


@pytest.mark.parametrize("failure", ["invalid", "duplicate", "objective", "partial"])
def test_extended_failure_evidence_is_retained(tmp_path, monkeypatch, failure):
    spec = extended_spec("mixed")
    spec["strategies"] = ["bo"]

    def bad_suggest(campaign):
        row = extended_suggest(campaign)
        if failure == "invalid":
            row.loc[0, "k"] = 5
        elif failure == "duplicate":
            row.loc[0, campaign.config.variable_names] = campaign.df.iloc[0][
                campaign.config.variable_names].values
        elif failure == "partial":
            return row.iloc[:0]
        return row

    def objective(problem, point):
        raise ValueError("Controlled objective failure")

    def failed_execute(directory, timeout):
        try:
            run_trial(directory, suggest=bad_suggest,
                      objective=objective if failure == "objective" else None)
        except ValueError as exc:
            runner._finalize_status(directory, "failed", str(exc), 1)

    monkeypatch.setattr(runner, "_execute_trial", failed_execute)
    run = runner.run_suite(spec_file(tmp_path, spec), tmp_path / "run")
    meta, traces, statuses = load_evidence(run)
    summary, _ = tables(meta, traces, statuses)
    assert statuses.status.tolist() == ["failed"]
    assert summary.complete.tolist() == [0] and summary.final_regret_median.isna().all()
    assert len(traces) == (0 if failure == "objective" else 4)


def test_pending_event_tampering_is_rejected_without_mutation(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "_execute_trial", execute)
    output = runner.run_suite(spec_file(tmp_path, extended_spec("pending_noisy")), tmp_path / "run")
    directory = next((output / "trials").iterdir())
    events, _ = read_trace(directory / "events.jsonl")
    events[-4]["pending_row_ids"] = []
    (directory / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    before = {p: p.read_bytes() for p in directory.iterdir() if p.is_file()}
    with pytest.raises(ValueError, match="workflow events.*expected.*actual"):
        load_evidence(output)
    assert all(p.read_bytes() == b for p, b in before.items())
