"""Coupled MO and continuous MF trials using the existing campaign engine."""

import time

import torch
import yaml

from benchmarks.definitions import campaign_config, definition
from benchmarks.designs import DRAW_LIMIT, Proposals, typed_digest
from benchmarks.multi_scores import OBJECTIVES, mf_score, mo_score
from benchmarks.storage import append_trace, sha256, utc_now, write_json
from benchmarks.workflows import _event, _submit
from bo_forge import CampaignSession


def problem_for_route(route):
    from botorch.test_functions.multi_fidelity import AugmentedBranin
    from botorch.test_functions.multi_objective import BraninCurrin

    cls = BraninCurrin if route == "multi_objective" else AugmentedBranin
    return cls(noise_std=None, negate=False).to(dtype=torch.double, device="cpu")


def evaluate(problem, point):
    with torch.no_grad():
        values = problem.evaluate_true(torch.tensor(point, dtype=torch.double))
    return values.tolist()


def input_metadata(trial):
    problem = definition(trial)
    result = {"bounds": problem["bounds"], "variables": problem["variables"],
              "optimum": problem["optimum"],
              "optimum_tolerance": trial["problem"]["optimum_tolerance"],
              "seed_mapping": trial["seeds"], "dtype": "typed_json", "direction": "minimize",
              "baseline_policy": trial["baseline_policy"],
              "initialization_policy": trial["initialization_policy"]}
    if trial["route"] == "multi_objective":
        result.update(objective_names=OBJECTIVES, objective_directions=["minimize"] * 2,
                      reference_point=trial["reference_point"])
    else:
        result["fidelity"] = trial["fidelity"]
    return result


def run_multi(directory, trial, *, suggest=None, objective=None):
    from benchmarks.trial import _warning_evidence

    config = directory / "campaign.yaml"
    with config.open("x", encoding="utf-8") as handle:
        yaml.safe_dump(campaign_config(trial), handle, sort_keys=False)
    campaign = CampaignSession.initialize(config, directory / "campaign.csv")
    proposals, initial = Proposals(trial), []
    try:
        for _ in range(trial["initial_observations"]):
            initial.append(proposals.draw("sobol", initial))
    finally:
        write_json(directory / "feasibility.json", {"draw_limit": DRAW_LIMIT, **proposals.counts})
    write_json(directory / "inputs.json", {
        **input_metadata(trial), "config_sha256": sha256(config.read_bytes()),
        "initial_design_sha256": typed_digest(initial),
    })
    problem, rows = problem_for_route(trial["route"]), []
    with _warning_evidence(directory):
        for index in range(trial["evaluations"]):
            submission = _submit(directory, campaign, trial, proposals, initial, index, suggest)
            _observe(directory, campaign, trial, problem, rows, submission, objective or evaluate)
    if len(campaign.pending_suggestions()) or len(campaign.observed_data()) != len(rows):
        raise ValueError("Completed workflow must drain its observation queue.")
    result = {"status": "complete", "completed_evaluations": len(rows), "message": "",
              "finished_at": utc_now()}
    write_json(directory / "status.json", result)
    return result


def _observe(directory, campaign, trial, problem, rows, submission, objective):
    point, row_id, index = submission["point"], submission["row_id"], submission["index"]
    write_json(directory / "status.json", {"status": "running", "phase": "observe",
                                           "completed_evaluations": len(rows)})
    start = time.monotonic()
    value = objective(problem, point)
    seconds = time.monotonic() - start
    oracle_seconds = 0.0
    if trial["route"] == "multi_objective":
        value = dict(zip(OBJECTIVES, value, strict=True)) if isinstance(value, list) else value
        scored = mo_score(rows, value, trial, row_id)
    start = time.monotonic()
    if trial["route"] == "multi_objective":
        campaign.mark_observed(row_id, objective_values=value)
    else:
        campaign.mark_observed(row_id, value)
    campaign.reload()
    campaign.validate()
    _event(directory, "observe", row_id, index, [])
    if len(campaign.observed_data()) != index + 1 or len(campaign.pending_suggestions()):
        raise ValueError("Reloaded observation count disagrees with the evaluation trace.")
    mutation_seconds = submission["submission_seconds"] + time.monotonic() - start
    if trial["route"] == "multi_fidelity":
        # Retain the actual observation even if the scoring-only oracle fails or is cancelled.
        start = time.monotonic()
        projected = objective(problem, [*point[:-1], trial["fidelity"]["target"]])
        oracle_seconds = time.monotonic() - start
        scored = mf_score(rows, value, projected, point, trial)
    row = {"evaluation": index + 1, "row_id": row_id, "x": point,
           "source": submission["source"], **scored,
           "suggestion_seconds": submission["suggestion_seconds"], "objective_seconds": seconds,
           "oracle_seconds": oracle_seconds,
           "mutation_seconds": mutation_seconds,
           "fit_evidence": submission["fit_evidence"], "pending_row_ids": [], "feasible": True}
    append_trace(directory / "trace.jsonl", row)
    rows.append(row)
