"""Version-2 mixed and delayed-observation routes over the existing campaign engine."""

import math
import time

import numpy as np
import pandas as pd
import yaml

from benchmarks.definitions import campaign_config, definition, expected_source
from benchmarks.designs import DRAW_LIMIT, Proposals, mixed_value, normalize, typed_digest
from benchmarks.problems import problem_for, score, true_value
from benchmarks.storage import append_trace, sha256, utc_now, write_json
from bo_forge import CampaignSession


def run_extended(directory, trial, *, suggest=None, objective=None):
    from benchmarks.trial import _warning_evidence

    config = directory / "campaign.yaml"
    with config.open("x", encoding="utf-8") as handle:
        yaml.safe_dump(campaign_config(trial), handle, sort_keys=False)
    campaign = CampaignSession.initialize(config, directory / "campaign.csv")
    proposals = Proposals(trial)
    initial = []
    try:
        for _ in range(trial["initial_observations"]):
            initial.append(proposals.draw("sobol", initial))
    finally:
        write_json(directory / "feasibility.json", {"draw_limit": DRAW_LIMIT, **proposals.counts})
    problem = definition(trial)
    write_json(directory / "inputs.json", {
        "bounds": problem["bounds"], "variables": problem["variables"],
        "optimum": problem["optimum"], "optimum_tolerance": trial["problem"]["optimum_tolerance"],
        "config_sha256": sha256(config.read_bytes()),
        "initial_design_sha256": typed_digest(initial),
        "seed_mapping": trial["seeds"], "dtype": "typed_json", "direction": "minimize",
    })
    noise = np.random.default_rng(trial["seeds"]["observation_noise"]).normal(
        size=trial["evaluations"],
    )
    native = problem_for("branin") if trial["route"] == "pending_noisy" else None
    rows = []
    with _warning_evidence(directory):
        index = 0
        while index < trial["evaluations"]:
            count = 2 if trial["route"] == "pending_noisy" and index >= len(initial) else 1
            batch = [_submit(directory, campaign, trial, proposals, initial, i, suggest)
                     for i in range(index, index + count)]
            # Neither objective is computed before both designs have been submitted and accepted.
            for submission in batch:
                _observe_submission(directory, campaign, trial, submission, rows, noise,
                                    native, objective)
            index += count
    if len(campaign.pending_suggestions()) or len(campaign.observed_data()) != trial["evaluations"]:
        raise ValueError("Completed workflow must drain its full observation queue.")
    result = {"status": "complete", "completed_evaluations": len(rows), "message": "",
              "finished_at": utc_now()}
    write_json(directory / "status.json", result)
    return result


def _event(directory, operation, row_id, index, pending):
    append_trace(directory / "events.jsonl", {"operation": operation, "row_id": row_id,
                 "submission": index + 1, "pending_row_ids": pending})


def _submit(directory, campaign, trial, proposals, initial, index, suggest):
    from benchmarks.trial import _external_row

    pending = campaign.pending_suggestions().row_id.astype(str).tolist()
    write_json(directory / "status.json", {"status": "running", "phase": "suggest",
                                           "completed_evaluations": len(campaign.observed_data())})
    start = time.monotonic()
    evidence = {}
    try:
        if index < len(initial):
            candidate = _external_row(campaign, initial[index], index, "sobol")
        elif trial["strategy"] != "bo":
            existing = campaign.df[campaign.config.variable_names].values.tolist()
            point = proposals.draw(trial["strategy"], existing)
            candidate = _external_row(campaign, point, index, trial["strategy"])
        else:
            candidate = suggest(campaign) if suggest else campaign.suggest_next(batch_size=1)
            evidence = {k: v for k, v in dict(campaign.model_summary().values).items()
                        if k.startswith("last_fit") or k == "fallback_status"}
    finally:
        write_json(directory / "feasibility.json", {"draw_limit": DRAW_LIMIT, **proposals.counts})
    seconds = time.monotonic() - start
    point = _validate(campaign, candidate, trial, index)
    row_id = str(candidate.iloc[0]["row_id"])
    start = time.monotonic()
    campaign.append_suggestions(candidate)
    _event(directory, "submit", row_id, index, pending)
    if campaign.config.review.enabled:
        campaign.review_suggestion(row_id, "accept")
        _event(directory, "accept", row_id, index,
               campaign.pending_suggestions().row_id.astype(str).tolist())
    return {"index": index, "row_id": row_id, "point": point, "pending": pending,
            "suggestion_seconds": seconds, "submission_seconds": time.monotonic() - start,
            "fit_evidence": evidence, "source": str(candidate.iloc[0]["source"])}


def _validate(campaign, candidate, trial, index):
    if not isinstance(candidate, pd.DataFrame) or len(candidate) != 1:
        raise ValueError("A benchmark submission requires exactly one candidate.")
    row = candidate.iloc[0]
    outcomes = (campaign.config.objective_names if campaign.config.is_multi_objective
                else [campaign.config.objective.name])
    if row["status"] != "suggested" or any(
        pd.notna(pd.to_numeric(row[name], errors="coerce")) for name in outcomes
    ):
        raise ValueError("Candidates must be unobserved suggestions.")
    if row["source"] != expected_source(trial, index):
        raise ValueError("Candidate source differs from the scheduled strategy.")
    point = normalize(row[campaign.config.variable_names].tolist(), trial)
    previous = campaign.df[campaign.config.variable_names].values.tolist()
    seen = [normalize(p, trial) for p in previous]
    if typed_digest(point) in {typed_digest(p) for p in seen}:
        raise ValueError("Duplicate benchmark candidate; no retry or baseline substitution.")
    return point


def _observe_submission(directory, campaign, trial, submission, rows, noise, native, objective):
    index, point = submission["index"], submission["point"]
    start = time.monotonic()
    if objective is not None:
        latent = objective(native, point)
    else:
        latent = mixed_value(point) if native is None else true_value(native, point)
    observed = float(latent + trial["problem"]["noise_std"] * noise[index])
    if not math.isfinite(latent) or not math.isfinite(observed):
        raise ValueError("Objective observation is not finite.")
    scored = score(rows, float(latent), observed, definition(trial)["optimum"],
                   trial["problem"]["optimum_tolerance"])
    seconds = time.monotonic() - start
    start = time.monotonic()
    campaign.mark_observed(submission["row_id"], observed)
    campaign.reload()
    campaign.validate()
    _event(directory, "observe", submission["row_id"], index,
           campaign.pending_suggestions().row_id.astype(str).tolist())
    if len(campaign.observed_data()) != index + 1:
        raise ValueError("Reloaded observation count disagrees with the evaluation trace.")
    row = {"evaluation": index + 1, "row_id": submission["row_id"], "x": point,
           "source": submission["source"], "noise": observed - latent, **scored,
           "suggestion_seconds": submission["suggestion_seconds"], "objective_seconds": seconds,
           "mutation_seconds": submission["submission_seconds"] + time.monotonic() - start,
           "fit_evidence": submission["fit_evidence"],
           "pending_row_ids": submission["pending"], "feasible": True}
    append_trace(directory / "trace.jsonl", row)
    rows.append(row)
