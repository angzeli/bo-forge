"""One synchronous trial using only the existing campaign mutation workflow."""

from __future__ import annotations

import json
import math
import time
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from benchmarks.problems import inputs, problem_for, score, true_value
from benchmarks.storage import append_trace, sha256, utc_now, write_json
from bo_forge import CampaignSession


def campaign_config(trial, bounds):
    """Only definitions/settings reach BO. No optimum, latent outcome, or noise parameter."""
    return {
        "campaign_name": trial["trial_id"],
        "objective": {"name": "outcome", "direction": "minimize"},
        "variables": [{"name": f"x{i + 1}", "type": "continuous",
                       "lower": float(lower), "upper": float(upper)}
                      for i, (lower, upper) in enumerate(zip(*bounds, strict=True))],
        "model": {"profile": "default"},
        "bo": {**trial["bo"], "batch_size": 1,
               "initial_design_size": trial["initial_observations"],
               "initial_design_method": "sobol", "random_seed": trial["seeds"]["fitting"],
               "acquisition": "qlog_nei" if trial["mode"] == "noisy" else "log_ei"},
    }


def _external_row(campaign, point, index, source):
    row = dict.fromkeys(campaign.df.columns, "")
    row.update(row_id=f"eval_{index + 1:06d}", iteration=index + 1,
               status="suggested", source=source)
    row.update(zip(campaign.config.variable_names, point, strict=True))
    return pd.DataFrame([row], columns=campaign.df.columns)


def _candidate(campaign, trial, index, sobol, random, suggest):
    if index < trial["initial_observations"]:
        return _external_row(campaign, sobol[index], index, "sobol"), {}
    if trial["strategy"] != "bo":
        points = sobol if trial["strategy"] == "sobol" else random
        return _external_row(campaign, points[index], index, trial["strategy"]), {}
    suggestions = suggest(campaign) if suggest else campaign.suggest_next(batch_size=1)
    evidence = dict(campaign.model_summary().values)
    return suggestions, {key: value for key, value in evidence.items()
                         if key.startswith("last_fit") or key == "fallback_status"}


def _validate_candidate(campaign, candidate, bounds):
    if not isinstance(candidate, pd.DataFrame) or len(candidate) != 1:
        raise ValueError("A synchronous benchmark step requires exactly one suggested row.")
    if candidate.iloc[0]["status"] != "suggested" or pd.notna(
        pd.to_numeric(candidate.iloc[0]["outcome"], errors="coerce")
    ):
        raise ValueError("Candidates must be unobserved suggestions.")
    point = candidate[campaign.config.variable_names].iloc[0].to_numpy(dtype=float)
    if not np.isfinite(point).all() or ((point < bounds[0]) | (point > bounds[1])).any():
        raise ValueError("Candidate is non-finite or outside the native problem bounds.")
    previous = campaign.df[campaign.config.variable_names].to_numpy(dtype=float)
    if len(previous) and np.any(np.all(previous == point, axis=1)):
        raise ValueError("Duplicate benchmark candidate; no retry or baseline fallback.")
    return point


@contextmanager
def _warning_evidence(directory):
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        previous = warnings.showwarning

        def record(message, category, filename, lineno, file=None, line=None):
            append_trace(directory / "warnings.jsonl", {
                "timestamp": utc_now(), "category": category.__name__, "message": str(message),
            })
            previous(message, category, filename, lineno, file=file, line=line)

        warnings.showwarning = record
        yield


def _observe(campaign, candidate, point, index, trial, problem, rows, noise, objective):
    start = time.monotonic()
    latent = (objective or true_value)(problem, point)
    deviation = trial["problem"]["noise_std"] if trial["mode"] == "noisy" else 0.0
    observed = float(latent + deviation * noise[index])
    if not math.isfinite(latent) or not math.isfinite(observed):
        raise ValueError("Objective observation is not finite.")
    scored = score(rows, float(latent), observed, float(problem.optimal_value),
                   trial["problem"]["optimum_tolerance"])
    objective_seconds = time.monotonic() - start
    start = time.monotonic()
    campaign.mark_observed(str(candidate.iloc[0]["row_id"]), observed)
    campaign.reload()
    campaign.validate()
    if len(campaign.observed_data()) != index + 1 or len(campaign.pending_suggestions()):
        raise ValueError("Reloaded campaign does not match the completed evaluation count.")
    return scored, objective_seconds, time.monotonic() - start


def run_trial(directory, *, suggest=None, objective=None):
    """Write durable evidence after each observed step; exceptions end the trial visibly."""
    directory = Path(directory)
    trial = json.loads((directory / "trial.json").read_text())
    problem = problem_for(trial["problem"]["name"])
    bounds = problem.bounds.numpy()
    config_path, log_path = directory / "campaign.yaml", directory / "campaign.csv"
    with config_path.open("x", encoding="utf-8") as handle:
        yaml.safe_dump(campaign_config(trial, bounds), handle, sort_keys=False)
    campaign = CampaignSession.initialize(config_path, log_path)
    sobol, random, noise = inputs(problem, trial)
    write_json(directory / "inputs.json", {
        "bounds": bounds.tolist(), "optimum": float(problem.optimal_value),
        "optimum_tolerance": trial["problem"]["optimum_tolerance"],
        "config_sha256": sha256(config_path.read_bytes()),
        "initial_design_sha256": sha256(sobol[:trial["initial_observations"]].tobytes()),
        "seed_mapping": trial["seeds"], "dtype": "float64", "direction": "minimize",
    })
    rows = []
    with _warning_evidence(directory):
        for index in range(trial["evaluations"]):
            _step(directory, campaign, trial, problem, sobol, random, noise,
                  index, rows, suggest, objective)
    result = {"status": "complete", "completed_evaluations": len(rows), "message": "",
              "finished_at": utc_now()}
    write_json(directory / "status.json", result)
    return result


def _step(directory, campaign, trial, problem, sobol, random, noise,
          index, rows, suggest, objective):
    write_json(directory / "status.json", {"status": "running", "phase": "suggest",
                                           "completed_evaluations": len(rows)})
    start = time.monotonic()
    candidate, evidence = _candidate(campaign, trial, index, sobol, random, suggest)
    suggestion_seconds = time.monotonic() - start
    point = _validate_candidate(campaign, candidate, problem.bounds.numpy())
    start = time.monotonic()
    campaign.append_suggestions(candidate)
    append_seconds = time.monotonic() - start
    write_json(directory / "status.json", {"status": "running", "phase": "observe",
                                           "completed_evaluations": len(rows)})
    scored, objective_seconds, observation_seconds = _observe(
        campaign, candidate, point, index, trial, problem, rows, noise, objective,
    )
    row = {"evaluation": index + 1, "row_id": str(candidate.iloc[0]["row_id"]),
           "x": point.tolist(), "source": str(candidate.iloc[0]["source"]),
           "noise": scored["observed"] - scored["latent"], **scored,
           "suggestion_seconds": suggestion_seconds, "objective_seconds": objective_seconds,
           "mutation_seconds": append_seconds + observation_seconds, "fit_evidence": evidence}
    append_trace(directory / "trace.jsonl", row)
    rows.append(row)
