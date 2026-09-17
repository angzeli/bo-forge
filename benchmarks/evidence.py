"""Validate recorded benchmark evidence without fitting or repairing campaigns."""

import json
import math

import pandas as pd

from benchmarks.storage import TERMINAL
from bo_forge._campaign.provenance_resume import inspect_provenance
from bo_forge.errors import BOForgeError

IDENTITY_FIELDS = {"trial_id", "problem", "mode", "strategy", "seed", "status"}
NUMBER_FIELDS = (
    "observed", "latent", "noise", "best_observed", "best_latent", "simple_regret_raw",
    "simple_regret", "incumbent_latent", "incumbent_regret_raw", "incumbent_regret",
    "suggestion_seconds", "objective_seconds", "mutation_seconds",
)


def read_object(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Benchmark metadata must be a JSON object: {path}")
    return value


def finite_number(value, label):
    if type(value) not in (int, float):
        raise ValueError(f"{label} must be a finite number.")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f"{label} must be a finite number.")


def validate_status(status):
    if not isinstance(status.get("status"), str) or status["status"] not in TERMINAL:
        raise ValueError("Missing or invalid terminal benchmark status.")
    count = status.get("completed_evaluations")
    if type(count) is not int or count < 0:
        raise ValueError("completed_evaluations must be a nonnegative integer.")
    if not isinstance(status.get("message", ""), str):
        raise ValueError("Benchmark status message must be text.")
    finite_number(status.get("wall_seconds", 0.0), "wall_seconds")
    if status.get("wall_seconds", 0.0) < 0:
        raise ValueError("wall_seconds must be nonnegative.")


def validate_rows(rows):
    for row in rows:
        if IDENTITY_FIELDS.intersection(row):
            raise ValueError("Trace rows must not override reserved trial identity fields.")
        if type(row.get("evaluation")) is not int or row["evaluation"] < 1:
            raise ValueError("Trace evaluation must be a positive integer.")
        for key in ("row_id", "source"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError(f"Trace {key} must be nonempty text.")
        if not isinstance(row.get("x"), list):
            raise ValueError("Trace x must be a list of finite coordinates.")
        for value in row["x"]:
            finite_number(value, "Trace coordinate")
        for key in NUMBER_FIELDS:
            finite_number(row.get(key), f"Trace {key}")
            if key.endswith("_seconds") and row[key] < 0:
                raise ValueError(f"Trace {key} must be nonnegative.")
        if not isinstance(row.get("fit_evidence"), dict):
            raise ValueError("Trace fit_evidence must be an object.")


def validate_inputs(inputs, trial):
    from benchmarks.spec import DIMENSIONS

    dimension = DIMENSIONS[trial["problem"]["name"]]
    bounds = inputs.get("bounds")
    if (not isinstance(bounds, list) or len(bounds) != 2
            or any(not isinstance(axis, list) or len(axis) != dimension for axis in bounds)):
        raise ValueError("Recorded bounds must match the configured problem dimension.")
    for axis in bounds:
        for value in axis:
            finite_number(value, "Recorded bound")
    finite_number(inputs.get("optimum"), "Recorded optimum")
    finite_number(inputs.get("optimum_tolerance"), "Recorded optimum tolerance")
    if inputs["optimum_tolerance"] != trial["problem"]["optimum_tolerance"]:
        raise ValueError("Recorded optimum tolerance differs from the trial specification.")
    if not isinstance(inputs.get("config_sha256"), str):
        raise ValueError("Recorded config_sha256 must be text.")


def campaign_evidence(directory, config, log, rows, complete):
    """Complete trials fail closed; incomplete trials disclose missing campaign evidence."""
    problems = []
    try:
        inspection = inspect_provenance(
            config, log, provenance_policy="required", include_environment=False,
        )
        if inspection.resume_status != "ready":
            problems.append(f"Provenance: {inspection.reason_code}")
    except (BOForgeError, OSError, ValueError) as exc:
        problems.append(f"Provenance: {exc}")
    if not log.exists():
        if rows:
            raise ValueError(f"Scored trace has no campaign CSV: {directory}")
        problems.append("Campaign CSV absent; no persisted observations can be verified.")
        observed = None
    else:
        frame = pd.read_csv(log, keep_default_na=False)
        if not {"row_id", "status", "outcome"}.issubset(frame.columns):
            raise ValueError(f"Campaign CSV lacks required benchmark columns: {directory}")
        if frame.row_id.duplicated().any():
            raise ValueError(f"Campaign CSV has duplicate row IDs: {directory}")
        observed = frame.loc[frame.status.eq("observed")].set_index("row_id")
        unscored = set(observed.index) - {row["row_id"] for row in rows}
        if unscored:
            problems.append(f"{len(unscored)} observed CSV row(s) have no scoring trace.")
        if complete and len(frame) != len(rows):
            problems.append("Completed campaign CSV and scoring trace have different row counts.")
    if complete and problems:
        message = "; ".join(problems)
        raise ValueError(f"Incomplete campaign evidence for {directory.name}: {message}")
    return observed, "; ".join(problems)
