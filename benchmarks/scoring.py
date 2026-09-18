"""Validate stored score evidence without loading or evaluating objective models."""

import math

from benchmarks.binding import require_equal
from benchmarks.definitions import definition, expected_source


def verify_rows(rows, observed, inputs, trial):
    best_latent, best_observed, incumbent = math.inf, math.inf, None
    seen = set()
    for index, row in enumerate(rows, 1):
        if row["evaluation"] != index or row["row_id"] in seen:
            raise ValueError("Trace evaluations and row IDs must be unique and ordered.")
        seen.add(row["row_id"])
        if row["row_id"] not in observed.index:
            raise ValueError("Trace row is missing from observed campaign history.")
        actual = observed.loc[row["row_id"]]
        require_equal(trial, f"trace row {index} source", expected_source(trial, index - 1),
                      row["source"])
        if actual.source != row["source"]:
            raise ValueError("Trace source differs from observed campaign history.")
        _close(float(actual.outcome), row["observed"], "campaign outcome")
        _verify_design(actual, row["x"], trial)
        _close(row["observed"] - row["latent"], row["noise"], "observation noise")
        if trial["mode"] == "deterministic":
            _close(row["noise"], 0.0, "deterministic observation")
        best_latent = min(best_latent, row["latent"])
        if row["observed"] < best_observed:
            incumbent = row["latent"]
        best_observed = min(best_observed, row["observed"])
        for name, value in (("best_latent", best_latent), ("best_observed", best_observed),
                            ("incumbent_latent", incumbent)):
            _close(row[name], value, name)
        for prefix, value in (("simple", best_latent), ("incumbent", incumbent)):
            raw = value - inputs["optimum"]
            if not math.isfinite(raw) or raw < -inputs["optimum_tolerance"]:
                raise ValueError("Stored scores imply an invalid negative regret.")
            _close(row[f"{prefix}_regret_raw"], raw, "raw regret")
            _close(row[f"{prefix}_regret"], max(raw, 0.0), "regret")


def _verify_design(actual, point, trial):
    variables = definition(trial)["variables"]
    if len(point) != len(variables):
        raise ValueError("Trace design dimension mismatch.")
    for variable, value in zip(variables, point, strict=True):
        if variable["type"] == "categorical":
            require_equal(trial, f"design.{variable['name']}", actual[variable["name"]], value)
        else:
            _close(float(actual[variable["name"]]), value, "campaign design")


def _close(actual, expected, label):
    if not math.isfinite(actual) or not math.isfinite(expected) or not math.isclose(
        actual, expected, rel_tol=1e-12, abs_tol=1e-12,
    ):
        raise ValueError(f"Inconsistent {label} in stored benchmark evidence.")
