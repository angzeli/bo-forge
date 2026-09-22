"""Fitting-free, objective-free validation of version-3 scoring artifacts."""

from benchmarks.binding import require_equal
from benchmarks.definitions import expected_source
from benchmarks.evidence import finite_number
from benchmarks.multi_scores import OBJECTIVES, mf_score, mo_score
from benchmarks.scoring import _close, _verify_design

COMMON = ["trial_id", "problem", "mode", "strategy", "seed", "status", "route", "evaluation",
          "row_id", "x", "source", "suggestion_seconds", "objective_seconds", "oracle_seconds",
          "mutation_seconds", "fit_evidence", "pending_row_ids", "feasible"]
MO = ["objective_names", "reference_point", "observed_objectives", "latent_objectives",
      "pareto_row_ids", "pareto_count", "hypervolume"]
MF = ["observed", "latent", "fidelity", "is_target", "modeled_evaluation_cost",
      "cumulative_modeled_cost", "target_observed_count", "best_target_observed",
      "target_regret_raw", "target_regret", "oracle_target_value", "oracle_best_target_value",
      "oracle_target_regret_raw", "oracle_target_regret"]


def trace_columns(route):
    return COMMON + (MO if route == "multi_objective" else MF)


def _equal_score(trial, key, expected, actual):
    if isinstance(expected, float):
        finite_number(actual, key)
        _close(actual, expected, key)
    else:
        require_equal(trial, key, expected, actual)


def verify_multi_rows(rows, observed, trial):
    seen, history = set(), []
    fields = MO if trial["route"] == "multi_objective" else MF
    allowed = set(trace_columns(trial["route"])) - {
        "trial_id", "problem", "mode", "strategy", "seed", "status", "route",
    }
    for index, row in enumerate(rows, 1):
        require_equal(trial, "trace fields", allowed, set(row))
        if row["evaluation"] != index or row["row_id"] in seen:
            raise ValueError(f"{trial['trial_id']}: Trace rows must be unique and ordered.")
        seen.add(row["row_id"])
        if row["row_id"] not in observed.index:
            raise ValueError(f"{trial['trial_id']}: Trace row missing from observed campaign.")
        actual = observed.loc[row["row_id"]]
        require_equal(trial, f"trace row {index} source", expected_source(trial, index - 1),
                      row["source"])
        require_equal(trial, "CSV source", row["source"], actual.source)
        _verify_design(actual, row["x"], trial)
        if trial["route"] == "multi_objective":
            values = row.get("observed_objectives")
            if not isinstance(values, list) or len(values) != 2:
                raise ValueError("Coupled trace must contain two ordered objective values.")
            for name, value in zip(OBJECTIVES, values, strict=True):
                finite_number(value, name)
                _close(float(actual[name]), value, name)
            expected = mo_score(history, dict(zip(OBJECTIVES, values, strict=True)),
                                trial, row["row_id"])
            _close(row["oracle_seconds"], 0.0, "MO oracle time")
        else:
            finite_number(row.get("observed"), "observed")
            finite_number(row.get("oracle_target_value"), "oracle_target_value")
            _close(float(actual.outcome), row["observed"], "campaign outcome")
            expected = mf_score(history, row["observed"], row["oracle_target_value"],
                                row["x"], trial)
            if row["x"][-1] == trial["fidelity"]["target"]:
                _close(row["observed"], row["oracle_target_value"], "target oracle agreement")
        for key in fields:
            _equal_score(trial, key, expected[key], row.get(key))
        history.append(row)
