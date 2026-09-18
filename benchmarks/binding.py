"""Bind all available campaign artifacts to their scheduled trial, never relabel them."""

import math
from dataclasses import asdict

import pandas as pd
import yaml

from benchmarks.definitions import campaign_config, definition, expected_source
from benchmarks.spec import seed_mapping
from bo_forge.config import parse_campaign_config


def require_equal(trial, field, expected, actual):
    if expected != actual:
        raise ValueError(f"{trial['trial_id']}: {field}: expected {expected!r}; actual {actual!r}")


def _compare(trial, expected, actual, path="config"):
    if isinstance(expected, dict) and isinstance(actual, dict):
        require_equal(trial, path + ".keys", set(expected), set(actual))
        for key, value in expected.items():
            _compare(trial, value, actual[key], f"{path}.{key}")
    else:
        require_equal(trial, path, expected, actual)


def verify_binding(directory, trial, inputs=None):
    route = trial.get("route")
    expected_seeds = seed_mapping(trial["problem"]["name"], trial["seed"], trial["mode"],
                                  route=route)
    require_equal(trial, "seeds", expected_seeds, trial["seeds"])
    config_path = directory / "campaign.yaml"
    if config_path.exists():
        actual = parse_campaign_config(yaml.safe_load(config_path.read_text(encoding="utf-8")))
        expected = parse_campaign_config(campaign_config(trial))
        _compare(trial, asdict(expected), asdict(actual))
    log = directory / "campaign.csv"
    if log.exists():
        frame = pd.read_csv(log, keep_default_na=False)
        for index, row in frame.iterrows():
            require_equal(trial, f"CSV row {index + 1} source", expected_source(trial, index),
                          row.get("source"))
    if inputs is not None:
        problem = definition(trial)
        for key, value in (("seed_mapping", expected_seeds), ("bounds", problem["bounds"]),
                           ("optimum", problem["optimum"]), ("direction", "minimize")):
            require_equal(trial, f"inputs.{key}", value, inputs.get(key))
        initial = _verify_representation(trial, inputs)
        if initial is not None and log.exists():
            _verify_initial_rows(trial, frame, initial)


def _verify_representation(trial, inputs):
    expected_dtype = "typed_json" if trial.get("schema_version") == 2 else "float64"
    require_equal(trial, "inputs.dtype", expected_dtype, inputs.get("dtype"))
    digest = inputs.get("initial_design_sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef"
                                                            for c in digest):
        raise ValueError(f"{trial['trial_id']}: inputs.initial_design_sha256 is not a SHA-256.")
    if trial.get("schema_version") != 2:
        return
    from benchmarks.designs import Proposals, typed_digest

    require_equal(trial, "inputs.variables", definition(trial)["variables"],
                  inputs.get("variables"))
    proposals, initial = Proposals(trial), []
    for _ in range(trial["initial_observations"]):
        initial.append(proposals.draw("sobol", initial))
    require_equal(trial, "inputs.initial_design_sha256", typed_digest(initial), digest)
    return initial


def _verify_initial_rows(trial, frame, initial):
    names = [v["name"] for v in definition(trial)["variables"]]
    for index, point in enumerate(frame[names].iloc[:len(initial)].values.tolist()):
        for name, expected, actual in zip(names, initial[index], point, strict=True):
            if isinstance(expected, str):
                require_equal(trial, f"initial row {index + 1}.{name}", expected, actual)
            elif not math.isclose(float(actual), expected, abs_tol=1e-12, rel_tol=1e-12):
                require_equal(trial, f"initial row {index + 1}.{name}", expected, actual)
