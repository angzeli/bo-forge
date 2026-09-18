"""Strict, bounded benchmark specifications and reproducible trial scheduling."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import yaml

from benchmarks import SPEC_VERSION

DIMENSIONS = {"branin": 2, "hartmann3": 3, "hartmann6": 6, "mixed_quadratic": 4}
ROUTES = ("mixed", "constrained_mixed", "pending_noisy")
MODES = ("deterministic", "noisy")
STRATEGIES = ("bo", "random", "sobol")
STREAMS = ("initialization", "baseline", "observation_noise", "fitting")


class SpecError(ValueError):
    """The repository benchmark specification is invalid."""


class _Loader(yaml.SafeLoader):
    pass


def _mapping(loader, node):
    result = {}
    for key, value in node.value:
        name = loader.construct_object(key, deep=True)
        if not isinstance(name, str) or name in result:
            raise SpecError("YAML keys must be unique strings.")
        result[name] = loader.construct_object(value, deep=True)
    return result


_Loader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def _keys(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise SpecError(f"{label} requires exactly these keys: {', '.join(expected)}.")


def _integer(value, low, high, label):
    if type(value) is not int or not low <= value <= high:
        raise SpecError(f"{label} must be an integer in {low}..{high}.")


def _number(value, low, high, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SpecError(f"{label} must be a finite number.")
    if not math.isfinite(value) or not low <= value <= high:
        raise SpecError(f"{label} must be finite and in {low}..{high}.")


def _choices(value, allowed, label):
    if not isinstance(value, list) or not value:
        raise SpecError(f"{label} must be a nonempty list.")
    if any(not isinstance(item, str) or item not in allowed for item in value):
        raise SpecError(f"{label} supports only {', '.join(allowed)}.")
    if len(set(value)) != len(value):
        raise SpecError(f"{label} must not contain duplicates.")


def validate_spec(spec):
    if not isinstance(spec, dict):
        raise SpecError("spec must be a mapping.")
    _integer(spec.get("schema_version"), 1, 2, "schema_version")
    keys = ("schema_version", "name", "problems", "seeds", "modes", "strategies",
            "evaluations", "initial_observations", "timeout_seconds", "bo")
    _keys(spec, keys + (("route",) if spec["schema_version"] == 2 else ()), "spec")
    if not isinstance(spec["name"], str) or not spec["name"].strip():
        raise SpecError("name must be a nonempty string.")
    _choices(spec["modes"], MODES, "modes")
    _choices(spec["strategies"], STRATEGIES, "strategies")
    seeds = spec["seeds"]
    if not isinstance(seeds, list) or not 1 <= len(seeds) <= 100:
        raise SpecError("seeds must contain 1..100 explicit integer seeds.")
    for seed in seeds:
        _integer(seed, 0, 2**32 - 1, "seed")
    if len(set(seeds)) != len(seeds):
        raise SpecError("seeds must not contain duplicates.")
    _integer(spec["evaluations"], 3, 1000, "evaluations")
    _number(spec["timeout_seconds"], 0.01, 600, "timeout_seconds")
    initial = spec["initial_observations"]
    if initial != "twice_dimension_plus_one":
        _integer(initial, 2, spec["evaluations"] - 1, "initial_observations")
    _validate_problems(spec)
    _validate_route(spec)
    _keys(spec["bo"], ("raw_samples", "num_restarts", "mc_samples",
                       "min_normalized_distance"), "bo")
    for key in ("raw_samples", "num_restarts", "mc_samples"):
        _integer(spec["bo"][key], 1, 4096, key)
    _number(spec["bo"]["min_normalized_distance"], 0, 1, "min_normalized_distance")
    return spec


def _validate_route(spec):
    names = [problem["name"] for problem in spec["problems"]]
    if spec["schema_version"] == 1:
        if "mixed_quadratic" in names:
            raise SpecError("mixed_quadratic requires a version-2 route.")
        return
    route = spec["route"]
    if route not in ROUTES:
        raise SpecError(f"route must be one of {ROUTES}.")
    pending = route == "pending_noisy"
    if names != (["branin"] if pending else ["mixed_quadratic"]):
        raise SpecError("Unsupported route/problem combination.")
    if spec["modes"] != (["noisy"] if pending else ["deterministic"]):
        raise SpecError("Unsupported route/mode combination.")
    if spec["problems"][0]["noise_std"] != (1.0 if pending else 0.0):
        raise SpecError("Route requires its fixed noise standard deviation.")
    if pending and (spec["evaluations"] - initial_count(spec, "branin")) % 2:
        raise SpecError("pending_noisy requires complete two-suggestion cycles.")


def _validate_problems(spec):
    problems = spec["problems"]
    supported = ("branin", "hartmann3", "hartmann6")
    if spec["schema_version"] == 2:
        supported = ("branin",) if spec["route"] == "pending_noisy" else ("mixed_quadratic",)
    if not isinstance(problems, list) or not 1 <= len(problems) <= len(DIMENSIONS):
        raise SpecError(f"problems must list supported problems: {', '.join(supported)}.")
    names = []
    for problem in problems:
        _keys(problem, ("name", "noise_std", "optimum_tolerance"), "problem")
        if not isinstance(problem["name"], str) or problem["name"] not in DIMENSIONS:
            raise SpecError(f"Unknown problem; use {', '.join(supported)} for this schema/route.")
        names.append(problem["name"])
        _number(problem["noise_std"], 0, 1000, "noise_std")
        if "noisy" in spec["modes"] and problem["noise_std"] == 0:
            raise SpecError("noisy mode requires positive noise_std.")
        _number(problem["optimum_tolerance"], 1e-12, 1e-4, "optimum_tolerance")
        if initial_count(spec, problem["name"]) >= spec["evaluations"]:
            raise SpecError("evaluations must exceed initial_observations for each problem.")
    if len(set(names)) != len(names):
        raise SpecError("problems must not contain duplicate names.")


def load_spec(path):
    return _parse_spec(Path(path).read_bytes())


def _parse_spec(payload):
    """Parse the same immutable bytes that execution and reporting retain and hash."""
    try:
        return validate_spec(yaml.load(payload, Loader=_Loader))
    except yaml.YAMLError as exc:
        raise SpecError(f"Invalid YAML: {exc}") from exc


def initial_count(spec, problem):
    value = spec["initial_observations"]
    return 2 * (DIMENSIONS[problem] + 1) if value == "twice_dimension_plus_one" else value


def seed_mapping(problem, seed, mode, *, route=None):
    """SHA-256 tagged streams, independent of strategy and process hash randomization."""
    return {
        stream: int.from_bytes(hashlib.sha256(json.dumps(
            ([SPEC_VERSION, problem, seed, mode, stream] if route is None
             else [2, route, problem, seed, mode, stream]), separators=(",", ":"),
        ).encode()).digest()[:4], "big") for stream in STREAMS
    }


def schedule(spec):
    trials = [
        {"trial_id": f"{problem['name']}-{mode}-{seed:010d}-{strategy}",
         "problem": problem, "mode": mode, "seed": seed, "strategy": strategy,
         "seeds": seed_mapping(problem["name"], seed, mode),
         "initial_observations": initial_count(spec, problem["name"]),
         "evaluations": spec["evaluations"], "bo": spec["bo"],
         "timeout_seconds": spec["timeout_seconds"]}
        for problem in spec["problems"] for mode in spec["modes"]
        for seed in spec["seeds"] for strategy in spec["strategies"]
    ]
    if spec["schema_version"] == 2:
        for trial in trials:
            trial.update(schema_version=2, route=spec["route"],
                         trial_id=f"{spec['route']}-{trial['trial_id']}",
                         seeds=seed_mapping(trial["problem"]["name"], trial["seed"],
                                            trial["mode"], route=spec["route"]))
    return trials
