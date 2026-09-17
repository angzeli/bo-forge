"""Benchmark protocol, strict schema, independent streams, and analytical scores."""

import copy
import math

import numpy as np
import pytest
import torch

from benchmarks.problems import inputs, problem_for, regret, score, true_value
from benchmarks.spec import SpecError, load_spec, schedule, seed_mapping, validate_spec
from tests._benchmark_support import ROOT, smoke_spec


def test_fixed_suites_and_budgets():
    smoke = load_spec(ROOT / "benchmarks/specs/smoke.yaml")
    standard = load_spec(ROOT / "benchmarks/specs/standard.yaml")
    trials = schedule(smoke)
    assert len(trials) == 6
    assert sum(t["evaluations"] - t["initial_observations"]
               for t in trials if t["strategy"] == "bo") == 4
    trials = schedule(standard)
    assert len(trials) == 90
    assert sum(t["evaluations"] for t in trials) == 2160
    assert {t["problem"]["name"]: t["initial_observations"] for t in trials} == {
        "branin": 6, "hartmann3": 8, "hartmann6": 14,
    }
    assert {t["problem"]["name"]: t["problem"]["noise_std"] for t in trials} == {
        "branin": 1.0, "hartmann3": 0.05, "hartmann6": 0.05,
    }
    assert all(t["timeout_seconds"] == 600 for t in trials)


@pytest.mark.parametrize("key,value", [
    ("schema_version", 2), ("schema_version", True), ("seeds", []), ("seeds", [True]),
    ("seeds", [0, 0]), ("seeds", [-1]), ("modes", ["noisier"]), ("strategies", ["bo", "bo"]),
    ("strategies", ["qmf_kg"]), ("timeout_seconds", math.nan), ("timeout_seconds", 0),
    ("timeout_seconds", 601), ("evaluations", 4), ("initial_observations", True),
    ("initial_observations", "guess"), ("unknown", "bad"), ("name", ""),
])
def test_invalid_protocol_is_rejected(key, value):
    spec = smoke_spec()
    spec[key] = value
    with pytest.raises(SpecError):
        validate_spec(spec)


@pytest.mark.parametrize("section,key,value", [
    ("problem", "name", "rastrigin"), ("problem", "noise_std", -1),
    ("problem", "noise_std", 0), ("problem", "optimum_tolerance", 1),
    ("bo", "raw_samples", 0), ("bo", "num_restarts", True), ("bo", "mc_samples", 1.2),
    ("bo", "min_normalized_distance", float("inf")), ("bo", "new_optimizer", "bad"),
])
def test_nested_invalid_settings(section, key, value):
    spec = smoke_spec()
    target = spec["problems"][0] if section == "problem" else spec[section]
    target[key] = value
    with pytest.raises(SpecError):
        validate_spec(spec)


def test_duplicate_yaml_keys_fail(tmp_path):
    path = tmp_path / "invalid.yaml"
    path.write_text("name: first\nname: second\n")
    with pytest.raises(SpecError, match="unique"):
        load_spec(path)


def test_separated_streams_pair_initial_points_and_noise_without_global_rng():
    trial = schedule(smoke_spec())[0]
    problem = problem_for("branin")
    mapping = seed_mapping("branin", 0, "deterministic")
    assert mapping == trial["seeds"]
    assert len(set(mapping.values())) == 4
    expected = inputs(problem, trial)
    with torch.random.fork_rng():
        torch.manual_seed(997)
        torch.randn(100)
        actual = inputs(problem, {**trial, "strategy": "random"})
    for left, right in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(left, right)
    changed = copy.deepcopy(trial)
    changed["seeds"]["fitting"] += 1
    for left, right in zip(expected, inputs(problem, changed), strict=True):
        np.testing.assert_array_equal(left, right)
    changed["seeds"]["baseline"] += 1
    new = inputs(problem, changed)
    np.testing.assert_array_equal(expected[0], new[0])
    np.testing.assert_array_equal(expected[2], new[2])
    assert not np.array_equal(expected[1], new[1])


@pytest.mark.parametrize("name", ["branin", "hartmann3", "hartmann6"])
def test_native_botorch_bounds_minimization_and_rounded_optima(name):
    problem = problem_for(name)
    point = problem.optimizers[0].tolist()
    value = true_value(problem, point)
    tolerance = 1e-6 if name == "branin" else 1e-5
    assert abs(value - problem.optimal_value) < tolerance
    assert problem.negate is False and problem.noise_std is None
    if name == "branin":
        assert problem.bounds.tolist() == [[-5.0, 0.0], [10.0, 15.0]]
    else:
        assert problem.bounds.tolist() == [[0.0] * problem.dim, [1.0] * problem.dim]


def test_minimization_and_noisy_incumbent_are_distinct():
    first = score([], 5.0, 10.0, 2.0, 1e-6)
    second = score([first], 7.0, 1.0, 2.0, 1e-6)
    assert second["simple_regret"] == 3
    assert second["incumbent_regret"] == 5
    assert second["best_observed"] == 1
    tied = score([first, second], 3.0, 1.0, 2.0, 1e-6)
    assert tied["incumbent_latent"] == 7
    assert tied["simple_regret"] == 1
    raw, corrected = regret(2 - 5e-7, 2.0, 1e-6)
    assert raw < 0 and corrected == 0
    with pytest.raises(ValueError, match="Invalid simple regret"):
        regret(1.9, 2.0, 1e-6)
