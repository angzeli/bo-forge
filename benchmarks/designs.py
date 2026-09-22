"""Typed designs and bounded feasible proposals for the version-2 benchmark routes."""

import json
import math

import numpy as np
import torch

from benchmarks.definitions import definition
from benchmarks.storage import sha256

DRAW_LIMIT = 10_000


def typed_digest(points):
    return sha256(json.dumps(points, separators=(",", ":"), allow_nan=False).encode("utf-8"))


def feasible(point, trial):
    if trial.get("route") != "constrained_mixed":
        return True
    x, k, _, c = point
    return (c != "B" or k >= 3) and x + .125 * k <= .625


def normalize(point, trial):
    variables = definition(trial)["variables"]
    if len(point) != len(variables):
        raise ValueError("Candidate dimension differs from the scheduled design.")
    result = []
    for value, variable in zip(point, variables, strict=True):
        if variable["type"] == "categorical":
            if value not in variable["values"]:
                raise ValueError("Invalid categorical candidate.")
            result.append(str(value))
            continue
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("Non-finite candidate.")
        if variable["type"] == "discrete":
            valid = numeric in variable["values"]
        else:
            valid = variable["lower"] <= numeric <= variable["upper"]
        if not valid or (variable["type"] == "integer" and not numeric.is_integer()):
            raise ValueError("Candidate outside its configured bounds or finite choices.")
        result.append(int(numeric) if variable["type"] == "integer" else numeric)
    if not feasible(result, trial):
        raise ValueError("Candidate violates benchmark feasibility constraints.")
    return result


def decode(unit, trial):
    result = []
    for u, variable in zip(unit, definition(trial)["variables"], strict=True):
        if variable["type"] == "continuous":
            value = variable["lower"] + float(u) * (variable["upper"] - variable["lower"])
        else:
            choices = (list(range(variable["lower"], variable["upper"] + 1))
                       if variable["type"] == "integer" else variable["values"])
            value = choices[min(int(float(u) * len(choices)), len(choices) - 1)]
        result.append(value)
    return result


class Proposals:
    """One total draw limit covers shared initialization and subsequent baseline proposals."""

    def __init__(self, trial):
        self.trial = trial
        dimension = len(definition(trial)["variables"])
        self.sobol = torch.quasirandom.SobolEngine(
            dimension, scramble=True, seed=trial["seeds"]["initialization"],
        )
        self.random = np.random.default_rng(trial["seeds"]["baseline"])
        self.counts = {"draws": 0, "infeasible": 0, "duplicate": 0}

    def draw(self, source, existing):
        seen = {typed_digest(point) for point in existing}
        while self.counts["draws"] < DRAW_LIMIT:
            self.counts["draws"] += 1
            unit = (self.sobol.draw(1, dtype=torch.double)[0].tolist() if source == "sobol"
                    else self.random.random(len(definition(self.trial)["variables"])).tolist())
            point = decode(unit, self.trial)
            if self.trial.get("route") == "multi_fidelity" and (
                len(existing) < 2 or (len(existing) >= self.trial["initial_observations"]
                                      and self.trial["strategy"] != "bo")
            ):
                point[-1] = self.trial["fidelity"]["target"]
            if not feasible(point, self.trial):
                self.counts["infeasible"] += 1
            elif typed_digest(point) in seen:
                self.counts["duplicate"] += 1
            else:
                return point
        raise ValueError(f"Baseline proposal limit exhausted after {DRAW_LIMIT} draws.")


def mixed_value(point):
    x, k, z, c = point
    return (x - .25)**2 + .125 * (k - 2)**2 + .25 * (z - .5)**2 + {
        "A": .5, "B": 0.0, "C": .25,
    }[c]
