"""Use installed BoTorch synthetic objectives, native bounds, and minimization."""

from __future__ import annotations

import math

import numpy as np
import torch
from botorch.test_functions import Branin, Hartmann


def problem_for(name):
    if name == "branin":
        return Branin(noise_std=None, negate=False).to(dtype=torch.double, device="cpu")
    if name in {"hartmann3", "hartmann6"}:
        return Hartmann(dim=int(name[-1]), noise_std=None, negate=False).to(
            dtype=torch.double, device="cpu",
        )
    raise ValueError(f"Unsupported problem: {name}")


def inputs(problem, trial):
    """Separate local streams; Sobol baseline continues the shared initial sequence."""
    count, dimension = trial["evaluations"], problem.dim
    unit = torch.quasirandom.SobolEngine(
        dimension, scramble=True, seed=trial["seeds"]["initialization"],
    ).draw(count, dtype=torch.double).numpy()
    bounds = problem.bounds.numpy()
    sobol = bounds[0] + unit * (bounds[1] - bounds[0])
    generator = np.random.default_rng(trial["seeds"]["baseline"])
    random = generator.uniform(bounds[0], bounds[1], size=(count, dimension))
    noise = np.random.default_rng(trial["seeds"]["observation_noise"]).normal(size=count)
    return sobol, random, noise


def true_value(problem, candidate):
    with torch.no_grad():
        value = float(problem.evaluate_true(torch.tensor(candidate, dtype=torch.double)).item())
    if not math.isfinite(value):
        raise ValueError("Simulator returned a non-finite objective.")
    return value


def regret(value, optimum, tolerance):
    raw = float(value - optimum)
    if not math.isfinite(raw) or raw < -tolerance:
        raise ValueError(f"Invalid simple regret {raw}; optimum={optimum}, tolerance={tolerance}.")
    # BoTorch publishes rounded optima. Only roundoff inside the stated tolerance is zeroed.
    return raw, max(raw, 0.0)


def score(rows, latent, observed, optimum, tolerance):
    best_latent = min([latent, *(row["latent"] for row in rows)])
    previous = min(rows, key=lambda row: row["observed"], default=None)
    incumbent = latent if previous is None or observed < previous["observed"] else (
        previous["latent"]
    )
    best_observed = min([observed, *(row["observed"] for row in rows)])
    raw, simple = regret(best_latent, optimum, tolerance)
    incumbent_raw, incumbent_regret = regret(incumbent, optimum, tolerance)
    return {"latent": latent, "observed": observed, "best_observed": best_observed,
            "best_latent": best_latent, "simple_regret_raw": raw, "simple_regret": simple,
            "incumbent_latent": incumbent, "incumbent_regret_raw": incumbent_raw,
            "incumbent_regret": incumbent_regret}
