"""Scoring-only vector and fidelity diagnostics; no simulator calls during reporting."""

import math

from benchmarks.problems import regret

OBJECTIVES = ["branin", "currin"]


def pareto_indices(values):
    """Minimization membership retains ties as distinct observed rows."""
    return [i for i, point in enumerate(values) if not any(
        all(a <= b for a, b in zip(other, point, strict=True))
        and any(a < b for a, b in zip(other, point, strict=True)) for other in values
    )]


def hypervolume_2d(values, reference):
    """Union of dominated rectangles in user-space minimization coordinates."""
    height, area = reference[1], 0.0
    for x, y in sorted(values):
        if x < reference[0] and y < height:
            area += (reference[0] - x) * (height - y)
            height = y
    return area


def mo_score(rows, observed, trial, row_id):
    if not isinstance(observed, dict) or set(observed) != set(OBJECTIVES):
        raise ValueError("Coupled observations require exactly branin and currin.")
    vector = [float(observed[name]) for name in OBJECTIVES]
    if not all(math.isfinite(value) for value in vector):
        raise ValueError("Coupled objective observations must be finite.")
    values = [r["observed_objectives"] for r in rows] + [vector]
    members = pareto_indices(values)
    ids = [r["row_id"] for r in rows] + [row_id]
    return {"objective_names": OBJECTIVES.copy(), "reference_point": trial["reference_point"],
            "observed_objectives": vector, "latent_objectives": vector.copy(),
            "pareto_row_ids": [ids[i] for i in members], "pareto_count": len(members),
            "hypervolume": hypervolume_2d(values, trial["reference_point"])}


def is_target(value, trial):
    return math.isclose(value, trial["fidelity"]["target"], rel_tol=1e-9, abs_tol=1e-9)


def mf_score(rows, observed, projected, point, trial):
    observed, projected = float(observed), float(projected)
    if not math.isfinite(observed) or not math.isfinite(projected):
        raise ValueError("Fidelity and oracle outcomes must be finite.")
    fidelity = trial["fidelity"]
    target = is_target(point[-1], trial)
    cost = fidelity["fixed_cost"] + fidelity["fidelity_cost_weight"] * point[-1]
    targets = [r["observed"] for r in rows if r["is_target"]]
    if target:
        targets.append(observed)
    best = min(targets) if targets else None
    oracle_best = min([projected, *(r["oracle_target_value"] for r in rows)])
    tolerance = trial["problem"]["optimum_tolerance"]
    raw, primary = regret(best, .397887, tolerance) if best is not None else (None, None)
    oracle_raw, oracle = regret(oracle_best, .397887, tolerance)
    return {"observed": observed, "latent": observed, "fidelity": point[-1],
            "is_target": target, "modeled_evaluation_cost": cost,
            "cumulative_modeled_cost": sum(r["modeled_evaluation_cost"] for r in rows) + cost,
            "target_observed_count": len(targets), "best_target_observed": best,
            "target_regret_raw": raw, "target_regret": primary,
            "oracle_target_value": projected, "oracle_best_target_value": oracle_best,
            "oracle_target_regret_raw": oracle_raw, "oracle_target_regret": oracle}
