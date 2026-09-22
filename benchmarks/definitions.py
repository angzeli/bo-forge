"""Declarative benchmark inputs shared by execution and fitting-free verification."""

from copy import deepcopy

NATIVE = {
    "branin": ([[-5.0, 0.0], [10.0, 15.0]], 0.397887),
    "hartmann3": ([[0.0] * 3, [1.0] * 3], -3.86278),
    "hartmann6": ([[0.0] * 6, [1.0] * 6], -3.32237),
    "branin_currin": ([[0.0, 0.0], [1.0, 1.0]], None),
    "augmented_branin": ([[-5.0, 0.0, 0.0], [10.0, 15.0, 1.0]], 0.397887),
}
MIXED_VARIABLES = [
    {"name": "x", "type": "continuous", "lower": -1.0, "upper": 1.0},
    {"name": "k", "type": "integer", "lower": 0, "upper": 4},
    {"name": "z", "type": "discrete", "values": [0.0, 0.5, 1.0]},
    {"name": "c", "type": "categorical", "values": ["A", "B", "C"]},
]
MIXED_CONSTRAINTS = [
    {"name": "category_k", "expression": "c != 'B' or k >= 3"},
    {"name": "resource", "expression": "x + 0.125*k <= 0.625"},
]


def definition(trial):
    name = trial["problem"]["name"]
    if name == "mixed_quadratic":
        constrained = trial["route"] == "constrained_mixed"
        return {"variables": deepcopy(MIXED_VARIABLES),
                "bounds": [[-1.0, 0.0, 0.0, 0.0], [1.0, 4.0, 1.0, 2.0]],
                "constraints": deepcopy(MIXED_CONSTRAINTS) if constrained else [],
                "optimum": 0.125 if constrained else 0.0}
    bounds, optimum = NATIVE[name]
    return {"bounds": deepcopy(bounds), "optimum": optimum, "constraints": [],
            "variables": [{"name": f"x{i + 1}", "type": "continuous",
                           "lower": lower, "upper": upper}
                          for i, (lower, upper) in enumerate(zip(*bounds, strict=True))]}


def campaign_config(trial, bounds=None):
    """Campaign inputs deliberately exclude simulator optimum and noise parameters."""
    problem = definition(trial)
    result = {
        "campaign_name": trial["trial_id"],
        "objective": {"name": "outcome", "direction": "minimize"},
        "variables": problem["variables"], "model": {"profile": "default"},
        "bo": {**trial["bo"], "batch_size": 1,
               "initial_design_size": trial["initial_observations"],
               "initial_design_method": "sobol", "random_seed": trial["seeds"]["fitting"],
               "acquisition": "qlog_nei" if trial["mode"] == "noisy" else "log_ei"},
    }
    if problem["constraints"]:
        result["constraints"] = problem["constraints"]
    if trial.get("route") == "pending_noisy":
        result["review"] = {"enabled": True}
    if trial.get("route") == "multi_objective":
        result.pop("objective")
        result["objectives"] = [
            {"name": name, "direction": "minimize", "reference_point": reference}
            for name, reference in zip(("branin", "currin"), trial["reference_point"], strict=True)
        ]
        result["bo"]["acquisition"] = "qlog_ehvi"
    elif trial.get("route") == "multi_fidelity":
        result["fidelity"] = {"variable": "x3", **trial["fidelity"]}
        result["bo"]["acquisition"] = "qmf_kg"
    return result


def expected_source(trial, index):
    if index < trial["initial_observations"]:
        return "sobol"
    if trial["strategy"] != "bo":
        return trial["strategy"]
    if trial.get("schema_version") == 3:
        return "qlog_ehvi" if trial["route"] == "multi_objective" else "qmf_kg"
    return "qlog_nei" if trial["mode"] == "noisy" else "log_ei"
