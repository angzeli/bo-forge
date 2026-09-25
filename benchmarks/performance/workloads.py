"""Frozen first-model-call inputs shared by both compared installations.

``materialize(root)`` creates ``<route>/campaign.yaml`` and ``campaign.csv``.
Returned ``config_path``/``log_path`` are POSIX paths relative to ``root``;
``config_sha256``/``log_sha256`` identify their exact bytes. Each route also
records ``expected_source``, ``initial_observations``, ``seed``, ``seeds``,
``settings`` (the entire serialized config), and ``origin``. The caller owns
the run manifest and must reuse these files, not regenerate them per version.
No campaign manifest is created or changed here.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import yaml

from benchmarks.definitions import campaign_config
from benchmarks.designs import Proposals
from benchmarks.multi import evaluate, problem_for_route
from benchmarks.problems import inputs, problem_for, true_value
from benchmarks.spec import load_spec, schedule
from bo_forge.config import CampaignConfig
from bo_forge.validation import canonical_columns, validate_campaign_data

ROOT = Path(__file__).resolve().parents[2]


def _standard(root: Path, route: str, spec_name: str, problem_name: str) -> dict:
    spec_path = ROOT / "benchmarks" / "specs" / spec_name
    trial = next(item for item in schedule(load_spec(spec_path))
                 if item["problem"]["name"] == problem_name and item["seed"] == 0
                 and item["mode"] == "deterministic" and item["strategy"] == "bo")
    directory = root / route
    directory.mkdir(parents=True)
    config_path = directory / "campaign.yaml"
    config_path.write_text(yaml.safe_dump(campaign_config(trial), sort_keys=False),
                           encoding="utf-8")
    config = CampaignConfig.from_yaml(config_path)
    count = trial["initial_observations"]
    if route == "log_ei":
        problem = problem_for(problem_name)
        points = inputs(problem, trial)[0][:count].tolist()
        values = [[true_value(problem, point)] for point in points]
    else:
        proposals, points = Proposals(trial), []
        for _ in range(count):
            points.append(proposals.draw("sobol", points))
        problem = problem_for_route(trial["route"])
        values = [evaluate(problem, point) for point in points]
    columns = canonical_columns(config)
    objectives = config.objective_names if config.is_multi_objective else [config.objective.name]
    rows = []
    for index, (point, value) in enumerate(zip(points, values, strict=True)):
        row = dict.fromkeys(columns, "")
        row.update(row_id=f"eval_{index + 1:06d}", iteration=index + 1,
                   status="observed", source="sobol")
        row.update(zip(config.variable_names, point, strict=True))
        row.update(zip(objectives, value, strict=True))
        rows.append(row)
    frame = pd.DataFrame(rows, columns=columns)
    validate_campaign_data(config, frame)
    frame.to_csv(directory / "campaign.csv", index=False)
    return _metadata(root, route, trial["seed"], trial["seeds"], {
        "spec_path": spec_path.relative_to(ROOT).as_posix(),
        "spec_sha256": _sha256(spec_path), "trial_id": trial["trial_id"],
        "initialization_policy": "shared_sobol", "objective_policy": "initial_rows_only",
    })


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metadata(root: Path, route: str, seed: int, seeds: dict, origin: dict) -> dict:
    config_path, log_path = (root / route / name for name in ("campaign.yaml", "campaign.csv"))
    config = CampaignConfig.from_yaml(config_path)
    frame = pd.read_csv(log_path)
    validate_campaign_data(config, frame)
    if list(frame.columns) != canonical_columns(config) or not frame.status.eq("observed").all():
        raise ValueError("Performance inputs must contain canonical observed rows only.")
    if len(frame) != config.bo.initial_design_size:
        raise ValueError("Performance inputs must end at the first model-based call.")
    return {
        "config_path": config_path.relative_to(root).as_posix(),
        "log_path": log_path.relative_to(root).as_posix(),
        "config_sha256": _sha256(config_path), "log_sha256": _sha256(log_path),
        "expected_source": route, "initial_observations": len(frame),
        "seed": seed, "seeds": seeds,
        "settings": yaml.safe_load(config_path.read_text(encoding="utf-8")), "origin": origin,
    }


def materialize(root: Path) -> dict:
    """Create the three immutable input pairs, refusing existing route directories."""
    root = Path(root)
    routes = ("log_ei", "qlog_ehvi", "qmf_kg")
    for route in routes:
        if (root / route).exists():
            raise FileExistsError(root / route)
    result = {
        "log_ei": _standard(root, "log_ei", "standard.yaml", "branin"),
        "qlog_ehvi": _standard(root, "qlog_ehvi", "multi_objective_standard.yaml", "branin_currin"),
    }
    directory = root / "qmf_kg"
    directory.mkdir(parents=True)
    config_source = ROOT / "configs/22_discrete_multi_fidelity_qmfkg.yaml"
    log_source = ROOT / "examples/22_discrete_multi_fidelity_qmfkg_campaign_log.csv"
    (directory / "campaign.yaml").write_bytes(config_source.read_bytes())
    (directory / "campaign.csv").write_bytes(log_source.read_bytes())
    result["qmf_kg"] = _metadata(root, "qmf_kg", 22, {"fitting": 22}, {
        "config_path": config_source.relative_to(ROOT).as_posix(),
        "log_path": log_source.relative_to(ROOT).as_posix(),
        "initialization_policy": "exact_supplied_six_observations",
        "objective_policy": "supplied_observations_no_evaluation",
    })
    return result
