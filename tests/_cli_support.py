import subprocess
import sys
from pathlib import Path

import matplotlib
import pandas as pd
import pytest
import torch

import bo_forge.cli as cli
import bo_forge.suggestions as suggestions_module
from bo_forge.cli import run
from bo_forge.config import (
    BOConfig,
    CampaignConfig,
    CostConfig,
    FidelityConfig,
    ObjectiveConfig,
    ReplicateConfig,
    ReviewConfig,
    VariableConfig,
)
from bo_forge.errors import LogBusyError, LogConflictError
from bo_forge.io import empty_campaign_log
from bo_forge.logs import load_campaign_log
from bo_forge.validation import canonical_columns

matplotlib.use("Agg")


def write_config(path: Path, *, initial_design_size: int = 2) -> Path:
    path.write_text(
        f"""
campaign_name: cli_test
objective:
  name: score
  direction: maximize
variables:
  - name: x
    type: continuous
    lower: 0
    upper: 1
bo:
  batch_size: 1
  initial_design_size: {initial_design_size}
  acquisition: log_ei
  random_seed: 7
  raw_samples: 16
  num_restarts: 2
  mc_samples: 16
""",
        encoding="utf-8",
    )
    return path


def write_mixed_config(path: Path, *, initial_design_size: int = 3) -> Path:
    path.write_text(
        f"""
campaign_name: mixed_cli_test
objective:
  name: score
  direction: maximize
variables:
  - name: x
    type: continuous
    lower: 0
    upper: 1
  - name: repeats
    type: integer
    lower: 1
    upper: 3
  - name: dose
    type: discrete
    values: [0.1, 0.2, 0.5]
  - name: solvent
    type: categorical
    values: [MeCN, EtOH]
bo:
  batch_size: 1
  initial_design_size: {initial_design_size}
  acquisition: log_ei
  random_seed: 7
  raw_samples: 16
  num_restarts: 2
  mc_samples: 16
""",
        encoding="utf-8",
    )
    return path


def write_cost_review_config(path: Path, *, initial_design_size: int = 2) -> Path:
    path.write_text(
        f"""
campaign_name: cost_review_cli_test
objective:
  name: score
  direction: maximize
variables:
  - name: x
    type: continuous
    lower: 0
    upper: 1
cost:
  expression: "1.0 + x"
  weight: 0.5
  budget: 10
  candidate_pool_size: 16
  top_k: 8
review:
  enabled: true
bo:
  batch_size: 1
  initial_design_size: {initial_design_size}
  acquisition: log_ei
  random_seed: 7
  raw_samples: 16
  num_restarts: 2
  mc_samples: 16
""",
        encoding="utf-8",
    )
    return path


def write_replicate_config(path: Path, *, initial_design_size: int = 2) -> Path:
    path.write_text(
        f"""
campaign_name: replicate_cli_test
objective:
  name: score
  direction: maximize
variables:
  - name: x
    type: continuous
    lower: 0
    upper: 1
replicates:
  enabled: true
bo:
  batch_size: 1
  initial_design_size: {initial_design_size}
  acquisition: log_ei
  random_seed: 7
  raw_samples: 16
  num_restarts: 2
  mc_samples: 16
""",
        encoding="utf-8",
    )
    return path


def write_multi_objective_config(path: Path, *, initial_design_size: int = 10) -> Path:
    path.write_text(
        f"""
campaign_name: multi_cli_test
objectives:
  - name: yield_score
    direction: maximize
    reference_point: 40
  - name: waste_score
    direction: minimize
    reference_point: 25
variables:
  - name: temperature
    type: continuous
    lower: 20
    upper: 100
bo:
  batch_size: 1
  initial_design_size: {initial_design_size}
  acquisition: qlog_ehvi
  random_seed: 7
  raw_samples: 8
  num_restarts: 2
  mc_samples: 8
""",
        encoding="utf-8",
    )
    return path


def write_multi_objective_cost_config(path: Path, *, initial_design_size: int = 10) -> Path:
    path.write_text(
        f"""
campaign_name: multi_cost_cli_test
objectives:
  - name: yield_score
    direction: maximize
    reference_point: 40
  - name: waste_score
    direction: minimize
    reference_point: 25
variables:
  - name: temperature
    type: continuous
    lower: 20
    upper: 100
cost:
  expression: "1.0 + 0.02 * temperature"
  weight: 0.5
  budget: 20
  candidate_pool_size: 16
  top_k: 8
bo:
  batch_size: 1
  initial_design_size: {initial_design_size}
  acquisition: qlog_ehvi
  random_seed: 7
  raw_samples: 8
  num_restarts: 2
  mc_samples: 8
""",
        encoding="utf-8",
    )
    return path


def write_fidelity_config(
    path: Path,
    *,
    initial_design_size: int = 3,
    review: bool = False,
    timeout_seconds: float | None = None,
) -> Path:
    review_block = "review:\n  enabled: true\n" if review else ""
    timeout_line = (
        ""
        if timeout_seconds is None
        else f"  optimizer_timeout_seconds: {timeout_seconds:g}\n"
    )
    path.write_text(
        f"""
campaign_name: fidelity_cli_test
objective:
  name: activity
  direction: maximize
variables:
  - name: x
    type: continuous
    lower: 0
    upper: 1
  - name: fidelity
    type: continuous
    lower: 0.2
    upper: 1.0
fidelity:
  variable: fidelity
  target: 1.0
  num_fantasies: 8
{timeout_line}\
{review_block}\
bo:
  batch_size: 1
  initial_design_size: {initial_design_size}
  acquisition: qmf_kg
  random_seed: 7
  raw_samples: 8
  num_restarts: 1
  mc_samples: 8
""",
        encoding="utf-8",
    )
    return path


def config(initial_design_size: int = 2) -> CampaignConfig:
    return CampaignConfig(
        campaign_name="cli_test",
        objective=ObjectiveConfig(name="score", direction="maximize"),
        variables=(VariableConfig("x", "continuous", 0.0, 1.0),),
        bo=BOConfig(
            batch_size=1,
            initial_design_size=initial_design_size,
            random_seed=7,
            raw_samples=16,
            num_restarts=2,
            mc_samples=16,
        ),
    )


def fidelity_config(
    initial_design_size: int = 3,
    *,
    review: bool = False,
) -> CampaignConfig:
    return CampaignConfig(
        campaign_name="fidelity_cli_test",
        objective=ObjectiveConfig(name="activity", direction="maximize"),
        variables=(
            VariableConfig("x", "continuous", 0.0, 1.0),
            VariableConfig("fidelity", "continuous", 0.2, 1.0),
        ),
        bo=BOConfig(
            batch_size=1,
            initial_design_size=initial_design_size,
            acquisition="qmf_kg",
            random_seed=7,
            raw_samples=8,
            num_restarts=1,
            mc_samples=8,
        ),
        fidelity=FidelityConfig(variable="fidelity", target=1.0, num_fantasies=8),
        review=ReviewConfig(enabled=review),
    )


def mixed_config(initial_design_size: int = 3) -> CampaignConfig:
    return CampaignConfig(
        campaign_name="mixed_cli_test",
        objective=ObjectiveConfig(name="score", direction="maximize"),
        variables=(
            VariableConfig("x", "continuous", 0.0, 1.0),
            VariableConfig("repeats", "integer", 1.0, 3.0),
            VariableConfig("dose", "discrete", values=(0.1, 0.2, 0.5)),
            VariableConfig("solvent", "categorical", values=("MeCN", "EtOH")),
        ),
        bo=BOConfig(
            batch_size=1,
            initial_design_size=initial_design_size,
            random_seed=7,
            raw_samples=16,
            num_restarts=2,
            mc_samples=16,
        ),
    )


def cost_review_config(initial_design_size: int = 2) -> CampaignConfig:
    cfg = config(initial_design_size=initial_design_size)
    return CampaignConfig(
        campaign_name="cost_review_cli_test",
        objective=cfg.objective,
        variables=cfg.variables,
        bo=cfg.bo,
        cost=CostConfig(
            expression="1.0 + x",
            weight=0.5,
            budget=10.0,
            candidate_pool_size=16,
            top_k=8,
        ),
        review=ReviewConfig(enabled=True),
    )


def replicate_config(initial_design_size: int = 2) -> CampaignConfig:
    cfg = config(initial_design_size=initial_design_size)
    return CampaignConfig(
        campaign_name="replicate_cli_test",
        objective=cfg.objective,
        variables=cfg.variables,
        bo=cfg.bo,
        replicates=ReplicateConfig(enabled=True),
    )


def multi_objective_config(initial_design_size: int = 10, *, cost: bool = False) -> CampaignConfig:
    return CampaignConfig(
        campaign_name="multi_cli_test",
        objective=ObjectiveConfig(name="yield_score", direction="maximize", reference_point=40.0),
        objectives=(
            ObjectiveConfig(name="yield_score", direction="maximize", reference_point=40.0),
            ObjectiveConfig(name="waste_score", direction="minimize", reference_point=25.0),
        ),
        variables=(VariableConfig("temperature", "continuous", 20.0, 100.0),),
        bo=BOConfig(
            batch_size=1,
            initial_design_size=initial_design_size,
            acquisition="qlog_ehvi",
            random_seed=7,
            raw_samples=8,
            num_restarts=2,
            mc_samples=8,
        ),
        cost=CostConfig(
            expression="1.0 + 0.02 * temperature",
            weight=0.5,
            budget=20.0,
            candidate_pool_size=16,
            top_k=8,
        )
        if cost
        else None,
    )


def observed_log(cfg: CampaignConfig) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "row_id": "obs_0",
                "iteration": 0,
                "status": "observed",
                "source": "manual",
                "x": 0.2,
                "score": 1.0,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            },
            {
                "row_id": "obs_1",
                "iteration": 1,
                "status": "observed",
                "source": "manual",
                "x": 0.8,
                "score": 1.5,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            },
        ],
        columns=canonical_columns(cfg),
    )


def fidelity_observed_log(cfg: CampaignConfig) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "row_id": "mf_obs_0",
                "iteration": 0,
                "status": "observed",
                "source": "manual",
                "x": 0.1,
                "fidelity": 0.25,
                "activity": 0.7,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            },
            {
                "row_id": "mf_obs_1",
                "iteration": 1,
                "status": "observed",
                "source": "manual",
                "x": 0.3,
                "fidelity": 0.5,
                "activity": 1.1,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            },
            {
                "row_id": "mf_obs_2",
                "iteration": 2,
                "status": "observed",
                "source": "manual",
                "x": 0.6,
                "fidelity": 0.75,
                "activity": 1.4,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            },
            {
                "row_id": "mf_obs_3",
                "iteration": 3,
                "status": "observed",
                "source": "manual",
                "x": 0.85,
                "fidelity": 1.0,
                "activity": 1.3,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            },
        ],
        columns=canonical_columns(cfg),
    )


def mixed_observed_log(cfg: CampaignConfig) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "row_id": "mixed_obs_0",
                "iteration": 0,
                "status": "observed",
                "source": "manual",
                "x": 0.1,
                "repeats": 1,
                "dose": 0.1,
                "solvent": "MeCN",
                "score": 1.0,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            },
            {
                "row_id": "mixed_obs_1",
                "iteration": 1,
                "status": "observed",
                "source": "manual",
                "x": 0.4,
                "repeats": 2,
                "dose": 0.2,
                "solvent": "EtOH",
                "score": 1.5,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            },
            {
                "row_id": "mixed_obs_2",
                "iteration": 2,
                "status": "observed",
                "source": "manual",
                "x": 0.8,
                "repeats": 3,
                "dose": 0.5,
                "solvent": "MeCN",
                "score": 1.2,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            },
        ],
        columns=canonical_columns(cfg),
    )


def cost_review_log(cfg: CampaignConfig) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "row_id": "obs_0",
                "iteration": 0,
                "status": "observed",
                "source": "manual",
                "review_status": "accepted",
                "review_note": "",
                "x": 0.2,
                "score": 1.0,
                "cost_estimate": 1.2,
                "cost_actual": 1.1,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
                "utility": "",
            },
            {
                "row_id": "suggested_0",
                "iteration": 1,
                "status": "suggested",
                "source": "sobol",
                "review_status": "pending",
                "review_note": "",
                "x": 0.5,
                "score": "",
                "cost_estimate": 1.5,
                "cost_actual": "",
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
                "utility": "",
            },
        ],
        columns=canonical_columns(cfg),
    )


def replicate_log(cfg: CampaignConfig) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "row_id": "rep_0a",
                "iteration": 0,
                "status": "observed",
                "source": "manual",
                "replicate_group": "group_0",
                "replicate_index": 0,
                "x": 0.2,
                "score": 1.0,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            },
            {
                "row_id": "rep_0b",
                "iteration": 0,
                "status": "observed",
                "source": "manual",
                "replicate_group": "group_0",
                "replicate_index": 1,
                "x": 0.2,
                "score": 1.6,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            },
            {
                "row_id": "rep_1a",
                "iteration": 1,
                "status": "observed",
                "source": "manual",
                "replicate_group": "group_1",
                "replicate_index": 0,
                "x": 0.8,
                "score": 1.4,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            },
        ],
        columns=canonical_columns(cfg),
    )


def multi_objective_log(cfg: CampaignConfig) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "row_id": "obs_0",
                "iteration": 0,
                "status": "observed",
                "source": "manual",
                "temperature": 35.0,
                "yield_score": 55.0,
                "waste_score": 20.0,
                "cost_estimate": 1.7,
                "cost_actual": "",
                "predicted_mean_yield_score": "",
                "predicted_std_yield_score": "",
                "predicted_mean_waste_score": "",
                "predicted_std_waste_score": "",
                "acquisition": "",
                "utility": "",
            },
            {
                "row_id": "suggested_0",
                "iteration": 1,
                "status": "suggested",
                "source": "sobol",
                "temperature": 65.0,
                "yield_score": "",
                "waste_score": "",
                "cost_estimate": 2.3,
                "cost_actual": "",
                "predicted_mean_yield_score": "",
                "predicted_std_yield_score": "",
                "predicted_mean_waste_score": "",
                "predicted_std_waste_score": "",
                "acquisition": "",
                "utility": "",
            },
        ],
        columns=canonical_columns(cfg),
    )


def write_log(path: Path, cfg: CampaignConfig, df: pd.DataFrame | None = None) -> Path:
    if df is None:
        df = empty_campaign_log(cfg)
    df.to_csv(path, index=False)
    return path


def base_args(config_path: Path, log_path: Path) -> list[str]:
    return ["--config", str(config_path), "--log", str(log_path)]


def output_under_file_parent(tmp_path: Path, filename: str) -> Path:
    parent = tmp_path / "not_a_dir"
    parent.write_text("not a directory", encoding="utf-8")
    return parent / filename


def run_python_module(module: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", module, *args],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )

__all__ = [
    'BOConfig',
    'CampaignConfig',
    'CostConfig',
    'FidelityConfig',
    'LogBusyError',
    'LogConflictError',
    'ObjectiveConfig',
    'Path',
    'ReplicateConfig',
    'ReviewConfig',
    'VariableConfig',
    'base_args',
    'canonical_columns',
    'cli',
    'config',
    'cost_review_config',
    'cost_review_log',
    'empty_campaign_log',
    'fidelity_config',
    'fidelity_observed_log',
    'load_campaign_log',
    'matplotlib',
    'mixed_config',
    'mixed_observed_log',
    'multi_objective_config',
    'multi_objective_log',
    'observed_log',
    'output_under_file_parent',
    'pd',
    'pytest',
    'replicate_config',
    'replicate_log',
    'run',
    'run_python_module',
    'subprocess',
    'suggestions_module',
    'sys',
    'torch',
    'write_config',
    'write_cost_review_config',
    'write_fidelity_config',
    'write_log',
    'write_mixed_config',
    'write_multi_objective_config',
    'write_multi_objective_cost_config',
    'write_replicate_config',
]
