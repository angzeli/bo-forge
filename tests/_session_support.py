from dataclasses import replace
from pathlib import Path

import matplotlib
import pandas as pd
import pytest

import bo_forge.session as session_module
import bo_forge.suggestions as suggestions_module
from bo_forge import CampaignSession
from bo_forge.config import (
    BOConfig,
    CampaignConfig,
    CostConfig,
    ObjectiveConfig,
    ReplicateConfig,
    ReviewConfig,
    StageConfig,
    VariableConfig,
)
from bo_forge.errors import LogConflictError, LogValidationError, SuggestionError
from bo_forge.io import empty_campaign_log
from bo_forge.logs import append_suggestions, mark_observed
from bo_forge.validation import canonical_columns

matplotlib.use("Agg")


def write_config(path: Path, *, direction: str = "maximize", initial_design_size: int = 2) -> Path:
    path.write_text(
        f"""
campaign_name: session_test
objective:
  name: score
  direction: {direction}
variables:
  - name: x
    type: continuous
    lower: 0
    upper: 1
bo:
  batch_size: 1
  initial_design_size: {initial_design_size}
  acquisition: log_ei
  random_seed: 5
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
campaign_name: mixed_session_test
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
  random_seed: 5
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
campaign_name: cost_review_session_test
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
  random_seed: 5
  raw_samples: 16
  num_restarts: 2
  mc_samples: 16
""",
        encoding="utf-8",
    )
    return path


def config(direction: str = "maximize", initial_design_size: int = 2) -> CampaignConfig:
    return CampaignConfig(
        campaign_name="session_test",
        objective=ObjectiveConfig(name="score", direction=direction),
        variables=(VariableConfig("x", "continuous", 0.0, 1.0),),
        bo=BOConfig(
            batch_size=1,
            initial_design_size=initial_design_size,
            random_seed=5,
            raw_samples=16,
            num_restarts=2,
            mc_samples=16,
        ),
    )


def structured_config() -> CampaignConfig:
    cfg = config(initial_design_size=1)
    return CampaignConfig(
        campaign_name="structured_session_test",
        objective=cfg.objective,
        variables=(
            VariableConfig("x", "continuous", 0.0, 1.0),
            VariableConfig("temperature", "continuous", 300.0, 900.0),
        ),
        bo=cfg.bo,
        stages=(
            StageConfig("screen", ("x",)),
            StageConfig("refine", ("x", "temperature")),
        ),
    )


def structured_review_config() -> CampaignConfig:
    cfg = structured_config()
    return CampaignConfig(
        campaign_name=cfg.campaign_name,
        objective=cfg.objective,
        variables=cfg.variables,
        bo=cfg.bo,
        review=ReviewConfig(enabled=True),
        stages=cfg.stages,
    )


def structured_replicate_config() -> CampaignConfig:
    cfg = structured_config()
    return CampaignConfig(
        campaign_name="structured_replicate_session_test",
        objective=cfg.objective,
        variables=cfg.variables,
        bo=cfg.bo,
        replicates=ReplicateConfig(enabled=True),
        stages=cfg.stages,
    )


def structured_multi_objective_config() -> CampaignConfig:
    return CampaignConfig(
        campaign_name="structured_multi_session_test",
        objective=ObjectiveConfig("yield_score", "maximize", 0.0),
        objectives=(
            ObjectiveConfig("yield_score", "maximize", 0.0),
            ObjectiveConfig("waste_score", "minimize", 10.0),
        ),
        variables=(
            VariableConfig("x", "continuous", 0.0, 1.0),
            VariableConfig("temperature", "continuous", 300.0, 900.0),
        ),
        bo=BOConfig(
            batch_size=1,
            initial_design_size=1,
            acquisition="qlog_ehvi",
            random_seed=5,
            raw_samples=8,
            num_restarts=2,
            mc_samples=8,
        ),
        stages=(
            StageConfig("screen", ("x",)),
            StageConfig("refine", ("x", "temperature")),
        ),
    )


def mixed_config(initial_design_size: int = 3) -> CampaignConfig:
    return CampaignConfig(
        campaign_name="mixed_session_test",
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
            random_seed=5,
            raw_samples=16,
            num_restarts=2,
            mc_samples=16,
        ),
    )


def cost_review_config(initial_design_size: int = 2) -> CampaignConfig:
    cfg = config(initial_design_size=initial_design_size)
    return CampaignConfig(
        campaign_name="cost_review_session_test",
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
        campaign_name="replicate_session_test",
        objective=cfg.objective,
        variables=cfg.variables,
        bo=cfg.bo,
        replicates=ReplicateConfig(enabled=True),
    )


def observed_log(cfg: CampaignConfig, values: list[float]) -> pd.DataFrame:
    rows = []
    for index, value in enumerate(values):
        rows.append(
            {
                "row_id": f"obs_{index}",
                "iteration": index,
                "status": "observed",
                "source": "manual",
                "x": 0.2 + index * 0.2,
                "score": value,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            }
        )
    return pd.DataFrame(rows, columns=canonical_columns(cfg))


def structured_observed_log(cfg: CampaignConfig) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "row_id": "screen_0",
                "iteration": 0,
                "status": "observed",
                "source": "manual",
                "stage": "screen",
                "x": 0.2,
                "temperature": "",
                "score": 1.0,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            }
        ],
        columns=canonical_columns(cfg),
    )


def structured_pending_log(cfg: CampaignConfig) -> pd.DataFrame:
    row = {
        "row_id": "screen_1",
        "iteration": 1,
        "status": "suggested",
        "source": "manual",
        "stage": "screen",
        "x": 0.4,
        "temperature": "",
        "score": "",
        "predicted_mean": "",
        "predicted_std": "",
        "acquisition": "",
    }
    if cfg.review.enabled:
        row["review_status"] = "pending"
        row["review_note"] = ""
    return pd.DataFrame([row], columns=canonical_columns(cfg))


def structured_stage_log(cfg: CampaignConfig) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "row_id": "screen_0",
                "iteration": 0,
                "status": "observed",
                "source": "manual",
                "stage": "screen",
                "x": 0.2,
                "temperature": "",
                "score": 1.0,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            },
            {
                "row_id": "screen_1",
                "iteration": 1,
                "status": "observed",
                "source": "manual",
                "stage": "screen",
                "x": 0.7,
                "temperature": "",
                "score": 1.5,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            },
            {
                "row_id": "refine_pending",
                "iteration": 2,
                "status": "suggested",
                "source": "manual",
                "stage": "refine",
                "x": 0.6,
                "temperature": 650.0,
                "score": "",
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            },
        ],
        columns=canonical_columns(cfg),
    )


def structured_replicate_log(cfg: CampaignConfig) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "row_id": "group_0_rep_0",
                "iteration": 0,
                "status": "observed",
                "source": "manual",
                "stage": "screen",
                "replicate_group": "group_0",
                "replicate_index": 0,
                "x": 0.2,
                "temperature": "",
                "score": 1.0,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            },
            {
                "row_id": "group_0_rep_1",
                "iteration": 0,
                "status": "observed",
                "source": "manual",
                "stage": "screen",
                "replicate_group": "group_0",
                "replicate_index": 1,
                "x": 0.2,
                "temperature": "",
                "score": 3.0,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            },
            {
                "row_id": "group_1_rep_0",
                "iteration": 1,
                "status": "observed",
                "source": "manual",
                "stage": "screen",
                "replicate_group": "group_1",
                "replicate_index": 0,
                "x": 0.8,
                "temperature": "",
                "score": 2.5,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            },
        ],
        columns=canonical_columns(cfg),
    )


def structured_multi_objective_log(cfg: CampaignConfig) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "row_id": "screen_0",
                "iteration": 0,
                "status": "observed",
                "source": "manual",
                "stage": "screen",
                "x": 0.2,
                "temperature": "",
                "yield_score": 1.0,
                "waste_score": 5.0,
                "predicted_mean_yield_score": "",
                "predicted_std_yield_score": "",
                "predicted_mean_waste_score": "",
                "predicted_std_waste_score": "",
                "acquisition": "",
            },
            {
                "row_id": "screen_1",
                "iteration": 1,
                "status": "observed",
                "source": "manual",
                "stage": "screen",
                "x": 0.8,
                "temperature": "",
                "yield_score": 0.5,
                "waste_score": 1.0,
                "predicted_mean_yield_score": "",
                "predicted_std_yield_score": "",
                "predicted_mean_waste_score": "",
                "predicted_std_waste_score": "",
                "acquisition": "",
            },
        ],
        columns=canonical_columns(cfg),
    )


def mixed_observed_log(cfg: CampaignConfig) -> pd.DataFrame:
    rows = []
    for index, (x_value, repeats, dose, solvent, score) in enumerate(
        [
            (0.1, 1, 0.1, "MeCN", 1.0),
            (0.3, 2, 0.2, "EtOH", 1.4),
            (0.8, 3, 0.5, "MeCN", 1.2),
        ]
    ):
        rows.append(
            {
                "row_id": f"mixed_obs_{index}",
                "iteration": index,
                "status": "observed",
                "source": "manual",
                "x": x_value,
                "repeats": repeats,
                "dose": dose,
                "solvent": solvent,
                "score": score,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            }
        )
    return pd.DataFrame(rows, columns=canonical_columns(cfg))


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


def pending_log(cfg: CampaignConfig) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "row_id": "pending_0",
                "iteration": 0,
                "status": "suggested",
                "source": "sobol",
                "x": 0.5,
                "score": "",
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            }
        ],
        columns=canonical_columns(cfg),
    )


def write_log(path: Path, cfg: CampaignConfig, df: pd.DataFrame | None = None) -> Path:
    if df is None:
        df = empty_campaign_log(cfg)
    df.to_csv(path, index=False)
    return path


def summary_value(summary: pd.DataFrame, field: str):
    matches = summary.loc[summary["field"] == field, "value"]
    assert len(matches) == 1
    return matches.iloc[0]

__all__ = [
    'BOConfig',
    'CampaignConfig',
    'CampaignSession',
    'CostConfig',
    'LogConflictError',
    'LogValidationError',
    'ObjectiveConfig',
    'Path',
    'ReplicateConfig',
    'ReviewConfig',
    'StageConfig',
    'SuggestionError',
    'VariableConfig',
    'append_suggestions',
    'canonical_columns',
    'config',
    'cost_review_config',
    'cost_review_log',
    'empty_campaign_log',
    'mark_observed',
    'matplotlib',
    'mixed_config',
    'mixed_observed_log',
    'observed_log',
    'pd',
    'pending_log',
    'pytest',
    'replace',
    'replicate_config',
    'replicate_log',
    'session_module',
    'structured_config',
    'structured_multi_objective_config',
    'structured_multi_objective_log',
    'structured_observed_log',
    'structured_pending_log',
    'structured_replicate_config',
    'structured_replicate_log',
    'structured_review_config',
    'structured_stage_log',
    'suggestions_module',
    'summary_value',
    'write_config',
    'write_cost_review_config',
    'write_log',
    'write_mixed_config',
]
