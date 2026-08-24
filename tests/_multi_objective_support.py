from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from bo_forge.config import (
    BOConfig,
    CampaignConfig,
    ConstraintConfig,
    CostConfig,
    ObjectiveConfig,
    ReplicateConfig,
    ReviewConfig,
    VariableConfig,
)
from bo_forge.errors import ConfigError, LogValidationError, LogWriteError
from bo_forge.logs import append_suggestions, mark_observed, review_suggestion
from bo_forge.multi_objective import (
    hypervolume,
    hypervolume_progress,
    objectives_to_model_space,
    pareto_front,
    reference_point_to_model_space,
)
from bo_forge.session import CampaignSession
from bo_forge.suggestions import suggest_next
from bo_forge.validation import (
    canonical_columns,
    design_key_for_values,
    design_tuples,
    has_pending_suggestions,
    validate_campaign_data,
)


@pytest.fixture(autouse=True)
def close_matplotlib_figures() -> None:
    yield
    plt.close("all")


def multi_config(
    batch_size: int = 2,
    initial_design_size: int = 3,
    *,
    cost: bool = False,
    review: bool = False,
    replicates: bool = False,
) -> CampaignConfig:
    return CampaignConfig(
        campaign_name="multi",
        objective=ObjectiveConfig("yield_score", "maximize", 40.0),
        objectives=(
            ObjectiveConfig("yield_score", "maximize", 40.0),
            ObjectiveConfig("waste_score", "minimize", 25.0),
        ),
        variables=(
            VariableConfig("temperature", "continuous", 20.0, 100.0),
            VariableConfig("solvent", "categorical", values=("MeCN", "Water")),
        ),
        bo=BOConfig(
            batch_size=batch_size,
            initial_design_size=initial_design_size,
            acquisition="qlog_ehvi",
            random_seed=4,
            raw_samples=8,
            num_restarts=2,
            mc_samples=8,
        ),
        cost=CostConfig(
            expression="1.0 + 0.02 * temperature + 2.0 * (solvent == 'Water')",
            weight=0.5,
            budget=20.0,
            candidate_pool_size=16,
            top_k=8,
        )
        if cost
        else None,
        review=ReviewConfig(enabled=review),
        replicates=ReplicateConfig(enabled=replicates),
    )


def qlog_nehvi_config(
    batch_size: int = 1,
    initial_design_size: int = 3,
    *,
    review: bool = False,
) -> CampaignConfig:
    cfg = multi_config(
        batch_size=batch_size,
        initial_design_size=initial_design_size,
        review=review,
    )
    return replace(
        cfg,
        campaign_name="qlog_nehvi_multi",
        bo=BOConfig(
            batch_size=batch_size,
            initial_design_size=initial_design_size,
            acquisition="qlog_nehvi",
            random_seed=9,
            raw_samples=8,
            num_restarts=1,
            mc_samples=8,
        ),
    )


def four_objective_config(
    batch_size: int = 1,
    initial_design_size: int = 4,
) -> CampaignConfig:
    return CampaignConfig(
        campaign_name="four_objective",
        objective=ObjectiveConfig("yield", "maximize", 0.2),
        objectives=(
            ObjectiveConfig("yield", "maximize", 0.2),
            ObjectiveConfig("selectivity", "maximize", 0.2),
            ObjectiveConfig("waste", "minimize", 0.9),
            ObjectiveConfig("energy_use", "minimize", 0.9),
        ),
        variables=(
            VariableConfig("catalyst_loading", "continuous", 0.02, 0.20),
            VariableConfig("reaction_time", "integer", 20.0, 90.0),
            VariableConfig("base_equivalents", "discrete", values=(0.5, 1.0, 1.5)),
            VariableConfig("solvent", "categorical", values=("MeCN", "DMF", "Water")),
        ),
        constraints=(
            ConstraintConfig(
                "water_needs_time",
                "solvent != 'Water' or reaction_time >= 30",
            ),
        ),
        bo=BOConfig(
            batch_size=batch_size,
            initial_design_size=initial_design_size,
            acquisition="qlog_ehvi",
            random_seed=7,
            raw_samples=8,
            num_restarts=2,
            mc_samples=8,
        ),
    )


def observed_multi_log(cfg: CampaignConfig) -> pd.DataFrame:
    rows = []
    for index, (temperature, solvent, yield_score, waste_score) in enumerate(
        [
            (30.0, "MeCN", 50.0, 20.0),
            (45.0, "Water", 65.0, 18.0),
            (65.0, "MeCN", 58.0, 12.0),
            (85.0, "Water", 72.0, 16.0),
        ]
    ):
        row = {
            "row_id": f"obs_{index}",
            "iteration": index,
            "status": "observed",
            "source": "manual",
            "temperature": temperature,
            "solvent": solvent,
            "yield_score": yield_score,
            "waste_score": waste_score,
            "predicted_mean_yield_score": "",
            "predicted_std_yield_score": "",
            "predicted_mean_waste_score": "",
            "predicted_std_waste_score": "",
            "acquisition": "",
        }
        if cfg.review.enabled:
            row["review_status"] = "accepted"
            row["review_note"] = ""
        if cfg.replicates.enabled:
            row["replicate_group"] = f"group_{index}"
            row["replicate_index"] = 0
        if cfg.cost is not None:
            cost_estimate = 1.0 + 0.02 * temperature + (2.0 if solvent == "Water" else 0.0)
            row["cost_estimate"] = cost_estimate
            row["cost_actual"] = ""
            row["utility"] = ""
        rows.append(row)
    return pd.DataFrame(rows, columns=canonical_columns(cfg))


def observed_four_objective_log(cfg: CampaignConfig) -> pd.DataFrame:
    data = [
        ("obs_a", 0, 0.05, 30, 0.5, "MeCN", 0.55, 0.40, 0.65, 0.35),
        ("obs_b", 1, 0.12, 60, 1.0, "MeCN", 0.82, 0.68, 0.48, 0.62),
        ("obs_c", 2, 0.16, 80, 1.5, "DMF", 0.74, 0.75, 0.55, 0.82),
        ("obs_d", 3, 0.08, 50, 1.0, "Water", 0.58, 0.82, 0.30, 0.40),
        ("obs_e", 4, 0.18, 70, 0.5, "DMF", 0.68, 0.62, 0.72, 0.78),
        ("obs_f", 5, 0.11, 45, 1.5, "Water", 0.61, 0.88, 0.38, 0.58),
    ]
    rows = []
    for row_id, iteration, loading, time, base, solvent, yld, sel, waste, energy in data:
        rows.append(
            {
                "row_id": row_id,
                "iteration": iteration,
                "status": "observed",
                "source": "manual",
                "catalyst_loading": loading,
                "reaction_time": time,
                "base_equivalents": base,
                "solvent": solvent,
                "yield": yld,
                "selectivity": sel,
                "waste": waste,
                "energy_use": energy,
                "predicted_mean_yield": "",
                "predicted_std_yield": "",
                "predicted_mean_selectivity": "",
                "predicted_std_selectivity": "",
                "predicted_mean_waste": "",
                "predicted_std_waste": "",
                "predicted_mean_energy_use": "",
                "predicted_std_energy_use": "",
                "acquisition": "",
            }
        )
    return pd.DataFrame(rows, columns=canonical_columns(cfg))
















































































































def pd_to_tensor(df: pd.DataFrame):
    import torch

    return torch.tensor(df.astype(float).to_numpy(), dtype=torch.double)

__all__ = [
    'BOConfig',
    'CampaignConfig',
    'CampaignSession',
    'ConfigError',
    'ConstraintConfig',
    'CostConfig',
    'LogValidationError',
    'LogWriteError',
    'ObjectiveConfig',
    'Path',
    'ReplicateConfig',
    'ReviewConfig',
    'VariableConfig',
    'append_suggestions',
    'canonical_columns',
    'close_matplotlib_figures',
    'design_key_for_values',
    'design_tuples',
    'four_objective_config',
    'has_pending_suggestions',
    'hypervolume',
    'hypervolume_progress',
    'mark_observed',
    'multi_config',
    'objectives_to_model_space',
    'observed_four_objective_log',
    'observed_multi_log',
    'pareto_front',
    'pd',
    'pd_to_tensor',
    'plt',
    'pytest',
    'qlog_nehvi_config',
    'reference_point_to_model_space',
    'replace',
    'review_suggestion',
    'suggest_next',
    'validate_campaign_data',
]
