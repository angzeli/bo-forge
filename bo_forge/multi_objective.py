"""Two-objective utilities for coupled qLogEHVI campaigns."""

from __future__ import annotations

import pandas as pd
import torch
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated

from bo_forge.config import CampaignConfig
from bo_forge.validation import get_observed_data, validate_campaign_data


def objective_signs(config: CampaignConfig) -> torch.Tensor:
    """Return signs that convert configured objectives to maximization space."""
    return torch.tensor(
        [1.0 if objective.direction == "maximize" else -1.0 for objective in config.objectives],
        dtype=torch.double,
    )


def objectives_to_model_space(config: CampaignConfig, y_user: torch.Tensor) -> torch.Tensor:
    """Convert user-facing objective values to model-space maximization values."""
    return y_user * objective_signs(config).to(dtype=y_user.dtype, device=y_user.device)


def objectives_from_model_space(config: CampaignConfig, y_model: torch.Tensor) -> torch.Tensor:
    """Convert model-space maximization values back to user-facing objective values."""
    return y_model * objective_signs(config).to(dtype=y_model.dtype, device=y_model.device)


def reference_point_to_model_space(config: CampaignConfig) -> torch.Tensor:
    """Return the configured reference point in model-space maximization units."""
    return objectives_to_model_space(
        config,
        torch.tensor(
            [[float(objective.reference_point) for objective in config.objectives]],
            dtype=torch.double,
        ),
    ).squeeze(0)


def observed_objectives_to_model_tensor(
    config: CampaignConfig,
    observed_df: pd.DataFrame,
) -> torch.Tensor:
    """Return observed objective values in model-space maximization units."""
    y_user = torch.tensor(
        observed_df[config.objective_names].astype(float).to_numpy(),
        dtype=torch.double,
    )
    return objectives_to_model_space(config, y_user)


def pareto_front(config: CampaignConfig, df: pd.DataFrame) -> pd.DataFrame:
    """Return nondominated observed rows in user-facing units."""
    _require_multi_objective(config)
    validate_campaign_data(config, df)
    observed = get_observed_data(config, df)
    if observed.empty:
        return observed.copy()

    y_model = observed_objectives_to_model_tensor(config, observed)
    mask = is_non_dominated(y_model, maximize=True, deduplicate=False).cpu().numpy()
    return observed.loc[mask].copy()


def hypervolume(config: CampaignConfig, df: pd.DataFrame) -> float:
    """Return observed hypervolume, or 0.0 when no point dominates the reference point."""
    _require_multi_objective(config)
    validate_campaign_data(config, df)
    observed = get_observed_data(config, df)
    if observed.empty:
        return 0.0

    y_model = observed_objectives_to_model_tensor(config, observed)
    ref_point = reference_point_to_model_space(config)
    dominates_ref = ((y_model >= ref_point).all(dim=-1) & (y_model > ref_point).any(dim=-1))
    if not bool(dominates_ref.any()):
        return 0.0

    pareto_y = y_model[is_non_dominated(y_model, maximize=True, deduplicate=False)]
    return float(Hypervolume(ref_point=ref_point).compute(pareto_y))


def hypervolume_progress(config: CampaignConfig, df: pd.DataFrame) -> pd.DataFrame:
    """Return hypervolume after each observed row in campaign order."""
    _require_multi_objective(config)
    validate_campaign_data(config, df)
    observed = get_observed_data(config, df).sort_values(["iteration", "row_id"])
    rows = []
    for index in range(len(observed)):
        prefix = observed.iloc[: index + 1].copy()
        rows.append(
            {
                "observation": index + 1,
                "row_id": str(prefix["row_id"].iloc[-1]),
                "hypervolume": hypervolume(config, prefix),
            }
        )
    return pd.DataFrame(rows, columns=["observation", "row_id", "hypervolume"])


def pareto_summary(config: CampaignConfig, df: pd.DataFrame) -> pd.DataFrame:
    """Return a compact two-column Pareto and hypervolume summary."""
    _require_multi_objective(config)
    front = pareto_front(config, df)
    rows: list[tuple[str, object]] = [
        ("pareto_count", len(front)),
        ("hypervolume", hypervolume(config, df)),
    ]
    for objective in config.objectives:
        rows.extend(
            [
                (f"{objective.name}_direction", objective.direction),
                (f"{objective.name}_reference_point", objective.reference_point),
            ]
        )
    return pd.DataFrame(rows, columns=["field", "value"])


def _require_multi_objective(config: CampaignConfig) -> None:
    if not config.is_multi_objective:
        raise ValueError("This helper requires a multi-objective campaign config.")
