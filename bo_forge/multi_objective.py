"""Multi-objective utilities for coupled qLogEHVI campaigns."""

from __future__ import annotations

import pandas as pd
import torch
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated

from bo_forge.config import CampaignConfig
from bo_forge.replicates import aggregate_observed_replicates
from bo_forge.validation import canonical_columns, get_observed_data, validate_campaign_data


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
    observed = multi_objective_observed_data(config, df)
    if observed.empty:
        return pd.DataFrame(columns=_pareto_front_columns(config))

    y_model = observed_objectives_to_model_tensor(config, observed)
    mask = is_non_dominated(y_model, maximize=True, deduplicate=False).cpu().numpy()
    return sort_pareto_front(config, observed.loc[mask].copy())


def hypervolume(config: CampaignConfig, df: pd.DataFrame) -> float:
    """Return observed hypervolume, or 0.0 when no point dominates the reference point."""
    _require_multi_objective(config)
    validate_campaign_data(config, df)
    observed = multi_objective_observed_data(config, df)
    if observed.empty:
        return 0.0

    y_model = observed_objectives_to_model_tensor(config, observed)
    ref_point = reference_point_to_model_space(config)
    dominates_ref = ((y_model >= ref_point).all(dim=-1) & (y_model > ref_point).any(dim=-1))
    if not bool(dominates_ref.any()):
        return 0.0

    valid_y = y_model[dominates_ref]
    pareto_y = valid_y[is_non_dominated(valid_y, maximize=True, deduplicate=False)]
    return float(Hypervolume(ref_point=ref_point).compute(pareto_y))


def hypervolume_progress(config: CampaignConfig, df: pd.DataFrame) -> pd.DataFrame:
    """Return cumulative best-so-far hypervolume after each observed row."""
    _require_multi_objective(config)
    validate_campaign_data(config, df)
    observed = get_observed_data(config, df)
    rows = []
    best_hypervolume = 0.0
    for index in range(len(observed)):
        prefix = observed.iloc[: index + 1].copy()
        best_hypervolume = max(best_hypervolume, hypervolume(config, prefix))
        rows.append(
            {
                "observation": index + 1,
                "row_id": str(prefix["row_id"].iloc[-1]),
                "iteration": int(prefix["iteration"].iloc[-1]),
                "hypervolume": best_hypervolume,
            }
        )
    return pd.DataFrame(rows, columns=["observation", "row_id", "iteration", "hypervolume"])


def pareto_summary(config: CampaignConfig, df: pd.DataFrame) -> pd.DataFrame:
    """Return a compact two-column Pareto and hypervolume summary."""
    _require_multi_objective(config)
    validate_campaign_data(config, df)
    observed = multi_objective_observed_data(config, df)
    front = pareto_front(config, df)
    observed_count = len(observed)
    current_hypervolume = hypervolume(config, df)
    rows: list[tuple[str, object]] = [
        ("objective_count", len(config.objectives)),
        ("pareto_count", len(front)),
        ("pareto_fraction", len(front) / observed_count if observed_count else 0.0),
        ("hypervolume", current_hypervolume),
        ("hypervolume_is_zero", current_hypervolume == 0.0),
        (
            "zero_hypervolume_reason",
            "No observed point currently dominates the configured reference point "
            "in transformed objective space."
            if current_hypervolume == 0.0
            else None,
        ),
    ]
    for objective in config.objectives:
        rows.extend(
            [
                (f"{objective.name}_direction", objective.direction),
                (f"{objective.name}_reference_point", objective.reference_point),
            ]
        )
    return pd.DataFrame(rows, columns=["field", "value"])


def sort_pareto_front(config: CampaignConfig, front: pd.DataFrame) -> pd.DataFrame:
    """Sort Pareto rows by objective preference and row_id for stable display."""
    if front.empty:
        return front.copy()

    sortable = front.copy()
    sort_columns: list[str] = []
    ascending: list[bool] = []
    for index, objective in enumerate(config.objectives):
        sort_column = f"__objective_sort_{index}"
        sortable[sort_column] = pd.to_numeric(sortable[objective.name])
        sort_columns.append(sort_column)
        ascending.append(objective.direction == "minimize")
    tie_breaker = "row_id" if "row_id" in sortable.columns else "replicate_group"
    sortable["__row_id_sort"] = sortable[tie_breaker].astype(str)
    sort_columns.append("__row_id_sort")
    ascending.append(True)

    sorted_front = sortable.sort_values(sort_columns, ascending=ascending)
    return sorted_front[front.columns].copy()


def multi_objective_observed_data(config: CampaignConfig, df: pd.DataFrame) -> pd.DataFrame:
    """Return observed rows used for multi-objective Pareto and hypervolume utilities."""
    _require_multi_objective(config)
    observed = get_observed_data(config, df)
    if not config.replicates.enabled:
        return observed
    if observed.empty:
        return pd.DataFrame(columns=_pareto_front_columns(config))

    aggregate = aggregate_observed_replicates(config, observed)
    rows: list[dict[str, object]] = []
    for group, group_df in observed.groupby("replicate_group", sort=False):
        summary = aggregate.loc[aggregate["replicate_group"] == group].iloc[0]
        row: dict[str, object] = {
            "replicate_group": group,
            "n_replicates": int(summary["n_replicates"]),
            "first_row_id": str(group_df["row_id"].iloc[0]),
            "first_iteration": int(group_df["iteration"].iloc[0]),
            "last_iteration": int(group_df["iteration"].iloc[-1]),
        }
        for variable_name in config.variable_names:
            row[variable_name] = summary[variable_name]
        for objective in config.objectives:
            row[objective.name] = float(summary[f"{objective.name}_mean"])
        rows.append(row)
    return pd.DataFrame(rows, columns=_pareto_front_columns(config))


def _pareto_front_columns(config: CampaignConfig) -> list[str]:
    if config.replicates.enabled:
        return [
            "replicate_group",
            *config.variable_names,
            "n_replicates",
            *config.objective_names,
            "first_row_id",
            "first_iteration",
            "last_iteration",
        ]
    return canonical_columns(config)


def _require_multi_objective(config: CampaignConfig) -> None:
    if not config.is_multi_objective:
        raise ValueError("This helper requires a multi-objective campaign config.")
