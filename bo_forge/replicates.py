"""Replicate-aware aggregation helpers."""

from __future__ import annotations

import math

import pandas as pd

from bo_forge.config import CampaignConfig
from bo_forge.validation import get_observed_data, validate_campaign_data


def aggregate_columns(config: CampaignConfig) -> list[str]:
    """Return canonical replicate aggregate columns."""
    if config.is_multi_objective:
        objective_columns: list[str] = []
        for objective in config.objectives:
            objective_columns.extend(
                [
                    f"{objective.name}_mean",
                    f"{objective.name}_std",
                    f"{objective.name}_sem",
                    f"{objective.name}_min",
                    f"{objective.name}_max",
                ]
            )
        return [
            "replicate_group",
            *config.variable_names,
            "n_replicates",
            *objective_columns,
        ]
    return [
        "replicate_group",
        *config.variable_names,
        "n_replicates",
        "objective_mean",
        "objective_std",
        "objective_sem",
        "objective_min",
        "objective_max",
    ]


def aggregate_observed_replicates(
    config: CampaignConfig,
    observed_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate observed replicate groups by mean objective."""
    columns = aggregate_columns(config)
    if not config.replicates.enabled or observed_df.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for group, group_df in observed_df.groupby("replicate_group", sort=False):
        first = group_df.iloc[0]
        row: dict[str, object] = {"replicate_group": group}
        for variable_name in config.variable_names:
            row[variable_name] = first[variable_name]
        n_replicates = int(len(group_df))
        row["n_replicates"] = n_replicates
        if config.is_multi_objective:
            for objective in config.objectives:
                values = pd.to_numeric(group_df[objective.name], errors="raise")
                std = float(values.std(ddof=1)) if n_replicates > 1 else math.nan
                sem = float(std / math.sqrt(n_replicates)) if n_replicates > 1 else math.nan
                row.update(
                    {
                        f"{objective.name}_mean": float(values.mean()),
                        f"{objective.name}_std": std,
                        f"{objective.name}_sem": sem,
                        f"{objective.name}_min": float(values.min()),
                        f"{objective.name}_max": float(values.max()),
                    }
                )
            rows.append(row)
            continue

        objective = config.objective.name
        values = pd.to_numeric(group_df[objective], errors="raise")
        std = float(values.std(ddof=1)) if n_replicates > 1 else math.nan
        sem = float(std / math.sqrt(n_replicates)) if n_replicates > 1 else math.nan
        row.update(
            {
                "objective_mean": float(values.mean()),
                "objective_std": std,
                "objective_sem": sem,
                "objective_min": float(values.min()),
                "objective_max": float(values.max()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def modeling_observed_data(config: CampaignConfig, observed_df: pd.DataFrame) -> pd.DataFrame:
    """Return observed rows used for model fitting."""
    model_df, _ = modeling_observed_data_with_variance(config, observed_df)
    return model_df


def modeling_observed_data_with_variance(
    config: CampaignConfig,
    observed_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Return model-fitting rows and optional replicate-derived observation variance."""
    if not config.replicates.enabled:
        return observed_df.copy(), None
    aggregate = aggregate_observed_replicates(config, observed_df)
    rows: list[dict[str, object]] = []
    for _, row in aggregate.iterrows():
        model_row = {variable: row[variable] for variable in config.variable_names}
        if config.is_multi_objective:
            for objective in config.objectives:
                model_row[objective.name] = row[f"{objective.name}_mean"]
            rows.append(model_row)
            continue
        model_row[config.objective.name] = row["objective_mean"]
        rows.append(model_row)
    model_df = pd.DataFrame(rows, columns=[*config.variable_names, *config.objective_names])
    return model_df, _replicate_observation_variance(config, aggregate)


def _replicate_observation_variance(
    config: CampaignConfig,
    aggregate: pd.DataFrame,
) -> pd.DataFrame | None:
    if aggregate.empty or not (aggregate["n_replicates"].astype(int) > 1).any():
        return None

    variance_rows: list[dict[str, float]] = []
    pooled = {
        objective_name: _pooled_replicate_variance(config, aggregate, objective_name)
        for objective_name in config.objective_names
    }
    for _, row in aggregate.iterrows():
        n_replicates = int(row["n_replicates"])
        variance_row: dict[str, float] = {}
        for objective_name in config.objective_names:
            std_column = _std_column(config, objective_name)
            if n_replicates > 1:
                variance = float(row[std_column]) ** 2 / n_replicates
            else:
                variance = pooled[objective_name]
            variance_row[objective_name] = max(float(variance), config.replicates.noise_floor)
        variance_rows.append(variance_row)
    return pd.DataFrame(variance_rows, columns=config.objective_names)


def _pooled_replicate_variance(
    config: CampaignConfig,
    aggregate: pd.DataFrame,
    objective_name: str,
) -> float:
    std_column = _std_column(config, objective_name)
    numerator = 0.0
    denominator = 0
    for _, row in aggregate.iterrows():
        n_replicates = int(row["n_replicates"])
        if n_replicates <= 1:
            continue
        std = float(row[std_column])
        if math.isfinite(std):
            numerator += (n_replicates - 1) * std**2
            denominator += n_replicates - 1
    if denominator == 0:
        return config.replicates.noise_floor
    return max(numerator / denominator, config.replicates.noise_floor)


def _std_column(config: CampaignConfig, objective_name: str) -> str:
    if config.is_multi_objective:
        return f"{objective_name}_std"
    return "objective_std"


def replicate_summary(config: CampaignConfig, df: pd.DataFrame) -> pd.DataFrame:
    """Return observed replicate-group summary statistics."""
    validate_campaign_data(config, df)
    if not config.replicates.enabled:
        return pd.DataFrame(columns=aggregate_columns(config))
    return aggregate_observed_replicates(config, get_observed_data(config, df))


def best_replicate_group(config: CampaignConfig, df: pd.DataFrame) -> pd.DataFrame:
    """Return the best replicate group by mean objective."""
    if config.is_multi_objective:
        raise ValueError(
            "best_replicate_group() is only defined for single-objective replicate campaigns; "
            "use replicate_summary() or pareto_front() for multi-objective campaigns."
        )
    summary = replicate_summary(config, df)
    if summary.empty:
        return pd.DataFrame(columns=aggregate_columns(config))

    values = pd.to_numeric(summary["objective_mean"])
    if config.objective.direction == "maximize":
        best_index = values.idxmax()
    else:
        best_index = values.idxmin()
    return summary.loc[[best_index], aggregate_columns(config)].copy()
