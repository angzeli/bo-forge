"""Replicate-aware aggregation helpers."""

from __future__ import annotations

import math

import pandas as pd

from bo_forge.config import CampaignConfig
from bo_forge.validation import get_observed_data, validate_campaign_data


def aggregate_columns(config: CampaignConfig) -> list[str]:
    """Return canonical replicate aggregate columns."""
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
    objective = config.objective.name
    for group, group_df in observed_df.groupby("replicate_group", sort=False):
        values = pd.to_numeric(group_df[objective], errors="raise")
        n_replicates = int(len(values))
        std = float(values.std(ddof=1)) if n_replicates > 1 else math.nan
        sem = float(std / math.sqrt(n_replicates)) if n_replicates > 1 else math.nan
        first = group_df.iloc[0]
        row: dict[str, object] = {"replicate_group": group}
        for variable_name in config.variable_names:
            row[variable_name] = first[variable_name]
        row.update(
            {
                "n_replicates": n_replicates,
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
    if not config.replicates.enabled:
        return observed_df.copy()

    aggregate = aggregate_observed_replicates(config, observed_df)
    rows: list[dict[str, object]] = []
    for _, row in aggregate.iterrows():
        model_row = {variable: row[variable] for variable in config.variable_names}
        model_row[config.objective.name] = row["objective_mean"]
        rows.append(model_row)
    return pd.DataFrame(rows, columns=[*config.variable_names, config.objective.name])


def replicate_summary(config: CampaignConfig, df: pd.DataFrame) -> pd.DataFrame:
    """Return observed replicate-group summary statistics."""
    validate_campaign_data(config, df)
    if not config.replicates.enabled:
        return pd.DataFrame(columns=aggregate_columns(config))
    return aggregate_observed_replicates(config, get_observed_data(config, df))


def best_replicate_group(config: CampaignConfig, df: pd.DataFrame) -> pd.DataFrame:
    """Return the best replicate group by mean objective."""
    summary = replicate_summary(config, df)
    if summary.empty:
        return pd.DataFrame(columns=aggregate_columns(config))

    values = pd.to_numeric(summary["objective_mean"])
    if config.objective.direction == "maximize":
        best_index = values.idxmax()
    else:
        best_index = values.idxmin()
    return summary.loc[[best_index], aggregate_columns(config)].copy()
