"""Basic plotting helpers for campaign diagnostics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bo_forge.config import CampaignConfig
from bo_forge.plot_style import (
    add_legend,
    finalise_figure,
    new_figure,
    set_axis_labels,
    set_title,
    style_colorbar,
)
from bo_forge.validation import get_observed_data, validate_campaign_data


def plot_progress(
    config: CampaignConfig,
    df: pd.DataFrame,
    *,
    filename: str | Path | None = None,
    fig_folder: str | Path = "figures",
    show: bool = False,
):
    """Plot observed objective values and best-so-far progress."""
    validate_campaign_data(config, df)
    observed = get_observed_data(config, df)

    _, ax = new_figure(figsize=(8, 6))
    if observed.empty:
        set_title(ax, f"{config.campaign_name}: no observations yet")
        set_axis_labels(ax, "Observation", config.objective.name)
        return finalise_figure(
            ax,
            filename=filename,
            fig_folder=fig_folder,
            show=show,
        )

    values = pd.to_numeric(observed[config.objective.name])
    best = values.cummax() if config.objective.direction == "maximize" else values.cummin()
    x = range(1, len(observed) + 1)
    ax.plot(x, values, marker="o", label="observed")
    ax.plot(x, best, marker=".", label="best so far")
    set_title(ax, f"{config.campaign_name}: progress")
    set_axis_labels(ax, "Observation", config.objective.name)
    add_legend(ax)
    return finalise_figure(
        ax,
        filename=filename,
        fig_folder=fig_folder,
        show=show,
    )


def plot_diagnostics(
    config: CampaignConfig,
    df: pd.DataFrame,
    *,
    filename: str | Path | None = None,
    fig_folder: str | Path = "figures",
    show: bool = False,
):
    """Plot a simple 1D or 2D diagnostic view of observed campaign data."""
    validate_campaign_data(config, df)
    observed = get_observed_data(config, df)
    objective = config.objective.name

    fig, ax = new_figure()
    if observed.empty:
        set_title(ax, f"{config.campaign_name}: no observations yet")
        set_axis_labels(ax, "", "")
        return finalise_figure(
            ax,
            filename=filename,
            fig_folder=fig_folder,
            show=show,
        )

    if len(config.variables) == 1:
        variable = config.variables[0].name
        ax.scatter(observed[variable].astype(float), observed[objective].astype(float))
        set_title(ax, f"{config.campaign_name}: observed data")
        set_axis_labels(ax, variable, objective)
    else:
        x_name = config.variables[0].name
        y_name = config.variables[1].name
        scatter = ax.scatter(
            observed[x_name].astype(float),
            observed[y_name].astype(float),
            c=observed[objective].astype(float),
            cmap="viridis",
        )
        set_title(ax, f"{config.campaign_name}: observed data")
        set_axis_labels(ax, x_name, y_name)
        colorbar = fig.colorbar(scatter, ax=ax)
        style_colorbar(colorbar, objective)

    return finalise_figure(
        ax,
        filename=filename,
        fig_folder=fig_folder,
        show=show,
    )
