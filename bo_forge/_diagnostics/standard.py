"""Basic plotting helpers for campaign diagnostics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bo_forge._diagnostics.multi_objective import (
    _directional_best_so_far,
    _plot_high_dimensional_diagnostics,
    plot_hypervolume,
    plot_pareto,
)
from bo_forge._diagnostics.structured import _plot_multi_objective_replicates
from bo_forge.config import CampaignConfig
from bo_forge.costs import effective_row_cost
from bo_forge.multi_objective import (
    hypervolume_progress,
)
from bo_forge.plot_style import (
    NEUTRAL_COLOR,
    OBSERVED_COLOR,
    ORDERED_CMAP,
    add_legend,
    finalise_figure,
    new_figure,
    scoped_plot_style,
    set_axis_labels,
    set_title,
    style_colorbar,
)
from bo_forge.replicates import replicate_summary
from bo_forge.transforms import (
    has_mixed_variables,
)
from bo_forge.validation import get_observed_data, validate_campaign_data


@scoped_plot_style
def plot_progress(
    config: CampaignConfig,
    df: pd.DataFrame,
    *,
    filename: str | Path | None = None,
    fig_folder: str | Path = "figures",
    save_path: str | Path | None = None,
    show: bool = False,
):
    """Plot observed objective values and best-so-far progress."""
    validate_campaign_data(config, df)
    if config.is_multi_objective:
        return plot_hypervolume(
            config,
            df,
            filename=filename,
            fig_folder=fig_folder,
            save_path=save_path,
            show=show,
        )
    observed = get_observed_data(config, df)

    _, ax = new_figure(figsize=(8, 6))
    if observed.empty:
        set_title(ax, f"{config.campaign_name}: no observations yet")
        set_axis_labels(ax, "Observation", config.objective.name)
        return finalise_figure(
            ax,
            filename=filename,
            fig_folder=fig_folder,
            save_path=save_path,
            show=show,
        )

    values = pd.to_numeric(observed[config.objective.name])
    best = _directional_best_so_far(config, values)
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
        save_path=save_path,
        show=show,
    )


@scoped_plot_style
def plot_diagnostics(
    config: CampaignConfig,
    df: pd.DataFrame,
    *,
    filename: str | Path | None = None,
    fig_folder: str | Path = "figures",
    save_path: str | Path | None = None,
    show: bool = False,
):
    """Plot dimension-aware diagnostics for observed campaign data."""
    validate_campaign_data(config, df)
    if config.is_multi_objective:
        return plot_pareto(
            config,
            df,
            filename=filename,
            fig_folder=fig_folder,
            save_path=save_path,
            show=show,
        )
    observed = get_observed_data(config, df)
    objective = config.objective.name

    if observed.empty:
        _, ax = new_figure()
        set_title(ax, f"{config.campaign_name}: no observations yet")
        set_axis_labels(ax, "", "")
        return finalise_figure(
            ax,
            filename=filename,
            fig_folder=fig_folder,
            save_path=save_path,
            show=show,
        )

    if has_mixed_variables(config):
        return _plot_high_dimensional_diagnostics(
            config,
            observed,
            filename=filename,
            fig_folder=fig_folder,
            save_path=save_path,
            show=show,
        )

    if len(config.variables) == 1:
        fig, ax = new_figure()
        variable = config.variables[0].name
        ax.scatter(observed[variable].astype(float), observed[objective].astype(float))
        set_title(ax, f"{config.campaign_name}: observed data")
        set_axis_labels(ax, variable, objective)
    elif len(config.variables) == 2:
        fig, ax = new_figure()
        x_name = config.variables[0].name
        y_name = config.variables[1].name
        scatter = ax.scatter(
            observed[x_name].astype(float),
            observed[y_name].astype(float),
            c=observed[objective].astype(float),
            cmap=ORDERED_CMAP,
        )
        set_title(ax, f"{config.campaign_name}: observed data")
        set_axis_labels(ax, x_name, y_name)
        colorbar = fig.colorbar(scatter, ax=ax)
        style_colorbar(colorbar, objective)
    else:
        return _plot_high_dimensional_diagnostics(
            config,
            observed,
            filename=filename,
            fig_folder=fig_folder,
            save_path=save_path,
            show=show,
        )

    return finalise_figure(
        ax,
        filename=filename,
        fig_folder=fig_folder,
        save_path=save_path,
        show=show,
    )


@scoped_plot_style
def plot_cost_progress(
    config: CampaignConfig,
    df: pd.DataFrame,
    *,
    filename: str | Path | None = None,
    fig_folder: str | Path = "figures",
    save_path: str | Path | None = None,
    show: bool = False,
):
    """Plot best observed objective against cumulative effective cost."""
    validate_campaign_data(config, df)
    if config.cost is None:
        raise ValueError("plot_cost_progress() requires a config with a cost section.")
    observed = get_observed_data(config, df)

    _, ax = new_figure(figsize=(8, 6))
    if observed.empty:
        set_title(ax, f"{config.campaign_name}: no observations yet")
        y_label = "Hypervolume" if config.is_multi_objective else config.objective.name
        set_axis_labels(ax, "Cumulative cost", y_label)
        return finalise_figure(
            ax,
            filename=filename,
            fig_folder=fig_folder,
            save_path=save_path,
            show=show,
        )

    cumulative_costs = []
    running_cost = 0.0
    for _, row in observed.iterrows():
        running_cost += effective_row_cost(config, row)
        cumulative_costs.append(running_cost)

    if config.is_multi_objective:
        progress = hypervolume_progress(config, df)
        ax.plot(cumulative_costs, progress["hypervolume"], marker="o", label="hypervolume")
        set_title(ax, f"{config.campaign_name}: cost progress")
        set_axis_labels(ax, "Cumulative cost", "Hypervolume")
        add_legend(ax)
        return finalise_figure(
            ax,
            filename=filename,
            fig_folder=fig_folder,
            save_path=save_path,
            show=show,
        )

    values = pd.to_numeric(observed[config.objective.name])
    best = _directional_best_so_far(config, values)
    ax.plot(cumulative_costs, best, marker="o", label="best so far")
    set_title(ax, f"{config.campaign_name}: cost progress")
    set_axis_labels(ax, "Cumulative cost", config.objective.name)
    add_legend(ax)
    return finalise_figure(
        ax,
        filename=filename,
        fig_folder=fig_folder,
        save_path=save_path,
        show=show,
    )


@scoped_plot_style
def plot_replicates(
    config: CampaignConfig,
    df: pd.DataFrame,
    *,
    filename: str | Path | None = None,
    fig_folder: str | Path = "figures",
    save_path: str | Path | None = None,
    show: bool = False,
):
    """Plot raw replicate observations and replicate-group mean summaries."""
    validate_campaign_data(config, df)
    if not config.replicates.enabled:
        raise ValueError("plot_replicates() requires replicates.enabled: true.")

    observed = get_observed_data(config, df)
    summary = replicate_summary(config, df)
    if config.is_multi_objective:
        return _plot_multi_objective_replicates(
            config,
            observed,
            summary,
            filename=filename,
            fig_folder=fig_folder,
            save_path=save_path,
            show=show,
        )
    _, ax = new_figure(figsize=(9, 6))
    if summary.empty:
        set_title(ax, f"{config.campaign_name}: no replicate observations yet")
        set_axis_labels(ax, "Replicate group index", config.objective.name)
        return finalise_figure(
            ax,
            filename=filename,
            fig_folder=fig_folder,
            save_path=save_path,
            show=show,
        )

    group_positions = {
        group: index + 1
        for index, group in enumerate(summary["replicate_group"].astype(str).tolist())
    }
    raw_x = observed["replicate_group"].astype(str).map(group_positions)
    raw_y = pd.to_numeric(observed[config.objective.name])
    ax.scatter(raw_x, raw_y, color=NEUTRAL_COLOR, label="raw observation")

    x = list(range(1, len(summary) + 1))
    mean = pd.to_numeric(summary["objective_mean"])
    sem = pd.to_numeric(summary["objective_sem"])
    ax.errorbar(
        x,
        mean,
        yerr=sem,
        fmt="o-",
        color=OBSERVED_COLOR,
        ecolor=OBSERVED_COLOR,
        capsize=4,
        label="group mean +/- SEM",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(index) for index in x])
    set_title(ax, f"{config.campaign_name}: replicate summary")
    set_axis_labels(ax, "Replicate group index", config.objective.name)
    add_legend(ax)
    return finalise_figure(
        ax,
        filename=filename,
        fig_folder=fig_folder,
        save_path=save_path,
        show=show,
    )
