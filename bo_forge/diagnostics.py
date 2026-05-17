"""Basic plotting helpers for campaign diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

from bo_forge.config import CampaignConfig
from bo_forge.plot_style import (
    add_legend,
    configure_plot_style,
    finalise_axes,
    finalise_figure,
    new_figure,
    set_axis_labels,
    set_title,
    style_colorbar,
)
from bo_forge.validation import get_observed_data, validate_campaign_data

_HIGH_DIM_TITLE_SIZE = 16
_HIGH_DIM_AXIS_LABEL_SIZE = 14
_HIGH_DIM_TICK_LABEL_SIZE = 10
_HIGH_DIM_LEGEND_SIZE = 9
_HIGH_DIM_COLORBAR_LABEL_SIZE = 12
_HIGH_DIM_COLORBAR_TICK_SIZE = 10


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
            cmap="viridis",
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


def _plot_high_dimensional_diagnostics(
    config: CampaignConfig,
    observed: pd.DataFrame,
    *,
    filename: str | Path | None,
    fig_folder: str | Path,
    save_path: str | Path | None,
    show: bool,
):
    objective = config.objective.name
    values = pd.to_numeric(observed[objective])
    best = _directional_best_so_far(config, values)
    x = range(1, len(observed) + 1)
    coverage = _normalised_variable_coverage(config, observed)

    configure_plot_style()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 6),
        facecolor="white",
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.0, 1.35]},
    )
    layout_engine = fig.get_layout_engine()
    if layout_engine is not None:
        layout_engine.set(w_pad=0.08, h_pad=0.2, wspace=0.06)
    progress_ax, coverage_ax = axes

    progress_ax.plot(x, values, marker="o", label="observed")
    progress_ax.plot(x, best, marker=".", label="best so far")
    progress_ax.margins(x=0.05, y=0.18)
    progress_ax.set_xlim(0.5, len(observed) + 0.5)
    progress_ax.set_xticks(_observation_tick_positions(len(observed)))
    progress_ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
    _set_compact_title(progress_ax, "Objective progress")
    _set_compact_axis_labels(progress_ax, "Observation", objective)
    _add_compact_legend(progress_ax)

    image = coverage_ax.imshow(
        coverage.to_numpy(dtype=float),
        aspect="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    coverage_ax.set_xticks(range(len(config.variables)))
    variable_labels = [name.replace("_", "\n") for name in config.variable_names]
    coverage_ax.set_xticklabels(variable_labels, rotation=0, ha="center")
    coverage_ax.set_yticks(range(len(observed)))
    coverage_ax.set_yticklabels(range(1, len(observed) + 1))
    _set_compact_title(coverage_ax, "Variable coverage")
    _set_compact_axis_labels(coverage_ax, "Variable", "Observation")
    colorbar = fig.colorbar(image, ax=coverage_ax)
    _style_compact_colorbar(colorbar, "Normalised value")

    return finalise_axes(
        fig,
        axes,
        filename=filename,
        fig_folder=fig_folder,
        save_path=save_path,
        show=show,
        tick_label_size=_HIGH_DIM_TICK_LABEL_SIZE,
    )


def _directional_best_so_far(config: CampaignConfig, values: pd.Series) -> pd.Series:
    return values.cummax() if config.objective.direction == "maximize" else values.cummin()


def _observation_tick_positions(count: int) -> list[int]:
    if count <= 8:
        return list(range(1, count + 1))

    step = max(round(count / 6), 1)
    ticks = list(range(1, count + 1, step))
    if ticks[-1] != count:
        ticks.append(count)
    return ticks


def _set_compact_title(ax, title: str) -> None:
    ax.set_title(
        title,
        fontsize=_HIGH_DIM_TITLE_SIZE,
        fontweight="bold",
        color="black",
    )


def _set_compact_axis_labels(ax, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(
        xlabel,
        fontsize=_HIGH_DIM_AXIS_LABEL_SIZE,
        fontweight="bold",
        color="black",
    )
    ax.set_ylabel(
        ylabel,
        fontsize=_HIGH_DIM_AXIS_LABEL_SIZE,
        fontweight="bold",
        color="black",
    )


def _add_compact_legend(ax) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return

    legend = ax.legend(
        handles,
        labels,
        frameon=True,
        prop={"size": _HIGH_DIM_LEGEND_SIZE, "weight": "bold"},
        loc="best",
        facecolor="white",
        edgecolor="black",
    )
    for text in legend.get_texts():
        text.set_color("black")


def _style_compact_colorbar(colorbar, label: str) -> None:
    colorbar.set_label(
        label,
        fontsize=_HIGH_DIM_COLORBAR_LABEL_SIZE,
        fontweight="bold",
        color="black",
    )
    colorbar.ax.tick_params(
        labelsize=_HIGH_DIM_COLORBAR_TICK_SIZE,
        colors="black",
    )
    colorbar.outline.set_edgecolor("black")
    colorbar.outline.set_linewidth(1.8)
    for tick_label in colorbar.ax.get_yticklabels():
        tick_label.set_fontweight("bold")
        tick_label.set_color("black")


def _normalised_variable_coverage(
    config: CampaignConfig,
    observed: pd.DataFrame,
) -> pd.DataFrame:
    data = {}
    for variable in config.variables:
        values = pd.to_numeric(observed[variable.name])
        width = variable.upper - variable.lower
        normalised = (values - variable.lower) / width
        data[variable.name] = normalised.clip(lower=0.0, upper=1.0)
    return pd.DataFrame(data, index=observed.index)
