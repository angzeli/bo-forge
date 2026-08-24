"""Basic plotting helpers for campaign diagnostics."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
from matplotlib.ticker import MaxNLocator

from bo_forge._diagnostics.multi_objective import _objective_axis_label
from bo_forge.config import CampaignConfig
from bo_forge.plot_style import (
    HIGHLIGHT_COLOR,
    NEUTRAL_COLOR,
    OBSERVED_COLOR,
    ORDERED_CMAP,
    add_legend,
    finalise_axes,
    new_subplots,
    scoped_plot_style,
    set_axis_labels,
    set_title,
    style_colorbar,
)
from bo_forge.structured import stage_summary
from bo_forge.validation import validate_campaign_data


@scoped_plot_style
def plot_stage_diagnostics(
    config: CampaignConfig,
    df: pd.DataFrame,
    *,
    filename: str | Path | None = None,
    fig_folder: str | Path = "figures",
    save_path: str | Path | None = None,
    show: bool = False,
):
    """Plot structured-campaign stage counts and active-variable coverage."""
    validate_campaign_data(config, df)
    if not config.is_structured_campaign:
        raise ValueError("plot_stage_diagnostics() requires a structured campaign config.")

    summary = stage_summary(config, df)
    fig, axes = new_subplots(
        1,
        2,
        figsize=(15, 5.5),
        facecolor="white",
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.1, 1.25]},
    )
    counts_ax, variables_ax = axes
    x = list(range(len(summary)))
    stage_labels = summary["stage"].astype(str).tolist()
    observed = pd.to_numeric(summary["observed_rows"])
    suggested = pd.to_numeric(summary["suggested_rows"])
    pending = pd.to_numeric(summary["pending_rows"])

    counts_ax.bar(x, observed, color=OBSERVED_COLOR, label="observed")
    counts_ax.bar(x, suggested, bottom=observed, color=NEUTRAL_COLOR, label="suggested")
    for position, value in zip(x, pending, strict=True):
        if value > 0:
            counts_ax.text(
                position,
                observed.iloc[position] + suggested.iloc[position] + 0.05,
                f"pending {int(value)}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=HIGHLIGHT_COLOR,
            )
    counts_ax.set_xticks(x)
    counts_ax.set_xticklabels(stage_labels, rotation=0)
    counts_ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    set_title(counts_ax, "Rows by stage")
    set_axis_labels(counts_ax, "Stage", "Rows")
    add_legend(counts_ax)

    matrix = [
        [
            1.0 if variable.name in set(stage.variables) else 0.0
            for variable in config.variables
        ]
        for stage in config.stages
    ]
    image = variables_ax.imshow(
        matrix,
        aspect="auto",
        cmap=ORDERED_CMAP,
        vmin=0.0,
        vmax=1.0,
    )
    variables_ax.set_xticks(range(len(config.variables)))
    variables_ax.set_xticklabels(
        [variable.name.replace("_", "\n") for variable in config.variables],
        rotation=0,
        ha="center",
    )
    variables_ax.set_yticks(range(len(config.stages)))
    variables_ax.set_yticklabels(stage_labels)
    set_title(variables_ax, "Active variable map")
    set_axis_labels(variables_ax, "Variable", "Stage")
    colorbar = fig.colorbar(image, ax=variables_ax, ticks=[0.0, 1.0])
    colorbar.ax.set_yticklabels(["inactive", "active"])
    style_colorbar(colorbar, "Stage variable state")

    fig.suptitle(
        f"{config.campaign_name}: stage diagnostics",
        fontsize=18,
        fontweight="bold",
        color="black",
    )
    return finalise_axes(
        fig,
        axes,
        filename=filename,
        fig_folder=fig_folder,
        save_path=save_path,
        show=show,
        tick_label_size=10,
    )


def _plot_multi_objective_replicates(
    config: CampaignConfig,
    observed: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    filename: str | Path | None,
    fig_folder: str | Path,
    save_path: str | Path | None,
    show: bool,
):
    objective_count = len(config.objectives)
    column_count = min(2, objective_count)
    row_count = math.ceil(objective_count / column_count)
    fig, axes = new_subplots(
        row_count,
        column_count,
        figsize=(7.2 * column_count, 4.8 * row_count),
        facecolor="white",
        constrained_layout=True,
        squeeze=False,
    )
    flat_axes = list(axes.flat)
    if summary.empty:
        first_ax = flat_axes[0]
        first_ax.text(
            0.5,
            0.5,
            "No replicate observations yet.",
            ha="center",
            va="center",
            transform=first_ax.transAxes,
        )
        set_title(first_ax, f"{config.campaign_name}: no replicate observations yet")
        set_axis_labels(first_ax, "Replicate group index", "Objective value")
        for ax in flat_axes[1:]:
            ax.set_visible(False)
        return finalise_axes(
            fig,
            axes,
            filename=filename,
            fig_folder=fig_folder,
            save_path=save_path,
            show=show,
        )

    group_positions = {
        group: index + 1
        for index, group in enumerate(summary["replicate_group"].astype(str).tolist())
    }
    x = list(range(1, len(summary) + 1))
    for ax, objective in zip(flat_axes, config.objectives, strict=False):
        raw_x = observed["replicate_group"].astype(str).map(group_positions)
        raw_y = pd.to_numeric(observed[objective.name])
        ax.scatter(raw_x, raw_y, color=NEUTRAL_COLOR, label="raw observation")
        mean = pd.to_numeric(summary[f"{objective.name}_mean"])
        sem = pd.to_numeric(summary[f"{objective.name}_sem"])
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
        set_title(ax, f"{objective.name} replicates")
        set_axis_labels(ax, "Replicate group index", _objective_axis_label(config, objective.name))
        add_legend(ax)

    for ax in flat_axes[objective_count:]:
        ax.set_visible(False)
    fig.suptitle(
        f"{config.campaign_name}: replicate summaries",
        fontsize=18,
        fontweight="bold",
        color="black",
    )
    return finalise_axes(
        fig,
        axes,
        filename=filename,
        fig_folder=fig_folder,
        save_path=save_path,
        show=show,
        tick_label_size=10,
    )
