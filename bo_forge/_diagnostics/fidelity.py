"""Basic plotting helpers for campaign diagnostics."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
from matplotlib.ticker import MaxNLocator

from bo_forge.config import CampaignConfig
from bo_forge.multifidelity import (
    _active_fidelity_suggestions,
    _is_target_fidelity,
)
from bo_forge.plot_style import (
    HIGHLIGHT_COLOR,
    NEUTRAL_COLOR,
    OBSERVED_COLOR,
    TARGET_COLOR,
    WARNING_COLOR,
    add_legend,
    finalise_axes,
    new_subplots,
    scoped_plot_style,
    set_axis_labels,
    set_title,
)
from bo_forge.validation import get_observed_data, validate_campaign_data


@scoped_plot_style
def plot_fidelity_diagnostics(
    config: CampaignConfig,
    df: pd.DataFrame,
    *,
    filename: str | Path | None = None,
    fig_folder: str | Path = "figures",
    save_path: str | Path | None = None,
    show: bool = False,
):
    """Plot observed objective values against fidelity and fidelity coverage."""
    validate_campaign_data(config, df)
    if config.fidelity is None:
        raise ValueError("plot_fidelity_diagnostics() requires a config with fidelity.")

    observed = get_observed_data(config, df)
    fidelity_name = config.fidelity.variable
    objective = config.objective.name
    target = float(config.fidelity.target)

    fig, axes = new_subplots(
        1,
        2,
        figsize=(14, 5.5),
        facecolor="white",
        constrained_layout=True,
    )
    scatter_ax, count_ax = axes

    _render_fidelity_diagnostic_panels(
        config,
        observed,
        scatter_ax,
        count_ax,
        fidelity_name=fidelity_name,
        objective=objective,
        target=target,
    )
    _label_fidelity_diagnostic_panels(
        config,
        scatter_ax,
        count_ax,
        fidelity_name=fidelity_name,
        objective=objective,
    )
    fig.suptitle(
        f"{config.campaign_name}: fidelity diagnostics",
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


def _render_fidelity_diagnostic_panels(
    config: CampaignConfig,
    observed: pd.DataFrame,
    scatter_ax,
    count_ax,
    *,
    fidelity_name: str,
    objective: str,
    target: float,
) -> None:

    if observed.empty:
        scatter_ax.text(
            0.5,
            0.5,
            "No observed fidelity data yet.",
            ha="center",
            va="center",
            transform=scatter_ax.transAxes,
        )
        scatter_ax.axvline(
            target,
            color=HIGHLIGHT_COLOR,
            linestyle="--",
            linewidth=2,
            label=f"target fidelity = {target:g}",
        )
        count_ax.text(
            0.5,
            0.5,
            "No observations yet.",
            ha="center",
            va="center",
            transform=count_ax.transAxes,
        )
    else:
        fidelity_values = pd.to_numeric(observed[fidelity_name])
        objective_values = pd.to_numeric(observed[objective])
        scatter_ax.scatter(
            fidelity_values,
            objective_values,
            color=OBSERVED_COLOR,
            label="observed",
        )
        scatter_ax.axvline(
            target,
            color=HIGHLIGHT_COLOR,
            linestyle="--",
            linewidth=2,
            label=f"target fidelity = {target:g}",
        )
        if config.fidelity.levels is None:
            bin_count = min(10, max(1, int(fidelity_values.nunique())))
            count_ax.hist(
                fidelity_values,
                bins=bin_count,
                color=NEUTRAL_COLOR,
                edgecolor="black",
            )
    if config.fidelity.levels is not None:
        levels = list(config.fidelity.levels)
        counts = [
            0
            if observed.empty
            else int(
                pd.to_numeric(observed[fidelity_name]).map(
                    lambda value, level=level: math.isclose(
                        float(value), level, rel_tol=1e-9, abs_tol=1e-9
                    )
                ).sum()
            )
            for level in levels
        ]
        count_ax.bar(
            levels,
            counts,
            width=_discrete_fidelity_bar_width(levels),
            color=NEUTRAL_COLOR,
            edgecolor="black",
        )
        scatter_ax.set_xticks(levels)
        count_ax.set_xticks(levels)

    count_ax.axvline(
        target,
        color=HIGHLIGHT_COLOR,
        linestyle="--",
        linewidth=2,
        label=f"target fidelity = {target:g}",
    )



def _label_fidelity_diagnostic_panels(
    config: CampaignConfig,
    scatter_ax,
    count_ax,
    *,
    fidelity_name: str,
    objective: str,
) -> None:
    set_title(scatter_ax, "Objective vs fidelity")
    set_axis_labels(scatter_ax, fidelity_name, objective)
    add_legend(scatter_ax)
    count_title = (
        "Observed fidelity level counts"
        if config.fidelity.levels is not None
        else "Observed fidelity distribution"
    )
    set_title(count_ax, count_title)
    set_axis_labels(count_ax, fidelity_name, "Observed rows")
    count_ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    add_legend(count_ax)


@scoped_plot_style
def plot_fidelity_progress(
    config: CampaignConfig,
    df: pd.DataFrame,
    *,
    filename: str | Path | None = None,
    fig_folder: str | Path = "figures",
    save_path: str | Path | None = None,
    show: bool = False,
):
    """Plot fidelity usage and target-fidelity objective progress by iteration."""
    validate_campaign_data(config, df)
    if config.fidelity is None:
        raise ValueError("plot_fidelity_progress() requires a config with fidelity.")

    observed = get_observed_data(config, df)
    active = _active_fidelity_suggestions(config, df)
    fidelity_name = config.fidelity.variable
    objective = config.objective.name
    target = float(config.fidelity.target)

    fig, axes = new_subplots(
        1,
        2,
        figsize=(14, 5.5),
        facecolor="white",
        constrained_layout=True,
    )
    fidelity_ax, objective_ax = axes

    _render_fidelity_iteration_panel(
        config,
        observed,
        active,
        fidelity_ax,
        fidelity_name=fidelity_name,
        target=target,
    )
    _render_target_fidelity_progress(
        config,
        observed,
        objective_ax,
        fidelity_name=fidelity_name,
        objective=objective,
        target=target,
    )
    fig.suptitle(
        f"{config.campaign_name}: fidelity progress",
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


def _render_fidelity_iteration_panel(
    config: CampaignConfig,
    observed: pd.DataFrame,
    active: pd.DataFrame,
    fidelity_ax,
    *,
    fidelity_name: str,
    target: float,
) -> None:

    if observed.empty:
        fidelity_ax.text(
            0.5,
            0.5,
            "No observed fidelity data yet.",
            ha="center",
            va="center",
            transform=fidelity_ax.transAxes,
        )
    else:
        qmfkg = observed["source"].astype(str) == "qmf_kg"
        initial_or_manual = observed.loc[~qmfkg]
        qmfkg_observed = observed.loc[qmfkg]
        if not initial_or_manual.empty:
            fidelity_ax.scatter(
                pd.to_numeric(initial_or_manual["iteration"]),
                pd.to_numeric(initial_or_manual[fidelity_name]),
                color=OBSERVED_COLOR,
                marker="o",
                label="manual / initial observed",
            )
        if not qmfkg_observed.empty:
            fidelity_ax.scatter(
                pd.to_numeric(qmfkg_observed["iteration"]),
                pd.to_numeric(qmfkg_observed[fidelity_name]),
                color=TARGET_COLOR,
                marker="s",
                label="qMFKG observed",
            )
    if not active.empty:
        fidelity_ax.scatter(
            pd.to_numeric(active["iteration"]),
            pd.to_numeric(active[fidelity_name]),
            facecolors="none",
            edgecolors=WARNING_COLOR,
            linewidths=1.5,
            marker="o",
            label="active suggestions",
        )
    fidelity_ax.axhline(
        target,
        color=HIGHLIGHT_COLOR,
        linestyle="--",
        linewidth=2,
        label=f"target fidelity = {target:g}",
    )
    if config.fidelity.levels is not None:
        fidelity_ax.set_yticks(list(config.fidelity.levels))
    set_title(fidelity_ax, "Fidelity by iteration")
    set_axis_labels(fidelity_ax, "Campaign iteration", fidelity_name)
    add_legend(fidelity_ax)


def _render_target_fidelity_progress(
    config: CampaignConfig,
    observed: pd.DataFrame,
    objective_ax,
    *,
    fidelity_name: str,
    objective: str,
    target: float,
) -> None:
    if observed.empty:
        target_observed = observed
    else:
        target_mask = pd.to_numeric(observed[fidelity_name]).map(
            lambda value: _is_target_fidelity(value, target)
        )
        target_observed = observed.loc[target_mask].copy()
    if target_observed.empty:
        objective_ax.text(
            0.5,
            0.5,
            "No target-fidelity observations yet.",
            ha="center",
            va="center",
            transform=objective_ax.transAxes,
        )
    else:
        iterations = pd.to_numeric(target_observed["iteration"])
        objective_values = pd.to_numeric(target_observed[objective])
        objective_ax.scatter(
            iterations,
            objective_values,
            color=OBSERVED_COLOR,
            label="target-fidelity observed",
        )
        per_iteration = (
            target_observed.assign(
                _iteration=iterations,
                _objective=objective_values,
            )
            .groupby("_iteration", sort=True)["_objective"]
            .agg("max" if config.objective.direction == "maximize" else "min")
        )
        best_so_far = (
            per_iteration.cummax()
            if config.objective.direction == "maximize"
            else per_iteration.cummin()
        )
        objective_ax.plot(
            best_so_far.index,
            best_so_far.values,
            color=TARGET_COLOR,
            marker="o",
            label="best so far",
        )
    set_title(objective_ax, "Target-fidelity objective progress")
    set_axis_labels(objective_ax, "Campaign iteration", objective)
    add_legend(objective_ax)


def _discrete_fidelity_bar_width(levels: list[float]) -> float:
    if len(levels) < 2:
        return 0.1
    return min(
        current - previous
        for previous, current in zip(levels, levels[1:], strict=False)
    ) * 0.6
