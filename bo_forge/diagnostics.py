"""Basic plotting helpers for campaign diagnostics."""

from __future__ import annotations

import math
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

from bo_forge.config import CampaignConfig
from bo_forge.contextual import context_summary
from bo_forge.costs import effective_row_cost
from bo_forge.models import (
    dataframe_to_training_tensors,
    fit_gp_model,
    model_profile_comparison,
)
from bo_forge.multi_objective import (
    hypervolume_progress,
    multi_objective_observed_data,
    pareto_front,
)
from bo_forge.noisy import qlog_nei_summary
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
from bo_forge.replicates import replicate_summary
from bo_forge.structured import stage_summary
from bo_forge.transforms import (
    dataframe_to_variable_coverage,
    has_mixed_variables,
    objective_from_model_space,
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
    ax.scatter(raw_x, raw_y, color="#64748b", alpha=0.75, label="raw observation")

    x = list(range(1, len(summary) + 1))
    mean = pd.to_numeric(summary["objective_mean"])
    sem = pd.to_numeric(summary["objective_sem"])
    ax.errorbar(
        x,
        mean,
        yerr=sem,
        fmt="o-",
        color="#2563eb",
        ecolor="#1d4ed8",
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

    configure_plot_style()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5.5),
        facecolor="white",
        constrained_layout=True,
    )
    scatter_ax, count_ax = axes

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
            color="#d97706",
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
            color="#2563eb",
            alpha=0.8,
            label="observed",
        )
        scatter_ax.axvline(
            target,
            color="#d97706",
            linestyle="--",
            linewidth=2,
            label=f"target fidelity = {target:g}",
        )
        bin_count = min(10, max(1, int(fidelity_values.nunique())))
        count_ax.hist(
            fidelity_values,
            bins=bin_count,
            color="#64748b",
            edgecolor="black",
            alpha=0.85,
        )
        count_ax.axvline(
            target,
            color="#d97706",
            linestyle="--",
            linewidth=2,
            label=f"target fidelity = {target:g}",
        )

    set_title(scatter_ax, "Objective vs fidelity")
    set_axis_labels(scatter_ax, fidelity_name, objective)
    add_legend(scatter_ax)
    set_title(count_ax, "Observed fidelity distribution")
    set_axis_labels(count_ax, fidelity_name, "Observed rows")
    count_ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    add_legend(count_ax)
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


def plot_context_diagnostics(
    config: CampaignConfig,
    df: pd.DataFrame,
    *,
    filename: str | Path | None = None,
    fig_folder: str | Path = "figures",
    save_path: str | Path | None = None,
    show: bool = False,
):
    """Plot observed counts and best objective by context combination."""
    validate_campaign_data(config, df)
    if config.context is None:
        raise ValueError("plot_context_diagnostics() requires a config with context.")

    summary = context_summary(config, df)
    observed_total = (
        0 if summary.empty else int(pd.to_numeric(summary["observed_rows"]).sum())
    )
    configure_plot_style()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15, 5.5),
        facecolor="white",
        constrained_layout=True,
    )
    count_ax, best_ax = axes

    if observed_total == 0:
        count_ax.text(
            0.5,
            0.5,
            "No observed contextual rows yet.",
            ha="center",
            va="center",
            transform=count_ax.transAxes,
        )
        best_ax.text(
            0.5,
            0.5,
            "No best objective by context yet.",
            ha="center",
            va="center",
            transform=best_ax.transAxes,
        )
    else:
        plotted = (
            summary.sort_values(
                by=["observed_rows", "context_key"],
                ascending=[False, True],
                kind="stable",
            )
            .head(20)
            .copy()
        )
        labels = plotted["context_key"].astype(str).tolist()
        x = list(range(len(plotted)))
        counts = pd.to_numeric(plotted["observed_rows"])
        count_ax.bar(x, counts, color="#2563eb", alpha=0.88)
        count_ax.set_xticks(x)
        count_ax.set_xticklabels(labels, rotation=45, ha="right")
        count_ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        best_rows = plotted.loc[plotted["best_objective"].notna()].copy()
        if best_rows.empty:
            best_ax.text(
                0.5,
                0.5,
                "No best objective by context yet.",
                ha="center",
                va="center",
                transform=best_ax.transAxes,
            )
        else:
            best_labels = best_rows["context_key"].astype(str).tolist()
            best_x = list(range(len(best_rows)))
            best_values = pd.to_numeric(best_rows["best_objective"])
            best_ax.bar(best_x, best_values, color="#0f766e", alpha=0.88)
            best_ax.set_xticks(best_x)
            best_ax.set_xticklabels(best_labels, rotation=45, ha="right")

    count_title = (
        "Observed rows by context (top 20)"
        if len(summary) > 20
        else "Observed rows by context"
    )
    best_title = (
        "Best objective by context (top 20)"
        if len(summary) > 20
        else "Best objective by context"
    )
    set_title(count_ax, count_title)
    set_axis_labels(count_ax, "Context", "Observed rows")
    set_title(best_ax, best_title)
    set_axis_labels(
        best_ax,
        "Context",
        f"{config.objective.name} ({config.objective.direction})",
    )
    fig.suptitle(
        f"{config.campaign_name}: context diagnostics",
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


def plot_qlog_nei_diagnostics(
    config: CampaignConfig,
    df: pd.DataFrame,
    *,
    filename: str | Path | None = None,
    fig_folder: str | Path = "figures",
    save_path: str | Path | None = None,
    show: bool = False,
):
    """Plot read-only qLogNEI pending-state diagnostics."""
    if config.bo.acquisition != "qlog_nei":
        raise ValueError("plot_qlog_nei_diagnostics() requires bo.acquisition: qlog_nei.")
    summary = qlog_nei_summary(config, df)
    values = {str(row["field"]): row["value"] for _, row in summary.iterrows()}

    configure_plot_style()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5.5),
        facecolor="white",
        constrained_layout=True,
    )
    count_ax, readiness_ax = axes
    labels = [
        "Observed\nbaseline",
        "Active\npending",
        "Blocking\nreview",
        "Rejected /\ndeferred",
    ]
    counts = [
        int(values["observed_baseline_rows"]),
        int(values["active_pending_rows"]),
        int(values["blocking_review_pending_rows"]),
        int(values["rejected_or_deferred_pending_rows"]),
    ]
    count_ax.bar(
        range(len(counts)),
        counts,
        color=["#2563eb", "#0f766e", "#dc2626", "#64748b"],
        alpha=0.88,
    )
    count_ax.set_xticks(range(len(counts)))
    count_ax.set_xticklabels(labels)
    count_ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    ready = bool(values["ready_for_qlog_nei"])
    blocking = int(values["blocking_review_pending_rows"])
    remaining = int(values["initial_design_remaining"])
    active_pending_initial = int(values["active_pending_initial_rows"])
    if ready:
        message = (
            "Ready for model-based qLogNEI.\n"
            f"X_pending used: {values['x_pending_used']}."
        )
        color = "#0f766e"
    elif blocking > 0:
        message = (
            "Blocked by review-pending rows.\n"
            "Accept, reject, or defer them first."
        )
        color = "#dc2626"
    elif active_pending_initial > 0:
        message = (
            "Accepted pending initial-design rows must be observed.\n"
            "Mark them observed before model-based qLogNEI."
        )
        color = "#d97706"
    else:
        message = (
            "Observed initial design is incomplete.\n"
            f"Observed rows still needed: {remaining}."
        )
        color = "#d97706"
    readiness_ax.text(
        0.5,
        0.62,
        message,
        ha="center",
        va="center",
        transform=readiness_ax.transAxes,
        fontsize=13,
        color=color,
        fontweight="bold",
    )
    readiness_ax.text(
        0.5,
        0.30,
        (
            f"initial_design_size = {values['initial_design_size']}\n"
            f"train_yvar_available = {values['train_yvar_available']}\n"
            f"model_profile = {values['model_profile']}"
        ),
        ha="center",
        va="center",
        transform=readiness_ax.transAxes,
        fontsize=11,
        color="#334155",
    )
    readiness_ax.set_xticks([])
    readiness_ax.set_yticks([])

    set_title(count_ax, "qLogNEI pending-state counts")
    set_axis_labels(count_ax, "Row state", "Rows")
    set_title(readiness_ax, "qLogNEI readiness")
    fig.suptitle(
        f"{config.campaign_name}: qLogNEI diagnostics",
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


def plot_model_diagnostics(
    config: CampaignConfig,
    df: pd.DataFrame,
    *,
    filename: str | Path | None = None,
    fig_folder: str | Path = "figures",
    save_path: str | Path | None = None,
    show: bool = False,
):
    """Plot observed objective values against posterior mean and residuals."""
    validate_campaign_data(config, df)
    if config.is_multi_objective:
        raise ValueError("plot_model_diagnostics() requires a single-objective config.")
    if config.fidelity is not None:
        raise ValueError("plot_model_diagnostics() does not support fidelity configs.")
    if config.is_structured_campaign:
        raise ValueError("plot_model_diagnostics() does not support structured configs.")

    observed = get_observed_data(config, df)
    configure_plot_style()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5.5),
        facecolor="white",
        constrained_layout=True,
    )
    fit_ax, residual_ax = axes

    if observed.empty:
        fit_ax.text(
            0.5,
            0.5,
            "No observed rows available for model diagnostics.",
            ha="center",
            va="center",
            transform=fit_ax.transAxes,
        )
        residual_ax.text(
            0.5,
            0.5,
            "No residuals available.",
            ha="center",
            va="center",
            transform=residual_ax.transAxes,
        )
    else:
        training = dataframe_to_training_tensors(config, observed)
        model = fit_gp_model(config, observed)
        posterior = model.posterior(training.train_x)
        predicted_user = objective_from_model_space(
            config,
            posterior.mean.squeeze(-1).detach(),
        )
        observed_user = objective_from_model_space(
            config,
            training.train_y.squeeze(-1).detach(),
        )
        residuals = observed_user - predicted_user

        observed_values = observed_user.cpu().numpy()
        predicted_values = predicted_user.cpu().numpy()
        residual_values = residuals.cpu().numpy()
        x = list(range(1, len(observed_values) + 1))

        fit_ax.scatter(observed_values, predicted_values, color="#2563eb", alpha=0.85)
        min_value = float(min(observed_values.min(), predicted_values.min()))
        max_value = float(max(observed_values.max(), predicted_values.max()))
        if math.isclose(min_value, max_value):
            min_value -= 0.5
            max_value += 0.5
        fit_ax.plot(
            [min_value, max_value],
            [min_value, max_value],
            color="#64748b",
            linestyle="--",
            linewidth=1.5,
            label="ideal",
        )
        add_legend(fit_ax)

        residual_ax.axhline(0.0, color="#64748b", linestyle="--", linewidth=1.5)
        residual_ax.scatter(x, residual_values, color="#dc2626", alpha=0.85)
        residual_ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    set_title(fit_ax, "Observed vs posterior mean")
    set_axis_labels(fit_ax, f"Observed {config.objective.name}", "Posterior mean")
    set_title(residual_ax, "Residuals on fitting rows")
    set_axis_labels(residual_ax, "Fitting row", "Observed - posterior mean")
    fig.suptitle(
        f"{config.campaign_name}: model diagnostics ({config.model.profile})",
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


def plot_model_comparison(
    config: CampaignConfig,
    df: pd.DataFrame,
    *,
    filename: str | Path | None = None,
    fig_folder: str | Path = "figures",
    save_path: str | Path | None = None,
    show: bool = False,
):
    """Plot read-only model-profile comparison diagnostics."""
    comparison = model_profile_comparison(config, df)
    configure_plot_style()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5.5),
        facecolor="white",
        constrained_layout=True,
    )
    error_ax, std_ax = axes
    metric_rows = comparison.copy()
    metric_rows["rmse_model_space"] = pd.to_numeric(
        metric_rows["rmse_model_space"], errors="coerce"
    )
    metric_rows["mae_model_space"] = pd.to_numeric(
        metric_rows["mae_model_space"], errors="coerce"
    )
    metric_rows["mean_predicted_std"] = pd.to_numeric(
        metric_rows["mean_predicted_std"], errors="coerce"
    )
    fitted = metric_rows[metric_rows["fit_status"] != "insufficient_observed"].copy()
    fitted["rmse_model_space"] = pd.to_numeric(
        fitted["rmse_model_space"], errors="coerce"
    )
    fitted["mae_model_space"] = pd.to_numeric(
        fitted["mae_model_space"], errors="coerce"
    )
    fitted["mean_predicted_std"] = pd.to_numeric(
        fitted["mean_predicted_std"], errors="coerce"
    )
    fitted = fitted.dropna(subset=["rmse_model_space", "mae_model_space"])
    status_note = _model_comparison_status_note(
        metric_rows.loc[~metric_rows.index.isin(fitted.index)]
    )

    if fitted.empty:
        insufficient = comparison["fit_status"].eq("insufficient_observed").all()
        if insufficient:
            message = "At least two fitting rows are required for model comparison."
            secondary_message = "No predicted standard deviations available."
        else:
            message = "Model comparison fits failed."
            secondary_message = "Run model-compare to inspect profile fit statuses."
        error_ax.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            transform=error_ax.transAxes,
        )
        std_ax.text(
            0.5,
            0.5,
            secondary_message,
            ha="center",
            va="center",
            transform=std_ax.transAxes,
        )
        if status_note:
            std_ax.text(
                0.5,
                0.25,
                status_note,
                ha="center",
                va="center",
                fontsize=9,
                color="#475569",
                transform=std_ax.transAxes,
            )
    else:
        x = list(range(len(fitted)))
        labels = fitted["model_profile"].astype(str).tolist()
        width = 0.35
        error_ax.bar(
            [position - width / 2 for position in x],
            fitted["rmse_model_space"],
            width=width,
            color="#2563eb",
            label="RMSE",
        )
        error_ax.bar(
            [position + width / 2 for position in x],
            fitted["mae_model_space"],
            width=width,
            color="#0891b2",
            label="MAE",
        )
        error_ax.set_xticks(x)
        error_ax.set_xticklabels(labels)
        add_legend(error_ax)

        std_ax.bar(
            x,
            fitted["mean_predicted_std"],
            color="#7c3aed",
            label="Mean predicted std",
        )
        std_ax.set_xticks(x)
        std_ax.set_xticklabels(labels)
        add_legend(std_ax)
        if status_note:
            error_ax.text(
                0.02,
                0.98,
                status_note,
                ha="left",
                va="top",
                fontsize=9,
                color="#475569",
                transform=error_ax.transAxes,
                bbox={
                    "boxstyle": "round,pad=0.35",
                    "facecolor": "#f8fafc",
                    "edgecolor": "#cbd5e1",
                    "alpha": 0.95,
                },
            )

    set_title(error_ax, "Model-space residual metrics")
    set_axis_labels(error_ax, "Model profile", "Error")
    set_title(std_ax, "Mean predicted uncertainty")
    set_axis_labels(std_ax, "Model profile", "Posterior std")
    fig.suptitle(
        f"{config.campaign_name}: model profile comparison (diagnostic only)",
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


def _model_comparison_status_note(rows: pd.DataFrame) -> str:
    if rows.empty:
        return ""
    pieces: list[str] = []
    for _, row in rows.iterrows():
        profile = str(row.get("model_profile", "unknown"))
        status = str(row.get("fit_status", "not_plotted"))
        message = str(row.get("fit_message", "") or "").strip()
        detail = f"{profile}={status}"
        if message:
            shortened = message if len(message) <= 80 else f"{message[:77]}..."
            detail = f"{detail} ({shortened})"
        pieces.append(detail)
    return "No metric bars: " + "; ".join(pieces)


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
    configure_plot_style()
    fig, axes = plt.subplots(
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

    counts_ax.bar(x, observed, color="#2563eb", label="observed")
    counts_ax.bar(x, suggested, bottom=observed, color="#94a3b8", label="suggested")
    for position, value in zip(x, pending, strict=True):
        if value > 0:
            counts_ax.text(
                position,
                observed.iloc[position] + suggested.iloc[position] + 0.05,
                f"pending {int(value)}",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#92400e",
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
        cmap="Blues",
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
    configure_plot_style()
    fig, axes = plt.subplots(
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
        ax.scatter(raw_x, raw_y, color="#64748b", alpha=0.75, label="raw observation")
        mean = pd.to_numeric(summary[f"{objective.name}_mean"])
        sem = pd.to_numeric(summary[f"{objective.name}_sem"])
        ax.errorbar(
            x,
            mean,
            yerr=sem,
            fmt="o-",
            color="#2563eb",
            ecolor="#1d4ed8",
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


def plot_pareto(
    config: CampaignConfig,
    df: pd.DataFrame,
    *,
    filename: str | Path | None = None,
    fig_folder: str | Path = "figures",
    save_path: str | Path | None = None,
    show: bool = False,
):
    """Plot observed Pareto diagnostics for a multi-objective campaign."""
    validate_campaign_data(config, df)
    if not config.is_multi_objective:
        raise ValueError("plot_pareto() requires a multi-objective config.")
    if len(config.objectives) > 2:
        return _plot_pairwise_pareto(
            config,
            df,
            filename=filename,
            fig_folder=fig_folder,
            save_path=save_path,
            show=show,
        )
    observed = multi_objective_observed_data(config, df)
    front = pareto_front(config, df)
    x_name, y_name = config.objective_names

    _, ax = new_figure(figsize=(8, 6))
    if observed.empty:
        set_title(ax, f"{config.campaign_name}: no observations yet")
        ax.text(
            0.5,
            0.5,
            "No observed objective values yet.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        set_axis_labels(
            ax,
            _objective_axis_label(config, x_name),
            _objective_axis_label(config, y_name),
        )
        return finalise_figure(
            ax,
            filename=filename,
            fig_folder=fig_folder,
            save_path=save_path,
            show=show,
        )

    ax.scatter(
        observed[x_name].astype(float),
        observed[y_name].astype(float),
        color="#64748b",
        alpha=0.65,
        label="observed",
    )
    if not front.empty:
        ax.plot(
            front[x_name].astype(float),
            front[y_name].astype(float),
            color="#d97706",
            marker="o",
            label="Pareto front",
        )
    set_title(ax, f"{config.campaign_name}: Pareto front")
    set_axis_labels(
        ax,
        _objective_axis_label(config, x_name),
        _objective_axis_label(config, y_name),
    )
    add_legend(ax)
    return finalise_figure(
        ax,
        filename=filename,
        fig_folder=fig_folder,
        save_path=save_path,
        show=show,
    )


def plot_pareto_parallel(
    config: CampaignConfig,
    df: pd.DataFrame,
    *,
    filename: str | Path | None = None,
    fig_folder: str | Path = "figures",
    save_path: str | Path | None = None,
    show: bool = False,
):
    """Plot Pareto-front rows with normalized parallel coordinates."""
    validate_campaign_data(config, df)
    if not config.is_multi_objective:
        raise ValueError("plot_pareto_parallel() requires a multi-objective config.")
    if len(config.objectives) < 3:
        raise ValueError(
            "plot_pareto_parallel() requires at least three objectives; use "
            "plot_pareto() for a 2D Pareto scatter."
        )
    observed = multi_objective_observed_data(config, df)
    front = pareto_front(config, df)

    _, ax = new_figure(figsize=(10, 6))
    if observed.empty or front.empty:
        set_title(ax, f"{config.campaign_name}: no Pareto observations yet")
        ax.text(
            0.5,
            0.5,
            "No Pareto-front rows to display yet.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        set_axis_labels(ax, "Objective", "Normalised value")
        return finalise_figure(
            ax,
            filename=filename,
            fig_folder=fig_folder,
            save_path=save_path,
            show=show,
        )

    normalized = _normalise_pareto_parallel_values(config, observed, front)
    x = list(range(len(config.objectives)))
    for _, row in normalized.iterrows():
        ax.plot(x, row.tolist(), color="#d97706", alpha=0.65, linewidth=1.8)
    ax.set_xticks(x)
    ax.set_xticklabels([_objective_axis_label(config, name) for name in config.objective_names])
    ax.set_ylim(-0.05, 1.05)
    set_title(ax, f"{config.campaign_name}: Pareto trade-off profiles")
    set_axis_labels(ax, "Objective", "Normalised value")
    return finalise_figure(
        ax,
        filename=filename,
        fig_folder=fig_folder,
        save_path=save_path,
        show=show,
    )


def plot_hypervolume(
    config: CampaignConfig,
    df: pd.DataFrame,
    *,
    filename: str | Path | None = None,
    fig_folder: str | Path = "figures",
    save_path: str | Path | None = None,
    show: bool = False,
):
    """Plot hypervolume progress for a multi-objective campaign."""
    validate_campaign_data(config, df)
    if not config.is_multi_objective:
        raise ValueError("plot_hypervolume() requires a multi-objective config.")
    progress = hypervolume_progress(config, df)

    _, ax = new_figure(figsize=(8, 6))
    if progress.empty:
        set_title(ax, f"{config.campaign_name}: no observations yet")
        ax.text(
            0.5,
            0.5,
            "No observed objective values yet.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        set_axis_labels(ax, "Observation", "Hypervolume")
        return finalise_figure(
            ax,
            filename=filename,
            fig_folder=fig_folder,
            save_path=save_path,
            show=show,
        )

    ax.plot(progress["observation"], progress["hypervolume"], marker="o")
    set_title(ax, f"{config.campaign_name}: hypervolume progress")
    set_axis_labels(ax, "Observation", "Hypervolume")
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


def _plot_pairwise_pareto(
    config: CampaignConfig,
    df: pd.DataFrame,
    *,
    filename: str | Path | None,
    fig_folder: str | Path,
    save_path: str | Path | None,
    show: bool,
):
    observed = multi_objective_observed_data(config, df)
    front = pareto_front(config, df)
    objective_pairs = list(combinations(config.objective_names, 2))
    column_count = min(3, len(objective_pairs))
    row_count = math.ceil(len(objective_pairs) / column_count)

    configure_plot_style()
    fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(5.2 * column_count, 4.5 * row_count),
        facecolor="white",
        constrained_layout=True,
        squeeze=False,
    )
    flat_axes = list(axes.flat)
    if observed.empty:
        first_ax = flat_axes[0]
        first_ax.text(
            0.5,
            0.5,
            "No observed objective values yet.",
            ha="center",
            va="center",
            transform=first_ax.transAxes,
        )
        set_title(first_ax, f"{config.campaign_name}: no observations yet")
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

    for ax, (x_name, y_name) in zip(flat_axes, objective_pairs, strict=False):
        ax.scatter(
            observed[x_name].astype(float),
            observed[y_name].astype(float),
            color="#64748b",
            alpha=0.55,
            label="observed",
        )
        if not front.empty:
            ax.scatter(
                front[x_name].astype(float),
                front[y_name].astype(float),
                color="#d97706",
                edgecolor="black",
                linewidth=0.5,
                label="full-space Pareto",
            )
        ax.set_title(f"{x_name} vs {y_name}", fontsize=12, fontweight="bold")
        ax.set_xlabel(_objective_axis_label(config, x_name), fontsize=12, fontweight="bold")
        ax.set_ylabel(_objective_axis_label(config, y_name), fontsize=12, fontweight="bold")
        add_legend(ax)

    for ax in flat_axes[len(objective_pairs) :]:
        ax.set_visible(False)

    fig.suptitle(
        f"{config.campaign_name}: pairwise Pareto projections",
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


def _normalise_pareto_parallel_values(
    config: CampaignConfig,
    observed: pd.DataFrame,
    front: pd.DataFrame,
) -> pd.DataFrame:
    normalized = pd.DataFrame(index=front.index)
    for objective in config.objectives:
        observed_values = pd.to_numeric(observed[objective.name])
        front_values = pd.to_numeric(front[objective.name])
        minimum = float(observed_values.min())
        maximum = float(observed_values.max())
        if math.isclose(minimum, maximum):
            display_values = pd.Series(0.5, index=front.index)
        else:
            display_values = (front_values - minimum) / (maximum - minimum)
            if objective.direction == "minimize":
                display_values = 1.0 - display_values
        normalized[objective.name] = display_values.astype(float)
    return normalized


def _objective_axis_label(config: CampaignConfig, objective_name: str) -> str:
    objective = next(item for item in config.objectives if item.name == objective_name)
    marker = "max" if objective.direction == "maximize" else "min"
    return f"{objective.name} ({marker})"


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
    unit = dataframe_to_variable_coverage(config, observed)
    data = unit.detach().cpu().numpy()
    return pd.DataFrame(data, columns=config.variable_names, index=observed.index).clip(
        lower=0.0,
        upper=1.0,
    )
