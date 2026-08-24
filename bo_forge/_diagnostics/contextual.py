"""Basic plotting helpers for campaign diagnostics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from matplotlib.ticker import MaxNLocator

from bo_forge.config import CampaignConfig
from bo_forge.contextual import context_summary
from bo_forge.noisy import qlog_nei_summary
from bo_forge.plot_style import (
    HIGHLIGHT_COLOR,
    NEUTRAL_COLOR,
    OBSERVED_COLOR,
    TARGET_COLOR,
    WARNING_COLOR,
    finalise_axes,
    new_subplots,
    scoped_plot_style,
    set_axis_labels,
    set_title,
)
from bo_forge.validation import validate_campaign_data


@scoped_plot_style
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
    fig, axes = new_subplots(
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
        count_ax.bar(x, counts, color=OBSERVED_COLOR)
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
            best_ax.bar(best_x, best_values, color=TARGET_COLOR)
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


@scoped_plot_style
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

    fig, axes = new_subplots(
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
        color=[OBSERVED_COLOR, TARGET_COLOR, WARNING_COLOR, NEUTRAL_COLOR],
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
        color = TARGET_COLOR
    elif blocking > 0:
        message = (
            "Blocked by review-pending rows.\n"
            "Accept, reject, or defer them first."
        )
        color = WARNING_COLOR
    elif active_pending_initial > 0:
        message = (
            "Accepted pending initial-design rows must be observed.\n"
            "Mark them observed before model-based qLogNEI."
        )
        color = HIGHLIGHT_COLOR
    else:
        message = (
            "Observed initial design is incomplete.\n"
            f"Observed rows still needed: {remaining}."
        )
        color = HIGHLIGHT_COLOR
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
        color=NEUTRAL_COLOR,
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
