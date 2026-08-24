"""Basic plotting helpers for campaign diagnostics."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
from matplotlib.ticker import MaxNLocator

from bo_forge.config import CampaignConfig
from bo_forge.models import (
    dataframe_to_training_tensors,
    fit_gp_model,
    model_profile_comparison,
)
from bo_forge.plot_style import (
    MODEL_COLOR,
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
from bo_forge.transforms import (
    objective_from_model_space,
)
from bo_forge.validation import get_observed_data, validate_campaign_data


@scoped_plot_style
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
    fig, axes = new_subplots(
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

        fit_ax.scatter(observed_values, predicted_values, color=OBSERVED_COLOR)
        min_value = float(min(observed_values.min(), predicted_values.min()))
        max_value = float(max(observed_values.max(), predicted_values.max()))
        if math.isclose(min_value, max_value):
            min_value -= 0.5
            max_value += 0.5
        fit_ax.plot(
            [min_value, max_value],
            [min_value, max_value],
            color=NEUTRAL_COLOR,
            linestyle="--",
            linewidth=1.5,
            label="ideal",
        )
        add_legend(fit_ax)

        residual_ax.axhline(0.0, color=NEUTRAL_COLOR, linestyle="--", linewidth=1.5)
        residual_ax.scatter(x, residual_values, color=WARNING_COLOR)
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


@scoped_plot_style
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
    fig, axes = new_subplots(
        1,
        2,
        figsize=(14, 5.5),
        facecolor="white",
        constrained_layout=True,
    )
    error_ax, std_ax = axes
    metric_rows, fitted, status_note = _model_comparison_plot_rows(comparison)
    _render_model_comparison_panels(
        comparison,
        metric_rows,
        fitted,
        status_note,
        error_ax,
        std_ax,
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


def _model_comparison_plot_rows(
    comparison: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
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
    return metric_rows, fitted, status_note


def _render_model_comparison_panels(
    comparison: pd.DataFrame,
    metric_rows: pd.DataFrame,
    fitted: pd.DataFrame,
    status_note: str,
    error_ax,
    std_ax,
) -> None:
    del metric_rows

    if fitted.empty:
        _render_empty_model_comparison(comparison, status_note, error_ax, std_ax)
        return
    _render_fitted_model_comparison(fitted, status_note, error_ax, std_ax)


def _render_empty_model_comparison(
    comparison: pd.DataFrame,
    status_note: str,
    error_ax,
    std_ax,
) -> None:
    insufficient = comparison["fit_status"].eq("insufficient_observed").all()
    message = (
        "At least two fitting rows are required for model comparison."
        if insufficient
        else "Model comparison fits failed."
    )
    secondary = (
        "No predicted standard deviations available."
        if insufficient
        else "Run model-compare to inspect profile fit statuses."
    )
    error_ax.text(0.5, 0.5, message, ha="center", va="center", transform=error_ax.transAxes)
    std_ax.text(0.5, 0.5, secondary, ha="center", va="center", transform=std_ax.transAxes)
    if status_note:
        std_ax.text(
            0.5,
            0.25,
            status_note,
            ha="center",
            va="center",
            fontsize=9,
            color=NEUTRAL_COLOR,
            transform=std_ax.transAxes,
        )


def _render_fitted_model_comparison(fitted, status_note: str, error_ax, std_ax) -> None:
    x = list(range(len(fitted)))
    labels = fitted["model_profile"].astype(str).tolist()
    width = 0.35
    error_ax.bar(
        [position - width / 2 for position in x],
        fitted["rmse_model_space"],
        width=width,
        color=OBSERVED_COLOR,
        label="RMSE",
    )
    error_ax.bar(
        [position + width / 2 for position in x],
        fitted["mae_model_space"],
        width=width,
        color=TARGET_COLOR,
        label="MAE",
    )
    error_ax.set_xticks(x)
    error_ax.set_xticklabels(labels)
    add_legend(error_ax)
    std_ax.bar(x, fitted["mean_predicted_std"], color=MODEL_COLOR, label="Mean predicted std")
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
            color=NEUTRAL_COLOR,
            transform=error_ax.transAxes,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "black"},
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
