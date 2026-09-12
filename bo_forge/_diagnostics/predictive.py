"""Figures from completed held-out results; this module never fits a model."""

from __future__ import annotations

import numpy as np
from matplotlib.ticker import MaxNLocator

from bo_forge.plot_style import (
    MODEL_COLOR,
    NEUTRAL_COLOR,
    finalise_axes,
    new_subplots,
    scoped_plot_style,
    set_axis_labels,
    set_title,
)


def _panels(result):
    profiles = result.summary.model_profile.tolist()
    fig, axes = new_subplots(
        1,
        len(profiles),
        figsize=(7 * len(profiles), 6),
        squeeze=False,
        constrained_layout=True,
    )
    return fig, axes[0], profiles


def _profile_data(result, profile, ax):
    rows = result.predictions.loc[
        result.predictions.model_profile.eq(profile) & result.predictions.fit_status.eq("complete")
    ]
    summary = result.summary.loc[result.summary.model_profile.eq(profile)].iloc[0]
    if summary.fit_status != "complete":
        note = "Incomplete evaluation; aggregate metrics withheld"
    else:
        note = f"95% interval coverage: {summary.interval_coverage:.1%} (held-out)"
    ax.text(
        0.02,
        0.98,
        note,
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        fontweight="bold",
        wrap=True,
    )
    if rows.empty:
        ax.text(
            0.5,
            0.5,
            "No valid held-out predictions",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
    return rows


@scoped_plot_style
def plot_predictions(result, *, save_path=None):
    """Display original-unit held-out predictions, separated by requested profile."""
    fig, axes, profiles = _panels(result)
    objective = result.metadata["objective_name"]
    for ax, profile in zip(axes, profiles, strict=True):
        rows = _profile_data(result, profile, ax)
        if not rows.empty:
            observed = rows.observed.to_numpy(dtype=float)
            mean = rows.predicted_mean.to_numpy(dtype=float)
            lo, hi = min(observed.min(), mean.min()), max(observed.max(), mean.max())
            if lo == hi:
                lo, hi = lo - 0.5, hi + 0.5
            ax.plot([lo, hi], [lo, hi], color=NEUTRAL_COLOR, linestyle="--")
            ax.scatter(observed, mean, color=MODEL_COLOR, edgecolors="black", linewidths=0.8)
        set_title(ax, f"{profile}: held-out predictions")
        set_axis_labels(ax, f"Observed {objective}", f"Held-out predicted {objective}")
    return finalise_axes(fig, axes, save_path=save_path)


@scoped_plot_style
def plot_residuals(result, *, save_path=None):
    """Show residuals standardized by observation-inclusive predictive uncertainty."""
    fig, axes, profiles = _panels(result)
    for ax, profile in zip(axes, profiles, strict=True):
        rows = _profile_data(result, profile, ax)
        if not rows.empty:
            ax.scatter(
                np.arange(1, len(rows) + 1),
                rows.standardized_residual,
                color=MODEL_COLOR,
                edgecolors="black",
                linewidths=0.8,
            )
        ax.axhline(0, color=NEUTRAL_COLOR)
        for bound in (-1.959963984540054, 1.959963984540054):
            ax.axhline(bound, color=NEUTRAL_COLOR, linestyle="--")
        ax.margins(y=0.18)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        set_title(ax, f"{profile}: held-out residuals")
        set_axis_labels(ax, "Held-out row", "Standardized residual")
    return finalise_axes(fig, axes, save_path=save_path)
