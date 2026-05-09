"""Shared report-ready plotting style for BO Forge diagnostics."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

FIGSIZE = (8, 6)
AXIS_LABEL_SIZE = 22
TICK_LABEL_SIZE = 14
TITLE_LABEL_SIZE = 18
SPINE_WIDTH = 1.8
LEGEND_SIZE = 10
COLORBAR_LABEL_SIZE = 20
COLORBAR_TICK_SIZE = 12

REPORT_READY_RCPARAMS = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.transparent": False,
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "text.color": "black",
    "legend.facecolor": "white",
    "legend.edgecolor": "black",
}


def configure_plot_style() -> None:
    """Force report-ready black-on-white figures, independent of notebook theme."""
    plt.rcParams.update(REPORT_READY_RCPARAMS)


def new_figure(figsize: tuple[float, float] = FIGSIZE) -> tuple[Any, Any]:
    """Create a new white-background figure and axes."""
    configure_plot_style()
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.set_facecolor("white")
    return fig, ax


def style_axes(ax: Any) -> Any:
    """Apply common axis styling used throughout BO Forge diagnostics."""
    configure_plot_style()
    ax.figure.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_WIDTH)
        spine.set_color("black")

    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE, colors="black")
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")
    ax.title.set_color("black")

    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontweight("bold")
        tick_label.set_color("black")

    return ax


def style_ax(ax: Any) -> Any:
    """Backward-compatible alias for style_axes()."""
    return style_axes(ax)


def set_axis_labels(ax: Any, xlabel: str, ylabel: str) -> Any:
    """Apply common x/y label formatting."""
    ax.set_xlabel(
        xlabel,
        fontsize=AXIS_LABEL_SIZE,
        fontweight="bold",
        color="black",
    )
    ax.set_ylabel(
        ylabel,
        fontsize=AXIS_LABEL_SIZE,
        fontweight="bold",
        color="black",
    )
    return ax


def set_title(ax: Any, title: str) -> Any:
    """Apply common title formatting."""
    ax.set_title(title, fontsize=TITLE_LABEL_SIZE, fontweight="bold", color="black")
    return ax


def set_bold_labels(
    ax: Any,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    title_size: int = 18,
    label_size: int = 18,
) -> Any:
    """Set bold title and axis labels using the tutorial plotting convention."""
    ax.set_title(title, fontsize=title_size, fontweight="bold", color="black")
    ax.set_xlabel(xlabel, fontsize=label_size, fontweight="bold", color="black")
    ax.set_ylabel(ylabel, fontsize=label_size, fontweight="bold", color="black")
    return ax


def add_legend(ax: Any, *, loc: str = "best") -> Any:
    """Apply common legend formatting when labelled artists exist."""
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        legend = ax.legend(
            handles,
            labels,
            frameon=True,
            prop={"size": LEGEND_SIZE, "weight": "bold"},
            loc=loc,
            facecolor="white",
            edgecolor="black",
        )
        for text in legend.get_texts():
            text.set_color("black")
    return ax


def bold_legend(ax: Any, *, loc: str | None = None, size: int = 10) -> Any:
    """Backward-compatible legend helper."""
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        legend = ax.legend(
            handles,
            labels,
            frameon=True,
            prop={"size": size, "weight": "bold"},
            loc=loc,
            facecolor="white",
            edgecolor="black",
        )
        for text in legend.get_texts():
            text.set_color("black")
    return ax


def style_colorbar(colorbar: Any, label: str) -> Any:
    """Apply common formatting to a diagnostic colorbar."""
    colorbar.set_label(
        label,
        fontsize=COLORBAR_LABEL_SIZE,
        fontweight="bold",
        color="black",
    )
    colorbar.ax.tick_params(labelsize=COLORBAR_TICK_SIZE, colors="black")
    colorbar.outline.set_edgecolor("black")
    colorbar.outline.set_linewidth(SPINE_WIDTH)
    for tick_label in colorbar.ax.get_yticklabels():
        tick_label.set_fontweight("bold")
        tick_label.set_color("black")
    return colorbar


def finalise_figure(
    ax: Any,
    *,
    filename: str | Path | None = None,
    fig_folder: str | Path = "figures",
    dpi: int = 300,
    show: bool = False,
) -> tuple[Any, Any]:
    """Apply final formatting, optionally save, and return the figure and axes."""
    style_axes(ax)
    ax.figure.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.figure.tight_layout()

    if filename is not None:
        save_path = Path(filename)
        if not save_path.parent or str(save_path.parent) == ".":
            save_path = Path(fig_folder) / save_path
        os.makedirs(save_path.parent, exist_ok=True)
        ax.figure.savefig(
            save_path,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
            transparent=False,
        )

    if show:
        plt.show()

    return ax.figure, ax
