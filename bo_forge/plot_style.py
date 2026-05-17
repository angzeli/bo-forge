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


def style_axes(ax: Any, *, tick_label_size: int = TICK_LABEL_SIZE) -> Any:
    """Apply common axis styling used throughout BO Forge diagnostics."""
    configure_plot_style()
    ax.figure.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_WIDTH)
        spine.set_color("black")

    ax.tick_params(axis="both", labelsize=tick_label_size, colors="black")
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
    save_path: str | Path | None = None,
    dpi: int = 300,
    show: bool = False,
    tick_label_size: int = TICK_LABEL_SIZE,
) -> tuple[Any, Any]:
    """Apply final formatting, optionally save, and return the figure and axes."""
    style_axes(ax, tick_label_size=tick_label_size)
    ax.figure.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.figure.tight_layout()

    _save_figure(
        ax.figure,
        filename=filename,
        fig_folder=fig_folder,
        save_path=save_path,
        dpi=dpi,
    )

    if show:
        plt.show()

    return ax.figure, ax


def finalise_axes(
    fig: Any,
    axes: Any,
    *,
    filename: str | Path | None = None,
    fig_folder: str | Path = "figures",
    save_path: str | Path | None = None,
    dpi: int = 300,
    show: bool = False,
    tick_label_size: int = TICK_LABEL_SIZE,
) -> tuple[Any, Any]:
    """Apply final formatting to multiple axes, optionally save, and return them."""
    for ax in _iter_axes(axes):
        style_axes(ax, tick_label_size=tick_label_size)
    fig.patch.set_facecolor("white")
    if not (hasattr(fig, "get_constrained_layout") and fig.get_constrained_layout()):
        fig.tight_layout()
    _save_figure(
        fig,
        filename=filename,
        fig_folder=fig_folder,
        save_path=save_path,
        dpi=dpi,
    )
    if show:
        plt.show()
    return fig, axes


def _iter_axes(axes: Any) -> list[Any]:
    if isinstance(axes, (list, tuple)):
        result = []
        for item in axes:
            result.extend(_iter_axes(item))
        return result
    if hasattr(axes, "flat"):
        return list(axes.flat)
    return [axes]


def _save_figure(
    fig: Any,
    *,
    filename: str | Path | None,
    fig_folder: str | Path,
    save_path: str | Path | None,
    dpi: int,
) -> Path | None:
    if filename is not None and save_path is not None:
        raise ValueError("Pass either filename or save_path, not both.")
    if save_path is None and filename is None:
        return None

    path = Path(save_path) if save_path is not None else Path(fig_folder) / Path(filename)
    os.makedirs(path.parent, exist_ok=True)
    fig.savefig(
        path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
        transparent=False,
    )
    return path
