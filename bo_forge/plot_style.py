"""Scoped scientific plotting style shared by BO Forge diagnostics."""

from __future__ import annotations

import os
from collections.abc import Callable
from contextlib import AbstractContextManager
from functools import wraps
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap

FIGSIZE = (8, 6)
AXIS_LABEL_SIZE = 22
TICK_LABEL_SIZE = 14
TITLE_LABEL_SIZE = 18
SPINE_WIDTH = 1.8
LEGEND_SIZE = 10
COLORBAR_LABEL_SIZE = 20
COLORBAR_TICK_SIZE = 12
_P = ParamSpec("_P")
_R = TypeVar("_R")

OBSERVED_COLOR = "#0072B2"
HIGHLIGHT_COLOR = "#D55E00"
MODEL_COLOR = "#7A5195"
TARGET_COLOR = "#009E73"
WARNING_COLOR = "#C23B70"
ADDITIONAL_COLOR = "#7A8F00"
NEUTRAL_COLOR = "#4D4D4D"

SEMANTIC_COLORS = {
    "observed": OBSERVED_COLOR,
    "highlight": HIGHLIGHT_COLOR,
    "model": MODEL_COLOR,
    "target": TARGET_COLOR,
    "warning": WARNING_COLOR,
    "additional": ADDITIONAL_COLOR,
    "neutral": NEUTRAL_COLOR,
}
ORDERED_CMAP = LinearSegmentedColormap.from_list(
    "bo_forge_ordered",
    ["#D6E7F2", OBSERVED_COLOR, "#003F66"],
)

REPORT_READY_RCPARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "stixsans",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.prop_cycle": cycler(
        color=[
            OBSERVED_COLOR,
            HIGHLIGHT_COLOR,
            MODEL_COLOR,
            TARGET_COLOR,
            WARNING_COLOR,
            ADDITIONAL_COLOR,
            NEUTRAL_COLOR,
        ]
    ),
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",
    "savefig.transparent": False,
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "axes.linewidth": SPINE_WIDTH,
    "axes.grid": False,
    "axes.spines.bottom": True,
    "axes.spines.left": True,
    "axes.spines.right": True,
    "axes.spines.top": True,
    "axes.titlesize": TITLE_LABEL_SIZE,
    "axes.titleweight": "bold",
    "axes.labelsize": AXIS_LABEL_SIZE,
    "axes.labelweight": "bold",
    "xtick.color": "black",
    "ytick.color": "black",
    "xtick.labelsize": TICK_LABEL_SIZE,
    "ytick.labelsize": TICK_LABEL_SIZE,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.major.width": SPINE_WIDTH,
    "ytick.major.width": SPINE_WIDTH,
    "xtick.top": False,
    "ytick.right": False,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "lines.linewidth": 2.0,
    "lines.markersize": 5.5,
    "lines.markeredgecolor": "black",
    "lines.markeredgewidth": 0.8,
    "legend.fontsize": LEGEND_SIZE,
    "legend.frameon": True,
    "legend.framealpha": 1.0,
    "legend.facecolor": "white",
    "legend.edgecolor": "black",
    "text.color": "black",
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
}


def configure_plot_style() -> AbstractContextManager[None]:
    """Return the scoped BO Forge plotting context without changing global rcParams."""
    return plt.rc_context(REPORT_READY_RCPARAMS)


def scoped_plot_style(function: Callable[_P, _R]) -> Callable[_P, _R]:
    """Run one complete plot routine inside the scoped scientific style."""
    @wraps(function)
    def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        with configure_plot_style():
            return function(*args, **kwargs)

    return wrapped


def new_figure(figsize: tuple[float, float] = FIGSIZE) -> tuple[Any, Any]:
    """Create a white scientific figure under the scoped plotting contract."""
    with configure_plot_style():
        fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    style_axes(ax)
    return fig, ax


def new_subplots(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
    """Create a scoped multi-panel figure and style every returned axis."""
    kwargs.setdefault("facecolor", "white")
    with configure_plot_style():
        fig, axes = plt.subplots(*args, **kwargs)
    for ax in _iter_axes(axes):
        style_axes(ax)
    return fig, axes


def style_axes(ax: Any, *, tick_label_size: int = TICK_LABEL_SIZE) -> Any:
    """Apply four-spine, inward-tick scientific axis styling."""
    ax.figure.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.grid(False)
    ax.minorticks_off()
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(SPINE_WIDTH)
        spine.set_color("black")
    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        length=4,
        width=SPINE_WIDTH,
        labelsize=tick_label_size,
        colors="black",
        top=False,
        right=False,
    )
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
        label.set_color("black")
    return ax


def style_ax(ax: Any) -> Any:
    """Backward-compatible alias for :func:`style_axes`."""
    return style_axes(ax)


def set_axis_labels(ax: Any, xlabel: str, ylabel: str) -> Any:
    """Set bold 22-point axis labels."""
    ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_SIZE, fontweight="bold", color="black")
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_SIZE, fontweight="bold", color="black")
    return ax


def set_title(ax: Any, title: str) -> Any:
    """Set a compact diagnostic title."""
    ax.set_title(title, fontsize=TITLE_LABEL_SIZE, fontweight="bold", color="black")
    return ax


def set_bold_labels(
    ax: Any,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    title_size: int = TITLE_LABEL_SIZE,
    label_size: int = AXIS_LABEL_SIZE,
) -> Any:
    """Set title and axis labels using the scientific plotting convention."""
    ax.set_title(title, fontsize=title_size, fontweight="bold", color="black")
    ax.set_xlabel(xlabel, fontsize=label_size, fontweight="bold", color="black")
    ax.set_ylabel(ylabel, fontsize=label_size, fontweight="bold", color="black")
    return ax


def add_legend(ax: Any, *, loc: str = "best") -> Any:
    """Add an opaque in-axes legend when labelled artists exist."""
    return bold_legend(ax, loc=loc, size=LEGEND_SIZE)


def bold_legend(ax: Any, *, loc: str | None = None, size: int = LEGEND_SIZE) -> Any:
    """Add a bold, opaque legend when labelled artists exist."""
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        legend = ax.legend(
            handles,
            labels,
            frameon=True,
            framealpha=1.0,
            prop={"size": size, "weight": "bold"},
            loc=loc,
            facecolor="white",
            edgecolor="black",
        )
        for text in legend.get_texts():
            text.set_color("black")
    return ax


def style_colorbar(colorbar: Any, label: str) -> Any:
    """Style a diagnostic colorbar without changing process-global settings."""
    colorbar.set_label(label, fontsize=COLORBAR_LABEL_SIZE, fontweight="bold", color="black")
    colorbar.ax.tick_params(labelsize=COLORBAR_TICK_SIZE, colors="black")
    colorbar.outline.set_edgecolor("black")
    colorbar.outline.set_linewidth(SPINE_WIDTH)
    for label_artist in colorbar.ax.get_yticklabels():
        label_artist.set_fontweight("bold")
        label_artist.set_color("black")
    return colorbar


def finalise_figure(
    ax: Any,
    *,
    filename: str | Path | None = None,
    fig_folder: str | Path = "figures",
    save_path: str | Path | None = None,
    dpi: int = 600,
    show: bool = False,
    tick_label_size: int = TICK_LABEL_SIZE,
) -> tuple[Any, Any]:
    """Finalize one axis and write at most the explicitly requested output file."""
    style_axes(ax, tick_label_size=tick_label_size)
    ax.figure.tight_layout()
    _save_figure(ax.figure, filename=filename, fig_folder=fig_folder, save_path=save_path, dpi=dpi)
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
    dpi: int = 600,
    show: bool = False,
    tick_label_size: int = TICK_LABEL_SIZE,
) -> tuple[Any, Any]:
    """Finalize multiple axes and write at most one explicit output file."""
    for ax in _iter_axes(axes):
        style_axes(ax, tick_label_size=tick_label_size)
    fig.patch.set_facecolor("white")
    if not (hasattr(fig, "get_constrained_layout") and fig.get_constrained_layout()):
        fig.tight_layout()
    _save_figure(fig, filename=filename, fig_folder=fig_folder, save_path=save_path, dpi=dpi)
    if show:
        plt.show()
    return fig, axes


def _iter_axes(axes: Any) -> list[Any]:
    if isinstance(axes, (list, tuple)):
        return [axis for item in axes for axis in _iter_axes(item)]
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
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white", transparent=False)
    return path
