"""Internal plot routing shared by CLI and app adapters."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class _PlotRoute:
    """Map one internal plot kind to its label and session method."""

    label: str
    session_method: str


_PLOT_ROUTES = {
    "progress": _PlotRoute("Progress", "plot_progress"),
    "diagnostics": _PlotRoute("Diagnostics", "plot_diagnostics"),
    "pareto": _PlotRoute("Pareto", "plot_pareto"),
    "pareto_parallel": _PlotRoute("Pareto Parallel", "plot_pareto_parallel"),
    "hypervolume": _PlotRoute("Hypervolume", "plot_hypervolume"),
    "stage_diagnostics": _PlotRoute("Stage Diagnostics", "plot_stage_diagnostics"),
    "fidelity_diagnostics": _PlotRoute("Fidelity Diagnostics", "plot_fidelity_diagnostics"),
    "context_diagnostics": _PlotRoute("Context Diagnostics", "plot_context_diagnostics"),
    "qlog_nei_diagnostics": _PlotRoute("qLogNEI Diagnostics", "plot_qlog_nei_diagnostics"),
    "model_diagnostics": _PlotRoute("Model Diagnostics", "plot_model_diagnostics"),
    "model_comparison": _PlotRoute("Model Comparison", "plot_model_comparison"),
    "cost_progress": _PlotRoute("Cost Progress", "plot_cost_progress"),
    "replicates": _PlotRoute("Replicates", "plot_replicates"),
}


def _canonical_plot_kind(kind: str) -> str:
    """Normalize CLI hyphens to the app/service plot-kind convention."""
    return kind.replace("-", "_")
