"""Compatibility facade for campaign diagnostic plots."""

from __future__ import annotations

import sys
from types import ModuleType

from bo_forge._diagnostics import contextual, fidelity, model, multi_objective, standard, structured
from bo_forge._diagnostics.contextual import plot_context_diagnostics, plot_qlog_nei_diagnostics
from bo_forge._diagnostics.fidelity import plot_fidelity_diagnostics, plot_fidelity_progress
from bo_forge._diagnostics.model import plot_model_comparison, plot_model_diagnostics
from bo_forge._diagnostics.multi_objective import (
    plot_hypervolume,
    plot_pareto,
    plot_pareto_parallel,
)
from bo_forge._diagnostics.standard import (
    plot_cost_progress,
    plot_diagnostics,
    plot_progress,
    plot_replicates,
)
from bo_forge._diagnostics.structured import plot_stage_diagnostics

__all__ = [
    "plot_progress", "plot_diagnostics", "plot_cost_progress", "plot_replicates",
    "plot_fidelity_diagnostics", "plot_fidelity_progress", "plot_context_diagnostics",
    "plot_qlog_nei_diagnostics", "plot_model_diagnostics", "plot_model_comparison",
    "plot_stage_diagnostics", "plot_pareto", "plot_pareto_parallel", "plot_hypervolume",
]

_COMPATIBILITY_MODULES = (contextual, fidelity, model, multi_objective, standard, structured)
_PATCH_TARGETS = {"model_profile_comparison": (model,)}

class _DiagnosticsFacade(ModuleType):
    """Expose legacy implementation helpers while plots live in focused modules."""

    def __getattr__(self, name: str) -> object:
        for module in _COMPATIBILITY_MODULES:
            if name in vars(module):
                return vars(module)[name]
        raise AttributeError(f"module {self.__name__!r} has no attribute {name!r}")

    def __setattr__(self, name: str, value: object) -> None:
        super().__setattr__(name, value)
        for module in _PATCH_TARGETS.get(name, ()):
            setattr(module, name, value)

sys.modules[__name__].__class__ = _DiagnosticsFacade
