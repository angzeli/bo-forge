"""Compatibility facade for candidate suggestion generation."""

from __future__ import annotations

import sys
from types import ModuleType

from bo_forge._optimization import (
    common,
    initial_design,
    multi_objective,
    multifidelity,
    replicates,
    router,
    single_objective,
)
from bo_forge._optimization.common import (
    MAX_DECODE_RETRIES,
    SUGGESTION_QUALITY_COLUMNS,
    suggestion_quality_summary,
)
from bo_forge._optimization.router import (
    suggest_next,
)

__all__ = [
    "MAX_DECODE_RETRIES",
    "SUGGESTION_QUALITY_COLUMNS",
    "suggest_next",
    "suggestion_quality_summary",
]

_PATCH_TARGETS = {
    "fit_gp_model": (single_objective, replicates, multi_objective, multifidelity),
    "fit_multi_fidelity_gp_model": (multifidelity,),
    "optimize_log_ei": (single_objective,),
    "optimize_qlog_nei": (single_objective,),
    "optimize_qlog_nehvi": (multi_objective,),
    "optimize_qmf_kg": (multifidelity,),
    "optimize_posterior_mean_at_target_fidelity": (multifidelity,),
    "time": (multifidelity,),
    "MAX_INITIAL_DESIGN_BATCHES": (initial_design,),
    "_suggest_model_based": (router, replicates),
    "_suggest_cost_aware_model_based": (router, replicates),
    "_cost_aware_candidate_pool": (single_objective,),
    "_score_cost_aware_candidate": (single_objective,),
}
_COMPATIBILITY_MODULES = (
    common,
    initial_design,
    multifidelity,
    multi_objective,
    replicates,
    router,
    single_objective,
)

class _SuggestionsFacade(ModuleType):
    """Forward legacy test hooks to the internal function owner."""

    def __getattr__(self, name: str) -> object:
        for module in _COMPATIBILITY_MODULES:
            if name in vars(module):
                return vars(module)[name]
        raise AttributeError(f"module {self.__name__!r} has no attribute {name!r}")

    def __setattr__(self, name: str, value: object) -> None:
        super().__setattr__(name, value)
        for module in _PATCH_TARGETS.get(name, ()):
            setattr(module, name, value)

sys.modules[__name__].__class__ = _SuggestionsFacade
