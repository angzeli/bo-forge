"""BO Forge v3.1.1."""

from importlib import import_module

__version__ = "3.1.1"

from bo_forge.config import (
    BOConfig,
    CampaignConfig,
    ConstraintConfig,
    ContextConfig,
    CostConfig,
    FidelityConfig,
    ModelConfig,
    ObjectiveConfig,
    ReplicateConfig,
    ReviewConfig,
    StageConfig,
    VariableConfig,
    active_variables_for_stage,
    configured_stage_names,
    is_structured_campaign,
)
from bo_forge.errors import (
    BOForgeError,
    ConfigError,
    LogBusyError,
    LogConflictError,
    LogValidationError,
    LogWriteError,
    ProvenanceError,
    ProvenanceRecoveryRequired,
    SuggestionError,
)

__all__ = [
    "BOConfig",
    "BOForgeError",
    "CampaignConfig",
    "CampaignSession",
    "ConfigError",
    "ConstraintConfig",
    "ContextConfig",
    "CostConfig",
    "FidelityConfig",
    "LogBusyError",
    "LogConflictError",
    "LogValidationError",
    "LogWriteError",
    "ModelConfig",
    "ObjectiveConfig",
    "ProvenanceError",
    "ProvenanceRecoveryRequired",
    "ReplicateConfig",
    "ReviewConfig",
    "StageConfig",
    "SuggestionError",
    "VariableConfig",
    "__version__",
    "active_variables_for_stage",
    "append_suggestions",
    "aggregate_observed_replicates",
    "best_replicate_group",
    "configured_stage_names",
    "context_summary",
    "evaluate_cost",
    "fidelity_coverage",
    "fidelity_summary",
    "get_observed_data",
    "hypervolume",
    "hypervolume_progress",
    "is_structured_campaign",
    "load_campaign_log",
    "mark_observed",
    "model_summary",
    "model_profile_comparison",
    "pareto_front",
    "pareto_summary",
    "provenance_summary",
    "recover_provenance",
    "qlog_nei_summary",
    "review_suggestion",
    "replicate_summary",
    "suggest_next",
    "suggestion_quality_summary",
    "stage_summary",
    "validate_campaign_data",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "CampaignSession": ("bo_forge.session", "CampaignSession"),
    "aggregate_observed_replicates": (
        "bo_forge.replicates",
        "aggregate_observed_replicates",
    ),
    "append_suggestions": ("bo_forge.logs", "append_suggestions"),
    "best_replicate_group": ("bo_forge.replicates", "best_replicate_group"),
    "context_summary": ("bo_forge.contextual", "context_summary"),
    "evaluate_cost": ("bo_forge.costs", "evaluate_cost"),
    "fidelity_summary": ("bo_forge.multifidelity", "fidelity_summary"),
    "fidelity_coverage": ("bo_forge.multifidelity", "fidelity_coverage"),
    "get_observed_data": ("bo_forge.validation", "get_observed_data"),
    "hypervolume": ("bo_forge.multi_objective", "hypervolume"),
    "hypervolume_progress": ("bo_forge.multi_objective", "hypervolume_progress"),
    "load_campaign_log": ("bo_forge.logs", "load_campaign_log"),
    "mark_observed": ("bo_forge.logs", "mark_observed"),
    "model_profile_comparison": ("bo_forge.models", "model_profile_comparison"),
    "model_summary": ("bo_forge.models", "model_summary"),
    "pareto_front": ("bo_forge.multi_objective", "pareto_front"),
    "pareto_summary": ("bo_forge.multi_objective", "pareto_summary"),
    "provenance_summary": ("bo_forge.provenance", "provenance_summary"),
    "recover_provenance": ("bo_forge.provenance", "recover_provenance"),
    "qlog_nei_summary": ("bo_forge.noisy", "qlog_nei_summary"),
    "replicate_summary": ("bo_forge.replicates", "replicate_summary"),
    "review_suggestion": ("bo_forge.logs", "review_suggestion"),
    "stage_summary": ("bo_forge.structured", "stage_summary"),
    "suggest_next": ("bo_forge.suggestions", "suggest_next"),
    "suggestion_quality_summary": (
        "bo_forge.suggestions",
        "suggestion_quality_summary",
    ),
    "validate_campaign_data": ("bo_forge.validation", "validate_campaign_data"),
}


def __getattr__(name: str) -> object:
    """Resolve heavy public exports only when callers request them."""
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazy public exports in interactive module discovery."""
    return sorted(set(globals()) | set(__all__))
