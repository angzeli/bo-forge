"""BO Forge v1.3.1."""

__version__ = "1.3.1"

from bo_forge.config import (
    BOConfig,
    CampaignConfig,
    ConstraintConfig,
    CostConfig,
    ObjectiveConfig,
    ReplicateConfig,
    ReviewConfig,
    StageConfig,
    VariableConfig,
    active_variables_for_stage,
    configured_stage_names,
    is_structured_campaign,
)
from bo_forge.costs import evaluate_cost
from bo_forge.errors import (
    BOForgeError,
    ConfigError,
    LogValidationError,
    LogWriteError,
    SuggestionError,
)
from bo_forge.logs import append_suggestions, load_campaign_log, mark_observed, review_suggestion
from bo_forge.multi_objective import (
    hypervolume,
    hypervolume_progress,
    pareto_front,
    pareto_summary,
)
from bo_forge.replicates import (
    aggregate_observed_replicates,
    best_replicate_group,
    replicate_summary,
)
from bo_forge.session import CampaignSession
from bo_forge.suggestions import suggest_next, suggestion_quality_summary
from bo_forge.validation import get_observed_data, validate_campaign_data

__all__ = [
    "BOConfig",
    "BOForgeError",
    "CampaignConfig",
    "CampaignSession",
    "ConfigError",
    "ConstraintConfig",
    "CostConfig",
    "LogValidationError",
    "LogWriteError",
    "ObjectiveConfig",
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
    "evaluate_cost",
    "get_observed_data",
    "hypervolume",
    "hypervolume_progress",
    "is_structured_campaign",
    "load_campaign_log",
    "mark_observed",
    "pareto_front",
    "pareto_summary",
    "review_suggestion",
    "replicate_summary",
    "suggest_next",
    "suggestion_quality_summary",
    "validate_campaign_data",
]
