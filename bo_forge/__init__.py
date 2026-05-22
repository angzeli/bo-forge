"""BO Forge v0.4.2."""

__version__ = "0.4.2"

from bo_forge.config import (
    BOConfig,
    CampaignConfig,
    ConstraintConfig,
    ObjectiveConfig,
    VariableConfig,
)
from bo_forge.errors import (
    BOForgeError,
    ConfigError,
    LogValidationError,
    LogWriteError,
    SuggestionError,
)
from bo_forge.logs import append_suggestions, load_campaign_log, mark_observed
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
    "LogValidationError",
    "LogWriteError",
    "ObjectiveConfig",
    "SuggestionError",
    "VariableConfig",
    "__version__",
    "append_suggestions",
    "get_observed_data",
    "load_campaign_log",
    "mark_observed",
    "suggest_next",
    "suggestion_quality_summary",
    "validate_campaign_data",
]
