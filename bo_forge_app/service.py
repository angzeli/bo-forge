"""Compatibility shim for the v3 application service."""

from bo_forge.application import (
    AppendResult,
    CampaignAppService,
    CampaignViewData,
    MutationResult,
    PlotResult,
    StagedSuggestionResult,
    ValidationResult,
)

__all__ = [
    "AppendResult", "CampaignAppService", "CampaignViewData", "MutationResult",
    "PlotResult", "StagedSuggestionResult", "ValidationResult",
]
