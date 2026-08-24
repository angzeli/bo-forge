"""Internal candidate suggestion generation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from bo_forge._optimization.common import _suggest_structured_stage
from bo_forge._optimization.initial_design import _suggest_initial_design
from bo_forge._optimization.multi_objective import (
    _suggest_cost_aware_multi_objective_model_based,
    _suggest_multi_objective_model_based,
    _suggest_qlog_nehvi_model_based,
)
from bo_forge._optimization.multifidelity import _suggest_multi_fidelity_model_based
from bo_forge._optimization.replicates import (
    _fill_replicate_batch_with_exploration,
    _suggest_uncertain_best_replicate,
)
from bo_forge._optimization.single_objective import (
    _suggest_cost_aware_model_based,
    _suggest_model_based,
    _suggest_qlog_nei_model_based,
)
from bo_forge.config import CampaignConfig
from bo_forge.contextual import (
    resolve_context_values,
)
from bo_forge.errors import SuggestionError
from bo_forge.replicates import modeling_observed_data
from bo_forge.validation import (
    get_observed_data,
    has_blocking_qlog_nehvi_review_suggestions,
    has_blocking_qlog_nei_review_suggestions,
    has_pending_suggestions,
    qlog_nehvi_active_pending_suggestions,
    qlog_nei_active_pending_suggestions,
    validate_campaign_data,
)

MAX_CATEGORICAL_COMBINATIONS = 64
MAX_DECODE_RETRIES = 8
MAX_INITIAL_DESIGN_BATCHES = 1000
_GENERATION_FAILURE_HINT = (
    "The feasible design space may be exhausted, constraints may be too restrictive, "
    "or bo.min_normalized_distance may be too large."
)
SUGGESTION_QUALITY_COLUMNS = [
    "row_id",
    "is_feasible",
    "violated_constraints",
    "is_exact_duplicate",
    "duplicate_allowed_by_replicates",
    "nearest_existing_distance",
    "nearest_batch_distance",
    "passes_distance_threshold",
]

@dataclass(frozen=True)
class _SuggestionState:
    """Validated inputs shared by initial-design and model-based routing."""

    uses_qlog_nei: bool
    uses_qlog_nehvi: bool
    resolved_context: dict[str, object]
    batch_size: int
    observed_df: pd.DataFrame
    training_observed_df: pd.DataFrame
    active_pending_df: pd.DataFrame


def suggest_next(
    config: CampaignConfig,
    df: pd.DataFrame,
    batch_size: int | None = None,
    stage: str | None = None,
    context_values: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Suggest the next experiment or batch for a campaign."""
    validate_campaign_data(config, df)
    if config.is_structured_campaign:
        if context_values:
            raise SuggestionError("Context values are only valid for contextual campaigns.")
        return _suggest_structured_stage(
            config=config,
            df=df,
            batch_size=batch_size,
            stage=stage,
        )
    if stage is not None:
        raise SuggestionError("--stage is only valid for structured campaign configs.")
    state = _prepare_suggestion_state(config, df, batch_size, context_values)
    initial_suggestions = _initial_design_suggestions(config, df, state)
    if initial_suggestions is not None:
        return initial_suggestions
    return _dispatch_model_based_suggestions(config, df, state)


def _prepare_suggestion_state(
    config: CampaignConfig,
    df: pd.DataFrame,
    batch_size: int | None,
    context_values: dict[str, object] | None,
) -> _SuggestionState:
    uses_qlog_nei = config.bo.acquisition == "qlog_nei"
    uses_qlog_nehvi = config.bo.acquisition == "qlog_nehvi"
    if uses_qlog_nei:
        if has_blocking_qlog_nei_review_suggestions(df, config):
            raise SuggestionError(
                "Cannot generate qLogNEI suggestions while review_status='pending' "
                "rows await review; accept, reject, or defer them first."
            )
    elif uses_qlog_nehvi:
        if has_blocking_qlog_nehvi_review_suggestions(df, config):
            raise SuggestionError(
                "Cannot generate qLogNEHVI suggestions while review_status='pending' "
                "rows await review; accept, reject, or defer them first. Accepted "
                "suggestions are allowed as X_pending."
            )
    elif has_pending_suggestions(df, config):
        raise SuggestionError(
            "Cannot generate new suggestions while unresolved status='suggested' rows exist."
        )
    resolved_context = resolve_context_values(config, context_values)

    requested_batch_size = batch_size if batch_size is not None else config.bo.batch_size
    if requested_batch_size < 1:
        raise SuggestionError(f"batch_size must be >= 1: value={requested_batch_size}.")
    if config.fidelity is not None and requested_batch_size > 4:
        raise SuggestionError(
            "qMFKG supports batch_size from 1 through 4: "
            f"requested={requested_batch_size}."
        )

    observed_df = get_observed_data(config, df)
    training_observed_df = modeling_observed_data(config, observed_df)
    active_pending_df = _active_pending_suggestions(
        config,
        df,
        uses_qlog_nei=uses_qlog_nei,
        uses_qlog_nehvi=uses_qlog_nehvi,
    )
    return _SuggestionState(
        uses_qlog_nei=uses_qlog_nei,
        uses_qlog_nehvi=uses_qlog_nehvi,
        resolved_context=resolved_context,
        batch_size=requested_batch_size,
        observed_df=observed_df,
        training_observed_df=training_observed_df,
        active_pending_df=active_pending_df,
    )


def _active_pending_suggestions(
    config: CampaignConfig,
    df: pd.DataFrame,
    *,
    uses_qlog_nei: bool,
    uses_qlog_nehvi: bool,
) -> pd.DataFrame:
    if uses_qlog_nei:
        return qlog_nei_active_pending_suggestions(df, config)
    if uses_qlog_nehvi:
        return qlog_nehvi_active_pending_suggestions(df, config)
    return df.iloc[0:0].copy()


def _initial_design_suggestions(
    config: CampaignConfig,
    df: pd.DataFrame,
    state: _SuggestionState,
) -> pd.DataFrame | None:
    pending_initial_count = (
        int(state.active_pending_df["source"].isin({"sobol", "random"}).sum())
        if (state.uses_qlog_nei or state.uses_qlog_nehvi)
        and not state.active_pending_df.empty
        else 0
    )
    remaining_initial = config.bo.initial_design_size - len(state.training_observed_df)
    if state.uses_qlog_nei or state.uses_qlog_nehvi:
        remaining_initial -= pending_initial_count
    if remaining_initial > 0:
        return _suggest_initial_design(
            config=config,
            df=df,
            count=min(state.batch_size, remaining_initial),
            context_values=state.resolved_context,
        )
    if state.uses_qlog_nei and len(state.training_observed_df) < config.bo.initial_design_size:
        raise SuggestionError(
            "qLogNEI requires observed initial-design rows before model-based "
            "suggestions; observe accepted pending initial suggestions first."
        )
    if state.uses_qlog_nehvi and len(state.training_observed_df) < config.bo.initial_design_size:
        raise SuggestionError(
            "qLogNEHVI requires observed initial-design rows before model-based "
            "suggestions; observe accepted pending initial suggestions first."
        )
    return None


def _dispatch_model_based_suggestions(
    config: CampaignConfig,
    df: pd.DataFrame,
    state: _SuggestionState,
) -> pd.DataFrame:

    if config.fidelity is not None:
        return _suggest_multi_fidelity_model_based(
            config=config,
            df=df,
            observed_df=state.observed_df,
            batch_size=state.batch_size,
        )

    if config.is_multi_objective:
        return _dispatch_multi_objective_suggestions(config, df, state)

    replicate_suggestions = _dispatch_replicate_suggestions(config, df, state)
    if replicate_suggestions is not None:
        return replicate_suggestions

    if config.cost is not None:
        return _suggest_cost_aware_model_based(
            config=config,
            df=df,
            observed_df=state.observed_df,
            batch_size=state.batch_size,
            context_values=state.resolved_context,
        )

    if state.uses_qlog_nei:
        return _suggest_qlog_nei_model_based(
            config=config,
            df=df,
            observed_df=state.observed_df,
            active_pending_df=state.active_pending_df,
            batch_size=state.batch_size,
        )

    return _suggest_model_based(
        config=config,
        df=df,
        observed_df=state.observed_df,
        batch_size=state.batch_size,
        context_values=state.resolved_context,
    )


def _dispatch_multi_objective_suggestions(
    config: CampaignConfig,
    df: pd.DataFrame,
    state: _SuggestionState,
) -> pd.DataFrame:
    if state.uses_qlog_nehvi:
        return _suggest_qlog_nehvi_model_based(
            config=config,
            df=df,
            observed_df=state.observed_df,
            active_pending_df=state.active_pending_df,
            batch_size=state.batch_size,
        )
    if config.cost is not None:
        return _suggest_cost_aware_multi_objective_model_based(
            config=config,
            df=df,
            observed_df=state.observed_df,
            batch_size=state.batch_size,
        )
    return _suggest_multi_objective_model_based(
        config=config,
        df=df,
        observed_df=state.observed_df,
        batch_size=state.batch_size,
    )


def _dispatch_replicate_suggestions(
    config: CampaignConfig,
    df: pd.DataFrame,
    state: _SuggestionState,
) -> pd.DataFrame | None:
    if not (
        config.replicates.enabled
        and config.replicates.suggestion_policy == "uncertain_best"
    ):
        return None
    repeat_suggestions = _suggest_uncertain_best_replicate(
        config=config,
        df=df,
        observed_df=state.observed_df,
        batch_size=state.batch_size,
        context_values=state.resolved_context,
    )
    if repeat_suggestions is None or len(repeat_suggestions) >= state.batch_size:
        return repeat_suggestions
    return _fill_replicate_batch_with_exploration(
        config=config,
        df=df,
        observed_df=state.observed_df,
        repeat_suggestions=repeat_suggestions,
        batch_size=state.batch_size,
        context_values=state.resolved_context,
    )
