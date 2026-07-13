"""Candidate suggestion generation."""

from __future__ import annotations

import math
import uuid
from dataclasses import replace

import pandas as pd
import torch
from botorch.acquisition import LogExpectedImprovement
from botorch.exceptions.errors import BotorchError
from torch.quasirandom import SobolEngine

from bo_forge.acquisition import (
    build_qlog_ehvi_acquisition,
    build_qlog_nehvi_acquisition,
    optimize_log_ei,
    optimize_posterior_mean_at_target_fidelity,
    optimize_qlog_ehvi,
    optimize_qlog_nehvi,
    optimize_qlog_nei,
    optimize_qmf_kg,
)
from bo_forge.config import CampaignConfig
from bo_forge.constraints import constraint_variable_names, constraint_violations_for_values
from bo_forge.contextual import (
    apply_context_to_candidate,
    contextual_categorical_combination_count,
    contextual_fixed_feature_assignments,
    normalize_context_value,
    resolve_context_values,
)
from bo_forge.costs import budget_remaining, evaluate_cost
from bo_forge.errors import SuggestionError
from bo_forge.models import (
    dataframe_to_tensors,
    dataframe_to_training_tensors,
    fit_gp_model,
    fit_multi_fidelity_gp_model,
)
from bo_forge.multi_objective import (
    objectives_from_model_space,
    reference_point_to_model_space,
)
from bo_forge.replicates import aggregate_observed_replicates, modeling_observed_data
from bo_forge.transforms import (
    categorical_combination_count,
    categorical_feature_assignments,
    dataframe_to_unit_cube,
    encoded_dimension,
    objective_from_model_space,
    unit_cube_to_design_values,
    unit_cube_to_user_values,
    values_to_unit_cube,
)
from bo_forge.validation import (
    canonical_columns,
    design_key_for_values,
    design_tuples,
    get_observed_data,
    has_blocking_qlog_nehvi_review_suggestions,
    has_blocking_qlog_nei_review_suggestions,
    has_pending_suggestions,
    next_iteration,
    qlog_nehvi_active_pending_suggestions,
    qlog_nei_active_pending_suggestions,
    validate_campaign_data,
)

MAX_CATEGORICAL_COMBINATIONS = 64
MAX_DECODE_RETRIES = 8
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


class _CandidateGenerationExhausted(SuggestionError):
    """Internal signal for expected budget or design-space exhaustion."""


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

    observed_df = get_observed_data(config, df)
    training_observed_df = modeling_observed_data(config, observed_df)
    if uses_qlog_nei:
        active_pending_df = qlog_nei_active_pending_suggestions(df, config)
    elif uses_qlog_nehvi:
        active_pending_df = qlog_nehvi_active_pending_suggestions(df, config)
    else:
        active_pending_df = df.iloc[0:0].copy()
    pending_initial_count = (
        int(active_pending_df["source"].isin({"sobol", "random"}).sum())
        if (uses_qlog_nei or uses_qlog_nehvi) and not active_pending_df.empty
        else 0
    )
    remaining_initial = config.bo.initial_design_size - len(training_observed_df)
    if uses_qlog_nei or uses_qlog_nehvi:
        remaining_initial -= pending_initial_count
    if remaining_initial > 0:
        return _suggest_initial_design(
            config=config,
            df=df,
            count=min(requested_batch_size, remaining_initial),
            context_values=resolved_context,
        )
    if uses_qlog_nei and len(training_observed_df) < config.bo.initial_design_size:
        raise SuggestionError(
            "qLogNEI requires observed initial-design rows before model-based "
            "suggestions; observe accepted pending initial suggestions first."
        )
    if uses_qlog_nehvi and len(training_observed_df) < config.bo.initial_design_size:
        raise SuggestionError(
            "qLogNEHVI requires observed initial-design rows before model-based "
            "suggestions; observe accepted pending initial suggestions first."
        )

    if config.fidelity is not None:
        return _suggest_multi_fidelity_model_based(
            config=config,
            df=df,
            observed_df=observed_df,
            batch_size=requested_batch_size,
        )

    if config.is_multi_objective:
        if uses_qlog_nehvi:
            return _suggest_qlog_nehvi_model_based(
                config=config,
                df=df,
                observed_df=observed_df,
                active_pending_df=active_pending_df,
                batch_size=requested_batch_size,
            )
        if config.cost is not None:
            return _suggest_cost_aware_multi_objective_model_based(
                config=config,
                df=df,
                observed_df=observed_df,
                batch_size=requested_batch_size,
            )
        return _suggest_multi_objective_model_based(
            config=config,
            df=df,
            observed_df=observed_df,
            batch_size=requested_batch_size,
        )

    if (
        config.replicates.enabled
        and config.replicates.suggestion_policy == "uncertain_best"
    ):
        repeat_suggestions = _suggest_uncertain_best_replicate(
            config=config,
            df=df,
            observed_df=observed_df,
            batch_size=requested_batch_size,
        )
        if repeat_suggestions is not None:
            if len(repeat_suggestions) >= requested_batch_size:
                return repeat_suggestions
            return _fill_replicate_batch_with_exploration(
                config=config,
                df=df,
                observed_df=observed_df,
                repeat_suggestions=repeat_suggestions,
                batch_size=requested_batch_size,
            )

    if config.cost is not None:
        return _suggest_cost_aware_model_based(
            config=config,
            df=df,
            observed_df=observed_df,
            batch_size=requested_batch_size,
            context_values=resolved_context,
        )

    if uses_qlog_nei:
        return _suggest_qlog_nei_model_based(
            config=config,
            df=df,
            observed_df=observed_df,
            active_pending_df=active_pending_df,
            batch_size=requested_batch_size,
        )

    return _suggest_model_based(
        config=config,
        df=df,
        observed_df=observed_df,
        batch_size=requested_batch_size,
        context_values=resolved_context,
    )


def _suggest_structured_stage(
    *,
    config: CampaignConfig,
    df: pd.DataFrame,
    batch_size: int | None,
    stage: str | None,
) -> pd.DataFrame:
    stage_name = _resolve_structured_stage(config, stage)
    if config.cost is not None:
        raise SuggestionError(
            "Structured campaign suggestions with cost are not supported in v1.4.0."
        )
    if has_pending_suggestions(df, config):
        raise SuggestionError(
            "Cannot generate new suggestions while unresolved status='suggested' rows exist."
        )

    stage_config = _stage_local_config(config, stage_name)
    stage_df = _stage_local_dataframe(config, df, stage_name, stage_config)
    local_suggestions = suggest_next(stage_config, stage_df, batch_size=batch_size)
    suggestions = _expand_stage_suggestions(
        config=config,
        stage_name=stage_name,
        local_suggestions=local_suggestions,
        iteration=next_iteration(df),
    )
    combined = pd.concat([df, suggestions], ignore_index=True)
    validate_campaign_data(config, combined)
    return suggestions


def _resolve_structured_stage(config: CampaignConfig, stage: str | None) -> str:
    if stage is None:
        if len(config.stages) == 1:
            stage = config.stages[0].name
        else:
            raise SuggestionError(
                "Structured campaign suggestions require an explicit stage. "
                f"Pass stage=... or CLI --stage with one of {config.stage_names}."
            )
    if not isinstance(stage, str) or not stage.strip() or stage.strip() != stage:
        raise SuggestionError(f"Invalid structured campaign stage: value={stage!r}.")
    if stage not in config.stage_names:
        raise SuggestionError(
            f"Unknown structured campaign stage '{stage}'. Expected one of {config.stage_names}."
        )
    active_variables = config.active_variable_names_for_stage(stage)
    if not active_variables:
        raise SuggestionError(f"Structured campaign stage '{stage}' has no active variables.")
    return stage


def _stage_local_config(config: CampaignConfig, stage_name: str) -> CampaignConfig:
    active_names = set(config.active_variable_names_for_stage(stage_name))
    active_variables = tuple(
        variable
        for variable in config.variables
        if variable.name in active_names
    )
    applicable_constraints = tuple(
        constraint
        for constraint in config.constraints
        if constraint_variable_names(constraint.expression).issubset(active_names)
    )
    return replace(
        config,
        variables=active_variables,
        constraints=applicable_constraints,
        stages=(),
    )


def _stage_local_dataframe(
    config: CampaignConfig,
    df: pd.DataFrame,
    stage_name: str,
    stage_config: CampaignConfig,
) -> pd.DataFrame:
    stage_rows = df.loc[df["stage"] == stage_name]
    columns = canonical_columns(stage_config)
    local = pd.DataFrame(columns=columns)
    for column in columns:
        if column in stage_rows.columns:
            local[column] = stage_rows[column].to_numpy()
    return local.loc[:, columns].copy()


def _expand_stage_suggestions(
    *,
    config: CampaignConfig,
    stage_name: str,
    local_suggestions: pd.DataFrame,
    iteration: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, local_row in local_suggestions.iterrows():
        row = _empty_row(config)
        for column in local_suggestions.columns:
            if column in row:
                row[column] = local_row[column]
        row["stage"] = stage_name
        row["iteration"] = iteration
        rows.append(row)
    return pd.DataFrame(rows, columns=canonical_columns(config))


def _suggest_initial_design(
    config: CampaignConfig,
    df: pd.DataFrame,
    count: int,
    context_values: dict[str, object] | None = None,
) -> pd.DataFrame:
    source = config.bo.initial_design_method
    candidates = _initial_user_candidates(
        config,
        df=df,
        count=count,
        method=source,
        context_values=context_values,
    )
    rows = []
    iteration = next_iteration(df)
    for candidate in candidates:
        row = _empty_row(config)
        row["row_id"] = uuid.uuid4().hex
        row["iteration"] = iteration
        row["status"] = "suggested"
        row["source"] = source
        _populate_replicate_fields(config, row)
        _populate_review_fields(config, row)
        for name, value in zip(config.variable_names, candidate, strict=True):
            row[name] = value
        _populate_cost_fields(config, row, candidate)
        rows.append(row)
    return pd.DataFrame(rows, columns=canonical_columns(config))


def _suggest_multi_fidelity_model_based(
    config: CampaignConfig,
    df: pd.DataFrame,
    observed_df: pd.DataFrame,
    batch_size: int,
) -> pd.DataFrame:
    if batch_size != 1:
        raise SuggestionError(
            "qMFKG model-based suggestions support batch_size=1 in v1.4.0: "
            f"requested={batch_size}."
        )
    try:
        torch.manual_seed(config.bo.random_seed)
        model = fit_multi_fidelity_gp_model(config, observed_df)
        fixed_features_list = categorical_feature_assignments(config)
        model_dim = encoded_dimension(config)
        current_value = optimize_posterior_mean_at_target_fidelity(
            config=config,
            model=model,
            model_dim=model_dim,
            fixed_features_list=fixed_features_list,
        )
        user_candidates: list[tuple[object, ...]] | None = None
        acquisition_value: torch.Tensor | None = None
        rejection_message = "no candidate was decoded"

        for attempt in range(MAX_DECODE_RETRIES):
            torch.manual_seed(config.bo.random_seed + attempt)
            x_unit_raw, acquisition_value, _ = optimize_qmf_kg(
                config=config,
                model=model,
                current_value=current_value,
                model_dim=model_dim,
                fixed_features_list=fixed_features_list,
            )
            decoded_candidates = unit_cube_to_user_values(config, x_unit_raw)
            rejection_message = _candidate_batch_rejection_message(
                config,
                df,
                decoded_candidates,
            )
            if rejection_message is None:
                user_candidates = decoded_candidates
                break

        if user_candidates is None or acquisition_value is None:
            raise _CandidateGenerationExhausted(
                "Could not generate a feasible, non-duplicate qMFKG suggestion after "
                f"{MAX_DECODE_RETRIES} retries. {_GENERATION_FAILURE_HINT} "
                f"Last rejection: {rejection_message}"
            )

        x_unit_repaired = values_to_unit_cube(config, user_candidates)
        with torch.no_grad():
            posterior = model.posterior(x_unit_repaired)
            mean_model = posterior.mean.squeeze(-1)
            std = posterior.variance.clamp_min(0.0).sqrt().squeeze(-1)
            mean_user = objective_from_model_space(config, mean_model)
    except SuggestionError:
        raise
    except (BotorchError, RuntimeError, ValueError) as exc:
        raise SuggestionError(f"Could not generate qMFKG suggestion: {exc}") from exc

    row = _empty_row(config)
    row["row_id"] = uuid.uuid4().hex
    row["iteration"] = next_iteration(df)
    row["status"] = "suggested"
    row["source"] = "qmf_kg"
    _populate_replicate_fields(config, row)
    _populate_review_fields(config, row)
    for name, value in zip(config.variable_names, user_candidates[0], strict=True):
        row[name] = value
    row["predicted_mean"] = float(mean_user.reshape(-1)[0])
    row["predicted_std"] = float(std.reshape(-1)[0])
    row["acquisition"] = float(acquisition_value.reshape(-1)[0])
    return pd.DataFrame([row], columns=canonical_columns(config))


def _suggest_multi_objective_model_based(
    config: CampaignConfig,
    df: pd.DataFrame,
    observed_df: pd.DataFrame,
    batch_size: int,
) -> pd.DataFrame:
    torch.manual_seed(config.bo.random_seed)
    combination_count = categorical_combination_count(config)
    if combination_count > MAX_CATEGORICAL_COMBINATIONS:
        raise SuggestionError(
            "Multi-objective mixed-variable suggestions support at most "
            f"{MAX_CATEGORICAL_COMBINATIONS} categorical combinations: "
            f"configured={combination_count}."
        )

    model = fit_gp_model(config, observed_df)
    _, train_y_model = dataframe_to_tensors(config, observed_df)
    fixed_features_list = categorical_feature_assignments(config)
    ref_point = reference_point_to_model_space(config)
    user_candidates: list[tuple[object, ...]] | None = None
    acquisition_value: torch.Tensor | None = None
    rejection_message = "no candidate was decoded"
    for attempt in range(MAX_DECODE_RETRIES):
        torch.manual_seed(config.bo.random_seed + attempt)
        x_unit_raw, acquisition_value, _ = optimize_qlog_ehvi(
            config=config,
            model=model,
            train_y_model=train_y_model,
            ref_point=ref_point,
            batch_size=batch_size,
            model_dim=encoded_dimension(config),
            fixed_features_list=fixed_features_list,
        )
        decoded_candidates = unit_cube_to_user_values(config, x_unit_raw)
        rejection_message = _candidate_batch_rejection_message(config, df, decoded_candidates)
        if rejection_message is None:
            user_candidates = decoded_candidates
            break

    if user_candidates is None or acquisition_value is None:
        user_candidates, acquisition_value = _fallback_qlog_ehvi_candidate_batch(
            config=config,
            df=df,
            model=model,
            train_y_model=train_y_model,
            ref_point=ref_point,
            batch_size=batch_size,
            fixed_features_list=fixed_features_list,
            rejection_message=rejection_message,
        )

    x_unit_repaired = values_to_unit_cube(config, user_candidates)
    with torch.no_grad():
        posterior = model.posterior(x_unit_repaired)
        mean_user = objectives_from_model_space(config, posterior.mean)
        std = posterior.variance.clamp_min(0.0).sqrt()

    rows = []
    iteration = next_iteration(df)
    acquisition_scalar = float(acquisition_value.reshape(-1)[0])
    for index in range(batch_size):
        row = _empty_row(config)
        row["row_id"] = uuid.uuid4().hex
        row["iteration"] = iteration
        row["status"] = "suggested"
        row["source"] = "qlog_ehvi"
        _populate_replicate_fields(config, row)
        _populate_review_fields(config, row)
        for name, value in zip(config.variable_names, user_candidates[index], strict=True):
            row[name] = value
        for objective_index, objective in enumerate(config.objectives):
            row[f"predicted_mean_{objective.name}"] = float(mean_user[index, objective_index])
            row[f"predicted_std_{objective.name}"] = float(std[index, objective_index])
        row["acquisition"] = acquisition_scalar
        rows.append(row)

    return pd.DataFrame(rows, columns=canonical_columns(config))


def _suggest_qlog_nehvi_model_based(
    config: CampaignConfig,
    df: pd.DataFrame,
    observed_df: pd.DataFrame,
    active_pending_df: pd.DataFrame,
    batch_size: int,
) -> pd.DataFrame:
    torch.manual_seed(config.bo.random_seed)
    combination_count = categorical_combination_count(config)
    if combination_count > MAX_CATEGORICAL_COMBINATIONS:
        raise SuggestionError(
            "qLogNEHVI mixed-variable suggestions support at most "
            f"{MAX_CATEGORICAL_COMBINATIONS} categorical combinations: "
            f"configured={combination_count}."
        )

    model = fit_gp_model(config, observed_df)
    x_baseline, _ = dataframe_to_tensors(config, observed_df)
    x_pending = _x_pending_for_qlog_nehvi(config, active_pending_df)
    fixed_features_list = categorical_feature_assignments(config)
    ref_point = reference_point_to_model_space(config)
    user_candidates: list[tuple[object, ...]] | None = None
    acquisition_value: torch.Tensor | None = None
    rejection_message = "no candidate was decoded"
    for attempt in range(MAX_DECODE_RETRIES):
        torch.manual_seed(config.bo.random_seed + attempt)
        x_unit_raw, acquisition_value, _ = optimize_qlog_nehvi(
            config=config,
            model=model,
            x_baseline=x_baseline,
            ref_point=ref_point,
            batch_size=batch_size,
            model_dim=encoded_dimension(config),
            x_pending=x_pending,
            fixed_features_list=fixed_features_list,
        )
        decoded_candidates = unit_cube_to_user_values(config, x_unit_raw)
        rejection_message = _candidate_batch_rejection_message(config, df, decoded_candidates)
        if rejection_message is None:
            user_candidates = decoded_candidates
            break

    if user_candidates is None or acquisition_value is None:
        user_candidates, acquisition_value = _fallback_qlog_nehvi_candidate_batch(
            config=config,
            df=df,
            model=model,
            x_baseline=x_baseline,
            ref_point=ref_point,
            x_pending=x_pending,
            batch_size=batch_size,
            fixed_features_list=fixed_features_list,
            rejection_message=rejection_message,
        )

    x_unit_repaired = values_to_unit_cube(config, user_candidates)
    with torch.no_grad():
        posterior = model.posterior(x_unit_repaired)
        mean_user = objectives_from_model_space(config, posterior.mean)
        std = posterior.variance.clamp_min(0.0).sqrt()

    rows = []
    iteration = next_iteration(df)
    acquisition_scalar = float(acquisition_value.reshape(-1)[0])
    for index in range(batch_size):
        row = _empty_row(config)
        row["row_id"] = uuid.uuid4().hex
        row["iteration"] = iteration
        row["status"] = "suggested"
        row["source"] = "qlog_nehvi"
        _populate_replicate_fields(config, row)
        _populate_review_fields(config, row)
        for name, value in zip(config.variable_names, user_candidates[index], strict=True):
            row[name] = value
        for objective_index, objective in enumerate(config.objectives):
            row[f"predicted_mean_{objective.name}"] = float(mean_user[index, objective_index])
            row[f"predicted_std_{objective.name}"] = float(std[index, objective_index])
        row["acquisition"] = acquisition_scalar
        rows.append(row)

    return pd.DataFrame(rows, columns=canonical_columns(config))


def _x_pending_for_qlog_nehvi(
    config: CampaignConfig,
    active_pending_df: pd.DataFrame,
) -> torch.Tensor | None:
    if active_pending_df.empty:
        return None
    pending_candidates = [
        tuple(row[variable.name] for variable in config.variables)
        for _, row in active_pending_df.iterrows()
    ]
    return values_to_unit_cube(config, pending_candidates)


def _suggest_cost_aware_multi_objective_model_based(
    config: CampaignConfig,
    df: pd.DataFrame,
    observed_df: pd.DataFrame,
    batch_size: int,
) -> pd.DataFrame:
    torch.manual_seed(config.bo.random_seed)
    combination_count = categorical_combination_count(config)
    if combination_count > MAX_CATEGORICAL_COMBINATIONS:
        raise SuggestionError(
            "Cost-aware multi-objective mixed-variable suggestions support at most "
            f"{MAX_CATEGORICAL_COMBINATIONS} categorical combinations: "
            f"configured={combination_count}."
        )

    model = fit_gp_model(config, observed_df)
    _, train_y_model = dataframe_to_tensors(config, observed_df)
    fixed_features_list = categorical_feature_assignments(config)
    ref_point = reference_point_to_model_space(config)
    acquisition = build_qlog_ehvi_acquisition(
        config=config,
        model=model,
        train_y_model=train_y_model,
        ref_point=ref_point,
    )
    remaining_budget = budget_remaining(config, df)
    scored_batches: list[dict[str, object]] = []
    rejection_message = "no cost-aware multi-objective candidate batches were evaluated"

    for attempt in range(MAX_DECODE_RETRIES):
        torch.manual_seed(config.bo.random_seed + attempt)
        for candidates in _cost_aware_multi_objective_candidate_batches(
            config=config,
            model=model,
            train_y_model=train_y_model,
            ref_point=ref_point,
            fixed_features_list=fixed_features_list,
            batch_size=batch_size,
            attempt=attempt,
        ):
            scored = _score_cost_aware_multi_objective_batch(
                config=config,
                df=df,
                model=model,
                acquisition=acquisition,
                candidates=candidates,
                remaining_budget=remaining_budget,
            )
            if "rejection_message" in scored:
                rejection_message = str(scored["rejection_message"])
                continue
            scored_batches.append(scored)

    if not scored_batches:
        remaining = "unbounded" if remaining_budget is None else f"{remaining_budget:.6g}"
        raise SuggestionError(
            "Could not generate a budget-feasible cost-aware qLogEHVI batch after "
            f"{MAX_DECODE_RETRIES} retries. remaining_budget={remaining}. "
            f"Last rejection: {rejection_message}"
        )

    assert config.cost is not None
    shortlisted = sorted(
        scored_batches,
        key=lambda item: float(item["acquisition"]),
        reverse=True,
    )[: config.cost.top_k]
    chosen = max(
        shortlisted,
        key=lambda item: (
            float(item["utility"]),
            float(item["acquisition"]),
            -float(item["total_batch_cost"]),
        ),
    )

    candidates = chosen["candidates"]
    mean_user = chosen["predicted_mean"]
    std = chosen["predicted_std"]
    costs = chosen["cost_estimates"]
    acquisition_scalar = float(chosen["acquisition"])
    utility_scalar = float(chosen["utility"])
    rows = []
    iteration = next_iteration(df)
    for index in range(batch_size):
        row = _empty_row(config)
        row["row_id"] = uuid.uuid4().hex
        row["iteration"] = iteration
        row["status"] = "suggested"
        row["source"] = "cost_qlog_ehvi"
        _populate_replicate_fields(config, row)
        _populate_review_fields(config, row)
        for name, value in zip(config.variable_names, candidates[index], strict=True):
            row[name] = value
        for objective_index, objective in enumerate(config.objectives):
            row[f"predicted_mean_{objective.name}"] = float(mean_user[index, objective_index])
            row[f"predicted_std_{objective.name}"] = float(std[index, objective_index])
        row["cost_estimate"] = float(costs[index])
        row["cost_actual"] = ""
        row["acquisition"] = acquisition_scalar
        row["utility"] = utility_scalar
        rows.append(row)

    return pd.DataFrame(rows, columns=canonical_columns(config))


def _cost_aware_multi_objective_candidate_batches(
    *,
    config: CampaignConfig,
    model,
    train_y_model: torch.Tensor,
    ref_point: torch.Tensor,
    fixed_features_list: list[dict[int, float]],
    batch_size: int,
    attempt: int,
) -> list[list[tuple[object, ...]]]:
    batches: list[list[tuple[object, ...]]] = []
    try:
        x_unit_raw, _, _ = optimize_qlog_ehvi(
            config=config,
            model=model,
            train_y_model=train_y_model,
            ref_point=ref_point,
            batch_size=batch_size,
            model_dim=encoded_dimension(config),
            fixed_features_list=fixed_features_list,
        )
    except (BotorchError, RuntimeError, ValueError):
        pass
    else:
        batches.append(unit_cube_to_user_values(config, x_unit_raw))

    assert config.cost is not None
    assignments = fixed_features_list or [{}]
    model_dim = encoded_dimension(config)
    pool_size = max(config.cost.candidate_pool_size, 1)
    for assignment_index, fixed_features in enumerate(assignments):
        engine = SobolEngine(
            dimension=model_dim,
            scramble=True,
            seed=config.bo.random_seed + 131071 + attempt * 101 + assignment_index,
        )
        unit = engine.draw(pool_size * batch_size).to(dtype=torch.double)
        candidate_batches = unit.reshape(pool_size, batch_size, model_dim)
        for candidate_batch in candidate_batches:
            for feature_index, value in fixed_features.items():
                candidate_batch[:, feature_index] = value
            batches.append(unit_cube_to_user_values(config, candidate_batch))
    return batches


def _score_cost_aware_multi_objective_batch(
    *,
    config: CampaignConfig,
    df: pd.DataFrame,
    model,
    acquisition,
    candidates: list[tuple[object, ...]],
    remaining_budget: float | None,
) -> dict[str, object]:
    rejection = _candidate_batch_rejection_message(config, df, candidates)
    if rejection is not None:
        return {"rejection_message": rejection}

    costs = [evaluate_cost(config, candidate) for candidate in candidates]
    total_cost = float(sum(costs))
    if remaining_budget is not None and total_cost > remaining_budget:
        return {
            "rejection_message": (
                "batch exceeds remaining budget: "
                f"total_batch_cost={total_cost:.6g}, "
                f"remaining_budget={remaining_budget:.6g}, candidates={candidates}."
            )
        }

    x_unit = values_to_unit_cube(config, candidates)
    with torch.no_grad():
        acquisition_value = float(acquisition(x_unit).reshape(-1)[0])
        posterior = model.posterior(x_unit)
        mean_user = objectives_from_model_space(config, posterior.mean)
        std = posterior.variance.clamp_min(0.0).sqrt()
    if not math.isfinite(acquisition_value):
        return {
            "rejection_message": (
                "batch produced a non-finite qLogEHVI acquisition value: "
                f"acquisition={acquisition_value!r}, candidates={candidates}."
            )
        }

    assert config.cost is not None
    utility = acquisition_value - config.cost.weight * total_cost
    return {
        "candidates": candidates,
        "cost_estimates": costs,
        "total_batch_cost": total_cost,
        "acquisition": acquisition_value,
        "utility": float(utility),
        "predicted_mean": mean_user,
        "predicted_std": std,
    }


def _fallback_qlog_ehvi_candidate_batch(
    *,
    config: CampaignConfig,
    df: pd.DataFrame,
    model,
    train_y_model: torch.Tensor,
    ref_point: torch.Tensor,
    batch_size: int,
    fixed_features_list: list[dict[int, float]],
    rejection_message: str,
) -> tuple[list[tuple[object, ...]], torch.Tensor]:
    acquisition = build_qlog_ehvi_acquisition(
        config=config,
        model=model,
        train_y_model=train_y_model,
        ref_point=ref_point,
    )
    assignments = fixed_features_list or [{}]
    best_candidates: list[tuple[object, ...]] | None = None
    best_value: torch.Tensor | None = None
    best_scalar = -math.inf
    model_dim = encoded_dimension(config)
    pool_size = max(config.bo.raw_samples, 64)

    for attempt in range(MAX_DECODE_RETRIES):
        for assignment_index, fixed_features in enumerate(assignments):
            engine = SobolEngine(
                dimension=model_dim,
                scramble=True,
                seed=config.bo.random_seed + 104729 + attempt * 101 + assignment_index,
            )
            unit = engine.draw(pool_size * batch_size).to(dtype=torch.double)
            candidate_batches = unit.reshape(pool_size, batch_size, model_dim)
            for candidate_batch in candidate_batches:
                for feature_index, value in fixed_features.items():
                    candidate_batch[:, feature_index] = value
                decoded_candidates = unit_cube_to_user_values(config, candidate_batch)
                rejection = _candidate_batch_rejection_message(config, df, decoded_candidates)
                if rejection is not None:
                    rejection_message = rejection
                    continue
                repaired = values_to_unit_cube(config, decoded_candidates)
                with torch.no_grad():
                    value = acquisition(repaired).reshape(-1)[0]
                scalar = float(value)
                if math.isfinite(scalar) and scalar > best_scalar:
                    best_candidates = decoded_candidates
                    best_value = value.detach().reshape(1)
                    best_scalar = scalar

    if best_candidates is None or best_value is None:
        raise SuggestionError(
            "Could not generate enough feasible, non-duplicate multi-objective "
            f"suggestions after {MAX_DECODE_RETRIES} retries. {_GENERATION_FAILURE_HINT} "
            f"Last rejection: {rejection_message}"
        )
    return best_candidates, best_value


def _fallback_qlog_nehvi_candidate_batch(
    *,
    config: CampaignConfig,
    df: pd.DataFrame,
    model,
    x_baseline: torch.Tensor,
    ref_point: torch.Tensor,
    x_pending: torch.Tensor | None,
    batch_size: int,
    fixed_features_list: list[dict[int, float]],
    rejection_message: str,
) -> tuple[list[tuple[object, ...]], torch.Tensor]:
    acquisition = build_qlog_nehvi_acquisition(
        config=config,
        model=model,
        x_baseline=x_baseline,
        ref_point=ref_point,
        x_pending=x_pending,
    )
    assignments = fixed_features_list or [{}]
    best_candidates: list[tuple[object, ...]] | None = None
    best_value: torch.Tensor | None = None
    best_scalar = -math.inf
    model_dim = encoded_dimension(config)
    pool_size = max(config.bo.raw_samples, 64)

    for attempt in range(MAX_DECODE_RETRIES):
        for assignment_index, fixed_features in enumerate(assignments):
            engine = SobolEngine(
                dimension=model_dim,
                scramble=True,
                seed=config.bo.random_seed + 130363 + attempt * 101 + assignment_index,
            )
            unit = engine.draw(pool_size * batch_size).to(dtype=torch.double)
            candidate_batches = unit.reshape(pool_size, batch_size, model_dim)
            for candidate_batch in candidate_batches:
                for feature_index, value in fixed_features.items():
                    candidate_batch[:, feature_index] = value
                decoded_candidates = unit_cube_to_user_values(config, candidate_batch)
                rejection = _candidate_batch_rejection_message(config, df, decoded_candidates)
                if rejection is not None:
                    rejection_message = rejection
                    continue
                repaired = values_to_unit_cube(config, decoded_candidates)
                with torch.no_grad():
                    value = acquisition(repaired).reshape(-1)[0]
                scalar = float(value)
                if math.isfinite(scalar) and scalar > best_scalar:
                    best_candidates = decoded_candidates
                    best_value = value.detach().reshape(1)
                    best_scalar = scalar

    if best_candidates is None or best_value is None:
        raise SuggestionError(
            "Could not generate enough feasible, non-duplicate qLogNEHVI "
            f"suggestions after {MAX_DECODE_RETRIES} retries. {_GENERATION_FAILURE_HINT} "
            f"Last rejection: {rejection_message}"
        )
    return best_candidates, best_value


def _suggest_uncertain_best_replicate(
    *,
    config: CampaignConfig,
    df: pd.DataFrame,
    observed_df: pd.DataFrame,
    batch_size: int,
) -> pd.DataFrame | None:
    aggregate = aggregate_observed_replicates(config, observed_df)
    if aggregate.empty:
        return None

    model = fit_gp_model(config, observed_df)
    model_df = modeling_observed_data(config, observed_df)
    x_unit = values_to_unit_cube(
        config,
        [
            tuple(row[variable.name] for variable in config.variables)
            for _, row in model_df.iterrows()
        ],
    )
    with torch.no_grad():
        posterior = model.posterior(x_unit)
        mean_model = posterior.mean.squeeze(-1)
        std = posterior.variance.clamp_min(0.0).sqrt().squeeze(-1)

    best_index = int(torch.argmax(mean_model).item())
    best_group = aggregate.iloc[best_index]
    n_replicates = int(best_group["n_replicates"])
    posterior_std = float(std[best_index])
    if (
        posterior_std <= config.replicates.replicate_threshold
        and n_replicates >= config.replicates.min_repeats_at_best
    ):
        return None
    if n_replicates >= config.replicates.max_repeats_per_group:
        return None

    candidate = tuple(best_group[variable.name] for variable in config.variables)
    repeat_count = 1
    if n_replicates < config.replicates.min_repeats_at_best:
        repeat_count = config.replicates.min_repeats_at_best - n_replicates
    repeat_count = min(
        batch_size,
        repeat_count,
        config.replicates.max_repeats_per_group - n_replicates,
    )
    if config.cost is not None:
        remaining = budget_remaining(config, df)
        if remaining is not None:
            repeat_cost = evaluate_cost(config, candidate)
            if repeat_cost > 0:
                repeat_count = min(repeat_count, int(remaining // repeat_cost))
                if repeat_count < 1:
                    return None

    group = str(best_group["replicate_group"])
    group_rows = df.loc[df["replicate_group"].astype(str) == group]
    group_row_count = int(len(group_rows))
    if group_row_count >= config.replicates.max_repeats_per_group:
        return None
    repeat_count = min(
        repeat_count,
        config.replicates.max_repeats_per_group - group_row_count,
    )
    if repeat_count < 1:
        return None
    next_replicate_index = int(pd.to_numeric(group_rows["replicate_index"]).max()) + 1
    iteration = next_iteration(df)
    source = "log_ei" if repeat_count == 1 else "qlog_ei"
    mean_user = objective_from_model_space(config, mean_model[best_index])
    rows = []
    for offset in range(repeat_count):
        row = _empty_row(config)
        row["row_id"] = uuid.uuid4().hex
        row["iteration"] = iteration
        row["status"] = "suggested"
        row["source"] = source
        row["replicate_group"] = group
        row["replicate_index"] = next_replicate_index + offset
        _populate_review_fields(config, row)
        for name, value in zip(config.variable_names, candidate, strict=True):
            row[name] = value
        row["predicted_mean"] = float(mean_user)
        row["predicted_std"] = posterior_std
        row["acquisition"] = 0.0
        _populate_cost_fields(config, row, candidate)
        rows.append(row)
    return pd.DataFrame(rows, columns=canonical_columns(config))


def _fill_replicate_batch_with_exploration(
    *,
    config: CampaignConfig,
    df: pd.DataFrame,
    observed_df: pd.DataFrame,
    repeat_suggestions: pd.DataFrame,
    batch_size: int,
) -> pd.DataFrame:
    remaining = batch_size - len(repeat_suggestions)
    if remaining <= 0:
        return repeat_suggestions

    df_with_repeats = pd.concat([df, repeat_suggestions], ignore_index=True)
    filler_config = _config_with_repeat_budget_reserved(config, repeat_suggestions)
    try:
        if config.cost is not None:
            filler = _suggest_cost_aware_model_based(
                config=filler_config,
                df=df_with_repeats,
                observed_df=observed_df,
                batch_size=remaining,
            )
        else:
            filler = _suggest_model_based(
                config=config,
                df=df_with_repeats,
                observed_df=observed_df,
                batch_size=remaining,
            )
    except _CandidateGenerationExhausted:
        return repeat_suggestions
    except SuggestionError as exc:
        raise SuggestionError(
            "Repeat suggestions were generated, but exploration fill failed: "
            f"{exc}"
        ) from exc

    filler = filler.copy()
    filler.loc[:, "iteration"] = repeat_suggestions["iteration"].iloc[0]
    return pd.concat([repeat_suggestions, filler], ignore_index=True).loc[
        :,
        canonical_columns(config),
    ]


def _config_with_repeat_budget_reserved(
    config: CampaignConfig,
    repeat_suggestions: pd.DataFrame,
) -> CampaignConfig:
    if config.cost is None or config.cost.budget is None or repeat_suggestions.empty:
        return config
    repeat_cost = sum(
        evaluate_cost(
            config,
            tuple(row[variable.name] for variable in config.variables),
        )
        for _, row in repeat_suggestions.iterrows()
    )
    if repeat_cost <= 0:
        return config
    return replace(
        config,
        cost=replace(
            config.cost,
            budget=max(0.0, config.cost.budget - float(repeat_cost)),
        ),
    )


def _suggest_model_based(
    config: CampaignConfig,
    df: pd.DataFrame,
    observed_df: pd.DataFrame,
    batch_size: int,
    context_values: dict[str, object] | None = None,
) -> pd.DataFrame:
    torch.manual_seed(config.bo.random_seed)
    combination_count = (
        contextual_categorical_combination_count(config)
        if config.context is not None
        else categorical_combination_count(config)
    )
    if combination_count > MAX_CATEGORICAL_COMBINATIONS:
        raise SuggestionError(
            "Model-based mixed-variable suggestions support at most "
            f"{MAX_CATEGORICAL_COMBINATIONS} categorical combinations: "
            f"configured={combination_count}."
        )

    model = fit_gp_model(config, observed_df)
    _, train_y_model = dataframe_to_tensors(config, observed_df)
    fixed_features_list = (
        contextual_fixed_feature_assignments(config, context_values or {})
        if config.context is not None
        else categorical_feature_assignments(config)
    )
    user_candidates: list[tuple[object, ...]] | None = None
    acquisition_value: torch.Tensor | None = None
    source: str | None = None
    rejection_message = "no candidate was decoded"
    for attempt in range(MAX_DECODE_RETRIES):
        torch.manual_seed(config.bo.random_seed + attempt)
        x_unit_raw, acquisition_value, source = optimize_log_ei(
            config=config,
            model=model,
            train_y_model=train_y_model,
            batch_size=batch_size,
            model_dim=encoded_dimension(config),
            fixed_features_list=fixed_features_list,
        )
        decoded_candidates = unit_cube_to_user_values(config, x_unit_raw)
        rejection_message = _candidate_batch_rejection_message(config, df, decoded_candidates)
        if rejection_message is None:
            user_candidates = decoded_candidates
            break

    if user_candidates is None or acquisition_value is None or source is None:
        raise _CandidateGenerationExhausted(
            "Could not generate enough feasible, non-duplicate suggestions after "
            f"{MAX_DECODE_RETRIES} retries. {_GENERATION_FAILURE_HINT} "
            f"Last rejection: {rejection_message}"
        )

    x_unit_repaired = values_to_unit_cube(config, user_candidates)

    with torch.no_grad():
        posterior = model.posterior(x_unit_repaired)
        mean_model = posterior.mean.squeeze(-1)
        std = posterior.variance.clamp_min(0.0).sqrt().squeeze(-1)
        mean_user = objective_from_model_space(config, mean_model)

    rows = []
    iteration = next_iteration(df)
    acquisition_scalar = float(acquisition_value.reshape(-1)[0])
    for index in range(batch_size):
        row = _empty_row(config)
        row["row_id"] = uuid.uuid4().hex
        row["iteration"] = iteration
        row["status"] = "suggested"
        row["source"] = source
        _populate_replicate_fields(config, row)
        _populate_review_fields(config, row)
        for name, value in zip(config.variable_names, user_candidates[index], strict=True):
            row[name] = value
        row["predicted_mean"] = float(mean_user[index])
        row["predicted_std"] = float(std[index])
        row["acquisition"] = acquisition_scalar
        rows.append(row)

    return pd.DataFrame(rows, columns=canonical_columns(config))


def _suggest_qlog_nei_model_based(
    config: CampaignConfig,
    df: pd.DataFrame,
    observed_df: pd.DataFrame,
    active_pending_df: pd.DataFrame,
    batch_size: int,
) -> pd.DataFrame:
    torch.manual_seed(config.bo.random_seed)
    combination_count = categorical_combination_count(config)
    if combination_count > MAX_CATEGORICAL_COMBINATIONS:
        raise SuggestionError(
            "qLogNEI mixed-variable suggestions support at most "
            f"{MAX_CATEGORICAL_COMBINATIONS} categorical combinations: "
            f"configured={combination_count}."
        )

    model = fit_gp_model(config, observed_df)
    training = dataframe_to_training_tensors(config, observed_df)
    fixed_features_list = categorical_feature_assignments(config)
    x_pending = (
        dataframe_to_unit_cube(config, active_pending_df)
        if not active_pending_df.empty
        else None
    )
    user_candidates: list[tuple[object, ...]] | None = None
    acquisition_value: torch.Tensor | None = None
    rejection_message = "no candidate was decoded"
    for attempt in range(MAX_DECODE_RETRIES):
        torch.manual_seed(config.bo.random_seed + attempt)
        x_unit_raw, acquisition_value, _source = optimize_qlog_nei(
            config=config,
            model=model,
            x_baseline=training.train_x,
            x_pending=x_pending,
            batch_size=batch_size,
            model_dim=encoded_dimension(config),
            fixed_features_list=fixed_features_list,
        )
        decoded_candidates = unit_cube_to_user_values(config, x_unit_raw)
        rejection_message = _candidate_batch_rejection_message(config, df, decoded_candidates)
        if rejection_message is None:
            user_candidates = decoded_candidates
            break

    if user_candidates is None or acquisition_value is None:
        raise _CandidateGenerationExhausted(
            "Could not generate enough feasible, non-duplicate qLogNEI suggestions "
            f"after {MAX_DECODE_RETRIES} retries. {_GENERATION_FAILURE_HINT} "
            f"Last rejection: {rejection_message}"
        )

    x_unit_repaired = values_to_unit_cube(config, user_candidates)

    with torch.no_grad():
        posterior = model.posterior(x_unit_repaired)
        mean_model = posterior.mean.squeeze(-1)
        std = posterior.variance.clamp_min(0.0).sqrt().squeeze(-1)
        mean_user = objective_from_model_space(config, mean_model)

    rows = []
    iteration = next_iteration(df)
    acquisition_scalar = float(acquisition_value.reshape(-1)[0])
    for index in range(batch_size):
        row = _empty_row(config)
        row["row_id"] = uuid.uuid4().hex
        row["iteration"] = iteration
        row["status"] = "suggested"
        row["source"] = "qlog_nei"
        _populate_replicate_fields(config, row)
        _populate_review_fields(config, row)
        for name, value in zip(config.variable_names, user_candidates[index], strict=True):
            row[name] = value
        row["predicted_mean"] = float(mean_user[index])
        row["predicted_std"] = float(std[index])
        row["acquisition"] = acquisition_scalar
        rows.append(row)

    return pd.DataFrame(rows, columns=canonical_columns(config))


def _suggest_cost_aware_model_based(
    config: CampaignConfig,
    df: pd.DataFrame,
    observed_df: pd.DataFrame,
    batch_size: int,
    context_values: dict[str, object] | None = None,
) -> pd.DataFrame:
    if config.context is not None and context_values is None:
        raise SuggestionError(
            "Contextual cost-aware suggestions require resolved context values."
        )
    torch.manual_seed(config.bo.random_seed)
    combination_count = (
        contextual_categorical_combination_count(config)
        if config.context is not None
        else categorical_combination_count(config)
    )
    if combination_count > MAX_CATEGORICAL_COMBINATIONS:
        raise SuggestionError(
            "Cost-aware mixed-variable suggestions support at most "
            f"{MAX_CATEGORICAL_COMBINATIONS} categorical combinations: "
            f"configured={combination_count}."
        )

    model = fit_gp_model(config, observed_df)
    _, train_y_model = dataframe_to_tensors(config, observed_df)
    acquisition = LogExpectedImprovement(model=model, best_f=train_y_model.max())
    fixed_features_list = (
        contextual_fixed_feature_assignments(config, context_values or {})
        if config.context is not None
        else categorical_feature_assignments(config)
    )
    remaining_budget = budget_remaining(config, df)
    selected: list[tuple[object, ...]] = []
    rows = []
    iteration = next_iteration(df)

    for batch_index in range(batch_size):
        chosen = _choose_cost_aware_candidate(
            config=config,
            df=df,
            model=model,
            acquisition=acquisition,
            train_y_model=train_y_model,
            fixed_features_list=fixed_features_list,
            selected=selected,
            remaining_budget=remaining_budget,
            attempt_offset=batch_index,
            context_values=context_values,
        )
        if remaining_budget is not None:
            remaining_budget -= chosen["cost_estimate"]
        selected.append(chosen["candidate"])

        row = _empty_row(config)
        row["row_id"] = uuid.uuid4().hex
        row["iteration"] = iteration
        row["status"] = "suggested"
        row["source"] = "cost_log_ei"
        _populate_replicate_fields(config, row)
        _populate_review_fields(config, row)
        for name, value in zip(config.variable_names, chosen["candidate"], strict=True):
            row[name] = value
        row["predicted_mean"] = chosen["predicted_mean"]
        row["predicted_std"] = chosen["predicted_std"]
        row["acquisition"] = chosen["acquisition"]
        row["cost_estimate"] = chosen["cost_estimate"]
        row["utility"] = chosen["utility"]
        rows.append(row)

    return pd.DataFrame(rows, columns=canonical_columns(config))


def _choose_cost_aware_candidate(
    *,
    config: CampaignConfig,
    df: pd.DataFrame,
    model,
    acquisition,
    train_y_model: torch.Tensor,
    fixed_features_list: list[dict[int, float]],
    selected: list[tuple[object, ...]],
    remaining_budget: float | None,
    attempt_offset: int,
    context_values: dict[str, object] | None = None,
) -> dict[str, object]:
    rejection_message = "no cost-aware candidates were evaluated"
    for attempt in range(MAX_DECODE_RETRIES):
        torch.manual_seed(config.bo.random_seed + attempt_offset * 101 + attempt)
        pool = _cost_aware_candidate_pool(
            config=config,
            model=model,
            train_y_model=train_y_model,
            fixed_features_list=fixed_features_list,
            attempt=attempt_offset * 101 + attempt,
            context_values=context_values,
        )
        seen: set[tuple[object, ...]] = set()
        scored_candidates: list[dict[str, object]] = []
        for candidate in pool:
            candidate_key = design_key_for_values(config, candidate)
            if candidate_key in seen:
                continue
            seen.add(candidate_key)
            rejection_message = _candidate_rejection_message(config, df, candidate, selected)
            if rejection_message is not None:
                continue
            cost_estimate = evaluate_cost(config, candidate)
            if remaining_budget is not None and cost_estimate > remaining_budget:
                rejection_message = (
                    "candidate exceeds remaining budget: "
                    f"cost_estimate={cost_estimate:.6g}, "
                    f"remaining_budget={remaining_budget:.6g}, candidate={candidate}."
                )
                continue
            scored = _score_cost_aware_candidate(
                config=config,
                model=model,
                acquisition=acquisition,
                candidate=candidate,
                cost_estimate=cost_estimate,
            )
            scored_candidates.append(scored)
        if scored_candidates:
            assert config.cost is not None
            shortlisted = sorted(
                scored_candidates,
                key=lambda item: float(item["acquisition"]),
                reverse=True,
            )[: config.cost.top_k]
            return max(
                shortlisted,
                key=lambda item: (
                    float(item["utility"]),
                    float(item["acquisition"]),
                    -float(item["cost_estimate"]),
                ),
            )

    raise _CandidateGenerationExhausted(
        "Could not generate enough budget-feasible cost-aware suggestions after "
        f"{MAX_DECODE_RETRIES} retries. The feasible design space may be exhausted, "
        "constraints may be too restrictive, bo.min_normalized_distance may be too "
        f"large, or the remaining budget may be too small. Last rejection: {rejection_message}"
    )


def _cost_aware_candidate_pool(
    *,
    config: CampaignConfig,
    model,
    train_y_model: torch.Tensor,
    fixed_features_list: list[dict[int, float]],
    attempt: int,
    context_values: dict[str, object] | None = None,
) -> list[tuple[object, ...]]:
    candidates: list[tuple[object, ...]] = []
    try:
        x_unit_raw, _, _ = optimize_log_ei(
            config=config,
            model=model,
            train_y_model=train_y_model,
            batch_size=1,
            model_dim=encoded_dimension(config),
            fixed_features_list=fixed_features_list,
        )
    except (BotorchError, RuntimeError, ValueError):
        pass
    else:
        candidates.extend(
            _apply_context_to_candidates(
                config,
                unit_cube_to_user_values(config, x_unit_raw),
                context_values,
            )
        )

    assert config.cost is not None
    pool_size = config.cost.candidate_pool_size
    engine = SobolEngine(
        dimension=encoded_dimension(config),
        scramble=True,
        seed=config.bo.random_seed + 7919 + attempt,
    )
    sobol = engine.draw(pool_size).to(dtype=torch.double)
    candidates.extend(
        _apply_context_to_candidates(
            config,
            unit_cube_to_user_values(config, sobol),
            context_values,
        )
    )
    return candidates


def _apply_context_to_candidates(
    config: CampaignConfig,
    candidates: list[tuple[object, ...]],
    context_values: dict[str, object] | None,
) -> list[tuple[object, ...]]:
    if not context_values:
        return candidates
    return [
        apply_context_to_candidate(config, candidate, context_values)
        for candidate in candidates
    ]


def _score_cost_aware_candidate(
    *,
    config: CampaignConfig,
    model,
    acquisition,
    candidate: tuple[object, ...],
    cost_estimate: float,
) -> dict[str, object]:
    x_unit = values_to_unit_cube(config, [candidate])
    with torch.no_grad():
        acquisition_value = float(acquisition(x_unit.unsqueeze(1)).reshape(-1)[0])
        posterior = model.posterior(x_unit)
        mean_model = posterior.mean.squeeze(-1)
        std = posterior.variance.clamp_min(0.0).sqrt().squeeze(-1)
        mean_user = objective_from_model_space(config, mean_model)
    utility = acquisition_value - config.cost.weight * cost_estimate
    return {
        "candidate": candidate,
        "cost_estimate": float(cost_estimate),
        "acquisition": acquisition_value,
        "utility": float(utility),
        "predicted_mean": float(mean_user[0]),
        "predicted_std": float(std[0]),
    }


def _initial_user_candidates(
    config: CampaignConfig,
    df: pd.DataFrame,
    count: int,
    method: str,
    context_values: dict[str, object] | None = None,
) -> list[tuple[object, ...]]:
    existing = design_tuples(config, df)
    finite_size = _finite_design_space_size(
        config,
        fixed_variable_names=set(context_values or {}),
    )
    existing_for_space = (
        design_tuples(config, _rows_matching_context(config, df, context_values))
        if context_values
        else existing
    )
    if finite_size is not None and len(existing_for_space) + count > finite_size:
        raise SuggestionError(
            "Could not generate non-duplicate initial suggestions because the finite "
            f"design space is exhausted: requested={count}, "
            f"existing={len(existing_for_space)}, "
            f"space_size={finite_size}."
        )

    engine = None
    generator = None
    if method == "sobol":
        engine = SobolEngine(
            dimension=len(config.variables),
            scramble=True,
            seed=config.bo.random_seed,
        )
    elif method == "random":
        generator = torch.Generator()
        generator.manual_seed(config.bo.random_seed)
    else:
        raise SuggestionError(
            f"Unsupported initial_design_method '{method}'. Expected 'sobol' or 'random'."
        )

    selected: list[tuple[object, ...]] = []
    seen = set(existing)
    batches_drawn = 0
    initial_remaining_budget = budget_remaining(config, df)
    if (
        config.cost is not None
        and initial_remaining_budget is not None
        and initial_remaining_budget <= 0
    ):
        raise SuggestionError(
            "Could not generate enough budget-feasible initial suggestions. "
            "The remaining budget may be too small."
        )

    while len(selected) < count:
        draw_count = max(count * 16, 64)
        if engine is not None:
            unit = engine.draw(draw_count).to(dtype=torch.double)
        else:
            unit = torch.rand(
                draw_count,
                len(config.variables),
                generator=generator,
                dtype=torch.double,
            )
        for candidate in unit_cube_to_design_values(config, unit):
            if context_values:
                candidate = apply_context_to_candidate(config, candidate, context_values)
            rejection_message = _candidate_rejection_message(config, df, candidate, selected)
            if rejection_message is not None:
                continue
            if config.cost is not None and initial_remaining_budget is not None:
                selected_cost = sum(evaluate_cost(config, item) for item in selected)
                candidate_cost = evaluate_cost(config, candidate)
                if candidate_cost > initial_remaining_budget - selected_cost:
                    continue
            selected.append(candidate)
            candidate_key = design_key_for_values(config, candidate)
            seen.add(candidate_key)
            if len(selected) == count:
                break
        batches_drawn += 1
        if batches_drawn > 1000 or len(seen) > 100_000:
            if config.cost is not None and initial_remaining_budget is not None:
                raise SuggestionError(
                    "Could not generate enough budget-feasible initial suggestions. "
                    "The feasible design space may be exhausted or the remaining "
                    "budget may be too small."
                )
            raise SuggestionError(
                "Could not generate enough feasible, non-duplicate suggestions after "
                f"{batches_drawn} retries. {_GENERATION_FAILURE_HINT}"
            )

    return selected


def _candidate_batch_rejection_message(
    config: CampaignConfig,
    df: pd.DataFrame,
    candidates: list[tuple[object, ...]],
) -> str | None:
    accepted: list[tuple[object, ...]] = []
    for candidate in candidates:
        rejection_message = _candidate_rejection_message(config, df, candidate, accepted)
        if rejection_message is not None:
            return rejection_message
        accepted.append(candidate)
    return None


def _candidate_rejection_message(
    config: CampaignConfig,
    df: pd.DataFrame,
    candidate: tuple[object, ...],
    batch_candidates: list[tuple[object, ...]],
) -> str | None:
    violations = constraint_violations_for_values(config, candidate)
    if violations:
        names = [constraint.name for constraint in violations]
        return f"candidate violates constraint(s) {names}: candidate={candidate}."

    candidate_key = design_key_for_values(config, candidate)
    if candidate_key in design_tuples(config, df):
        return f"candidate duplicates an existing design exactly: candidate={candidate}."

    batch_keys = {design_key_for_values(config, existing) for existing in batch_candidates}
    if candidate_key in batch_keys:
        return f"candidate duplicates another same-batch design exactly: candidate={candidate}."

    threshold = config.bo.min_normalized_distance
    if threshold > 0:
        comparisons = _candidate_values_from_df(config, df) + batch_candidates
        nearest = _nearest_normalized_distance(config, candidate, comparisons)
        if nearest is not None and nearest < threshold:
            return (
                "candidate is too close to an existing or same-batch design in encoded "
                f"model space: distance={nearest:.6g}, "
                f"min_normalized_distance={threshold:.6g}, candidate={candidate}."
            )

    return None


def suggestion_quality_summary(
    config: CampaignConfig,
    df: pd.DataFrame,
    suggestions: pd.DataFrame,
) -> pd.DataFrame:
    """Return read-only quality diagnostics for suggested rows."""
    validate_campaign_data(config, df)
    if config.is_structured_campaign:
        return _structured_suggestion_quality_summary(config, df, suggestions)
    required_columns = {"row_id", *config.variable_names}
    missing = sorted(required_columns - set(suggestions.columns))
    if missing:
        raise SuggestionError(
            f"Suggestion quality summary is missing required columns: {missing}."
        )

    suggestion_candidates = _candidate_values_from_df(config, suggestions)
    existing_candidates = _candidate_values_from_df(config, df)
    existing_keys = design_tuples(config, df)
    suggestion_keys = [
        design_key_for_values(config, candidate) for candidate in suggestion_candidates
    ]
    threshold = config.bo.min_normalized_distance

    rows = []
    for index, candidate in enumerate(suggestion_candidates):
        row = suggestions.iloc[index]
        violations = constraint_violations_for_values(config, candidate)
        violation_names = [constraint.name for constraint in violations]
        candidate_key = suggestion_keys[index]
        existing_duplicate_allowed_by_replicates = _duplicate_allowed_by_replicates(
            config,
            df,
            row,
            candidate,
        )
        batch_duplicate_allowed_by_replicates = (
            _same_batch_duplicate_allowed_by_replicates(
                config,
                df,
                suggestions,
                candidate_key,
                suggestion_keys,
            )
        )
        duplicate_allowed_by_replicates = (
            existing_duplicate_allowed_by_replicates
            or batch_duplicate_allowed_by_replicates
        )
        is_exact_duplicate = (
            (
                candidate_key in existing_keys
                and not existing_duplicate_allowed_by_replicates
            )
            or (
                suggestion_keys.count(candidate_key) > 1
                and not batch_duplicate_allowed_by_replicates
            )
        )
        batch_comparisons = [
            other_candidate
            for other_index, other_candidate in enumerate(suggestion_candidates)
            if other_index != index
        ]
        nearest_existing = _nearest_normalized_distance(
            config,
            candidate,
            existing_candidates,
        )
        nearest_batch = _nearest_normalized_distance(
            config,
            candidate,
            batch_comparisons,
        )
        distances = [
            distance for distance in (nearest_existing, nearest_batch) if distance is not None
        ]
        passes_distance_threshold = duplicate_allowed_by_replicates or (
            True if threshold <= 0 or not distances else min(distances) >= threshold
        )
        rows.append(
            {
                "row_id": row["row_id"],
                "is_feasible": len(violation_names) == 0,
                "violated_constraints": ", ".join(violation_names),
                "is_exact_duplicate": is_exact_duplicate,
                "duplicate_allowed_by_replicates": duplicate_allowed_by_replicates,
                "nearest_existing_distance": nearest_existing,
                "nearest_batch_distance": nearest_batch,
                "passes_distance_threshold": passes_distance_threshold,
            }
        )

    return pd.DataFrame(
        rows,
        columns=SUGGESTION_QUALITY_COLUMNS,
    )


def _structured_suggestion_quality_summary(
    config: CampaignConfig,
    df: pd.DataFrame,
    suggestions: pd.DataFrame,
) -> pd.DataFrame:
    validate_campaign_data(config, suggestions)
    rows: list[pd.DataFrame] = []
    for stage_name in config.stage_names:
        stage_suggestions = suggestions.loc[suggestions["stage"] == stage_name]
        if stage_suggestions.empty:
            continue
        stage_config = _stage_local_config(config, stage_name)
        stage_df = _stage_local_dataframe(config, df, stage_name, stage_config)
        local_suggestions = _stage_local_dataframe(
            config,
            stage_suggestions,
            stage_name,
            stage_config,
        )
        quality = suggestion_quality_summary(stage_config, stage_df, local_suggestions)
        rows.append(quality)
    if not rows:
        return pd.DataFrame(columns=SUGGESTION_QUALITY_COLUMNS)
    return pd.concat(rows, ignore_index=True)


def _candidate_values_from_df(
    config: CampaignConfig,
    df: pd.DataFrame,
) -> list[tuple[object, ...]]:
    return [
        tuple(row[variable.name] for variable in config.variables)
        for _, row in df.iterrows()
    ]


def _duplicate_allowed_by_replicates(
    config: CampaignConfig,
    df: pd.DataFrame,
    row: pd.Series,
    candidate: tuple[object, ...],
) -> bool:
    if not config.replicates.enabled or "replicate_group" not in df.columns:
        return False
    if "replicate_group" not in row or "replicate_index" not in row:
        return False

    group = str(row["replicate_group"])
    group_rows = df.loc[df["replicate_group"].astype(str) == group]
    if group_rows.empty:
        return False

    candidate_key = design_key_for_values(config, candidate)
    group_keys = {
        design_key_for_values(
            config,
            [existing_row[variable.name] for variable in config.variables],
        )
        for _, existing_row in group_rows.iterrows()
    }
    if group_keys != {candidate_key}:
        return False

    replicate_index = pd.to_numeric(pd.Series([row["replicate_index"]]), errors="coerce").iloc[0]
    if pd.isna(replicate_index) or not math.isfinite(float(replicate_index)):
        return False
    existing_indices = set(pd.to_numeric(group_rows["replicate_index"], errors="coerce"))
    return int(replicate_index) not in {int(index) for index in existing_indices}


def _same_batch_duplicate_allowed_by_replicates(
    config: CampaignConfig,
    df: pd.DataFrame,
    suggestions: pd.DataFrame,
    candidate_key: tuple[object, ...],
    suggestion_keys: list[tuple[object, ...]],
) -> bool:
    if not config.replicates.enabled or "replicate_group" not in suggestions.columns:
        return False
    matching_indices = [
        index for index, key in enumerate(suggestion_keys) if key == candidate_key
    ]
    if len(matching_indices) <= 1:
        return False

    matching_rows = suggestions.iloc[matching_indices]
    groups = {str(value) for value in matching_rows["replicate_group"]}
    if len(groups) != 1:
        return False
    group = next(iter(groups))
    group_rows = df.loc[df["replicate_group"].astype(str) == group]
    if group_rows.empty:
        return False

    group_keys = {
        design_key_for_values(
            config,
            [existing_row[variable.name] for variable in config.variables],
        )
        for _, existing_row in group_rows.iterrows()
    }
    if group_keys != {candidate_key}:
        return False

    suggested_indices = pd.to_numeric(
        matching_rows["replicate_index"],
        errors="coerce",
    )
    if suggested_indices.isna().any():
        return False
    suggested_index_values = [int(value) for value in suggested_indices]
    if len(suggested_index_values) != len(set(suggested_index_values)):
        return False

    existing_indices = {
        int(index)
        for index in pd.to_numeric(group_rows["replicate_index"], errors="coerce")
    }
    return existing_indices.isdisjoint(suggested_index_values)


def _nearest_normalized_distance(
    config: CampaignConfig,
    candidate: tuple[object, ...],
    comparison_candidates: list[tuple[object, ...]],
) -> float | None:
    if not comparison_candidates:
        return None

    candidate_tensor = values_to_unit_cube(config, [candidate])
    comparison_tensor = values_to_unit_cube(config, comparison_candidates)
    distance = torch.cdist(candidate_tensor, comparison_tensor).min().item()
    return float(distance / math.sqrt(encoded_dimension(config)))


def _empty_row(config: CampaignConfig) -> dict[str, object]:
    return {column: "" for column in canonical_columns(config)}


def _populate_review_fields(config: CampaignConfig, row: dict[str, object]) -> None:
    if config.review.enabled:
        row["review_status"] = "pending"
        row["review_note"] = ""


def _populate_replicate_fields(config: CampaignConfig, row: dict[str, object]) -> None:
    if config.replicates.enabled:
        row["replicate_group"] = row["row_id"]
        row["replicate_index"] = 0


def _populate_cost_fields(
    config: CampaignConfig,
    row: dict[str, object],
    candidate: tuple[object, ...],
) -> None:
    if config.cost is None:
        return
    row["cost_estimate"] = evaluate_cost(config, candidate)
    row["cost_actual"] = ""
    row["utility"] = ""


def _finite_design_space_size(
    config: CampaignConfig,
    fixed_variable_names: set[str] | None = None,
) -> int | None:
    fixed_names = fixed_variable_names or set()
    sizes = []
    for variable in config.variables:
        if variable.name in fixed_names:
            continue
        if variable.type == "continuous":
            return None
        if variable.type == "integer":
            if variable.lower is None or variable.upper is None:
                raise SuggestionError(f"Variable '{variable.name}' is missing integer bounds.")
            sizes.append(int(variable.upper) - int(variable.lower) + 1)
        elif variable.type in {"discrete", "categorical"}:
            sizes.append(len(variable.values))
        else:
            raise SuggestionError(
                f"Variable '{variable.name}' has unsupported type '{variable.type}'."
            )
    return math.prod(sizes)


def _rows_matching_context(
    config: CampaignConfig,
    df: pd.DataFrame,
    context_values: dict[str, object] | None,
) -> pd.DataFrame:
    if not context_values or df.empty:
        return df
    variables_by_name = {variable.name: variable for variable in config.variables}
    mask = pd.Series(True, index=df.index)
    for name, value in context_values.items():
        variable = variables_by_name[name]
        expected = normalize_context_value(variable, value, f"context '{name}'")
        matches = df[name].map(
            lambda row_value, variable=variable, expected=expected, name=name: (
                normalize_context_value(variable, row_value, f"context '{name}'")
                == expected
            )
        )
        mask &= matches
    return df.loc[mask]
