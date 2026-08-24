"""Internal candidate suggestion generation."""

from __future__ import annotations

import uuid

import pandas as pd
import torch
from botorch.acquisition import LogExpectedImprovement
from botorch.exceptions.errors import BotorchError
from torch.quasirandom import SobolEngine

from bo_forge._optimization.common import (
    _candidate_batch_rejection_message,
    _candidate_rejection_message,
    _CandidateGenerationExhausted,
    _empty_row,
    _populate_replicate_fields,
    _populate_review_fields,
)
from bo_forge.acquisition import (
    optimize_log_ei,
    optimize_qlog_nei,
)
from bo_forge.config import CampaignConfig
from bo_forge.contextual import (
    apply_context_to_candidate,
    contextual_categorical_combination_count,
    contextual_fixed_feature_assignments,
)
from bo_forge.costs import budget_remaining, evaluate_cost
from bo_forge.errors import SuggestionError
from bo_forge.models import (
    dataframe_to_tensors,
    dataframe_to_training_tensors,
    fit_gp_model,
)
from bo_forge.transforms import (
    categorical_combination_count,
    categorical_feature_assignments,
    dataframe_to_unit_cube,
    encoded_dimension,
    objective_from_model_space,
    unit_cube_to_user_values,
    values_to_unit_cube,
)
from bo_forge.validation import (
    canonical_columns,
    design_key_for_values,
    next_iteration,
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
            "Contextual cost-aware suggestions require resolved context values. "
            "Provide every required context value through context defaults, "
            "context_values=..., or CLI --context NAME=VALUE."
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
