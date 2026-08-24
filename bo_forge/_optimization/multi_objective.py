"""Internal candidate suggestion generation."""

from __future__ import annotations

import math
import uuid

import pandas as pd
import torch
from botorch.exceptions.errors import BotorchError
from torch.quasirandom import SobolEngine

from bo_forge._optimization.common import (
    _candidate_batch_rejection_message,
    _empty_row,
    _populate_replicate_fields,
    _populate_review_fields,
)
from bo_forge.acquisition import (
    build_qlog_ehvi_acquisition,
    build_qlog_nehvi_acquisition,
    optimize_qlog_ehvi,
    optimize_qlog_nehvi,
)
from bo_forge.config import CampaignConfig
from bo_forge.costs import budget_remaining, evaluate_cost
from bo_forge.errors import SuggestionError
from bo_forge.models import (
    dataframe_to_tensors,
    fit_gp_model,
)
from bo_forge.multi_objective import (
    objectives_from_model_space,
    reference_point_to_model_space,
)
from bo_forge.transforms import (
    categorical_combination_count,
    categorical_feature_assignments,
    encoded_dimension,
    unit_cube_to_user_values,
    values_to_unit_cube,
)
from bo_forge.validation import (
    canonical_columns,
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
