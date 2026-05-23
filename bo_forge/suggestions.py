"""Candidate suggestion generation."""

from __future__ import annotations

import math
import uuid

import pandas as pd
import torch
from botorch.acquisition import LogExpectedImprovement
from botorch.exceptions.errors import BotorchError
from torch.quasirandom import SobolEngine

from bo_forge.acquisition import optimize_log_ei
from bo_forge.config import CampaignConfig
from bo_forge.constraints import constraint_violations_for_values
from bo_forge.costs import budget_remaining, evaluate_cost
from bo_forge.errors import SuggestionError
from bo_forge.models import dataframe_to_tensors, fit_gp_model
from bo_forge.transforms import (
    categorical_combination_count,
    categorical_feature_assignments,
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
    has_pending_suggestions,
    next_iteration,
    validate_campaign_data,
)

MAX_CATEGORICAL_COMBINATIONS = 64
MAX_DECODE_RETRIES = 8
_GENERATION_FAILURE_HINT = (
    "The feasible design space may be exhausted, constraints may be too restrictive, "
    "or bo.min_normalized_distance may be too large."
)


def suggest_next(
    config: CampaignConfig,
    df: pd.DataFrame,
    batch_size: int | None = None,
) -> pd.DataFrame:
    """Suggest the next experiment or batch for a campaign."""
    validate_campaign_data(config, df)
    if has_pending_suggestions(df, config):
        raise SuggestionError(
            "Cannot generate new suggestions while unresolved status='suggested' rows exist."
        )

    requested_batch_size = batch_size if batch_size is not None else config.bo.batch_size
    if requested_batch_size < 1:
        raise SuggestionError(f"batch_size must be >= 1: value={requested_batch_size}.")

    observed_df = get_observed_data(config, df)
    remaining_initial = config.bo.initial_design_size - len(observed_df)
    if remaining_initial > 0:
        return _suggest_initial_design(
            config=config,
            df=df,
            count=min(requested_batch_size, remaining_initial),
        )

    if config.cost is not None:
        return _suggest_cost_aware_model_based(
            config=config,
            df=df,
            observed_df=observed_df,
            batch_size=requested_batch_size,
        )

    return _suggest_model_based(
        config=config,
        df=df,
        observed_df=observed_df,
        batch_size=requested_batch_size,
    )


def _suggest_initial_design(
    config: CampaignConfig,
    df: pd.DataFrame,
    count: int,
) -> pd.DataFrame:
    source = config.bo.initial_design_method
    candidates = _initial_user_candidates(
        config,
        df=df,
        count=count,
        method=source,
    )
    rows = []
    iteration = next_iteration(df)
    for candidate in candidates:
        row = _empty_row(config)
        row["row_id"] = uuid.uuid4().hex
        row["iteration"] = iteration
        row["status"] = "suggested"
        row["source"] = source
        _populate_review_fields(config, row)
        for name, value in zip(config.variable_names, candidate, strict=True):
            row[name] = value
        _populate_cost_fields(config, row, candidate)
        rows.append(row)
    return pd.DataFrame(rows, columns=canonical_columns(config))


def _suggest_model_based(
    config: CampaignConfig,
    df: pd.DataFrame,
    observed_df: pd.DataFrame,
    batch_size: int,
) -> pd.DataFrame:
    torch.manual_seed(config.bo.random_seed)
    combination_count = categorical_combination_count(config)
    if combination_count > MAX_CATEGORICAL_COMBINATIONS:
        raise SuggestionError(
            "Model-based mixed-variable suggestions support at most "
            f"{MAX_CATEGORICAL_COMBINATIONS} categorical combinations: "
            f"configured={combination_count}."
        )

    model = fit_gp_model(config, observed_df)
    _, train_y_model = dataframe_to_tensors(config, observed_df)
    fixed_features_list = categorical_feature_assignments(config)
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
        raise SuggestionError(
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
) -> pd.DataFrame:
    torch.manual_seed(config.bo.random_seed)
    combination_count = categorical_combination_count(config)
    if combination_count > MAX_CATEGORICAL_COMBINATIONS:
        raise SuggestionError(
            "Cost-aware mixed-variable suggestions support at most "
            f"{MAX_CATEGORICAL_COMBINATIONS} categorical combinations: "
            f"configured={combination_count}."
        )

    model = fit_gp_model(config, observed_df)
    _, train_y_model = dataframe_to_tensors(config, observed_df)
    acquisition = LogExpectedImprovement(model=model, best_f=train_y_model.max())
    fixed_features_list = categorical_feature_assignments(config)
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
        )
        if remaining_budget is not None:
            remaining_budget -= chosen["cost_estimate"]
        selected.append(chosen["candidate"])

        row = _empty_row(config)
        row["row_id"] = uuid.uuid4().hex
        row["iteration"] = iteration
        row["status"] = "suggested"
        row["source"] = "cost_log_ei"
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

    raise SuggestionError(
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
        candidates.extend(unit_cube_to_user_values(config, x_unit_raw))

    assert config.cost is not None
    pool_size = config.cost.candidate_pool_size
    engine = SobolEngine(
        dimension=encoded_dimension(config),
        scramble=True,
        seed=config.bo.random_seed + 7919 + attempt,
    )
    sobol = engine.draw(pool_size).to(dtype=torch.double)
    candidates.extend(unit_cube_to_user_values(config, sobol))
    return candidates


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
) -> list[tuple[object, ...]]:
    existing = design_tuples(config, df)
    finite_size = _finite_design_space_size(config)
    if finite_size is not None and len(existing) + count > finite_size:
        raise SuggestionError(
            "Could not generate non-duplicate initial suggestions because the finite "
            f"design space is exhausted: requested={count}, existing={len(existing)}, "
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
        is_exact_duplicate = (
            candidate_key in existing_keys
            or suggestion_keys.count(candidate_key) > 1
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
        passes_distance_threshold = (
            True if threshold <= 0 or not distances else min(distances) >= threshold
        )
        rows.append(
            {
                "row_id": row["row_id"],
                "is_feasible": len(violation_names) == 0,
                "violated_constraints": ", ".join(violation_names),
                "is_exact_duplicate": is_exact_duplicate,
                "nearest_existing_distance": nearest_existing,
                "nearest_batch_distance": nearest_batch,
                "passes_distance_threshold": passes_distance_threshold,
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "row_id",
            "is_feasible",
            "violated_constraints",
            "is_exact_duplicate",
            "nearest_existing_distance",
            "nearest_batch_distance",
            "passes_distance_threshold",
        ],
    )


def _candidate_values_from_df(
    config: CampaignConfig,
    df: pd.DataFrame,
) -> list[tuple[object, ...]]:
    return [
        tuple(row[variable.name] for variable in config.variables)
        for _, row in df.iterrows()
    ]


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


def _finite_design_space_size(config: CampaignConfig) -> int | None:
    sizes = []
    for variable in config.variables:
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
