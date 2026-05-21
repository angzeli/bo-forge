"""Candidate suggestion generation."""

from __future__ import annotations

import math
import uuid

import pandas as pd
import torch
from torch.quasirandom import SobolEngine

from bo_forge.acquisition import optimize_log_ei
from bo_forge.config import CampaignConfig
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


def suggest_next(
    config: CampaignConfig,
    df: pd.DataFrame,
    batch_size: int | None = None,
) -> pd.DataFrame:
    """Suggest the next experiment or batch for a campaign."""
    validate_campaign_data(config, df)
    if has_pending_suggestions(df):
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
    existing = design_tuples(config, df)
    source = config.bo.initial_design_method
    candidates = _initial_user_candidates(
        config,
        count=count,
        existing=existing,
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
        for name, value in zip(config.variable_names, candidate, strict=True):
            row[name] = value
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
    duplicate_message = "no candidate was decoded"
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
        duplicate_message = _duplicate_candidate_message(config, df, decoded_candidates)
        if duplicate_message is None:
            user_candidates = decoded_candidates
            break

    if user_candidates is None or acquisition_value is None or source is None:
        raise SuggestionError(
            "Could not generate non-duplicate model-based suggestions after "
            f"{MAX_DECODE_RETRIES} decode retries. Last duplicate check: "
            f"{duplicate_message}"
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
        for name, value in zip(config.variable_names, user_candidates[index], strict=True):
            row[name] = value
        row["predicted_mean"] = float(mean_user[index])
        row["predicted_std"] = float(std[index])
        row["acquisition"] = acquisition_scalar
        rows.append(row)

    return pd.DataFrame(rows, columns=canonical_columns(config))


def _initial_user_candidates(
    config: CampaignConfig,
    count: int,
    existing: set[tuple[object, ...]],
    method: str,
) -> list[tuple[object, ...]]:
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
            candidate_key = design_key_for_values(config, candidate)
            if candidate_key in seen:
                continue
            selected.append(candidate)
            seen.add(candidate_key)
            if len(selected) == count:
                break
        batches_drawn += 1
        if batches_drawn > 1000 or len(seen) > 100_000:
            raise SuggestionError(
                f"Could not generate non-duplicate {method} suggestions."
            )

    return selected


def _duplicate_candidate_message(
    config: CampaignConfig,
    df: pd.DataFrame,
    candidates: list[tuple[object, ...]],
) -> str | None:
    existing = design_tuples(config, df)
    seen = set(existing)
    for candidate in candidates:
        candidate_key = design_key_for_values(config, candidate)
        if candidate_key in seen:
            return (
                "Model-based suggestion duplicated an existing or batch design exactly "
                f"after decoding: candidate={candidate}."
            )
        seen.add(candidate_key)
    return None


def _empty_row(config: CampaignConfig) -> dict[str, object]:
    return {column: "" for column in canonical_columns(config)}


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
