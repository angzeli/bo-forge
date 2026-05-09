"""Candidate suggestion generation."""

from __future__ import annotations

import uuid

import pandas as pd
import torch
from torch.quasirandom import SobolEngine

from bo_forge.acquisition import optimize_log_ei
from bo_forge.config import CampaignConfig
from bo_forge.errors import SuggestionError
from bo_forge.models import dataframe_to_tensors, fit_gp_model
from bo_forge.transforms import from_unit_cube, objective_from_model_space
from bo_forge.validation import (
    canonical_columns,
    design_tuples,
    get_observed_data,
    has_pending_suggestions,
    next_iteration,
    validate_campaign_data,
)


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
        return _suggest_sobol(
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


def _suggest_sobol(config: CampaignConfig, df: pd.DataFrame, count: int) -> pd.DataFrame:
    existing = design_tuples(config, df)
    candidates = _sobol_user_candidates(config, count=count, existing=existing)
    rows = []
    iteration = next_iteration(df)
    for candidate in candidates:
        row = _empty_row(config)
        row["row_id"] = uuid.uuid4().hex
        row["iteration"] = iteration
        row["status"] = "suggested"
        row["source"] = "sobol"
        for name, value in zip(config.variable_names, candidate, strict=True):
            row[name] = float(value)
        rows.append(row)
    return pd.DataFrame(rows, columns=canonical_columns(config))


def _suggest_model_based(
    config: CampaignConfig,
    df: pd.DataFrame,
    observed_df: pd.DataFrame,
    batch_size: int,
) -> pd.DataFrame:
    torch.manual_seed(config.bo.random_seed)
    model = fit_gp_model(config, observed_df)
    _, train_y_model = dataframe_to_tensors(config, observed_df)
    x_unit, acquisition_value, source = optimize_log_ei(
        config=config,
        model=model,
        train_y_model=train_y_model,
        batch_size=batch_size,
    )
    x_user = from_unit_cube(config, x_unit).detach()
    _raise_if_duplicate_candidates(config, df, x_user)

    with torch.no_grad():
        posterior = model.posterior(x_unit)
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
        for name, value in zip(config.variable_names, x_user[index].tolist(), strict=True):
            row[name] = float(value)
        row["predicted_mean"] = float(mean_user[index])
        row["predicted_std"] = float(std[index])
        row["acquisition"] = acquisition_scalar
        rows.append(row)

    return pd.DataFrame(rows, columns=canonical_columns(config))


def _sobol_user_candidates(
    config: CampaignConfig,
    count: int,
    existing: set[tuple[float, ...]],
) -> list[tuple[float, ...]]:
    engine = SobolEngine(
        dimension=len(config.variables),
        scramble=True,
        seed=config.bo.random_seed,
    )
    selected: list[tuple[float, ...]] = []
    seen = set(existing)

    while len(selected) < count:
        draw_count = max(count * 8, len(seen) + count + 8)
        unit = engine.draw(draw_count).to(dtype=torch.double)
        user = from_unit_cube(config, unit).tolist()
        for row in user:
            candidate = tuple(float(value) for value in row)
            candidate_key = _design_key(candidate)
            if candidate_key in seen:
                continue
            selected.append(candidate)
            seen.add(candidate_key)
            if len(selected) == count:
                break
        if len(seen) > 100_000:
            raise SuggestionError("Could not generate non-duplicate Sobol suggestions.")

    return selected


def _raise_if_duplicate_candidates(
    config: CampaignConfig,
    df: pd.DataFrame,
    x_user: torch.Tensor,
) -> None:
    existing = design_tuples(config, df)
    for row in x_user.tolist():
        candidate = tuple(float(value) for value in row)
        if _design_key(candidate) in existing:
            raise SuggestionError(
                "Model-based suggestion duplicated an existing design exactly: "
                f"candidate={candidate}."
            )


def _empty_row(config: CampaignConfig) -> dict[str, object]:
    return {column: "" for column in canonical_columns(config)}


def _design_key(candidate: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(round(value, 12) for value in candidate)
