"""Internal candidate suggestion generation."""

from __future__ import annotations

import uuid
from dataclasses import replace

import pandas as pd
import torch

from bo_forge._optimization.common import (
    _CandidateGenerationExhausted,
    _empty_row,
    _populate_cost_fields,
    _populate_review_fields,
    _rows_matching_context,
)
from bo_forge._optimization.single_objective import (
    _suggest_cost_aware_model_based,
    _suggest_model_based,
)
from bo_forge.config import CampaignConfig
from bo_forge.contextual import (
    apply_context_to_candidate,
)
from bo_forge.costs import budget_remaining, evaluate_cost
from bo_forge.errors import SuggestionError
from bo_forge.models import (
    fit_gp_model,
)
from bo_forge.replicates import aggregate_observed_replicates
from bo_forge.transforms import (
    objective_from_model_space,
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

def _suggest_uncertain_best_replicate(
    *,
    config: CampaignConfig,
    df: pd.DataFrame,
    observed_df: pd.DataFrame,
    batch_size: int,
    context_values: dict[str, object] | None = None,
) -> pd.DataFrame | None:
    model_choice = _best_replicate_model_choice(
        config,
        observed_df,
        context_values,
    )
    if model_choice is None:
        return None
    best_group, best_index, mean_model, posterior_std = model_choice
    n_replicates = int(best_group["n_replicates"])
    if (
        posterior_std <= config.replicates.replicate_threshold
        and n_replicates >= config.replicates.min_repeats_at_best
    ):
        return None
    if n_replicates >= config.replicates.max_repeats_per_group:
        return None

    candidate = tuple(best_group[variable.name] for variable in config.variables)
    if context_values:
        candidate = apply_context_to_candidate(config, candidate, context_values)
    repeat_count = _repeat_count_for_candidate(
        config,
        df,
        candidate,
        n_replicates,
        batch_size,
    )
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


def _best_replicate_model_choice(
    config: CampaignConfig,
    observed_df: pd.DataFrame,
    context_values: dict[str, object] | None,
) -> tuple[pd.Series, int, torch.Tensor, float] | None:
    aggregate = aggregate_observed_replicates(config, observed_df).reset_index(drop=True)
    if aggregate.empty:
        return None
    eligible = _rows_matching_context(config, aggregate, context_values)
    if eligible.empty:
        return None
    model = fit_gp_model(config, observed_df)
    x_unit = values_to_unit_cube(
        config,
        [
            tuple(row[variable.name] for variable in config.variables)
            for _, row in aggregate.iterrows()
        ],
    )
    with torch.no_grad():
        posterior = model.posterior(x_unit)
        mean_model = posterior.mean.squeeze(-1)
        std = posterior.variance.clamp_min(0.0).sqrt().squeeze(-1)
    eligible_positions = torch.tensor(eligible.index.tolist(), dtype=torch.long)
    eligible_best = int(torch.argmax(mean_model[eligible_positions]).item())
    best_index = int(eligible_positions[eligible_best].item())
    return aggregate.iloc[best_index], best_index, mean_model, float(std[best_index])


def _repeat_count_for_candidate(
    config: CampaignConfig,
    df: pd.DataFrame,
    candidate: tuple[object, ...],
    n_replicates: int,
    batch_size: int,
) -> int:
    repeat_count = 1
    if n_replicates < config.replicates.min_repeats_at_best:
        repeat_count = config.replicates.min_repeats_at_best - n_replicates
    repeat_count = min(
        batch_size,
        repeat_count,
        config.replicates.max_repeats_per_group - n_replicates,
    )
    if config.cost is None:
        return repeat_count
    remaining = budget_remaining(config, df)
    repeat_cost = evaluate_cost(config, candidate)
    if remaining is None or repeat_cost <= 0:
        return repeat_count
    return min(repeat_count, int(remaining // repeat_cost))


def _fill_replicate_batch_with_exploration(
    *,
    config: CampaignConfig,
    df: pd.DataFrame,
    observed_df: pd.DataFrame,
    repeat_suggestions: pd.DataFrame,
    batch_size: int,
    context_values: dict[str, object] | None = None,
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
                context_values=context_values,
            )
        else:
            filler = _suggest_model_based(
                config=config,
                df=df_with_repeats,
                observed_df=observed_df,
                batch_size=remaining,
                context_values=context_values,
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
