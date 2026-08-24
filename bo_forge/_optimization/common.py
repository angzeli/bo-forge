"""Internal candidate suggestion generation."""

from __future__ import annotations

import math
from dataclasses import replace

import pandas as pd
import torch

from bo_forge.config import CampaignConfig
from bo_forge.constraints import constraint_variable_names, constraint_violations_for_values
from bo_forge.contextual import (
    normalize_context_value,
)
from bo_forge.costs import evaluate_cost
from bo_forge.errors import SuggestionError
from bo_forge.transforms import (
    encoded_dimension,
    values_to_unit_cube,
)
from bo_forge.validation import (
    canonical_columns,
    design_key_for_values,
    design_tuples,
    has_pending_suggestions,
    next_iteration,
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



class _CandidateGenerationExhausted(SuggestionError):
    """Internal signal for expected budget or design-space exhaustion."""


def _suggest_structured_stage(
    *,
    config: CampaignConfig,
    df: pd.DataFrame,
    batch_size: int | None,
    stage: str | None,
) -> pd.DataFrame:
    from bo_forge._optimization.router import suggest_next

    stage_name = _resolve_structured_stage(config, stage)
    if config.cost is not None:
        raise SuggestionError(
            "Structured campaign suggestions with cost are currently unsupported."
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
        if (
            config.fidelity is not None
            and config.fidelity.levels is not None
            and variable.name == config.fidelity.variable
        ):
            sizes.append(len(config.fidelity.levels))
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
