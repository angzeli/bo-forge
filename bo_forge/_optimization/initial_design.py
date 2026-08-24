"""Internal candidate suggestion generation."""

from __future__ import annotations

import uuid

import pandas as pd
import torch
from torch.quasirandom import SobolEngine

from bo_forge._optimization.common import (
    _candidate_rejection_message,
    _empty_row,
    _finite_design_space_size,
    _populate_cost_fields,
    _populate_replicate_fields,
    _populate_review_fields,
    _rows_matching_context,
)
from bo_forge.config import CampaignConfig
from bo_forge.contextual import (
    apply_context_to_candidate,
)
from bo_forge.costs import budget_remaining, evaluate_cost
from bo_forge.errors import SuggestionError
from bo_forge.multifidelity import (
    map_initial_fidelity_to_levels,
)
from bo_forge.transforms import (
    unit_cube_to_design_values,
)
from bo_forge.validation import (
    canonical_columns,
    design_key_for_values,
    design_tuples,
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

    engine, generator = _initial_design_generators(config, method)

    selected: list[tuple[object, ...]] = []
    seen = set(existing)
    batches_drawn = 0
    minimum_rejected_candidate_cost: float | None = None
    initial_remaining_budget = budget_remaining(config, df)
    if (
        config.cost is not None
        and initial_remaining_budget is not None
        and initial_remaining_budget <= 0
    ):
        raise SuggestionError(
            "Could not generate enough budget-feasible initial suggestions. "
            f"remaining_budget={initial_remaining_budget:.6g}."
        )

    while len(selected) < count:
        draw_count = max(count * 16, 64)
        unit = _draw_initial_unit_candidates(config, draw_count, engine, generator)
        unit = map_initial_fidelity_to_levels(config, unit)
        minimum_rejected_candidate_cost = _select_initial_candidates_from_draw(
            config=config,
            df=df,
            candidates=unit_cube_to_design_values(config, unit),
            selected=selected,
            seen=seen,
            count=count,
            context_values=context_values,
            remaining_budget=initial_remaining_budget,
            minimum_rejected_cost=minimum_rejected_candidate_cost,
        )
        batches_drawn += 1
        if batches_drawn > MAX_INITIAL_DESIGN_BATCHES or len(seen) > 100_000:
            _raise_initial_generation_exhausted(
                config=config,
                selected=selected,
                batches_drawn=batches_drawn,
                remaining_budget=initial_remaining_budget,
                minimum_rejected_cost=minimum_rejected_candidate_cost,
            )

    return selected


def _initial_design_generators(
    config: CampaignConfig,
    method: str,
) -> tuple[SobolEngine | None, torch.Generator | None]:
    if method == "sobol":
        return (
            SobolEngine(
                dimension=len(config.variables),
                scramble=True,
                seed=config.bo.random_seed,
            ),
            None,
        )
    if method == "random":
        generator = torch.Generator()
        generator.manual_seed(config.bo.random_seed)
        return None, generator
    raise SuggestionError(
        f"Unsupported initial_design_method '{method}'. Expected 'sobol' or 'random'."
    )


def _draw_initial_unit_candidates(
    config: CampaignConfig,
    count: int,
    engine: SobolEngine | None,
    generator: torch.Generator | None,
) -> torch.Tensor:
    if engine is not None:
        return engine.draw(count).to(dtype=torch.double)
    return torch.rand(
        count,
        len(config.variables),
        generator=generator,
        dtype=torch.double,
    )


def _select_initial_candidates_from_draw(
    *,
    config: CampaignConfig,
    df: pd.DataFrame,
    candidates: list[tuple[object, ...]],
    selected: list[tuple[object, ...]],
    seen: set[tuple[object, ...]],
    count: int,
    context_values: dict[str, object] | None,
    remaining_budget: float | None,
    minimum_rejected_cost: float | None,
) -> float | None:
    for candidate in candidates:
        if context_values:
            candidate = apply_context_to_candidate(config, candidate, context_values)
        if _candidate_rejection_message(config, df, candidate, selected) is not None:
            continue
        if config.cost is not None and remaining_budget is not None:
            selected_cost = sum(evaluate_cost(config, item) for item in selected)
            candidate_cost = evaluate_cost(config, candidate)
            if candidate_cost > remaining_budget - selected_cost:
                minimum_rejected_cost = min(
                    candidate_cost,
                    minimum_rejected_cost
                    if minimum_rejected_cost is not None
                    else candidate_cost,
                )
                continue
        selected.append(candidate)
        seen.add(design_key_for_values(config, candidate))
        if len(selected) == count:
            break
    return minimum_rejected_cost


def _raise_initial_generation_exhausted(
    *,
    config: CampaignConfig,
    selected: list[tuple[object, ...]],
    batches_drawn: int,
    remaining_budget: float | None,
    minimum_rejected_cost: float | None,
) -> None:
    if config.cost is not None and remaining_budget is not None:
        selected_cost = sum(evaluate_cost(config, item) for item in selected)
        available_for_next = remaining_budget - selected_cost
        cost_detail = (
            ""
            if minimum_rejected_cost is None
            else ", minimum_rejected_candidate_cost="
            f"{minimum_rejected_cost:.6g}, "
            f"available_for_next_candidate={available_for_next:.6g}"
        )
        raise SuggestionError(
            "Could not generate enough budget-feasible initial suggestions. "
            "The feasible design space may be exhausted or the remaining "
            "budget may be too small: "
            f"remaining_budget={remaining_budget:.6g}{cost_detail}."
        )
    raise SuggestionError(
        "Could not generate enough feasible, non-duplicate suggestions after "
        f"{batches_drawn} retries. {_GENERATION_FAILURE_HINT}"
    )
