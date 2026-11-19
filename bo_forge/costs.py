"""Cost expression validation, evaluation, and budget summaries."""

from __future__ import annotations

import ast
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import pandas as pd

from bo_forge._expressions import (
    evaluate_expression,
    normalise_variable_value,
    parse_expression,
    validate_expression,
)
from bo_forge.config import CampaignConfig
from bo_forge.errors import ConfigError, LogValidationError


def validate_cost_expression(
    *,
    expression: str,
    variable_names: Iterable[str],
) -> None:
    """Validate a cost expression at config-load time."""
    validate_expression(
        _parse_expression(expression),
        variable_names,
        unknown_name=lambda variable: (
            f"Cost expression references unknown variable '{variable}'."
        ),
        unsupported=lambda node_type: f"Cost expression uses unsupported syntax: {node_type}.",
    )


def evaluate_cost(
    config: CampaignConfig,
    values: Mapping[str, object] | Sequence[object] | pd.Series[Any],
) -> float:
    """Evaluate the configured cost expression for one user-space design."""
    if config.cost is None:
        raise LogValidationError("Cannot evaluate cost because config has no cost section.")
    if isinstance(values, Mapping) or _contains_variable_names(config, values):
        context = {
            variable.name: normalise_variable_value(variable, values[variable.name])
            for variable in config.variables
        }
    else:
        context = {
            variable.name: normalise_variable_value(variable, value)
            for variable, value in zip(config.variables, values, strict=True)
        }
    result = _evaluate_cost_expression(config.cost.expression, context)
    if isinstance(result, bool) or not isinstance(result, int | float):
        raise LogValidationError(
            "Cost expression must evaluate to a numeric value, not "
            f"{type(result).__name__}."
        )
    cost = float(result)
    if not math.isfinite(cost):
        raise LogValidationError(f"Cost expression produced a non-finite value: {cost!r}.")
    if cost < 0:
        raise LogValidationError(f"Cost expression produced a negative value: {cost:g}.")
    return cost


def _contains_variable_names(config: CampaignConfig, values: object) -> bool:
    return all(
        hasattr(values, "__contains__") and variable.name in values
        for variable in config.variables
    )


def observed_effective_cost(config: CampaignConfig, df: pd.DataFrame) -> float:
    """Return total effective cost for observed rows."""
    if config.cost is None or df.empty:
        return 0.0
    total = 0.0
    for _, row in df.loc[df["status"] == "observed"].iterrows():
        total += effective_row_cost(config, row)
    return float(total)


def accepted_pending_estimated_cost(config: CampaignConfig, df: pd.DataFrame) -> float:
    """Return estimated cost reserved by accepted pending suggestions."""
    if config.cost is None or not config.review.enabled or df.empty:
        return 0.0
    mask = (df["status"] == "suggested") & (df["review_status"] == "accepted")
    total = 0.0
    for _, row in df.loc[mask].iterrows():
        total += estimated_row_cost(config, row)
    return float(total)


def budget_remaining(config: CampaignConfig, df: pd.DataFrame) -> float | None:
    """Return remaining budget, or None when no budget is configured."""
    if config.cost is None or config.cost.budget is None:
        return None
    return float(
        config.cost.budget
        - observed_effective_cost(config, df)
        - accepted_pending_estimated_cost(config, df)
    )


def effective_row_cost(config: CampaignConfig, row: Mapping[str, object]) -> float:
    """Return actual cost if present, otherwise estimated/evaluated cost."""
    if config.cost is None:
        return 0.0
    actual = row.get("cost_actual", "")
    if not _is_blank(actual):
        return _non_negative_float(actual, "cost_actual")
    return estimated_row_cost(config, row)


def estimated_row_cost(config: CampaignConfig, row: Mapping[str, object]) -> float:
    """Return estimated row cost, evaluating the expression when blank."""
    estimate = row.get("cost_estimate", "")
    if not _is_blank(estimate):
        return _non_negative_float(estimate, "cost_estimate")
    return evaluate_cost(config, row)


def _parse_expression(expression: str) -> ast.Expression:
    try:
        return parse_expression(expression)
    except SyntaxError as exc:
        raise ConfigError(f"Cost expression has invalid syntax: {expression!r}.") from exc


def _evaluate_cost_expression(
    expression: str,
    context: Mapping[str, object],
) -> object:
    tree = _parse_expression(expression)
    try:
        return evaluate_expression(
            tree.body,
            context,
            unsupported=lambda node_type: (
                f"Cost expression contains unsupported syntax: {node_type}."
            ),
        )
    except (KeyError, TypeError, ZeroDivisionError) as exc:
        raise LogValidationError(f"Cost expression could not be evaluated: {exc}.") from exc


def _non_negative_float(value: object, column: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise LogValidationError(f"{column} must be numeric: value={value!r}.") from exc
    if not math.isfinite(parsed):
        raise LogValidationError(f"{column} must be finite: value={value!r}.")
    if parsed < 0:
        raise LogValidationError(f"{column} must be >= 0: value={parsed:g}.")
    return parsed


def _is_blank(value: object) -> bool:
    if pd.isna(value):
        return True
    return str(value).strip() == ""
