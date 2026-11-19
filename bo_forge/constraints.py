"""Safe constraint expression validation and evaluation."""

from __future__ import annotations

import ast
from collections.abc import Iterable, Mapping, Sequence

from bo_forge._expressions import (
    evaluate_expression,
    normalise_variable_value,
    parse_expression,
    validate_expression,
)
from bo_forge.config import CampaignConfig, ConstraintConfig
from bo_forge.errors import ConfigError, LogValidationError


def validate_constraint_expression(
    *,
    name: str,
    expression: str,
    variable_names: Iterable[str],
) -> None:
    """Validate one constraint expression at config-load time."""
    validate_expression(
        _parse_expression(name, expression),
        variable_names,
        unknown_name=lambda variable: (
            f"Constraint '{name}' references unknown variable '{variable}'."
        ),
        unsupported=lambda node_type: (
            f"Constraint '{name}' uses unsupported syntax: {node_type}."
        ),
    )


def constraint_variable_names(expression: str) -> set[str]:
    """Return variable names referenced by a validated constraint expression."""
    tree = parse_expression(expression)
    return {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
    }


def constraint_violations_for_row(
    config: CampaignConfig,
    row: Mapping[str, object],
) -> list[ConstraintConfig]:
    """Return constraints violated by one campaign row."""
    context = {
        variable.name: normalise_variable_value(variable, row[variable.name])
        for variable in config.variables
    }
    return _constraint_violations(config.constraints, context)


def constraint_violations_for_values(
    config: CampaignConfig,
    values: Sequence[object],
) -> list[ConstraintConfig]:
    """Return constraints violated by one user-space candidate tuple."""
    context = {
        variable.name: normalise_variable_value(variable, value)
        for variable, value in zip(config.variables, values, strict=True)
    }
    return _constraint_violations(config.constraints, context)


def _constraint_violations(
    constraints: Sequence[ConstraintConfig],
    context: Mapping[str, object],
) -> list[ConstraintConfig]:
    violations: list[ConstraintConfig] = []
    for constraint in constraints:
        result = _evaluate_constraint(constraint, context)
        if result is not True:
            violations.append(constraint)
    return violations


def _parse_expression(name: str, expression: str) -> ast.Expression:
    try:
        return parse_expression(expression)
    except SyntaxError as exc:
        raise ConfigError(
            f"Constraint '{name}' has invalid expression syntax: {expression!r}."
        ) from exc


def _evaluate_constraint(
    constraint: ConstraintConfig,
    context: Mapping[str, object],
) -> bool:
    tree = _parse_expression(constraint.name, constraint.expression)
    try:
        result = evaluate_expression(
            tree.body,
            context,
            unsupported=lambda node_type: (
                f"Constraint expression contains unsupported syntax: {node_type}."
            ),
        )
    except (TypeError, ZeroDivisionError) as exc:
        raise LogValidationError(
            f"Constraint '{constraint.name}' could not be evaluated: {exc}."
        ) from exc
    if not isinstance(result, bool):
        raise LogValidationError(
            f"Constraint '{constraint.name}' did not evaluate to a boolean value."
        )
    return result
