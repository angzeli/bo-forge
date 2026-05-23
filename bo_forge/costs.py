"""Cost expression validation, evaluation, and budget summaries."""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Iterable, Mapping, Sequence

import pandas as pd

from bo_forge.config import CampaignConfig, VariableConfig
from bo_forge.errors import ConfigError, LogValidationError

_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_COMPARISON_OPERATORS = {
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}


def validate_cost_expression(
    *,
    expression: str,
    variable_names: Iterable[str],
) -> None:
    """Validate a cost expression at config-load time."""
    tree = _parse_expression(expression)
    validator = _CostAstValidator(set(variable_names))
    validator.visit(tree)


def evaluate_cost(
    config: CampaignConfig,
    values: Mapping[str, object] | Sequence[object],
) -> float:
    """Evaluate the configured cost expression for one user-space design."""
    if config.cost is None:
        raise LogValidationError("Cannot evaluate cost because config has no cost section.")
    if isinstance(values, Mapping) or _contains_variable_names(config, values):
        context = {
            variable.name: _normalise_variable_value(variable, values[variable.name])
            for variable in config.variables
        }
    else:
        context = {
            variable.name: _normalise_variable_value(variable, value)
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
        return ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ConfigError(f"Cost expression has invalid syntax: {expression!r}.") from exc


def _evaluate_cost_expression(
    expression: str,
    context: Mapping[str, object],
) -> object:
    tree = _parse_expression(expression)
    try:
        return _evaluate_node(tree.body, context)
    except (KeyError, TypeError, ZeroDivisionError) as exc:
        raise LogValidationError(f"Cost expression could not be evaluated: {exc}.") from exc


def _evaluate_node(node: ast.AST, context: Mapping[str, object]) -> object:
    if isinstance(node, ast.Constant):
        return node.value

    if isinstance(node, ast.Name):
        return context[node.id]

    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            for value in node.values:
                if not bool(_evaluate_node(value, context)):
                    return False
            return True
        if isinstance(node.op, ast.Or):
            for value in node.values:
                if bool(_evaluate_node(value, context)):
                    return True
            return False

    if isinstance(node, ast.UnaryOp):
        operand = _evaluate_node(node.operand, context)
        if isinstance(node.op, ast.Not):
            return not bool(operand)
        for operator_type, operation in _UNARY_OPERATORS.items():
            if isinstance(node.op, operator_type):
                return operation(operand)

    if isinstance(node, ast.BinOp):
        left = _evaluate_node(node.left, context)
        right = _evaluate_node(node.right, context)
        for operator_type, operation in _BINARY_OPERATORS.items():
            if isinstance(node.op, operator_type):
                return operation(left, right)

    if isinstance(node, ast.Compare):
        left = _evaluate_node(node.left, context)
        for op, comparator in zip(node.ops, node.comparators, strict=True):
            right = _evaluate_node(comparator, context)
            for operator_type, operation in _COMPARISON_OPERATORS.items():
                if isinstance(op, operator_type):
                    if not operation(left, right):
                        return False
                    break
            left = right
        return True

    raise LogValidationError(
        f"Cost expression contains unsupported syntax: {type(node).__name__}."
    )


class _CostAstValidator(ast.NodeVisitor):
    def __init__(self, variable_names: set[str]) -> None:
        self.variable_names = variable_names

    def visit_Expression(self, node: ast.Expression) -> None:
        self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> None:
        if not isinstance(node.value, str | int | float | bool):
            self._raise_unsupported(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id not in self.variable_names:
            raise ConfigError(f"Cost expression references unknown variable '{node.id}'.")

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        if not isinstance(node.op, ast.And | ast.Or):
            self._raise_unsupported(node)
        for value in node.values:
            self.visit(value)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        if not isinstance(node.op, ast.Not | ast.UAdd | ast.USub):
            self._raise_unsupported(node)
        self.visit(node.operand)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if not isinstance(node.op, ast.Add | ast.Sub | ast.Mult | ast.Div):
            self._raise_unsupported(node)
        self.visit(node.left)
        self.visit(node.right)

    def visit_Compare(self, node: ast.Compare) -> None:
        for op in node.ops:
            if not isinstance(op, ast.Lt | ast.LtE | ast.Gt | ast.GtE | ast.Eq | ast.NotEq):
                self._raise_unsupported(node)
        self.visit(node.left)
        for comparator in node.comparators:
            self.visit(comparator)

    def generic_visit(self, node: ast.AST) -> None:
        self._raise_unsupported(node)

    def _raise_unsupported(self, node: ast.AST) -> None:
        raise ConfigError(
            f"Cost expression uses unsupported syntax: {type(node).__name__}."
        )


def _normalise_variable_value(variable: VariableConfig, value: object) -> object:
    if variable.type == "continuous":
        return _finite_float(variable, value)
    if variable.type == "integer":
        parsed = _finite_float(variable, value)
        if parsed % 1 != 0:
            raise LogValidationError(
                f"Variable '{variable.name}' has non-integer value: value={value!r}."
            )
        return int(parsed)
    if variable.type == "discrete":
        parsed = _finite_float(variable, value)
        for allowed in [float(item) for item in variable.values]:
            if math.isclose(parsed, allowed, rel_tol=1e-12, abs_tol=1e-12):
                return float(allowed)
        raise LogValidationError(
            f"Variable '{variable.name}' has value outside configured choices: "
            f"value={value!r}."
        )
    if variable.type == "categorical":
        if not isinstance(value, str) or value == "" or value.strip() != value:
            raise LogValidationError(
                f"Variable '{variable.name}' has blank or whitespace-padded "
                f"categorical value: value={value!r}."
            )
        return value
    raise LogValidationError(
        f"Variable '{variable.name}' has unsupported type '{variable.type}'."
    )


def _finite_float(variable: VariableConfig, value: object) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise LogValidationError(
            f"Variable '{variable.name}' has non-numeric value: value={value!r}."
        ) from exc
    if not math.isfinite(parsed):
        raise LogValidationError(
            f"Variable '{variable.name}' has non-finite value: value={value!r}."
        )
    return parsed


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
