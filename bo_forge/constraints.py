"""Safe constraint expression validation and evaluation."""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Iterable, Mapping, Sequence

from bo_forge.config import CampaignConfig, ConstraintConfig, VariableConfig
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


def validate_constraint_expression(
    *,
    name: str,
    expression: str,
    variable_names: Iterable[str],
) -> None:
    """Validate one constraint expression at config-load time."""
    tree = _parse_expression(name, expression)
    allowed_names = set(variable_names)
    validator = _ConstraintAstValidator(name, allowed_names)
    validator.visit(tree)


def constraint_violations_for_row(
    config: CampaignConfig,
    row: Mapping[str, object],
) -> list[ConstraintConfig]:
    """Return constraints violated by one campaign row."""
    context = {
        variable.name: _normalise_variable_value(variable, row[variable.name])
        for variable in config.variables
    }
    return _constraint_violations(config.constraints, context)


def constraint_violations_for_values(
    config: CampaignConfig,
    values: Sequence[object],
) -> list[ConstraintConfig]:
    """Return constraints violated by one user-space candidate tuple."""
    context = {
        variable.name: _normalise_variable_value(variable, value)
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
        return ast.parse(expression, mode="eval")
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
        result = _evaluate_node(tree.body, context)
    except (TypeError, ZeroDivisionError) as exc:
        raise LogValidationError(
            f"Constraint '{constraint.name}' could not be evaluated: {exc}."
        ) from exc
    if not isinstance(result, bool):
        raise LogValidationError(
            f"Constraint '{constraint.name}' did not evaluate to a boolean value."
        )
    return result


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
        f"Constraint expression contains unsupported syntax: {type(node).__name__}."
    )


class _ConstraintAstValidator(ast.NodeVisitor):
    def __init__(self, name: str, variable_names: set[str]) -> None:
        self.name = name
        self.variable_names = variable_names

    def visit_Expression(self, node: ast.Expression) -> None:
        self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> None:
        if not isinstance(node.value, str | int | float | bool):
            self._raise_unsupported(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id not in self.variable_names:
            raise ConfigError(
                f"Constraint '{self.name}' references unknown variable '{node.id}'."
            )

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
            f"Constraint '{self.name}' uses unsupported syntax: "
            f"{type(node).__name__}."
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
