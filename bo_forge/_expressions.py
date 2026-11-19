"""Restricted expressions shared by constraint and cost configuration.

Constraints and cost models accept the same small grammar: constants, variable names,
``and``/``or``/``not``, unary ``+``/``-``, ``+ - * /``, and the six comparisons. Each
caller supplies its own messages and decides which evaluation errors to translate, so
user-facing errors stay specific to the feature that raised them.
"""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable, Iterable, Mapping
from functools import lru_cache

from bo_forge.config import VariableConfig
from bo_forge.errors import ConfigError, LogValidationError

MessageBuilder = Callable[[str], str]

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


@lru_cache(maxsize=256)
def parse_expression(expression: str) -> ast.Expression:
    """Parse an expression once; callers turn ``SyntaxError`` into their own message.

    Evaluation runs per log row and per candidate, so the parsed tree is cached. Trees
    are only read, never mutated, which makes sharing them safe.
    """
    return ast.parse(expression, mode="eval")


def validate_expression(
    tree: ast.Expression,
    variable_names: Iterable[str],
    *,
    unknown_name: MessageBuilder,
    unsupported: MessageBuilder,
) -> None:
    """Raise ``ConfigError`` for unknown names or syntax outside the shared grammar."""
    _ExpressionValidator(set(variable_names), unknown_name, unsupported).visit(tree)


def evaluate_expression(
    node: ast.AST,
    context: Mapping[str, object],
    *,
    unsupported: MessageBuilder,
) -> object:
    """Evaluate a validated expression node against variable values.

    Unsupported syntax raises ``LogValidationError``. ``KeyError``, ``TypeError``, and
    ``ZeroDivisionError`` propagate so each caller can report them in its own terms.
    """
    return _ExpressionEvaluator(context, unsupported).evaluate(node)


def normalise_variable_value(variable: VariableConfig, value: object) -> object:
    """Return a log or candidate value in the type an expression compares against."""
    if variable.type == "continuous":
        return finite_float(variable, value)
    if variable.type == "integer":
        parsed = finite_float(variable, value)
        if parsed % 1 != 0:
            raise LogValidationError(
                f"Variable '{variable.name}' has non-integer value: value={value!r}."
            )
        return int(parsed)
    if variable.type == "discrete":
        parsed = finite_float(variable, value)
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


def finite_float(variable: VariableConfig, value: object) -> float:
    """Return ``value`` as a finite float or raise ``LogValidationError``."""
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


class _ExpressionEvaluator:
    def __init__(self, context: Mapping[str, object], unsupported: MessageBuilder) -> None:
        self.context = context
        self.unsupported = unsupported

    def evaluate(self, node: ast.AST) -> object:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return self.context[node.id]
        if isinstance(node, ast.BoolOp):
            return self._bool_operation(node)
        if isinstance(node, ast.UnaryOp):
            return self._unary_operation(node)
        if isinstance(node, ast.BinOp):
            return self._binary_operation(node)
        if isinstance(node, ast.Compare):
            return self._comparison(node)
        self._raise_unsupported(node)

    def _bool_operation(self, node: ast.BoolOp) -> bool:
        if isinstance(node.op, ast.And):
            for value in node.values:
                if not bool(self.evaluate(value)):
                    return False
            return True
        if isinstance(node.op, ast.Or):
            for value in node.values:
                if bool(self.evaluate(value)):
                    return True
            return False
        self._raise_unsupported(node)

    def _unary_operation(self, node: ast.UnaryOp) -> object:
        operand = self.evaluate(node.operand)
        if isinstance(node.op, ast.Not):
            return not bool(operand)
        for operator_type, operation in _UNARY_OPERATORS.items():
            if isinstance(node.op, operator_type):
                return operation(operand)
        self._raise_unsupported(node)

    def _binary_operation(self, node: ast.BinOp) -> object:
        left = self.evaluate(node.left)
        right = self.evaluate(node.right)
        for operator_type, operation in _BINARY_OPERATORS.items():
            if isinstance(node.op, operator_type):
                return operation(left, right)
        self._raise_unsupported(node)

    def _comparison(self, node: ast.Compare) -> bool:
        left = self.evaluate(node.left)
        for comparison, comparator in zip(node.ops, node.comparators, strict=True):
            right = self.evaluate(comparator)
            operation = self._comparison_operation(comparison, node)
            if not operation(left, right):
                return False
            left = right
        return True

    def _comparison_operation(self, comparison: ast.cmpop, node: ast.AST):
        for operator_type, operation in _COMPARISON_OPERATORS.items():
            if isinstance(comparison, operator_type):
                return operation
        self._raise_unsupported(node)

    def _raise_unsupported(self, node: ast.AST) -> None:
        raise LogValidationError(self.unsupported(type(node).__name__))


class _ExpressionValidator(ast.NodeVisitor):
    def __init__(
        self,
        variable_names: set[str],
        unknown_name: MessageBuilder,
        unsupported: MessageBuilder,
    ) -> None:
        self.variable_names = variable_names
        self.unknown_name = unknown_name
        self.unsupported = unsupported

    def visit_Expression(self, node: ast.Expression) -> None:
        self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> None:
        if not isinstance(node.value, str | int | float | bool):
            self._raise_unsupported(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id not in self.variable_names:
            raise ConfigError(self.unknown_name(node.id))

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
        raise ConfigError(self.unsupported(type(node).__name__))
