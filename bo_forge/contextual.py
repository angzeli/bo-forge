"""Helpers for contextual single-objective BO campaigns."""

from __future__ import annotations

import math
from collections.abc import Mapping
from itertools import product

from bo_forge.config import CampaignConfig, VariableConfig
from bo_forge.errors import SuggestionError
from bo_forge.transforms import encoded_feature_indices, values_to_unit_cube


def resolve_context_values(
    config: CampaignConfig,
    supplied_values: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return normalized context values from defaults plus supplied overrides."""
    supplied = dict(supplied_values or {})
    if config.context is None:
        if supplied:
            raise SuggestionError("Context values are only valid for contextual campaigns.")
        return {}

    context_names = config.context_variable_names
    unknown = sorted(set(supplied) - set(context_names))
    if unknown:
        raise SuggestionError(
            f"Unknown context variable(s): {unknown}. Expected one of {context_names}."
        )

    resolved = dict(config.context.default_values)
    resolved.update(supplied)
    missing = [name for name in context_names if name not in resolved]
    if missing:
        raise SuggestionError(
            "Contextual suggestions require values for every context variable: "
            f"missing={missing}. Pass context_values=... or CLI --context NAME=VALUE."
        )

    variables = {variable.name: variable for variable in config.variables}
    return {
        name: normalize_context_value(variables[name], resolved[name], f"context '{name}'")
        for name in context_names
    }


def normalize_context_value(
    variable: VariableConfig,
    value: object,
    context: str,
) -> object:
    """Normalize one context value using the existing variable semantics."""
    if variable.type == "continuous":
        parsed = _finite_float(value, context)
        lower = _required_bound(variable, "lower")
        upper = _required_bound(variable, "upper")
        if parsed < lower or parsed > upper:
            raise SuggestionError(
                f"{context} is outside variable '{variable.name}' bounds: "
                f"value={parsed:g}, lower={lower:g}, upper={upper:g}."
            )
        return parsed
    if variable.type == "integer":
        parsed = _finite_float(value, context)
        if parsed % 1 != 0:
            raise SuggestionError(f"{context} must be integer-valued: value={value!r}.")
        lower = int(_required_bound(variable, "lower"))
        upper = int(_required_bound(variable, "upper"))
        if parsed < lower or parsed > upper:
            raise SuggestionError(
                f"{context} is outside variable '{variable.name}' bounds: "
                f"value={parsed:g}, lower={lower:g}, upper={upper:g}."
            )
        return int(parsed)
    if variable.type == "discrete":
        parsed = _finite_float(value, context)
        allowed = [float(item) for item in variable.values]
        for allowed_value in allowed:
            if math.isclose(parsed, allowed_value, rel_tol=1e-12, abs_tol=1e-12):
                return float(allowed_value)
        raise SuggestionError(
            f"{context} is not one of variable '{variable.name}' choices: "
            f"value={value!r}, allowed={allowed}."
        )
    if variable.type == "categorical":
        parsed = str(value)
        allowed = [str(item) for item in variable.values]
        if parsed not in allowed:
            raise SuggestionError(
                f"{context} is not one of variable '{variable.name}' choices: "
                f"value={value!r}, allowed={allowed}."
            )
        return parsed
    raise SuggestionError(f"Variable '{variable.name}' has unsupported type '{variable.type}'.")


def contextual_fixed_feature_assignments(
    config: CampaignConfig,
    context_values: Mapping[str, object],
) -> list[dict[int, float]]:
    """Return fixed-feature assignments for context and decision categoricals."""
    context_features = _context_fixed_features(config, context_values)
    categorical_decision_variables = [
        variable
        for variable in config.variables
        if variable.type == "categorical" and variable.name not in config.context_variable_names
    ]
    if not categorical_decision_variables:
        return [context_features]

    feature_indices = encoded_feature_indices(config)
    assignments: list[dict[int, float]] = []
    for category_indices in product(
        *[range(len(variable.values)) for variable in categorical_decision_variables]
    ):
        fixed_features = dict(context_features)
        for variable, active_index in zip(
            categorical_decision_variables,
            category_indices,
            strict=True,
        ):
            for offset, model_index in enumerate(feature_indices[variable.name]):
                fixed_features[model_index] = 1.0 if offset == active_index else 0.0
        assignments.append(fixed_features)
    return assignments


def contextual_categorical_combination_count(config: CampaignConfig) -> int:
    """Return categorical combinations after excluding fixed context variables."""
    context_names = set(config.context_variable_names)
    count = 1
    for variable in config.variables:
        if variable.type == "categorical" and variable.name not in context_names:
            count *= len(variable.values)
    return count


def apply_context_to_candidate(
    config: CampaignConfig,
    candidate: tuple[object, ...],
    context_values: Mapping[str, object],
) -> tuple[object, ...]:
    """Return a candidate tuple with context variable values fixed."""
    values = list(candidate)
    index_by_name = {variable.name: index for index, variable in enumerate(config.variables)}
    for name, value in context_values.items():
        values[index_by_name[name]] = value
    return tuple(values)


def context_metadata(
    config: CampaignConfig,
    context_values: Mapping[str, object],
) -> dict[str, object]:
    """Return staged-bundle-safe context metadata in configured order."""
    if config.context is None:
        return {}
    return {name: context_values[name] for name in config.context_variable_names}


def _context_fixed_features(
    config: CampaignConfig,
    context_values: Mapping[str, object],
) -> dict[int, float]:
    row = []
    for variable in config.variables:
        if variable.name in context_values:
            row.append(context_values[variable.name])
        else:
            row.append(_default_variable_value(variable))
    encoded = values_to_unit_cube(config, [row]).squeeze(0)
    feature_indices = encoded_feature_indices(config)
    fixed_features: dict[int, float] = {}
    for name in config.context_variable_names:
        for index in feature_indices[name]:
            fixed_features[index] = float(encoded[index])
    return fixed_features


def _default_variable_value(variable: VariableConfig) -> object:
    if variable.type in {"continuous", "integer"}:
        return _required_bound(variable, "lower")
    if variable.type == "discrete":
        return float(variable.values[0])
    if variable.type == "categorical":
        return str(variable.values[0])
    raise SuggestionError(f"Variable '{variable.name}' has unsupported type '{variable.type}'.")


def _required_bound(variable: VariableConfig, key: str) -> float:
    value = variable.lower if key == "lower" else variable.upper
    if value is None:
        raise SuggestionError(f"Variable '{variable.name}' is missing bound '{key}'.")
    return float(value)


def _finite_float(value: object, context: str) -> float:
    if isinstance(value, bool):
        raise SuggestionError(f"{context} must be numeric, not a boolean.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise SuggestionError(f"{context} must be numeric: value={value!r}.") from exc
    if not math.isfinite(parsed):
        raise SuggestionError(f"{context} must be finite: value={value!r}.")
    return parsed
