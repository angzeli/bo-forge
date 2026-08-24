"""Reusable constraint, numeric, and design-value validation rules."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import replace

import pandas as pd

from bo_forge.config import CampaignConfig, VariableConfig, fidelity_values_match
from bo_forge.constraints import constraint_variable_names, constraint_violations_for_row
from bo_forge.errors import LogValidationError


def _validate_constraints(config: CampaignConfig, df: pd.DataFrame) -> None:
    if config.is_structured_campaign:
        _validate_structured_constraints(config, df)
        return
    for _, row in df.iterrows():
        violations = constraint_violations_for_row(config, row)
        if violations:
            constraint = violations[0]
            row_id = str(row["row_id"])
            raise LogValidationError(
                f"Row '{row_id}' violates constraint '{constraint.name}': "
                f"{constraint.expression}."
            )


def _validate_structured_constraints(config: CampaignConfig, df: pd.DataFrame) -> None:
    references = {
        constraint.name: constraint_variable_names(constraint.expression)
        for constraint in config.constraints
    }
    active_by_stage = {
        stage.name: set(stage.variables)
        for stage in config.stages
    }
    for _, row in df.iterrows():
        active_variables = active_by_stage[str(row["stage"])]
        applicable_constraints = tuple(
            constraint
            for constraint in config.constraints
            if references[constraint.name].issubset(active_variables)
        )
        if not applicable_constraints:
            continue
        stage_variables = tuple(
            variable
            for variable in config.variables
            if variable.name in active_variables
        )
        stage_config = replace(
            config,
            variables=stage_variables,
            constraints=applicable_constraints,
        )
        violations = constraint_violations_for_row(stage_config, row)
        if violations:
            constraint = violations[0]
            row_id = str(row["row_id"])
            raise LogValidationError(
                f"Row '{row_id}' violates constraint '{constraint.name}': "
                f"{constraint.expression}."
            )


def _validate_cost_expression(config: CampaignConfig, df: pd.DataFrame) -> None:
    if config.cost is None:
        return

    from bo_forge.costs import evaluate_cost

    for _, row in df.iterrows():
        expected = evaluate_cost(config, row)
        estimate = row["cost_estimate"]
        if _is_blank_value(estimate):
            continue
        actual = float(estimate)
        if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9):
            row_id = str(row["row_id"])
            raise LogValidationError(
                f"Row '{row_id}' has cost_estimate inconsistent with cost expression: "
                f"cost_estimate={actual:g}, expected={expected:g}."
            )


def _validate_nullable_numeric_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        blank = _blank_mask(df[column])
        numeric = pd.to_numeric(df.loc[~blank, column], errors="coerce")
        invalid = numeric.isna() | ~numeric.map(math.isfinite)
        if invalid.any():
            row_id = str(df.loc[~blank].loc[invalid, "row_id"].iloc[0])
            value = df.loc[~blank].loc[invalid, column].iloc[0]
            raise LogValidationError(
                f"Row '{row_id}' has non-finite numeric value for column "
                f"'{column}': value={value!r}."
            )


def _validate_nonnegative_nullable_numeric_columns(
    df: pd.DataFrame,
    columns: Iterable[str],
) -> None:
    for column in columns:
        blank = _blank_mask(df[column])
        numeric = pd.to_numeric(df.loc[~blank, column], errors="coerce")
        invalid = numeric.isna() | ~numeric.map(math.isfinite)
        if invalid.any():
            row_id = str(df.loc[~blank].loc[invalid, "row_id"].iloc[0])
            value = df.loc[~blank].loc[invalid, column].iloc[0]
            raise LogValidationError(
                f"Row '{row_id}' has non-finite numeric value for column "
                f"'{column}': value={value!r}."
            )
        negative = numeric < 0
        if negative.any():
            row_id = str(df.loc[~blank].loc[negative, "row_id"].iloc[0])
            value = numeric.loc[negative].iloc[0]
            raise LogValidationError(
                f"Row '{row_id}' has negative value for column '{column}': value={value:g}."
            )


def _validate_finite_nullable_numeric_columns(
    df: pd.DataFrame,
    columns: Iterable[str],
) -> None:
    for column in columns:
        blank = _blank_mask(df[column])
        numeric = pd.to_numeric(df.loc[~blank, column], errors="coerce")
        invalid = numeric.isna() | ~numeric.map(math.isfinite)
        if invalid.any():
            row_id = str(df.loc[~blank].loc[invalid, "row_id"].iloc[0])
            value = df.loc[~blank].loc[invalid, column].iloc[0]
            raise LogValidationError(
                f"Row '{row_id}' has non-finite numeric value for column "
                f"'{column}': value={value!r}."
            )


def _blank_mask(series: pd.Series) -> pd.Series:
    return series.isna() | (series.astype(str).str.strip() == "")


def _is_blank_value(value: object) -> bool:
    return pd.isna(value) or str(value).strip() == ""


def _normalise_variable_value(variable: VariableConfig, value: object) -> object:
    if variable.type == "continuous":
        return round(_finite_float(variable, value), 12)
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


def _normalise_design_value(
    config: CampaignConfig,
    variable: VariableConfig,
    value: object,
) -> object:
    normalised = _normalise_variable_value(variable, value)
    if (
        config.fidelity is None
        or config.fidelity.levels is None
        or variable.name != config.fidelity.variable
    ):
        return normalised
    numeric = float(normalised)
    for level in config.fidelity.levels:
        if fidelity_values_match(numeric, level):
            return round(level, 12)
    return normalised


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


def _required_bound(variable: VariableConfig, key: str) -> float:
    value = variable.lower if key == "lower" else variable.upper
    if value is None:
        raise LogValidationError(f"Variable '{variable.name}' is missing bound '{key}'.")
    return float(value)
