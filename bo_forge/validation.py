"""Validation helpers for campaign logs."""

from __future__ import annotations

import math
from collections.abc import Iterable

import pandas as pd

from bo_forge.config import CampaignConfig, VariableConfig
from bo_forge.constraints import constraint_violations_for_row
from bo_forge.errors import LogValidationError

BASE_COLUMNS = ["row_id", "iteration", "status", "source"]
RESULT_COLUMNS = ["predicted_mean", "predicted_std", "acquisition"]
VALID_STATUSES = {"suggested", "observed"}
VALID_SOURCES = {"manual", "random", "sobol", "log_ei", "qlog_ei"}


def canonical_columns(config: CampaignConfig) -> list[str]:
    """Return canonical CSV columns for a campaign."""
    return [
        *BASE_COLUMNS,
        *config.variable_names,
        config.objective.name,
        *RESULT_COLUMNS,
    ]


def validate_campaign_data(config: CampaignConfig, df: pd.DataFrame) -> None:
    """Validate a campaign log DataFrame against the canonical CSV schema."""
    _validate_columns(config, df)
    if df.empty:
        return

    _validate_row_id(df)
    _validate_iteration(df)
    _validate_status(df)
    _validate_source(df)
    _validate_variables(config, df)
    _validate_constraints(config, df)
    _validate_objective(config, df)
    _validate_nullable_numeric_columns(df, RESULT_COLUMNS)


def get_observed_data(config: CampaignConfig, df: pd.DataFrame) -> pd.DataFrame:
    """Return observed rows after validating the campaign log."""
    validate_campaign_data(config, df)
    return df.loc[df["status"] == "observed"].copy()


def has_pending_suggestions(df: pd.DataFrame) -> bool:
    """Return True when a log contains unresolved suggested rows."""
    if "status" not in df.columns or df.empty:
        return False
    return bool((df["status"] == "suggested").any())


def design_tuples(config: CampaignConfig, df: pd.DataFrame) -> set[tuple[object, ...]]:
    """Return existing variable rows as stable typed tuples for duplicate checks."""
    if df.empty:
        return set()
    return {
        design_key_for_values(
            config,
            [row[variable.name] for variable in config.variables],
        )
        for _, row in df.iterrows()
    }


def design_key_for_values(
    config: CampaignConfig,
    values: Iterable[object],
) -> tuple[object, ...]:
    """Return one stable typed design key for user-facing variable values."""
    return tuple(
        _normalise_variable_value(variable, value)
        for variable, value in zip(config.variables, values, strict=True)
    )


def next_iteration(df: pd.DataFrame) -> int:
    """Return the next campaign iteration index."""
    if df.empty or "iteration" not in df.columns:
        return 0
    iteration = pd.to_numeric(df["iteration"], errors="coerce")
    if iteration.dropna().empty:
        return 0
    return int(iteration.max()) + 1


def _validate_columns(config: CampaignConfig, df: pd.DataFrame) -> None:
    expected = canonical_columns(config)
    actual = list(df.columns)
    if actual != expected:
        missing = [column for column in expected if column not in actual]
        extra = [column for column in actual if column not in expected]
        if missing:
            raise LogValidationError(f"Campaign log is missing required columns: {missing}.")
        if extra:
            raise LogValidationError(f"Campaign log has unsupported extra columns: {extra}.")
        raise LogValidationError(
            "Campaign log columns are not in canonical order: "
            f"expected={expected}, actual={actual}."
        )


def _validate_row_id(df: pd.DataFrame) -> None:
    row_ids = df["row_id"].astype(str)
    blank_mask = row_ids.str.strip() == ""
    if blank_mask.any():
        row_number = int(blank_mask[blank_mask].index[0])
        raise LogValidationError(f"Row at index {row_number} has blank row_id.")
    duplicated = row_ids[row_ids.duplicated()]
    if not duplicated.empty:
        raise LogValidationError(f"Duplicate row_id '{duplicated.iloc[0]}'.")


def _validate_iteration(df: pd.DataFrame) -> None:
    iteration = pd.to_numeric(df["iteration"], errors="coerce")
    invalid = iteration.isna() | (iteration < 0) | (iteration % 1 != 0)
    if invalid.any():
        row_id = str(df.loc[invalid, "row_id"].iloc[0])
        value = df.loc[invalid, "iteration"].iloc[0]
        raise LogValidationError(
            f"Row '{row_id}' has invalid iteration '{value}'. Expected a non-negative integer."
        )


def _validate_status(df: pd.DataFrame) -> None:
    invalid = ~df["status"].isin(VALID_STATUSES)
    if invalid.any():
        row_id = str(df.loc[invalid, "row_id"].iloc[0])
        value = df.loc[invalid, "status"].iloc[0]
        raise LogValidationError(
            f"Row '{row_id}' has invalid status '{value}'. "
            f"Expected one of {sorted(VALID_STATUSES)}."
        )


def _validate_source(df: pd.DataFrame) -> None:
    invalid = ~df["source"].isin(VALID_SOURCES)
    if invalid.any():
        row_id = str(df.loc[invalid, "row_id"].iloc[0])
        value = df.loc[invalid, "source"].iloc[0]
        raise LogValidationError(
            f"Row '{row_id}' has invalid source '{value}'. "
            f"Expected one of {sorted(VALID_SOURCES)}."
        )


def _validate_variables(config: CampaignConfig, df: pd.DataFrame) -> None:
    for variable in config.variables:
        if variable.type == "categorical":
            _validate_categorical_variable(variable, df)
            continue

        numeric = pd.to_numeric(df[variable.name], errors="coerce")
        invalid_numeric = numeric.isna() | ~numeric.map(math.isfinite)
        if invalid_numeric.any():
            row_id = str(df.loc[invalid_numeric, "row_id"].iloc[0])
            value = df.loc[invalid_numeric, variable.name].iloc[0]
            raise LogValidationError(
                f"Row '{row_id}' has non-numeric value for variable "
                f"'{variable.name}': value={value!r}."
            )

        if variable.type == "continuous":
            _validate_numeric_bounds(variable, df, numeric)
        elif variable.type == "integer":
            _validate_integer_variable(variable, df, numeric)
        elif variable.type == "discrete":
            _validate_discrete_variable(variable, df, numeric)
        else:
            raise LogValidationError(
                f"Variable '{variable.name}' has unsupported type '{variable.type}'."
            )


def _validate_numeric_bounds(
    variable: VariableConfig,
    df: pd.DataFrame,
    numeric: pd.Series,
) -> None:
    lower = _required_bound(variable, "lower")
    upper = _required_bound(variable, "upper")
    below = numeric < lower
    above = numeric > upper
    if below.any() or above.any():
        invalid = below | above
        row_id = str(df.loc[invalid, "row_id"].iloc[0])
        value = numeric.loc[invalid].iloc[0]
        raise LogValidationError(
            f"Row '{row_id}' has variable '{variable.name}' outside bounds: "
            f"value={value:g}, lower={lower:g}, upper={upper:g}."
        )


def _validate_integer_variable(
    variable: VariableConfig,
    df: pd.DataFrame,
    numeric: pd.Series,
) -> None:
    non_integer = numeric % 1 != 0
    if non_integer.any():
        row_id = str(df.loc[non_integer, "row_id"].iloc[0])
        value = df.loc[non_integer, variable.name].iloc[0]
        raise LogValidationError(
            f"Row '{row_id}' has non-integer value for variable "
            f"'{variable.name}': value={value!r}."
        )
    _validate_numeric_bounds(variable, df, numeric)


def _validate_discrete_variable(
    variable: VariableConfig,
    df: pd.DataFrame,
    numeric: pd.Series,
) -> None:
    allowed = [float(value) for value in variable.values]
    valid = numeric.map(
        lambda value: any(
            math.isclose(float(value), allowed_value, rel_tol=1e-12, abs_tol=1e-12)
            for allowed_value in allowed
        )
    )
    invalid = ~valid
    if invalid.any():
        row_id = str(df.loc[invalid, "row_id"].iloc[0])
        value = df.loc[invalid, variable.name].iloc[0]
        raise LogValidationError(
            f"Row '{row_id}' has value outside configured choices for variable "
            f"'{variable.name}': value={value!r}, allowed={allowed}."
        )


def _validate_categorical_variable(variable: VariableConfig, df: pd.DataFrame) -> None:
    allowed = set(str(value) for value in variable.values)
    for index, value in df[variable.name].items():
        if not isinstance(value, str) or value == "" or value.strip() != value:
            row_id = str(df.at[index, "row_id"])
            raise LogValidationError(
                f"Row '{row_id}' has blank or whitespace-padded categorical value "
                f"for variable '{variable.name}': value={value!r}."
            )
        if value not in allowed:
            row_id = str(df.at[index, "row_id"])
            raise LogValidationError(
                f"Row '{row_id}' has value outside configured choices for variable "
                f"'{variable.name}': value={value!r}, allowed={sorted(allowed)}."
            )


def _validate_objective(config: CampaignConfig, df: pd.DataFrame) -> None:
    objective_name = config.objective.name
    values = df[objective_name]
    blank = _blank_mask(values)
    observed = df["status"] == "observed"
    suggested = df["status"] == "suggested"

    missing_observed = observed & blank
    if missing_observed.any():
        row_id = str(df.loc[missing_observed, "row_id"].iloc[0])
        raise LogValidationError(
            f"Row '{row_id}' has status='observed' but objective "
            f"'{objective_name}' is blank."
        )

    filled_suggested = suggested & ~blank
    if filled_suggested.any():
        row_id = str(df.loc[filled_suggested, "row_id"].iloc[0])
        value = df.loc[filled_suggested, objective_name].iloc[0]
        raise LogValidationError(
            f"Row '{row_id}' has status='suggested' but objective "
            f"'{objective_name}' is filled: value={value!r}."
        )

    numeric = pd.to_numeric(values.loc[observed], errors="coerce")
    invalid_numeric = numeric.isna()
    if invalid_numeric.any():
        row_id = str(df.loc[observed].loc[invalid_numeric, "row_id"].iloc[0])
        value = df.loc[observed].loc[invalid_numeric, objective_name].iloc[0]
        raise LogValidationError(
            f"Row '{row_id}' has non-numeric objective '{objective_name}': value={value!r}."
        )
    non_finite = ~numeric.map(math.isfinite)
    if non_finite.any():
        row_id = str(df.loc[observed].loc[non_finite, "row_id"].iloc[0])
        value = df.loc[observed].loc[non_finite, objective_name].iloc[0]
        raise LogValidationError(
            f"Row '{row_id}' has non-finite objective '{objective_name}': value={value!r}."
        )


def _validate_constraints(config: CampaignConfig, df: pd.DataFrame) -> None:
    for _, row in df.iterrows():
        violations = constraint_violations_for_row(config, row)
        if violations:
            constraint = violations[0]
            row_id = str(row["row_id"])
            raise LogValidationError(
                f"Row '{row_id}' violates constraint '{constraint.name}': "
                f"{constraint.expression}."
            )


def _validate_nullable_numeric_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        blank = _blank_mask(df[column])
        numeric = pd.to_numeric(df.loc[~blank, column], errors="coerce")
        invalid = numeric.isna()
        if invalid.any():
            row_id = str(df.loc[~blank].loc[invalid, "row_id"].iloc[0])
            value = df.loc[~blank].loc[invalid, column].iloc[0]
            raise LogValidationError(
                f"Row '{row_id}' has non-numeric value for column '{column}': value={value!r}."
            )


def _blank_mask(series: pd.Series) -> pd.Series:
    return series.isna() | (series.astype(str).str.strip() == "")


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
