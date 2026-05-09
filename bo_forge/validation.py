"""Validation helpers for campaign logs."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from bo_forge.config import CampaignConfig
from bo_forge.errors import LogValidationError

BASE_COLUMNS = ["row_id", "iteration", "status", "source"]
RESULT_COLUMNS = ["predicted_mean", "predicted_std", "acquisition"]
VALID_STATUSES = {"suggested", "observed"}
VALID_SOURCES = {"manual", "sobol", "log_ei", "qlog_ei"}


def canonical_columns(config: CampaignConfig) -> list[str]:
    """Return canonical CSV columns for a campaign."""
    return [
        *BASE_COLUMNS,
        *config.variable_names,
        config.objective.name,
        *RESULT_COLUMNS,
    ]


def validate_campaign_data(config: CampaignConfig, df: pd.DataFrame) -> None:
    """Validate a campaign log DataFrame against the MVP v0.1 schema."""
    _validate_columns(config, df)
    if df.empty:
        return

    _validate_row_id(df)
    _validate_iteration(df)
    _validate_status(df)
    _validate_source(df)
    _validate_variables(config, df)
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


def design_tuples(config: CampaignConfig, df: pd.DataFrame) -> set[tuple[float, ...]]:
    """Return existing variable rows as stable float tuples for duplicate checks."""
    if df.empty:
        return set()
    return {
        tuple(round(float(row[name]), 12) for name in config.variable_names)
        for _, row in df.iterrows()
    }


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
        numeric = pd.to_numeric(df[variable.name], errors="coerce")
        invalid_numeric = numeric.isna()
        if invalid_numeric.any():
            row_id = str(df.loc[invalid_numeric, "row_id"].iloc[0])
            value = df.loc[invalid_numeric, variable.name].iloc[0]
            raise LogValidationError(
                f"Row '{row_id}' has non-numeric value for variable "
                f"'{variable.name}': value={value!r}."
            )

        below = numeric < variable.lower
        above = numeric > variable.upper
        if below.any() or above.any():
            invalid = below | above
            row_id = str(df.loc[invalid, "row_id"].iloc[0])
            value = numeric.loc[invalid].iloc[0]
            raise LogValidationError(
                f"Row '{row_id}' has variable '{variable.name}' outside bounds: "
                f"value={value:g}, lower={variable.lower:g}, upper={variable.upper:g}."
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
