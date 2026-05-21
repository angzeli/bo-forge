"""Campaign log loading and write transitions."""

from __future__ import annotations

import math
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd

from bo_forge.config import CampaignConfig
from bo_forge.errors import LogValidationError, LogWriteError
from bo_forge.io import empty_campaign_log
from bo_forge.validation import (
    BASE_COLUMNS,
    RESULT_COLUMNS,
    VALID_SOURCES,
    VALID_STATUSES,
    validate_campaign_data,
)


def load_campaign_log(path: str | Path, config: CampaignConfig) -> pd.DataFrame:
    """Load and validate a campaign log, returning an empty canonical log if missing."""
    log_path = Path(path)
    if not log_path.exists():
        return empty_campaign_log(config)

    df = _read_csv(log_path)
    validate_campaign_data(config, df)
    return df


def append_suggestions(log_path: str | Path, suggestions: pd.DataFrame) -> None:
    """Append suggested rows to a campaign log and validate the written file."""
    path = Path(log_path)
    if suggestions.empty:
        raise LogWriteError("append_suggestions() received an empty suggestions DataFrame.")

    _validate_suggestions_for_append(suggestions)

    if path.exists():
        existing = _read_csv(path)
        _validate_structural_log(existing)
        if list(existing.columns) != list(suggestions.columns):
            raise LogWriteError(
                "Suggestions columns do not match existing log columns: "
                f"expected={list(existing.columns)}, actual={list(suggestions.columns)}."
            )
    else:
        existing = pd.DataFrame(columns=suggestions.columns)

    duplicated = set(existing["row_id"].astype(str)) & set(suggestions["row_id"].astype(str))
    if duplicated:
        row_id = sorted(duplicated)[0]
        raise LogWriteError(f"Cannot append suggestions with duplicate row_id '{row_id}'.")

    combined = pd.concat([existing, suggestions], ignore_index=True)
    _validate_structural_log(combined)
    _atomic_write_and_validate(path, combined)


def mark_observed(log_path: str | Path, row_id: str, objective_value: float) -> None:
    """Mark a suggested row as observed by filling the objective value in place."""
    path = Path(log_path)
    if not path.exists():
        raise LogWriteError(
            f"Cannot mark row '{row_id}' observed because log '{path}' does not exist."
        )
    if not isinstance(row_id, str) or not row_id.strip():
        raise LogWriteError("row_id must be a non-empty string.")

    objective = _objective_column_from_columns(_read_csv(path).columns)
    try:
        objective_float = float(objective_value)
    except (TypeError, ValueError) as exc:
        raise LogWriteError(
            f"Objective value for row '{row_id}' must be numeric: value={objective_value!r}."
        ) from exc
    if not math.isfinite(objective_float):
        raise LogWriteError(
            f"Objective value for row '{row_id}' must be finite: value={objective_value!r}."
        )

    df = _read_csv(path)
    _validate_structural_log(df)

    matches = df["row_id"].astype(str) == row_id
    if not matches.any():
        raise LogWriteError(f"Cannot mark row '{row_id}' observed because row_id was not found.")
    if matches.sum() > 1:
        raise LogWriteError(f"Cannot mark row '{row_id}' observed because row_id is duplicated.")

    index = matches[matches].index[0]
    status = str(df.at[index, "status"])
    if status != "suggested":
        raise LogWriteError(
            f"Cannot mark row '{row_id}' observed because status is '{status}', not 'suggested'."
        )

    objective_cell = df.at[index, objective]
    if not _is_blank(objective_cell):
        raise LogWriteError(
            f"Cannot mark row '{row_id}' observed because objective '{objective}' "
            f"is already filled: value={objective_cell!r}."
        )

    df.at[index, objective] = f"{objective_float:.17g}"
    df.at[index, "status"] = "observed"
    _validate_structural_log(df)
    _atomic_write_and_validate(path, df)


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, keep_default_na=False)
    except OSError as exc:
        raise LogWriteError(f"Could not read campaign log '{path}': {exc}") from exc
    except pd.errors.ParserError as exc:
        raise LogWriteError(f"Could not parse campaign log '{path}': {exc}") from exc


def _atomic_write_and_validate(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
        df.to_csv(handle, index=False, float_format="%.17g")

    try:
        temp_df = _read_csv(temp_path)
        _validate_structural_log(temp_df)
        temp_path.replace(path)
        post_write_df = _read_csv(path)
        _validate_structural_log(post_write_df)
    except (OSError, LogValidationError, LogWriteError) as exc:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise LogWriteError(
            f"Post-write validation failed for campaign log '{path}': {exc}"
        ) from exc


def _validate_suggestions_for_append(suggestions: pd.DataFrame) -> None:
    _validate_structural_log(suggestions)
    invalid = suggestions["status"] != "suggested"
    if invalid.any():
        row_id = str(suggestions.loc[invalid, "row_id"].iloc[0])
        status = suggestions.loc[invalid, "status"].iloc[0]
        raise LogWriteError(
            f"append_suggestions() expected status='suggested' for row '{row_id}', "
            f"got status={status!r}."
        )


def _validate_structural_log(df: pd.DataFrame) -> None:
    _validate_structural_columns(df)
    if df.empty:
        return

    row_ids = df["row_id"].astype(str)
    blank = row_ids.str.strip() == ""
    if blank.any():
        row_number = int(blank[blank].index[0])
        raise LogValidationError(f"Row at index {row_number} has blank row_id.")
    duplicated = row_ids[row_ids.duplicated()]
    if not duplicated.empty:
        raise LogValidationError(f"Duplicate row_id '{duplicated.iloc[0]}'.")

    iteration = pd.to_numeric(df["iteration"], errors="coerce")
    invalid_iteration = iteration.isna() | (iteration < 0) | (iteration % 1 != 0)
    if invalid_iteration.any():
        row_id = str(df.loc[invalid_iteration, "row_id"].iloc[0])
        value = df.loc[invalid_iteration, "iteration"].iloc[0]
        raise LogValidationError(
            f"Row '{row_id}' has invalid iteration '{value}'. Expected a non-negative integer."
        )

    invalid_status = ~df["status"].isin(VALID_STATUSES)
    if invalid_status.any():
        row_id = str(df.loc[invalid_status, "row_id"].iloc[0])
        value = df.loc[invalid_status, "status"].iloc[0]
        raise LogValidationError(f"Row '{row_id}' has invalid status '{value}'.")

    invalid_source = ~df["source"].isin(VALID_SOURCES)
    if invalid_source.any():
        row_id = str(df.loc[invalid_source, "row_id"].iloc[0])
        value = df.loc[invalid_source, "source"].iloc[0]
        raise LogValidationError(f"Row '{row_id}' has invalid source '{value}'.")

    variable_columns, objective = _variable_and_objective_columns(df.columns)
    for column in variable_columns:
        invalid = df[column].map(_is_blank)
        if invalid.any():
            row_id = str(df.loc[invalid, "row_id"].iloc[0])
            value = df.loc[invalid, column].iloc[0]
            raise LogValidationError(
                f"Row '{row_id}' has blank value for variable '{column}': value={value!r}."
            )

    objective_blank = df[objective].map(_is_blank)
    observed = df["status"] == "observed"
    suggested = df["status"] == "suggested"
    missing_observed = observed & objective_blank
    if missing_observed.any():
        row_id = str(df.loc[missing_observed, "row_id"].iloc[0])
        raise LogValidationError(
            f"Row '{row_id}' has status='observed' but objective '{objective}' is blank."
        )
    filled_suggested = suggested & ~objective_blank
    if filled_suggested.any():
        row_id = str(df.loc[filled_suggested, "row_id"].iloc[0])
        value = df.loc[filled_suggested, objective].iloc[0]
        raise LogValidationError(
            f"Row '{row_id}' has status='suggested' but objective '{objective}' "
            f"is filled: value={value!r}."
        )

    numeric_objective = pd.to_numeric(df.loc[observed, objective], errors="coerce")
    invalid_objective = numeric_objective.isna()
    if invalid_objective.any():
        row_id = str(df.loc[observed].loc[invalid_objective, "row_id"].iloc[0])
        value = df.loc[observed].loc[invalid_objective, objective].iloc[0]
        raise LogValidationError(
            f"Row '{row_id}' has non-numeric objective '{objective}': value={value!r}."
        )

    for column in RESULT_COLUMNS:
        blank_result = df[column].map(_is_blank)
        numeric_result = pd.to_numeric(df.loc[~blank_result, column], errors="coerce")
        invalid_result = numeric_result.isna()
        if invalid_result.any():
            row_id = str(df.loc[~blank_result].loc[invalid_result, "row_id"].iloc[0])
            value = df.loc[~blank_result].loc[invalid_result, column].iloc[0]
            raise LogValidationError(
                f"Row '{row_id}' has non-numeric value for column '{column}': value={value!r}."
            )


def _validate_structural_columns(df: pd.DataFrame) -> None:
    columns = list(df.columns)
    minimum_columns = [*BASE_COLUMNS, "variable", "objective", *RESULT_COLUMNS]
    if len(columns) < len(minimum_columns):
        raise LogValidationError(
            "Campaign log has too few columns for canonical schema: "
            f"columns={columns}."
        )
    if columns[: len(BASE_COLUMNS)] != BASE_COLUMNS:
        raise LogValidationError(
            "Campaign log must start with canonical columns "
            f"{BASE_COLUMNS}: actual_start={columns[: len(BASE_COLUMNS)]}."
        )
    if columns[-len(RESULT_COLUMNS) :] != RESULT_COLUMNS:
        raise LogValidationError(
            "Campaign log must end with canonical result columns "
            f"{RESULT_COLUMNS}: actual_end={columns[-len(RESULT_COLUMNS):]}."
        )
    variable_columns, objective = _variable_and_objective_columns(columns)
    if not variable_columns:
        raise LogValidationError("Campaign log must contain at least one variable column.")
    if objective in variable_columns:
        raise LogValidationError(f"Objective column '{objective}' is duplicated as a variable.")


def _variable_and_objective_columns(columns: pd.Index | list[str]) -> tuple[list[str], str]:
    column_list = list(columns)
    middle = column_list[len(BASE_COLUMNS) : -len(RESULT_COLUMNS)]
    if len(middle) < 2:
        raise LogValidationError(
            "Campaign log must contain at least one variable column and one objective column."
        )
    return middle[:-1], middle[-1]


def _objective_column_from_columns(columns: pd.Index | list[str]) -> str:
    return _variable_and_objective_columns(columns)[1]


def _is_blank(value: object) -> bool:
    if pd.isna(value):
        return True
    return str(value).strip() == ""
