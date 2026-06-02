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
    COST_COLUMNS,
    REPLICATE_COLUMNS,
    RESULT_COLUMNS,
    REVIEW_COLUMNS,
    UTILITY_COLUMNS,
    VALID_MULTI_OBJECTIVE_SOURCES,
    VALID_REVIEW_STATUSES,
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


def append_suggestions(
    log_path: str | Path,
    suggestions: pd.DataFrame,
    config: CampaignConfig | None = None,
) -> None:
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
    if config is not None:
        validate_campaign_data(config, combined)
    elif _has_replicate_columns(combined.columns):
        raise LogWriteError(
            "Replicate append requires config-aware validation; use "
            "append_suggestions(..., config=config) or CampaignSession.append_suggestions()."
        )
    _atomic_write_and_validate(path, combined)


def mark_observed(
    log_path: str | Path,
    row_id: str,
    objective_value: float | None = None,
    objective_values: dict[str, float] | None = None,
    actual_cost: float | None = None,
) -> None:
    """Mark a suggested row as observed by filling the objective value in place."""
    path = Path(log_path)
    if not path.exists():
        raise LogWriteError(
            f"Cannot mark row '{row_id}' observed because log '{path}' does not exist."
        )
    if not isinstance(row_id, str) or not row_id.strip():
        raise LogWriteError("row_id must be a non-empty string.")

    columns = _read_csv(path).columns
    objective_columns = _variable_and_objective_columns(columns)[1]
    parsed_objective_values = _parse_mark_observed_objective_values(
        row_id=row_id,
        objective_columns=objective_columns,
        objective_value=objective_value,
        objective_values=objective_values,
    )
    actual_cost_text = None
    if actual_cost is not None:
        try:
            actual_cost_float = float(actual_cost)
        except (TypeError, ValueError) as exc:
            raise LogWriteError(
                f"actual_cost for row '{row_id}' must be numeric: value={actual_cost!r}."
            ) from exc
        if not math.isfinite(actual_cost_float) or actual_cost_float < 0:
            raise LogWriteError(
                f"actual_cost for row '{row_id}' must be finite and >= 0: "
                f"value={actual_cost!r}."
            )
        actual_cost_text = f"{actual_cost_float:.17g}"

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
    if _has_review_columns(df.columns):
        review_status = str(df.at[index, "review_status"])
        if review_status != "accepted":
            raise LogWriteError(
                f"Cannot mark row '{row_id}' observed because review_status is "
                f"'{review_status}', not 'accepted'."
            )
    if actual_cost_text is not None and not _has_cost_columns(df.columns):
        raise LogWriteError(
            f"Cannot record actual_cost for row '{row_id}' because the campaign log "
            "has no cost columns."
        )

    for objective in objective_columns:
        objective_cell = df.at[index, objective]
        if not _is_blank(objective_cell):
            raise LogWriteError(
                f"Cannot mark row '{row_id}' observed because objective '{objective}' "
                f"is already filled: value={objective_cell!r}."
            )

    for objective, objective_float in parsed_objective_values.items():
        df.at[index, objective] = f"{objective_float:.17g}"
    df.at[index, "status"] = "observed"
    if actual_cost_text is not None:
        df.at[index, "cost_actual"] = actual_cost_text
    _validate_structural_log(df)
    _atomic_write_and_validate(path, df)


def _parse_mark_observed_objective_values(
    *,
    row_id: str,
    objective_columns: list[str],
    objective_value: float | None,
    objective_values: dict[str, float] | None,
) -> dict[str, float]:
    if len(objective_columns) == 1:
        if objective_values is not None:
            expected = set(objective_columns)
            actual = set(objective_values)
            if actual != expected:
                raise LogWriteError(
                    "objective_values for a single-objective campaign must contain exactly "
                    f"{sorted(expected)}: actual={sorted(actual)}."
                )
            if objective_value is not None:
                raise LogWriteError(
                    "Pass either objective_value or objective_values, not both."
                )
            return {
                objective_columns[0]: _finite_objective_value(
                    row_id,
                    objective_columns[0],
                    objective_values[objective_columns[0]],
                )
            }
        if objective_value is None:
            raise LogWriteError(
                f"Objective value for row '{row_id}' is required for single-objective logs."
            )
        return {
            objective_columns[0]: _finite_objective_value(
                row_id,
                objective_columns[0],
                objective_value,
            )
        }

    if objective_value is not None:
        raise LogWriteError(
            "objective_value is not valid for multi-objective campaign logs; "
            "pass objective_values with every configured objective."
        )
    if objective_values is None:
        raise LogWriteError(
            "objective_values is required for multi-objective campaign logs."
        )
    expected = set(objective_columns)
    actual = set(objective_values)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise LogWriteError(
            "objective_values keys must exactly match configured objective columns: "
            f"missing={missing}, extra={extra}."
        )
    return {
        objective: _finite_objective_value(row_id, objective, objective_values[objective])
        for objective in objective_columns
    }


def _finite_objective_value(row_id: str, objective: str, value: object) -> float:
    try:
        objective_float = float(value)
    except (TypeError, ValueError) as exc:
        raise LogWriteError(
            f"Objective value for row '{row_id}' and objective '{objective}' must be "
            f"numeric: value={value!r}."
        ) from exc
    if not math.isfinite(objective_float):
        raise LogWriteError(
            f"Objective value for row '{row_id}' and objective '{objective}' must be "
            f"finite: value={value!r}."
        )
    return objective_float


def review_suggestion(
    log_path: str | Path,
    row_id: str,
    decision: str,
    note: str = "",
) -> None:
    """Record a human review decision for one suggested row."""
    path = Path(log_path)
    if not path.exists():
        raise LogWriteError(
            f"Cannot review row '{row_id}' because log '{path}' does not exist."
        )
    if not isinstance(row_id, str) or not row_id.strip():
        raise LogWriteError("row_id must be a non-empty string.")

    decision_map = {
        "accept": "accepted",
        "reject": "rejected",
        "defer": "deferred",
    }
    if decision not in decision_map:
        raise LogWriteError(
            f"Invalid review decision '{decision}'. Expected one of "
            f"{sorted(decision_map)}."
        )
    cleaned_note = str(note).strip()
    if "\n" in cleaned_note or "\r" in cleaned_note:
        raise LogWriteError("review_note cannot contain newline characters.")

    df = _read_csv(path)
    _validate_structural_log(df)
    if not _has_review_columns(df.columns):
        raise LogWriteError("Cannot review suggestions because review is not enabled.")

    matches = df["row_id"].astype(str) == row_id
    if not matches.any():
        raise LogWriteError(f"Cannot review row '{row_id}' because row_id was not found.")
    if matches.sum() > 1:
        raise LogWriteError(f"Cannot review row '{row_id}' because row_id is duplicated.")

    index = matches[matches].index[0]
    status = str(df.at[index, "status"])
    if status != "suggested":
        raise LogWriteError(
            f"Cannot review row '{row_id}' because status is '{status}', not 'suggested'."
        )

    df.at[index, "review_status"] = decision_map[decision]
    df.at[index, "review_note"] = cleaned_note
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
    if _has_review_columns(suggestions.columns):
        invalid_review = suggestions["review_status"] != "pending"
        if invalid_review.any():
            row_id = str(suggestions.loc[invalid_review, "row_id"].iloc[0])
            review_status = suggestions.loc[invalid_review, "review_status"].iloc[0]
            raise LogWriteError(
                f"append_suggestions() expected review_status='pending' for row "
                f"'{row_id}', got review_status={review_status!r}."
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
    if _has_multi_objective_columns(df.columns):
        invalid_source = ~df["source"].isin(VALID_MULTI_OBJECTIVE_SOURCES)
    if invalid_source.any():
        row_id = str(df.loc[invalid_source, "row_id"].iloc[0])
        value = df.loc[invalid_source, "source"].iloc[0]
        raise LogValidationError(f"Row '{row_id}' has invalid source '{value}'.")

    variable_columns, objective_columns = _variable_and_objective_columns(df.columns)
    for column in variable_columns:
        invalid = df[column].map(_is_blank)
        if invalid.any():
            row_id = str(df.loc[invalid, "row_id"].iloc[0])
            value = df.loc[invalid, column].iloc[0]
            raise LogValidationError(
                f"Row '{row_id}' has blank value for variable '{column}': value={value!r}."
            )

    observed = df["status"] == "observed"
    suggested = df["status"] == "suggested"
    for objective in objective_columns:
        objective_blank = df[objective].map(_is_blank)
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
        invalid_objective = numeric_objective.isna() | ~numeric_objective.map(math.isfinite)
        if invalid_objective.any():
            row_id = str(df.loc[observed].loc[invalid_objective, "row_id"].iloc[0])
            value = df.loc[observed].loc[invalid_objective, objective].iloc[0]
            raise LogValidationError(
                f"Row '{row_id}' has non-finite objective '{objective}': value={value!r}."
            )

    if _has_review_columns(df.columns):
        invalid_review = ~df["review_status"].isin(VALID_REVIEW_STATUSES)
        if invalid_review.any():
            row_id = str(df.loc[invalid_review, "row_id"].iloc[0])
            value = df.loc[invalid_review, "review_status"].iloc[0]
            raise LogValidationError(
                f"Row '{row_id}' has invalid review_status '{value}'."
            )
        observed_not_accepted = observed & (df["review_status"] != "accepted")
        if observed_not_accepted.any():
            row_id = str(df.loc[observed_not_accepted, "row_id"].iloc[0])
            raise LogValidationError(
                f"Row '{row_id}' has status='observed' but review_status is not 'accepted'."
            )
        review_newline = df["review_note"].astype(str).str.contains(r"[\r\n]", regex=True)
        if review_newline.any():
            row_id = str(df.loc[review_newline, "row_id"].iloc[0])
            raise LogValidationError(f"Row '{row_id}' has review_note containing a newline.")

    if _has_replicate_columns(df.columns):
        invalid_group = df["replicate_group"].map(
            lambda value: (
                not isinstance(value, str)
                or value == ""
                or value.strip() != value
                or "\n" in value
                or "\r" in value
            )
        )
        if invalid_group.any():
            row_id = str(df.loc[invalid_group, "row_id"].iloc[0])
            value = df.loc[invalid_group, "replicate_group"].iloc[0]
            raise LogValidationError(
                f"Row '{row_id}' has invalid replicate_group: value={value!r}."
            )

        replicate_index = pd.to_numeric(df["replicate_index"], errors="coerce")
        invalid_replicate = (
            replicate_index.isna()
            | (replicate_index < 0)
            | (replicate_index % 1 != 0)
        )
        if invalid_replicate.any():
            row_id = str(df.loc[invalid_replicate, "row_id"].iloc[0])
            value = df.loc[invalid_replicate, "replicate_index"].iloc[0]
            raise LogValidationError(
                f"Row '{row_id}' has invalid replicate_index '{value}'."
            )
        replicate_pairs = (
            df["replicate_group"].astype(str)
            + "\0"
            + replicate_index.astype(int).astype(str)
        )
        duplicated_pair = replicate_pairs[replicate_pairs.duplicated()]
        if not duplicated_pair.empty:
            index = int(duplicated_pair.index[0])
            group = str(df.at[index, "replicate_group"])
            replicate = int(replicate_index.at[index])
            raise LogValidationError(
                f"Duplicate replicate row for replicate_group='{group}', "
                f"replicate_index={replicate}."
            )

    numeric_columns = [*_result_columns_from_columns(df.columns)]
    if _has_cost_columns(df.columns):
        numeric_columns.extend([*COST_COLUMNS, *UTILITY_COLUMNS])

    for column in numeric_columns:
        blank_result = df[column].map(_is_blank)
        numeric_result = pd.to_numeric(df.loc[~blank_result, column], errors="coerce")
        invalid_result = numeric_result.isna() | ~numeric_result.map(math.isfinite)
        if invalid_result.any():
            row_id = str(df.loc[~blank_result].loc[invalid_result, "row_id"].iloc[0])
            value = df.loc[~blank_result].loc[invalid_result, column].iloc[0]
            raise LogValidationError(
                f"Row '{row_id}' has non-finite numeric value for column "
                f"'{column}': value={value!r}."
            )
        if column in set(COST_COLUMNS):
            negative = numeric_result < 0
            if negative.any():
                row_id = str(df.loc[~blank_result].loc[negative, "row_id"].iloc[0])
                value = numeric_result.loc[negative].iloc[0]
                raise LogValidationError(
                    f"Row '{row_id}' has negative value for column '{column}': "
                    f"value={value:g}."
                )


def _validate_structural_columns(df: pd.DataFrame) -> None:
    columns = list(df.columns)
    multi_objective_parts = _multi_objective_parts_from_columns(columns)
    if multi_objective_parts is not None:
        variable_columns, objective_columns, _ = multi_objective_parts
        if not variable_columns:
            raise LogValidationError("Campaign log must contain at least one variable column.")
        if len(objective_columns) < 2:
            raise LogValidationError(
                "Multi-objective campaign log must contain at least two objectives."
            )
        return

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
    has_utility = columns[-len(UTILITY_COLUMNS) :] == UTILITY_COLUMNS
    result_end = len(columns) - (len(UTILITY_COLUMNS) if has_utility else 0)
    if columns[result_end - len(RESULT_COLUMNS) : result_end] != RESULT_COLUMNS:
        raise LogValidationError(
            "Campaign log must end with canonical result columns "
            f"{RESULT_COLUMNS}: actual_end={columns[result_end - len(RESULT_COLUMNS):result_end]}."
        )
    if has_utility and not _has_cost_columns(columns):
        raise LogValidationError("Campaign log has utility column but no cost columns.")
    variable_columns, objective_columns = _variable_and_objective_columns(columns)
    if not variable_columns:
        raise LogValidationError("Campaign log must contain at least one variable column.")
    for objective in objective_columns:
        if objective in variable_columns:
            raise LogValidationError(f"Objective column '{objective}' is duplicated as a variable.")


def _variable_and_objective_columns(
    columns: pd.Index | list[str],
) -> tuple[list[str], list[str]]:
    column_list = list(columns)
    multi_objective_parts = _multi_objective_parts_from_columns(column_list)
    if multi_objective_parts is not None:
        variable_columns, objective_columns, _ = multi_objective_parts
        return variable_columns, objective_columns

    start = len(BASE_COLUMNS)
    if column_list[start : start + len(REVIEW_COLUMNS)] == REVIEW_COLUMNS:
        start += len(REVIEW_COLUMNS)
    if column_list[start : start + len(REPLICATE_COLUMNS)] == REPLICATE_COLUMNS:
        start += len(REPLICATE_COLUMNS)

    has_utility = column_list[-len(UTILITY_COLUMNS) :] == UTILITY_COLUMNS
    result_end = len(column_list) - (len(UTILITY_COLUMNS) if has_utility else 0)
    middle = column_list[start : result_end - len(RESULT_COLUMNS)]
    if middle[-len(COST_COLUMNS) :] == COST_COLUMNS:
        middle = middle[: -len(COST_COLUMNS)]
    elif has_utility:
        raise LogValidationError("Campaign log has utility column but no cost columns.")
    if len(middle) < 2:
        raise LogValidationError(
            "Campaign log must contain at least one variable column and one objective column."
        )
    return middle[:-1], [middle[-1]]


def _objective_column_from_columns(columns: pd.Index | list[str]) -> str:
    objective_columns = _variable_and_objective_columns(columns)[1]
    if len(objective_columns) != 1:
        raise LogWriteError(
            "objective_value is only valid for single-objective campaign logs."
        )
    return objective_columns[0]


def _result_columns_from_columns(columns: pd.Index | list[str]) -> list[str]:
    column_list = list(columns)
    multi_objective_parts = _multi_objective_parts_from_columns(column_list)
    if multi_objective_parts is not None:
        return multi_objective_parts[2]
    return [*RESULT_COLUMNS]


def _has_multi_objective_columns(columns: pd.Index | list[str]) -> bool:
    return _multi_objective_parts_from_columns(columns) is not None


def _multi_objective_parts_from_columns(
    columns: pd.Index | list[str],
) -> tuple[list[str], list[str], list[str]] | None:
    column_list = list(columns)
    if len(column_list) < len(BASE_COLUMNS) + 1 + 2 + 4 + 1:
        return None
    if column_list[: len(BASE_COLUMNS)] != BASE_COLUMNS or column_list[-1] != "acquisition":
        return None

    start = len(BASE_COLUMNS)
    if column_list[start : start + len(REVIEW_COLUMNS)] == REVIEW_COLUMNS:
        start += len(REVIEW_COLUMNS)
    if column_list[start : start + len(REPLICATE_COLUMNS)] == REPLICATE_COLUMNS:
        start += len(REPLICATE_COLUMNS)

    tail_length = len(column_list) - start - 1
    max_objectives = max((tail_length - 1) // 3, 1)
    for objective_count in range(max_objectives, 1, -1):
        result_start = len(column_list) - 1 - 2 * objective_count
        if result_start <= start:
            continue
        result_columns = column_list[result_start:-1]
        objective_names: list[str] = []
        valid_result_columns = True
        for index in range(0, len(result_columns), 2):
            mean_column = result_columns[index]
            std_column = result_columns[index + 1]
            if not mean_column.startswith("predicted_mean_"):
                valid_result_columns = False
                break
            objective_name = mean_column.removeprefix("predicted_mean_")
            if std_column != f"predicted_std_{objective_name}":
                valid_result_columns = False
                break
            objective_names.append(objective_name)
        if not valid_result_columns:
            continue

        middle = column_list[start:result_start]
        if len(middle) < objective_count + 1:
            continue
        if middle[-objective_count:] != objective_names:
            continue
        return middle[:-objective_count], objective_names, [*result_columns, "acquisition"]
    return None


def _is_blank(value: object) -> bool:
    if pd.isna(value):
        return True
    return str(value).strip() == ""


def _has_review_columns(columns: pd.Index | list[str]) -> bool:
    column_list = list(columns)
    start = len(BASE_COLUMNS)
    return column_list[start : start + len(REVIEW_COLUMNS)] == REVIEW_COLUMNS


def _has_replicate_columns(columns: pd.Index | list[str]) -> bool:
    column_list = list(columns)
    start = len(BASE_COLUMNS)
    if column_list[start : start + len(REVIEW_COLUMNS)] == REVIEW_COLUMNS:
        start += len(REVIEW_COLUMNS)
    return column_list[start : start + len(REPLICATE_COLUMNS)] == REPLICATE_COLUMNS


def _has_cost_columns(columns: pd.Index | list[str]) -> bool:
    column_list = list(columns)
    has_utility = column_list[-len(UTILITY_COLUMNS) :] == UTILITY_COLUMNS
    result_end = len(column_list) - (len(UTILITY_COLUMNS) if has_utility else 0)
    middle = column_list[len(BASE_COLUMNS) : result_end - len(RESULT_COLUMNS)]
    if middle[: len(REVIEW_COLUMNS)] == REVIEW_COLUMNS:
        middle = middle[len(REVIEW_COLUMNS) :]
    if middle[: len(REPLICATE_COLUMNS)] == REPLICATE_COLUMNS:
        middle = middle[len(REPLICATE_COLUMNS) :]
    return middle[-len(COST_COLUMNS) :] == COST_COLUMNS
