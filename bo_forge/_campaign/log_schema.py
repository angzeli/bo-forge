"""Structural validation for campaign CSV logs."""

from __future__ import annotations

import math

import pandas as pd

from bo_forge.errors import LogValidationError, LogWriteError
from bo_forge.validation import (
    BASE_COLUMNS,
    COST_COLUMNS,
    REPLICATE_COLUMNS,
    RESULT_COLUMNS,
    REVIEW_COLUMNS,
    STAGE_COLUMNS,
    UTILITY_COLUMNS,
    VALID_MULTI_OBJECTIVE_SOURCES,
    VALID_REVIEW_STATUSES,
    VALID_SOURCES,
    VALID_STATUSES,
)


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

    _validate_structural_identifiers(df)
    variable_columns, objective_columns = _variable_and_objective_columns(df.columns)
    _validate_structural_variables(df, variable_columns)
    observed = df["status"] == "observed"
    suggested = df["status"] == "suggested"
    _validate_structural_objectives(df, objective_columns, observed, suggested)
    _validate_structural_review(df, observed)
    _validate_structural_replicates(df)
    _validate_structural_numeric_results(df)


def _validate_structural_identifiers(df: pd.DataFrame) -> None:

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


def _validate_structural_variables(
    df: pd.DataFrame,
    variable_columns: list[str],
) -> None:
    if not _has_stage_column(df.columns):
        for column in variable_columns:
            invalid = df[column].map(_is_blank)
            if invalid.any():
                row_id = str(df.loc[invalid, "row_id"].iloc[0])
                value = df.loc[invalid, column].iloc[0]
                raise LogValidationError(
                    f"Row '{row_id}' has blank value for variable '{column}': value={value!r}."
                )


def _validate_structural_objectives(
    df: pd.DataFrame,
    objective_columns: list[str],
    observed: pd.Series,
    suggested: pd.Series,
) -> None:
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


def _validate_structural_review(df: pd.DataFrame, observed: pd.Series) -> None:
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


def _validate_structural_replicates(df: pd.DataFrame) -> None:
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


def _validate_structural_numeric_results(df: pd.DataFrame) -> None:
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
    if column_list[start : start + len(STAGE_COLUMNS)] == STAGE_COLUMNS:
        start += len(STAGE_COLUMNS)
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
    has_utility = column_list[-len(UTILITY_COLUMNS) :] == UTILITY_COLUMNS
    acquisition_index = len(column_list) - (len(UTILITY_COLUMNS) if has_utility else 0) - 1
    if (
        column_list[: len(BASE_COLUMNS)] != BASE_COLUMNS
        or acquisition_index < 0
        or column_list[acquisition_index] != "acquisition"
    ):
        return None

    start = len(BASE_COLUMNS)
    if column_list[start : start + len(STAGE_COLUMNS)] == STAGE_COLUMNS:
        start += len(STAGE_COLUMNS)
    if column_list[start : start + len(REVIEW_COLUMNS)] == REVIEW_COLUMNS:
        start += len(REVIEW_COLUMNS)
    if column_list[start : start + len(REPLICATE_COLUMNS)] == REPLICATE_COLUMNS:
        start += len(REPLICATE_COLUMNS)

    tail_length = acquisition_index - start
    max_objectives = max((tail_length - 1) // 3, 1)
    for objective_count in range(max_objectives, 1, -1):
        parts = _multi_objective_parts_for_count(
            column_list,
            start=start,
            acquisition_index=acquisition_index,
            objective_count=objective_count,
            has_utility=has_utility,
        )
        if parts is not None:
            return parts
    return None


def _multi_objective_parts_for_count(
    columns: list[str],
    *,
    start: int,
    acquisition_index: int,
    objective_count: int,
    has_utility: bool,
) -> tuple[list[str], list[str], list[str]] | None:
    result_start = acquisition_index - 2 * objective_count
    if result_start <= start:
        return None
    result_columns = columns[result_start:acquisition_index]
    objective_names = _objective_names_from_result_columns(result_columns)
    if objective_names is None:
        return None
    middle = columns[start:result_start]
    if middle[-len(COST_COLUMNS) :] == COST_COLUMNS:
        middle = middle[: -len(COST_COLUMNS)]
    elif has_utility:
        raise LogValidationError("Campaign log has utility column but no cost columns.")
    if len(middle) < objective_count + 1 or middle[-objective_count:] != objective_names:
        return None
    return middle[:-objective_count], objective_names, [*result_columns, "acquisition"]


def _objective_names_from_result_columns(
    result_columns: list[str],
) -> list[str] | None:
    objective_names: list[str] = []
    for index in range(0, len(result_columns), 2):
        mean_column = result_columns[index]
        std_column = result_columns[index + 1]
        if not mean_column.startswith("predicted_mean_"):
            return None
        objective_name = mean_column.removeprefix("predicted_mean_")
        if std_column != f"predicted_std_{objective_name}":
            return None
        objective_names.append(objective_name)
    return objective_names


def _is_blank(value: object) -> bool:
    if pd.isna(value):
        return True
    return str(value).strip() == ""


def _has_review_columns(columns: pd.Index | list[str]) -> bool:
    column_list = list(columns)
    start = len(BASE_COLUMNS)
    if column_list[start : start + len(STAGE_COLUMNS)] == STAGE_COLUMNS:
        start += len(STAGE_COLUMNS)
    return column_list[start : start + len(REVIEW_COLUMNS)] == REVIEW_COLUMNS


def _has_replicate_columns(columns: pd.Index | list[str]) -> bool:
    column_list = list(columns)
    start = len(BASE_COLUMNS)
    if column_list[start : start + len(STAGE_COLUMNS)] == STAGE_COLUMNS:
        start += len(STAGE_COLUMNS)
    if column_list[start : start + len(REVIEW_COLUMNS)] == REVIEW_COLUMNS:
        start += len(REVIEW_COLUMNS)
    return column_list[start : start + len(REPLICATE_COLUMNS)] == REPLICATE_COLUMNS


def _has_qmfkg_source(df: pd.DataFrame) -> bool:
    return "source" in df.columns and df["source"].astype(str).eq("qmf_kg").any()


def _has_qlog_nehvi_source(df: pd.DataFrame) -> bool:
    return "source" in df.columns and df["source"].astype(str).eq("qlog_nehvi").any()


def _has_stage_column(columns: pd.Index | list[str]) -> bool:
    column_list = list(columns)
    start = len(BASE_COLUMNS)
    return column_list[start : start + len(STAGE_COLUMNS)] == STAGE_COLUMNS


def _has_cost_columns(columns: pd.Index | list[str]) -> bool:
    column_list = list(columns)
    multi_objective_parts = _multi_objective_parts_from_columns(column_list)
    if multi_objective_parts is not None:
        variable_columns, objective_columns, _ = multi_objective_parts
        start = len(BASE_COLUMNS)
        if column_list[start : start + len(STAGE_COLUMNS)] == STAGE_COLUMNS:
            start += len(STAGE_COLUMNS)
        if column_list[start : start + len(REVIEW_COLUMNS)] == REVIEW_COLUMNS:
            start += len(REVIEW_COLUMNS)
        if column_list[start : start + len(REPLICATE_COLUMNS)] == REPLICATE_COLUMNS:
            start += len(REPLICATE_COLUMNS)
        cost_start = start + len(variable_columns) + len(objective_columns)
        return column_list[cost_start : cost_start + len(COST_COLUMNS)] == COST_COLUMNS
    has_utility = column_list[-len(UTILITY_COLUMNS) :] == UTILITY_COLUMNS
    result_end = len(column_list) - (len(UTILITY_COLUMNS) if has_utility else 0)
    start = len(BASE_COLUMNS)
    if column_list[start : start + len(STAGE_COLUMNS)] == STAGE_COLUMNS:
        start += len(STAGE_COLUMNS)
    middle = column_list[start : result_end - len(RESULT_COLUMNS)]
    if middle[: len(REVIEW_COLUMNS)] == REVIEW_COLUMNS:
        middle = middle[len(REVIEW_COLUMNS) :]
    if middle[: len(REPLICATE_COLUMNS)] == REPLICATE_COLUMNS:
        middle = middle[len(REPLICATE_COLUMNS) :]
    return middle[-len(COST_COLUMNS) :] == COST_COLUMNS
