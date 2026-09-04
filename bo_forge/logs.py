"""Campaign log loading and write transitions."""

from __future__ import annotations

import hashlib
import math
import os
import shutil
import stat
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd
from filelock import FileLock, Timeout

from bo_forge._campaign.log_schema import (
    _has_cost_columns,
    _has_qlog_nehvi_source,
    _has_qmfkg_source,
    _has_replicate_columns,
    _has_review_columns,
    _has_stage_column,
    _is_blank,
    _validate_structural_log,
    _validate_suggestions_for_append,
    _variable_and_objective_columns,
)
from bo_forge.config import CampaignConfig
from bo_forge.errors import (
    LogBusyError,
    LogConflictError,
    LogValidationError,
    LogWriteError,
)
from bo_forge.io import empty_campaign_log
from bo_forge.validation import (
    validate_campaign_data,
)

LOG_LOCK_TIMEOUT_SECONDS = 10.0
_SYSTEM_TEMP_DIRECTORY = Path(tempfile.gettempdir()) if os.name == "nt" else Path("/tmp")
_LOCK_OWNER = str(os.getuid()) if hasattr(os, "getuid") else "default"
_LOG_LOCK_DIRECTORY = _SYSTEM_TEMP_DIRECTORY / f"bo-forge-{_LOCK_OWNER}-log-locks"
_MISSING_LOG_FINGERPRINT = "<missing-log>"
_SESSION_FINGERPRINT_PREFIX = "bo-forge-session-v1"


def load_campaign_log(path: str | Path, config: CampaignConfig) -> pd.DataFrame:
    """Load and validate a campaign log, returning an empty canonical log if missing."""
    log_path = Path(path)
    if not log_path.exists():
        return empty_campaign_log(config)

    df = _read_csv(log_path)
    validate_campaign_data(config, df)
    return df


def _load_campaign_log_snapshot(
    path: str | Path,
    config: CampaignConfig,
) -> tuple[pd.DataFrame, str | None]:
    """Load a session-consistent log and byte fingerprint under its mutation lock."""
    log_path = _canonical_log_path(path)
    with _campaign_log_lock(log_path):
        df = load_campaign_log(log_path, config)
        return df, _log_file_fingerprint(log_path) or _MISSING_LOG_FINGERPRINT


def append_suggestions(
    log_path: str | Path,
    suggestions: pd.DataFrame,
    config: CampaignConfig | None = None,
    *,
    expected_log_fingerprint: str | None = None,
) -> None:
    """Append suggested rows to a campaign log and validate the written file."""
    path = _canonical_log_path(log_path)
    if suggestions.empty:
        raise LogWriteError("append_suggestions() received an empty suggestions DataFrame.")

    _validate_suggestions_for_append(suggestions)
    with _campaign_log_lock(path):
        _assert_expected_log_fingerprint(path, expected_log_fingerprint)
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

        duplicated = set(existing["row_id"].astype(str)) & set(
            suggestions["row_id"].astype(str)
        )
        if duplicated:
            row_id = sorted(duplicated)[0]
            raise LogWriteError(f"Cannot append suggestions with duplicate row_id '{row_id}'.")

        combined = pd.concat([existing, suggestions], ignore_index=True)
        _validate_structural_log(combined)
        if config is not None:
            validate_campaign_data(config, combined)
        elif _has_stage_column(combined.columns):
            raise LogWriteError(
                "Structured campaign append requires config-aware validation; use "
                "append_suggestions(..., config=config) or CampaignSession.append_suggestions()."
            )
        elif _has_replicate_columns(combined.columns):
            raise LogWriteError(
                "Replicate append requires config-aware validation; use "
                "append_suggestions(..., config=config) or CampaignSession.append_suggestions()."
            )
        elif _has_qmfkg_source(combined):
            raise LogWriteError(
                "qMFKG append requires config-aware validation; use "
                "append_suggestions(..., config=config) or CampaignSession.append_suggestions()."
            )
        elif _has_qlog_nehvi_source(combined):
            raise LogWriteError(
                "qLogNEHVI append requires config-aware validation; use "
                "append_suggestions(..., config=config) or CampaignSession.append_suggestions()."
            )
        _write_campaign_log(
            path,
            combined,
            config=config,
            operation="append_suggestions",
            affected_row_ids=suggestions["row_id"].astype(str).tolist(),
            metadata={"appended_row_count": len(suggestions)},
        )


def mark_observed(
    log_path: str | Path,
    row_id: str,
    objective_value: float | None = None,
    objective_values: dict[str, float] | None = None,
    actual_cost: float | None = None,
    config: CampaignConfig | None = None,
    *,
    expected_log_fingerprint: str | None = None,
) -> None:
    """Mark a suggested row as observed by filling the objective value in place."""
    path = _canonical_log_path(log_path)
    if not isinstance(row_id, str) or not row_id.strip():
        raise LogWriteError("row_id must be a non-empty string.")

    with _campaign_log_lock(path):
        _assert_expected_log_fingerprint(path, expected_log_fingerprint)
        if not path.exists():
            raise LogWriteError(
                f"Cannot mark row '{row_id}' observed because log '{path}' does not exist."
            )
        df = _read_csv(path)
        objective_columns = _variable_and_objective_columns(df.columns)[1]
        parsed_objective_values = _parse_mark_observed_objective_values(
            row_id=row_id,
            objective_columns=objective_columns,
            objective_value=objective_value,
            objective_values=objective_values,
        )
        actual_cost_text = _parse_actual_cost(row_id, actual_cost)
        _validate_log_for_mark_observed(df, config)
        index = _mark_observed_row_index(df, row_id)
        _validate_mark_observed_transition(
            df,
            index=index,
            row_id=row_id,
            objective_columns=objective_columns,
            actual_cost_text=actual_cost_text,
        )
        _apply_observation(
            df,
            index=index,
            objective_values=parsed_objective_values,
            actual_cost_text=actual_cost_text,
        )
        _validate_structural_log(df)
        if config is not None:
            validate_campaign_data(config, df)
        _write_campaign_log(
            path,
            df,
            config=config,
            operation="mark_observed",
            affected_row_ids=[row_id],
            metadata={
                "objective_count": len(parsed_objective_values),
                "actual_cost_recorded": actual_cost_text is not None,
            },
        )


def _parse_actual_cost(row_id: str, actual_cost: float | None) -> str | None:
    if actual_cost is None:
        return None
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
    return f"{actual_cost_float:.17g}"


def _validate_log_for_mark_observed(
    df: pd.DataFrame,
    config: CampaignConfig | None,
) -> None:
    _validate_structural_log(df)
    if config is not None:
        validate_campaign_data(config, df)
    elif _has_stage_column(df.columns):
        raise LogWriteError(
            "Structured campaign mark_observed requires config-aware validation; use "
            "mark_observed(..., config=config) or CampaignSession.mark_observed()."
        )
    elif _has_qmfkg_source(df):
        raise LogWriteError(
            "qMFKG mark_observed requires config-aware validation; use "
            "mark_observed(..., config=config) or CampaignSession.mark_observed()."
        )


def _mark_observed_row_index(df: pd.DataFrame, row_id: str) -> object:
    matches = df["row_id"].astype(str) == row_id
    if not matches.any():
        raise LogWriteError(f"Cannot mark row '{row_id}' observed because row_id was not found.")
    if matches.sum() > 1:
        raise LogWriteError(f"Cannot mark row '{row_id}' observed because row_id is duplicated.")
    return matches[matches].index[0]


def _validate_mark_observed_transition(
    df: pd.DataFrame,
    *,
    index: object,
    row_id: str,
    objective_columns: list[str],
    actual_cost_text: str | None,
) -> None:
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


def _apply_observation(
    df: pd.DataFrame,
    *,
    index: object,
    objective_values: dict[str, float],
    actual_cost_text: str | None,
) -> None:
    for objective, objective_float in objective_values.items():
        df.at[index, objective] = f"{objective_float:.17g}"
    df.at[index, "status"] = "observed"
    if actual_cost_text is not None:
        df.at[index, "cost_actual"] = actual_cost_text


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
    config: CampaignConfig | None = None,
    *,
    expected_log_fingerprint: str | None = None,
) -> None:
    """Record a human review decision for one suggested row."""
    path = _canonical_log_path(log_path)
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

    with _campaign_log_lock(path):
        _assert_expected_log_fingerprint(path, expected_log_fingerprint)
        if not path.exists():
            raise LogWriteError(
                f"Cannot review row '{row_id}' because log '{path}' does not exist."
            )
        df = _read_csv(path)
        _validate_structural_log(df)
        if config is not None:
            validate_campaign_data(config, df)
        elif _has_stage_column(df.columns):
            raise LogWriteError(
                "Structured campaign review_suggestion requires config-aware validation; use "
                "review_suggestion(..., config=config) or CampaignSession.review_suggestion()."
            )
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
        if config is not None:
            validate_campaign_data(config, df)
        _write_campaign_log(
            path,
            df,
            config=config,
            operation="review_suggestion",
            affected_row_ids=[row_id],
            metadata={"decision": decision_map[decision]},
        )


def _write_campaign_log(
    path: Path,
    df: pd.DataFrame,
    *,
    config: CampaignConfig | None,
    operation: str,
    affected_row_ids: list[str],
    metadata: dict[str, object],
) -> None:
    from bo_forge._campaign.provenance import write_managed_campaign_log

    if write_managed_campaign_log(
        path,
        df,
        config=config,
        operation=operation,
        affected_row_ids=affected_row_ids,
        metadata=metadata,
    ):
        return
    _atomic_write_and_validate(path, df, config=config)


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, keep_default_na=False)
    except OSError as exc:
        raise LogWriteError(f"Could not read campaign log '{path}': {exc}") from exc
    except pd.errors.ParserError as exc:
        raise LogWriteError(f"Could not parse campaign log '{path}': {exc}") from exc


def _atomic_write_and_validate(
    path: Path,
    df: pd.DataFrame,
    *,
    config: CampaignConfig | None = None,
) -> None:
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

    backup_path: Path | None = None
    try:
        temp_df = _read_csv(temp_path)
        _validate_structural_log(temp_df)
        if config is not None:
            validate_campaign_data(config, temp_df)
        if path.exists():
            temp_path.chmod(stat.S_IMODE(path.stat().st_mode))
            backup_path = _copy_log_backup(path)
        temp_path.replace(path)
    except (OSError, LogValidationError, LogWriteError) as exc:
        _remove_temporary_file(temp_path)
        _remove_temporary_file(backup_path)
        raise LogWriteError(
            f"Campaign log write failed before replacement for '{path}': {exc}"
        ) from exc

    try:
        post_write_df = _read_csv(path)
        _validate_structural_log(post_write_df)
        if config is not None:
            validate_campaign_data(config, post_write_df)
    except (OSError, LogValidationError, LogWriteError) as exc:
        rollback_error = _restore_replaced_log(path, backup_path)
        _remove_temporary_file(temp_path)
        _remove_temporary_file(backup_path)
        if rollback_error is not None:
            raise LogWriteError(
                f"Post-write validation failed for campaign log '{path}', and the "
                f"previous file could not be restored; campaign state is uncertain: "
                f"validation_error={exc}; rollback_error={rollback_error}"
            ) from exc
        raise LogWriteError(
            f"Post-write validation failed for campaign log '{path}'; the previous "
            f"file was restored: {exc}"
        ) from exc
    _remove_temporary_file(backup_path)


def _copy_log_backup(path: Path) -> Path:
    with NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".bak",
        delete=False,
    ) as handle:
        backup_path = Path(handle.name)
    try:
        shutil.copy2(path, backup_path)
    except OSError:
        _remove_temporary_file(backup_path)
        raise
    return backup_path


def _restore_replaced_log(path: Path, backup_path: Path | None) -> OSError | None:
    try:
        if backup_path is None:
            path.unlink(missing_ok=True)
        else:
            backup_path.replace(path)
    except OSError as exc:
        return exc
    return None


def _remove_temporary_file(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _canonical_log_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _log_lock_path(path: str | Path) -> Path:
    canonical = _canonical_log_path(path)
    digest = hashlib.sha256(str(canonical).encode("utf-8")).hexdigest()
    return _LOG_LOCK_DIRECTORY / f"{digest}.lock"


@contextmanager
def _campaign_log_lock(path: str | Path) -> Iterator[None]:
    canonical = _canonical_log_path(path)
    lock_path = _log_lock_path(canonical)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(lock_path, timeout=LOG_LOCK_TIMEOUT_SECONDS)
    try:
        lock.acquire()
    except Timeout as exc:
        raise LogBusyError(
            f"Campaign log '{canonical}' is busy; another process is writing it. "
            f"Try again after {LOG_LOCK_TIMEOUT_SECONDS:g} seconds."
        ) from exc
    try:
        yield
    finally:
        lock.release()


def _log_file_fingerprint(path: str | Path) -> str | None:
    canonical = _canonical_log_path(path)
    if not canonical.exists():
        return None
    digest = hashlib.sha256()
    try:
        with canonical.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise LogWriteError(f"Could not fingerprint campaign log '{canonical}': {exc}") from exc
    return digest.hexdigest()


def _assert_expected_log_fingerprint(path: Path, expected: str | None) -> None:
    if expected is None:
        return
    if expected.startswith(f"{_SESSION_FINGERPRINT_PREFIX}:"):
        _, expected_mode, expected = expected.split(":", maxsplit=2)
        manifest_exists = path.with_name(f"{path.name}.manifest.json").exists()
        if manifest_exists != (expected_mode == "managed"):
            raise LogConflictError(
                "Campaign provenance state changed after it was loaded. Reload the "
                f"campaign before retrying the mutation: log='{path}'."
            )
    current = _log_file_fingerprint(path) or _MISSING_LOG_FINGERPRINT
    if current != expected:
        raise LogConflictError(
            "Campaign log changed after it was loaded. Reload the campaign before retrying "
            f"the mutation: log='{path}'."
        )


def _session_log_fingerprint(fingerprint: str | None, *, managed: bool) -> str:
    value = fingerprint or _MISSING_LOG_FINGERPRINT
    mode = "managed" if managed else "legacy"
    return f"{_SESSION_FINGERPRINT_PREFIX}:{mode}:{value}"
