"""HTTP payload, serialization, and error contracts."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from bo_forge.application import (
    CampaignAppService,
    ValidationResult,
    staged_suggestions_from_bundle,
)
from bo_forge_api.stages import (
    StageSnapshot,
    StageSummary,
)


class ApiError(ValueError):
    """Structured user-facing API error."""

    def __init__(self, code: str, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


class TablePayload(BaseModel):
    """JSON-safe DataFrame payload."""

    columns: list[str]
    records: list[dict[str, Any]]


class CampaignRef(BaseModel):
    """Root-relative campaign file reference."""

    config_path: str
    log_path: str
    require_provenance: bool = False


class DryRunRequest(CampaignRef):
    """Request for non-mutating suggestion generation."""

    batch_size: int | None = Field(default=None, ge=1)
    stage: str | None = None
    context_values: dict[str, object] | None = None


class StagedBundlePayload(BaseModel):
    """JSON-safe staged suggestion bundle."""

    suggestions: TablePayload
    suggestions_fingerprint: str
    config_path: str
    config_fingerprint: str
    log_path: str
    log_fingerprint: str
    appended: bool = False
    context_values: dict[str, object] | None = None
    context_values_fingerprint: str | None = None
    stage: str | None = None


class AppendRequest(CampaignRef):
    """Request to append a staged suggestion bundle."""

    staged_bundle: StagedBundlePayload
    last_appended_fingerprint: str | None = None


class ReviewRequest(CampaignRef):
    """Request to apply a review decision."""

    row_id: str
    decision: str
    note: str = ""
    expected_log_fingerprint: str


class ObservationRequest(CampaignRef):
    """Request to mark a row observed."""

    row_id: str
    objective_value: float | None = None
    objective_values: dict[str, float] | None = None
    actual_cost: float | None = None
    expected_log_fingerprint: str


class ProvenanceRecoveryRequest(CampaignRef):
    """Request for explicit managed-campaign transaction recovery."""

    expected_log_fingerprint: str


def _resolve_campaign_paths(root: Path, request: CampaignRef) -> tuple[Path, Path]:
    return (
        _resolve_under_root(root, request.config_path, "config_path"),
        _resolve_under_root(root, request.log_path, "log_path"),
    )

def _resolve_under_root(root: Path, value: str, field: str) -> Path:
    requested = Path(value)
    if requested.is_absolute():
        raise ApiError("path_outside_root", f"{field} must be a relative path.")
    resolved = (root / requested).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ApiError("path_outside_root", f"{field} must stay under API root.") from exc
    return resolved


def _relative_to_root(root: Path, path_value: object) -> str:
    path = Path(str(path_value)).expanduser().resolve()
    try:
        return path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ApiError("path_outside_root", "Staged bundle path is outside API root.") from exc


def _table_payload(df: pd.DataFrame | None) -> dict[str, object]:
    if df is None:
        df = pd.DataFrame()
    records = [
        {str(column): _json_safe_value(value) for column, value in row.items()}
        for row in df.to_dict(orient="records")
    ]
    return {"columns": [str(column) for column in df.columns], "records": records}


def _table_to_dataframe(payload: TablePayload) -> pd.DataFrame:
    return pd.DataFrame(payload.records, columns=payload.columns)


def _json_safe_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, int | str | bool):
        return value
    return str(value)


def _staged_bundle_payload(bundle: dict[str, object], root: Path) -> dict[str, object]:
    suggestions = staged_suggestions_from_bundle(bundle)
    return {
        "suggestions": _table_payload(suggestions),
        "suggestions_fingerprint": str(bundle.get("suggestions_fingerprint", "")),
        "config_path": _relative_to_root(root, bundle.get("config_path", "")),
        "config_fingerprint": str(bundle.get("config_fingerprint", "")),
        "log_path": _relative_to_root(root, bundle.get("log_path", "")),
        "log_fingerprint": str(bundle.get("log_fingerprint", "")),
        "appended": bool(bundle.get("appended", False)),
        "context_values": bundle.get("context_values"),
        "context_values_fingerprint": bundle.get("context_values_fingerprint"),
        "stage": bundle.get("stage"),
    }


def _rehydrate_staged_bundle(payload: StagedBundlePayload, root: Path) -> dict[str, object]:
    bundle: dict[str, object] = {
        "suggestions": _table_to_dataframe(payload.suggestions),
        "suggestions_fingerprint": payload.suggestions_fingerprint,
        "config_path": str(_resolve_under_root(root, payload.config_path, "staged.config_path")),
        "config_fingerprint": payload.config_fingerprint,
        "log_path": str(_resolve_under_root(root, payload.log_path, "staged.log_path")),
        "log_fingerprint": payload.log_fingerprint,
        "appended": payload.appended,
    }
    if payload.context_values is not None:
        bundle["context_values"] = payload.context_values
        bundle["context_values_fingerprint"] = payload.context_values_fingerprint
    if payload.stage is not None:
        bundle["stage"] = payload.stage
    return bundle


def _stage_metadata_payload(staged: StageSnapshot, root: Path) -> dict[str, object]:
    return {
        "stage_id": staged.stage_id,
        "status": staged.status,
        "created_at": staged.created_at.isoformat(),
        "expires_at": staged.expires_at.isoformat(),
        "last_transition_at": staged.last_transition_at.isoformat(),
        "remaining_ttl_seconds": max(
            0.0,
            (staged.expires_at - datetime.now(UTC)).total_seconds(),
        ),
        "suggestion_count": len(staged.suggestions),
        "config_path": _relative_to_root(root, staged.config_path),
        "log_path": _relative_to_root(root, staged.log_path),
        "stage_selection": staged.stage_selection,
        "context_values": staged.context_values,
        "context_variable_names": sorted((staged.context_values or {}).keys()),
        "renewal_count": staged.renewal_count,
        "status_reason": staged.status_reason,
    }


def _stage_summary_payload(staged: StageSummary, root: Path) -> dict[str, object]:
    return {
        "stage_id": staged.stage_id,
        "status": staged.status,
        "created_at": staged.created_at.isoformat(),
        "expires_at": staged.expires_at.isoformat(),
        "last_transition_at": staged.last_transition_at.isoformat(),
        "remaining_ttl_seconds": staged.remaining_ttl_seconds,
        "suggestion_count": staged.suggestion_count,
        "config_path": _relative_to_root(root, staged.config_path),
        "log_path": _relative_to_root(root, staged.log_path),
        "stage_selection": staged.stage_selection,
        "context_variable_names": list(staged.context_variable_names),
        "renewal_count": staged.renewal_count,
        "status_reason": staged.status_reason,
    }


def _server_stage_payload(staged: StageSnapshot, root: Path) -> dict[str, object]:
    return {
        "stage": _stage_metadata_payload(staged, root),
        "suggestions": _table_payload(staged.suggestions),
        "quality": _table_payload(staged.quality),
    }


def _resolved_stage_selection(
    service: CampaignAppService,
    suggestions: pd.DataFrame,
) -> str | None:
    if not service.config.is_structured_campaign:
        return None
    stages = suggestions["stage"].dropna().astype(str).unique().tolist()
    if len(stages) != 1:
        raise ValueError("Structured suggestions must resolve to exactly one stage.")
    return stages[0]


def _resolved_context_values(
    service: CampaignAppService,
    suggestions: pd.DataFrame,
) -> dict[str, object] | None:
    context_names = service.config.context_variable_names
    if not context_names:
        return None
    first = suggestions.iloc[0]
    return {
        name: _json_safe_value(first[name])
        for name in context_names
    }


def _validation_payload(result: ValidationResult) -> dict[str, object]:
    return {"ok": result.ok, "label": result.label, "message": result.message}


def _error_response(
    code: str,
    message: str,
    status_code: int,
    *,
    reason_code: str | None = None,
    recovery_action: str | None = None,
) -> JSONResponse:
    retryable, suggested_action = _error_recovery(code)
    error: dict[str, object] = {
        "code": code,
        "message": message,
        "retryable": retryable,
        "suggested_action": suggested_action,
    }
    if reason_code is not None:
        error["reason_code"] = reason_code
    if recovery_action is not None:
        error["recovery_action"] = recovery_action
    return JSONResponse(
        status_code=status_code,
        content={
            "ok": False,
            "error": error,
        },
    )


def _error_recovery(code: str) -> tuple[bool, str]:
    recovery = {
        "stage_in_use": (
            True,
            "Wait for the current append attempt to finish, then retry.",
        ),
        "stage_capacity": (
            True,
            "Append or discard an active stage, wait for expiry, then retry.",
        ),
        "log_busy": (True, "Wait briefly for the other local writer, then retry."),
        "stage_expired": (
            False,
            "Generate a new dry-run; an expired stage cannot be renewed or appended.",
        ),
        "stage_consumed": (False, "Refresh campaign state before generating a new dry-run."),
        "stage_discarded": (False, "Generate a new dry-run if suggestions are still needed."),
        "stage_stale": (
            False,
            "Refresh campaign state and generate a new dry-run.",
        ),
        "stage_not_found": (
            False,
            "Check the stage ID or generate a new dry-run in this API process.",
        ),
        "stale_log": (
            False,
            "Refresh campaign state and resubmit with the new log fingerprint.",
        ),
        "provenance_recovery_required": (
            False,
            "Run provenance recovery, then reload the campaign and retry.",
        ),
        "path_outside_root": (
            False,
            "Use config and log paths that resolve inside the configured API root.",
        ),
        "client_bundle_append_disabled": (
            False,
            "Generate a dry-run and append its server-managed stage ID instead.",
        ),
        "request_validation": (False, "Correct the request fields before retrying."),
        "bo_forge_error": (False, "Correct the campaign state or request before retrying."),
        "value_error": (False, "Correct the request or campaign before retrying."),
    }
    return recovery.get(
        code,
        (False, "Correct the request or campaign state before retrying."),
    )
