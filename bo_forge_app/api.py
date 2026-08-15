"""Experimental FastAPI probe around the internal BO Forge app service."""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from bo_forge import __version__
from bo_forge.errors import BOForgeError, LogBusyError, LogConflictError
from bo_forge_app.service import CampaignAppService, ValidationResult
from bo_forge_app.stages import (
    STAGE_STATUSES,
    InMemoryStageStore,
    StageSnapshot,
    StageStoreError,
    StageSummary,
    StageValidationSnapshot,
)
from bo_forge_app.streamlit_helpers import file_fingerprint, staged_suggestions_from_bundle


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


def create_app(
    root: str | Path,
    *,
    stage_ttl_seconds: float = 1800.0,
    max_staged_batches: int = 128,
    server_stages_only: bool = False,
    interactive_docs: bool = True,
) -> FastAPI:
    """Create the experimental BO Forge FastAPI app rooted at one directory."""
    resolved_root = Path(root).expanduser().resolve()
    if not resolved_root.is_dir():
        raise ValueError(f"API root must be an existing directory: {resolved_root}")

    app = FastAPI(
        title="BO Forge API Probe",
        version=__version__,
        description="Experimental local/trusted-network API probe around CampaignAppService.",
        docs_url="/docs" if interactive_docs else None,
        redoc_url="/redoc" if interactive_docs else None,
        openapi_url="/openapi.json" if interactive_docs else None,
    )
    app.state.root = resolved_root
    stage_store = InMemoryStageStore(
        ttl_seconds=stage_ttl_seconds,
        max_active_stages=max_staged_batches,
    )
    app.state.stage_store = stage_store

    _register_error_handlers(app)

    @app.get("/health")
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "version": __version__,
            "experimental": True,
            "staging": stage_store.stats(),
            "deployment": {
                "authentication": "none",
                "trusted_network_only": True,
                "stage_storage": "process_memory",
                "client_carried_bundles": not server_stages_only,
                "interactive_docs": interactive_docs,
                "multi_worker_safe": False,
            },
        }

    @app.post("/campaign/validation")
    def validation(request: CampaignRef) -> dict[str, object]:
        config_path, log_path = _resolve_campaign_paths(resolved_root, request)
        try:
            service = CampaignAppService.load(config_path, log_path)
        except BOForgeError as exc:
            result = ValidationResult(False, "Validation issue", str(exc))
        else:
            result = service.validate()
        return {
            "validation": _validation_payload(result),
            "log_fingerprint": _safe_file_fingerprint(log_path),
        }

    @app.post("/campaign/summary")
    def summary(request: CampaignRef) -> dict[str, object]:
        service = _load_service(resolved_root, request)
        view_data = service.collect_view_data("Data")
        review_queue = service.review_queue() if service.config.review.enabled else None
        return {
            "summary": _table_payload(view_data.summary),
            "next_action": _table_payload(view_data.next_action),
            "observed": _table_payload(view_data.observed),
            "pending": _table_payload(view_data.pending),
            "review_queue": _table_payload(review_queue),
            "pareto_summary": _table_payload(view_data.pareto_summary),
            "pareto_front": _table_payload(view_data.pareto_front),
            "cost_summary": _table_payload(view_data.cost_summary),
            "replicate_summary": _table_payload(view_data.replicate_summary),
            "fidelity_summary": _table_payload(view_data.fidelity_summary),
            "fidelity_coverage": _table_payload(view_data.fidelity_coverage),
            "log_fingerprint": file_fingerprint(service.log_path),
        }

    @app.post("/campaign/suggestions/dry-run")
    def dry_run(request: DryRunRequest) -> dict[str, object]:
        service = _load_service(resolved_root, request)
        _prune_invalid_server_stages(stage_store)
        reservation = stage_store.reserve_capacity()
        try:
            batch_size = request.batch_size or service.config.bo.batch_size
            result = service.suggest_dry_run(
                batch_size=batch_size,
                stage=request.stage,
                context_values=request.context_values,
            )
            resolved_stage = _resolved_stage_selection(service, result.suggestions)
            resolved_context_values = _resolved_context_values(
                service,
                result.suggestions,
            )
            staged = stage_store.create(
                suggestions=result.suggestions,
                quality=result.quality,
                bundle=result.bundle,
                config_path=service.config_path,
                log_path=service.log_path,
                stage_selection=resolved_stage,
                context_values=resolved_context_values,
                reservation_token=reservation,
            )
        except Exception:
            stage_store.release_reservation(reservation)
            raise
        return {
            "suggestions": _table_payload(result.suggestions),
            "quality": _table_payload(result.quality),
            "staged_bundle": _staged_bundle_payload(result.bundle, resolved_root),
            "stage": _stage_metadata_payload(staged, resolved_root),
            "log_fingerprint": file_fingerprint(service.log_path),
        }

    @app.post("/campaign/suggestions/append")
    def append(request: AppendRequest) -> dict[str, object]:
        if server_stages_only:
            raise ApiError(
                "client_bundle_append_disabled",
                "Client-carried staged-bundle append is disabled for this deployment. "
                "Use a server-managed stage append instead.",
                403,
            )
        service = _load_service(resolved_root, request)
        bundle = _rehydrate_staged_bundle(request.staged_bundle, resolved_root)
        result = service.append_staged(
            bundle,
            last_appended_fingerprint=request.last_appended_fingerprint,
        )
        stage_store.retire_for_log_change(
            log_path=result.service.log_path,
            previous_log_fingerprint=request.staged_bundle.log_fingerprint,
            consumed_suggestions_fingerprint=(
                request.staged_bundle.suggestions_fingerprint
            ),
        )
        return {
            "validation": _validation_payload(result.validation),
            "appended_fingerprint": result.appended_fingerprint,
            "log_fingerprint": file_fingerprint(result.service.log_path),
        }

    @app.post("/campaign/review")
    def review(request: ReviewRequest) -> dict[str, object]:
        service = _load_service(resolved_root, request)
        result = service.review(
            request.row_id,
            request.decision,
            request.note,
            expected_log_fingerprint=request.expected_log_fingerprint,
        )
        stage_store.retire_for_log_change(
            log_path=result.service.log_path,
            previous_log_fingerprint=request.expected_log_fingerprint,
        )
        return {
            "validation": _validation_payload(result.validation),
            "log_fingerprint": file_fingerprint(result.service.log_path),
        }

    @app.post("/campaign/observations")
    def observations(request: ObservationRequest) -> dict[str, object]:
        service = _load_service(resolved_root, request)
        result = service.mark_observed(
            request.row_id,
            objective_value=request.objective_value,
            objective_values=request.objective_values,
            actual_cost=request.actual_cost,
            expected_log_fingerprint=request.expected_log_fingerprint,
        )
        stage_store.retire_for_log_change(
            log_path=result.service.log_path,
            previous_log_fingerprint=request.expected_log_fingerprint,
        )
        return {
            "validation": _validation_payload(result.validation),
            "log_fingerprint": file_fingerprint(result.service.log_path),
        }

    _register_server_stage_routes(app, resolved_root, stage_store)

    return app


def _register_error_handlers(app: FastAPI) -> None:

    @app.exception_handler(ApiError)
    async def _api_error_handler(_request: Request, exc: ApiError) -> JSONResponse:
        return _error_response(exc.code, exc.message, exc.status_code)

    @app.exception_handler(StageStoreError)
    async def _stage_store_error_handler(
        _request: Request,
        exc: StageStoreError,
    ) -> JSONResponse:
        return _error_response(exc.code, exc.message, exc.status_code)

    @app.exception_handler(LogBusyError)
    async def _log_busy_error_handler(_request: Request, exc: LogBusyError) -> JSONResponse:
        return _error_response("log_busy", str(exc), 409)

    @app.exception_handler(LogConflictError)
    async def _log_conflict_error_handler(
        _request: Request,
        exc: LogConflictError,
    ) -> JSONResponse:
        return _error_response("stale_log", str(exc), 400)

    @app.exception_handler(BOForgeError)
    async def _bo_forge_error_handler(_request: Request, exc: BOForgeError) -> JSONResponse:
        return _error_response("bo_forge_error", str(exc), 400)

    @app.exception_handler(ValueError)
    async def _value_error_handler(_request: Request, exc: ValueError) -> JSONResponse:
        return _error_response("value_error", str(exc), 400)

    @app.exception_handler(RequestValidationError)
    async def _validation_error_handler(
        _request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        details = [
            {"loc": list(error.get("loc", [])), "message": str(error.get("msg", ""))}
            for error in exc.errors()
        ]
        retryable, suggested_action = _error_recovery("request_validation")
        return JSONResponse(
            status_code=422,
            content={
                "ok": False,
                "error": {
                    "code": "request_validation",
                    "message": "Invalid request.",
                    "retryable": retryable,
                    "suggested_action": suggested_action,
                    "details": details,
                },
            },
        )

def _register_server_stage_routes(
    app: FastAPI,
    root: Path,
    stage_store: InMemoryStageStore,
) -> None:
    @app.get("/campaign/stages")
    def list_server_stages(
        include_terminal: bool = False,
        status: Annotated[list[str] | None, Query()] = None,
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
    ) -> dict[str, object]:
        requested_statuses = status or list(STAGE_STATUSES)
        unknown = set(requested_statuses).difference(STAGE_STATUSES)
        if unknown:
            allowed = ", ".join(STAGE_STATUSES)
            raise ApiError(
                "request_validation",
                f"Unknown stage status '{sorted(unknown)[0]}'. Expected one of: {allowed}.",
                422,
            )
        listing = stage_store.list_summaries(
            include_terminal=include_terminal,
            statuses=requested_statuses,
            limit=limit,
            validator=_server_stage_file_validator(),
        )
        return {
            "stages": [_stage_summary_payload(item, root) for item in listing.stages],
            "total": listing.total,
            "returned": listing.returned,
            "truncated": listing.truncated,
            "status_counts": listing.status_counts,
        }

    @app.get("/campaign/stages/{stage_id}")
    def get_stage(stage_id: str) -> dict[str, object]:
        staged = stage_store.get(stage_id)
        stale_reason = _server_stage_file_invalidation_reason(staged)
        if stale_reason is not None and stage_store.mark_stale_if_active(
            stage_id, stale_reason
        ):
            raise StageStoreError("stage_stale", stale_reason, 409)
        return _server_stage_payload(staged, root)

    @app.post("/campaign/stages/{stage_id}/renew")
    def renew_server_stage(stage_id: str) -> dict[str, object]:
        renewed = stage_store.renew(
            stage_id,
            validator=_server_stage_file_invalidation_reason,
        )
        return {"stage": _stage_metadata_payload(renewed, root)}

    @app.post("/campaign/stages/{stage_id}/append")
    def append_server_stage(stage_id: str) -> dict[str, object]:
        staged = stage_store.claim(stage_id)
        stale_reason = _server_stage_file_invalidation_reason(staged)
        if stale_reason is not None:
            stage_store.complete(stage_id, "stale", reason=stale_reason)
            raise StageStoreError("stage_stale", stale_reason, 409)
        try:
            service = CampaignAppService.load(staged.config_path, staged.log_path)
            bundled_context_values = staged.bundle.get("context_values")
            result = service.append_staged(
                staged.bundle,
                stage=staged.stage_selection,
                context_values=(
                    bundled_context_values
                    if isinstance(bundled_context_values, dict)
                    else None
                ),
            )
        except (LogConflictError, ValueError) as exc:
            stale_message = (
                "Staged batch failed append integrity checks and cannot be retried."
            )
            stage_store.complete(stage_id, "stale", reason=stale_message)
            raise StageStoreError(
                "stage_stale",
                stale_message,
                409,
            ) from exc
        except LogBusyError:
            stage_store.restore(stage_id)
            raise
        except Exception as exc:
            stale_reason = _server_stage_file_invalidation_reason(staged)
            if stale_reason is None:
                stage_store.restore(stage_id)
                raise
            stale_message = (
                "Campaign files changed during append, so retry safety cannot be proven."
            )
            stage_store.complete(stage_id, "stale", reason=stale_message)
            raise StageStoreError(
                "stage_stale",
                stale_message,
                409,
            ) from exc
        stage_store.retire_for_log_change(
            log_path=staged.log_path,
            previous_log_fingerprint=str(staged.bundle.get("log_fingerprint", "")),
        )
        terminal = stage_store.complete(stage_id, "consumed")
        return {
            "stage": _stage_metadata_payload(terminal, root),
            "validation": _validation_payload(result.validation),
            "appended_fingerprint": result.appended_fingerprint,
            "log_fingerprint": file_fingerprint(result.service.log_path),
        }

    @app.delete("/campaign/stages/{stage_id}")
    def discard_server_stage(stage_id: str) -> dict[str, object]:
        discarded = stage_store.discard(stage_id)
        return {"stage": _stage_metadata_payload(discarded, root)}


def _load_service(root: Path, request: CampaignRef) -> CampaignAppService:
    config_path, log_path = _resolve_campaign_paths(root, request)
    return CampaignAppService.load(config_path, log_path)


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


def _server_stage_file_invalidation_reason(
    staged: StageValidationSnapshot | StageSnapshot,
    *,
    fingerprint_cache: dict[Path, tuple[str | None, bool]] | None = None,
) -> str | None:
    config_fingerprint, config_unreadable = _cached_stage_file_fingerprint(
        staged.config_path,
        fingerprint_cache,
    )
    log_fingerprint, log_unreadable = _cached_stage_file_fingerprint(
        staged.log_path,
        fingerprint_cache,
    )
    if config_unreadable or log_unreadable:
        return "Staged batch files cannot be read."
    expected_config_fingerprint = (
        staged.config_fingerprint
        if isinstance(staged, StageValidationSnapshot)
        else staged.bundle.get("config_fingerprint")
    )
    expected_log_fingerprint = (
        staged.log_fingerprint
        if isinstance(staged, StageValidationSnapshot)
        else staged.bundle.get("log_fingerprint")
    )
    if config_fingerprint != expected_config_fingerprint:
        return "Config file changed after suggestions were staged."
    if log_fingerprint != expected_log_fingerprint:
        return "Log file changed after suggestions were staged."
    return None


def _cached_stage_file_fingerprint(
    path: Path,
    cache: dict[Path, tuple[str | None, bool]] | None,
) -> tuple[str | None, bool]:
    cache_key = path.resolve()
    if cache is not None and cache_key in cache:
        return cache[cache_key]
    try:
        result = (file_fingerprint(cache_key), False)
    except OSError:
        result = (None, True)
    if cache is not None:
        cache[cache_key] = result
    return result


def _server_stage_file_validator() -> Callable[
    [StageValidationSnapshot | StageSnapshot], str | None
]:
    fingerprint_cache: dict[Path, tuple[str | None, bool]] = {}

    def validate(staged: StageValidationSnapshot | StageSnapshot) -> str | None:
        return _server_stage_file_invalidation_reason(
            staged,
            fingerprint_cache=fingerprint_cache,
        )

    return validate


def _prune_invalid_server_stages(stage_store: InMemoryStageStore) -> None:
    validate = _server_stage_file_validator()
    for staged in stage_store.validation_snapshots():
        stale_reason = validate(staged)
        if stale_reason is not None:
            stage_store.mark_stale_if_active(staged.stage_id, stale_reason)


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


def _safe_file_fingerprint(path: Path) -> str | None:
    try:
        return file_fingerprint(path)
    except OSError:
        return None


def _error_response(code: str, message: str, status_code: int) -> JSONResponse:
    retryable, suggested_action = _error_recovery(code)
    return JSONResponse(
        status_code=status_code,
        content={
            "ok": False,
            "error": {
                "code": code,
                "message": message,
                "retryable": retryable,
                "suggested_action": suggested_action,
            },
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
