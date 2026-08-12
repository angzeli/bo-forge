"""Experimental FastAPI probe around the internal BO Forge app service."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from bo_forge import __version__
from bo_forge.errors import BOForgeError, LogBusyError, LogConflictError
from bo_forge_app.service import CampaignAppService, ValidationResult
from bo_forge_app.stages import InMemoryStageStore, StageSnapshot, StageStoreError
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
) -> FastAPI:
    """Create the experimental BO Forge FastAPI app rooted at one directory."""
    resolved_root = Path(root).expanduser().resolve()
    if not resolved_root.is_dir():
        raise ValueError(f"API root must be an existing directory: {resolved_root}")

    app = FastAPI(
        title="BO Forge API Probe",
        version=__version__,
        description="Experimental local/trusted-network API probe around CampaignAppService.",
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
            staged = stage_store.create(
                suggestions=result.suggestions,
                quality=result.quality,
                bundle=result.bundle,
                config_path=service.config_path,
                log_path=service.log_path,
                stage_selection=request.stage,
                context_values=request.context_values,
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
        return JSONResponse(
            status_code=422,
            content={
                "ok": False,
                "error": {
                    "code": "request_validation",
                    "message": "Invalid request.",
                    "details": details,
                },
            },
        )

def _register_server_stage_routes(
    app: FastAPI,
    root: Path,
    stage_store: InMemoryStageStore,
) -> None:
    @app.get("/campaign/stages/{stage_id}")
    def get_stage(stage_id: str) -> dict[str, object]:
        staged = stage_store.get(stage_id)
        stale_reason = _server_stage_file_invalidation_reason(staged)
        if stale_reason is not None and stage_store.mark_stale_if_active(stage_id):
            raise StageStoreError("stage_stale", stale_reason, 409)
        return _server_stage_payload(staged, root)

    @app.post("/campaign/stages/{stage_id}/append")
    def append_server_stage(stage_id: str) -> dict[str, object]:
        staged = stage_store.claim(stage_id)
        stale_reason = _server_stage_file_invalidation_reason(staged)
        if stale_reason is not None:
            stage_store.complete(stage_id, "stale")
            raise StageStoreError("stage_stale", stale_reason, 409)
        try:
            service = CampaignAppService.load(staged.config_path, staged.log_path)
            result = service.append_staged(
                staged.bundle,
                stage=staged.stage_selection,
                context_values=staged.context_values,
            )
        except (LogConflictError, ValueError) as exc:
            stage_store.complete(stage_id, "stale")
            raise StageStoreError(
                "stage_stale",
                f"Staged batch is stale and cannot be appended: {exc}",
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
            stage_store.complete(stage_id, "stale")
            raise StageStoreError(
                "stage_stale",
                "The campaign files changed during append, so this stage cannot be "
                f"retried safely: {exc}",
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
        "config_path": _relative_to_root(root, staged.config_path),
        "log_path": _relative_to_root(root, staged.log_path),
        "stage_selection": staged.stage_selection,
        "context_values": staged.context_values,
    }


def _server_stage_payload(staged: StageSnapshot, root: Path) -> dict[str, object]:
    return {
        "stage": _stage_metadata_payload(staged, root),
        "suggestions": _table_payload(staged.suggestions),
        "quality": _table_payload(staged.quality),
    }


def _server_stage_file_invalidation_reason(staged: StageSnapshot) -> str | None:
    try:
        config_fingerprint = file_fingerprint(staged.config_path)
        log_fingerprint = file_fingerprint(staged.log_path)
    except OSError as exc:
        return f"Staged batch files cannot be read: {exc}"
    if config_fingerprint != staged.bundle.get("config_fingerprint"):
        return "Config file changed after suggestions were staged."
    if log_fingerprint != staged.bundle.get("log_fingerprint"):
        return "Log file changed after suggestions were staged."
    return None


def _prune_invalid_server_stages(stage_store: InMemoryStageStore) -> None:
    for staged in stage_store.active_snapshots():
        if _server_stage_file_invalidation_reason(staged) is not None:
            stage_store.mark_stale_if_active(staged.stage_id)


def _validation_payload(result: ValidationResult) -> dict[str, object]:
    return {"ok": result.ok, "label": result.label, "message": result.message}


def _safe_file_fingerprint(path: Path) -> str | None:
    try:
        return file_fingerprint(path)
    except OSError:
        return None


def _error_response(code: str, message: str, status_code: int) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"ok": False, "error": {"code": code, "message": message}},
    )
