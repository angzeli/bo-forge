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
from bo_forge.errors import BOForgeError
from bo_forge_app.service import CampaignAppService, ValidationResult
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


def create_app(root: str | Path) -> FastAPI:
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

    @app.exception_handler(ApiError)
    async def _api_error_handler(_request: Request, exc: ApiError) -> JSONResponse:
        return _error_response(exc.code, exc.message, exc.status_code)

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

    @app.get("/health")
    def health() -> dict[str, object]:
        return {"status": "ok", "version": __version__, "experimental": True}

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
            "log_fingerprint": file_fingerprint(service.log_path),
        }

    @app.post("/campaign/suggestions/dry-run")
    def dry_run(request: DryRunRequest) -> dict[str, object]:
        service = _load_service(resolved_root, request)
        batch_size = request.batch_size or service.config.bo.batch_size
        result = service.suggest_dry_run(
            batch_size=batch_size,
            context_values=request.context_values,
        )
        return {
            "suggestions": _table_payload(result.suggestions),
            "quality": _table_payload(result.quality),
            "staged_bundle": _staged_bundle_payload(result.bundle, resolved_root),
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
        return {
            "validation": _validation_payload(result.validation),
            "appended_fingerprint": result.appended_fingerprint,
            "log_fingerprint": file_fingerprint(result.service.log_path),
        }

    @app.post("/campaign/review")
    def review(request: ReviewRequest) -> dict[str, object]:
        _assert_expected_log_fingerprint(resolved_root, request)
        service = _load_service(resolved_root, request)
        result = service.review(request.row_id, request.decision, request.note)
        return {
            "validation": _validation_payload(result.validation),
            "log_fingerprint": file_fingerprint(result.service.log_path),
        }

    @app.post("/campaign/observations")
    def observations(request: ObservationRequest) -> dict[str, object]:
        _assert_expected_log_fingerprint(resolved_root, request)
        service = _load_service(resolved_root, request)
        result = service.mark_observed(
            request.row_id,
            objective_value=request.objective_value,
            objective_values=request.objective_values,
            actual_cost=request.actual_cost,
        )
        return {
            "validation": _validation_payload(result.validation),
            "log_fingerprint": file_fingerprint(result.service.log_path),
        }

    return app


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


def _assert_expected_log_fingerprint(root: Path, request: CampaignRef) -> None:
    expected = getattr(request, "expected_log_fingerprint", None)
    if expected is None:
        return
    _, log_path = _resolve_campaign_paths(root, request)
    current = file_fingerprint(log_path)
    if current != expected:
        raise ApiError("stale_log", "Log file changed before this mutation.")


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
    return bundle


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
