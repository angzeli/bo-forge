"""Root-bounded preview/apply transport for explicit provenance lifecycle actions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI
from pydantic import model_validator

from bo_forge.application import CampaignAppService
from bo_forge_api.contracts import (
    ApiError,
    CampaignRef,
    _resolve_campaign_paths,
    _resolve_under_root,
)


class LifecycleRequest(CampaignRef):
    apply: bool = False
    reason: str = ""
    expected_identities: dict[str, Any] | None = None
    destination: str | None = None
    config_changes: dict[str, Any] | None = None

    @model_validator(mode="after")
    def require_preview_for_apply(self):
        if self.apply and (not self.reason.strip() or self.expected_identities is None):
            raise ValueError("Apply requires a nonempty reason and preview expected_identities.")
        return self


def register_lifecycle_routes(app: FastAPI, root: Path) -> None:
    for operation in ("adopt", "migrate", "accept-config", "fork"):
        app.add_api_route(
            f"/campaign/provenance/{operation}", _handler(root, operation), methods=["POST"]
        )


def _handler(root: Path, operation: str):
    def handle(request: LifecycleRequest) -> dict:
        config_path, log_path = _resolve_campaign_paths(root, request)
        _check_sidecar_boundary(root, log_path)
        destination = None
        if request.destination is not None:
            if (root / request.destination).is_symlink():
                raise ApiError("path_outside_root", "Fork destination cannot be a symlink.")
            destination = _resolve_under_root(root, request.destination, "destination")
        expected = (
            None if request.expected_identities is None else dict(request.expected_identities)
        )
        if expected is not None:
            for name in ("config_path", "log_path"):
                expected[name] = str(_resolve_under_root(root, str(expected.get(name, "")), name))
        result = CampaignAppService.provenance_lifecycle(
            operation,
            config_path,
            log_path,
            apply=request.apply,
            reason=request.reason,
            expected_identities=expected,
            destination=destination,
            config_changes=request.config_changes,
        )
        for name in ("config_path", "log_path"):
            result["expected_identities"][name] = (
                Path(result["expected_identities"][name]).relative_to(root).as_posix()
            )
        if result["destination"] is not None:
            result["destination"] = Path(result["destination"]).relative_to(root).as_posix()
        return result

    return handle


def _check_sidecar_boundary(root: Path, log_path: Path) -> None:
    """Reject sidecar/archive symlinks before reading campaign snapshots."""
    sidecar = log_path.with_name(f"{log_path.name}.manifest.json")
    if sidecar.is_symlink():
        raise ApiError("path_outside_root", "Lifecycle manifest cannot be a symlink.")
    for path in log_path.parent.glob(f"{sidecar.name}.*.archive.*"):
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ApiError(
                "path_outside_root", "Archive must stay under API root without symlinks."
            )
