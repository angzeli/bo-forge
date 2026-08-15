"""Behavior freeze for the completed v2.5.x operational contracts."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import bo_forge_app.api as api_module
import bo_forge_app.api_cli as api_cli
import bo_forge_app.cli as app_cli
from bo_forge_app.api import _error_response, create_app
from bo_forge_app.streamlit_helpers import file_fingerprint

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _copy_campaign(root: Path) -> tuple[dict[str, str], Path]:
    config_name = "01_simple_2d_maximise_logei.yaml"
    log_name = "01_simple_2d_maximise_logei_campaign_log.csv"
    (root / "configs").mkdir()
    (root / "examples").mkdir()
    shutil.copy2(PROJECT_ROOT / "configs" / config_name, root / "configs" / config_name)
    log_path = root / "examples" / log_name
    shutil.copy2(PROJECT_ROOT / "examples" / log_name, log_path)
    return (
        {
            "config_path": f"configs/{config_name}",
            "log_path": f"examples/{log_name}",
        },
        log_path,
    )


@pytest.mark.parametrize(
    ("server_stages_only", "interactive_docs", "bundles_enabled", "docs_status"),
    [
        (False, True, True, 200),
        (True, True, False, 200),
        (False, False, True, 404),
        (True, False, False, 404),
    ],
    ids=["default", "server-stages-only", "no-docs", "combined-strict"],
)
def test_api_deployment_mode_matrix(
    tmp_path: Path,
    server_stages_only: bool,
    interactive_docs: bool,
    bundles_enabled: bool,
    docs_status: int,
) -> None:
    campaign_ref, log_path = _copy_campaign(tmp_path)
    client = TestClient(
        create_app(
            tmp_path,
            server_stages_only=server_stages_only,
            interactive_docs=interactive_docs,
        )
    )

    health = client.get("/health")

    assert health.status_code == 200
    assert health.json()["deployment"] == {
        "authentication": "none",
        "trusted_network_only": True,
        "stage_storage": "process_memory",
        "client_carried_bundles": bundles_enabled,
        "interactive_docs": interactive_docs,
        "multi_worker_safe": False,
    }
    assert client.get("/docs").status_code == docs_status
    assert client.get("/redoc").status_code == docs_status
    assert client.get("/openapi.json").status_code == docs_status

    dry_run = client.post("/campaign/suggestions/dry-run", json=campaign_ref)
    assert dry_run.status_code == 200, dry_run.text
    staged = dry_run.json()
    before = log_path.read_bytes()
    compatibility_append = client.post(
        "/campaign/suggestions/append",
        json={
            **campaign_ref,
            "staged_bundle": staged["staged_bundle"],
        },
    )
    if server_stages_only:
        assert compatibility_append.status_code == 403
        assert (
            compatibility_append.json()["error"]["code"]
            == "client_bundle_append_disabled"
        )
        assert log_path.read_bytes() == before
        server_append = client.post(
            f"/campaign/stages/{staged['stage']['stage_id']}/append"
        )
        assert server_append.status_code == 200, server_append.text
        assert server_append.json()["stage"]["status"] == "consumed"
    else:
        assert compatibility_append.status_code == 200, compatibility_append.text
    assert log_path.read_bytes() != before


@pytest.mark.parametrize(
    ("host", "acknowledged", "rejected"),
    [
        ("127.0.0.1", False, False),
        ("localhost", False, False),
        ("::1", False, False),
        ("0.0.0.0", False, True),
        ("::", False, True),
        ("192.168.1.25", False, True),
        ("lab-server.local", False, True),
        ("0.0.0.0", True, False),
    ],
)
def test_launcher_network_acknowledgement_matrix(
    host: str,
    acknowledged: bool,
    rejected: bool,
) -> None:
    args = SimpleNamespace(host=host, allow_network_access=acknowledged)

    for check, error_type in (
        (app_cli._require_network_access_acknowledgement, app_cli.LauncherError),
        (api_cli._require_network_access_acknowledgement, api_cli.ApiLauncherError),
    ):
        if rejected:
            with pytest.raises(error_type, match="--allow-network-access"):
                check(args)
        else:
            check(args)


@pytest.mark.parametrize(
    ("code", "status_code", "retryable"),
    [
        ("stage_stale", 409, False),
        ("stage_expired", 410, False),
        ("stage_consumed", 409, False),
        ("stage_discarded", 409, False),
        ("stage_in_use", 409, True),
        ("log_busy", 409, True),
        ("stale_log", 400, False),
    ],
)
def test_operational_error_contract_matrix(
    code: str,
    status_code: int,
    retryable: bool,
) -> None:
    response = _error_response(code, "contract message", status_code)
    payload = json.loads(response.body)

    assert response.status_code == status_code
    assert payload["ok"] is False
    assert payload["error"]["code"] == code
    assert payload["error"]["message"] == "contract message"
    assert payload["error"]["retryable"] is retryable
    assert payload["error"]["suggested_action"]


def test_health_remains_file_free_and_contains_no_root_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("health must not load campaigns or fingerprint files")

    monkeypatch.setattr(api_module, "file_fingerprint", fail)
    monkeypatch.setattr(api_module, "_safe_file_fingerprint", fail)
    monkeypatch.setattr(api_module.CampaignAppService, "load", fail)

    response = TestClient(create_app(tmp_path)).get("/health")

    assert response.status_code == 200
    assert str(tmp_path) not in response.text


@pytest.mark.parametrize(
    ("endpoint", "payload"),
    [
        (
            "/campaign/review",
            {"row_id": "nei_pending_0", "decision": "invalid"},
        ),
        (
            "/campaign/observations",
            {
                "row_id": "nei_pending_0",
                "objective_value": 0.9,
                "actual_cost": -1.0,
            },
        ),
    ],
)
def test_failed_review_and_observation_are_byte_atomic(
    tmp_path: Path,
    endpoint: str,
    payload: dict[str, object],
) -> None:
    config_name = "18_noisy_pending_qlognei.yaml"
    log_name = "18_noisy_pending_qlognei_campaign_log.csv"
    shutil.copy2(PROJECT_ROOT / "configs" / config_name, tmp_path / config_name)
    shutil.copy2(PROJECT_ROOT / "examples" / log_name, tmp_path / log_name)
    log_path = tmp_path / log_name
    before = log_path.read_bytes()
    request = {
        "config_path": config_name,
        "log_path": log_name,
        "expected_log_fingerprint": file_fingerprint(log_path),
        **payload,
    }

    response = TestClient(create_app(tmp_path)).post(endpoint, json=request)

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "bo_forge_error"
    assert log_path.read_bytes() == before
