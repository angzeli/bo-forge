import json
import shutil
from pathlib import Path

from fastapi.testclient import TestClient

import bo_forge._campaign.provenance as provenance_module
from bo_forge.session import CampaignSession
from bo_forge_app.api import create_app
from bo_forge_app.streamlit_helpers import file_fingerprint

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _copy_campaign(root: Path) -> dict[str, str]:
    config_name = "01_simple_2d_maximise_logei.yaml"
    log_name = "01_simple_2d_maximise_logei_campaign_log.csv"
    (root / "configs").mkdir()
    (root / "examples").mkdir()
    shutil.copyfile(PROJECT_ROOT / "configs" / config_name, root / "configs" / config_name)
    shutil.copyfile(PROJECT_ROOT / "examples" / log_name, root / "examples" / log_name)
    return {
        "config_path": f"configs/{config_name}",
        "log_path": f"examples/{log_name}",
    }


def _client(root: Path) -> TestClient:
    return TestClient(create_app(root))


def test_api_provenance_returns_root_bounded_managed_summary(tmp_path: Path) -> None:
    ref = _copy_campaign(tmp_path)
    log_path = tmp_path / ref["log_path"]
    log_path.unlink()
    CampaignSession.initialize(tmp_path / ref["config_path"], log_path)

    response = _client(tmp_path).post("/campaign/provenance", json=ref)

    assert response.status_code == 200
    payload = response.json()["provenance"]
    assert payload["columns"] == ["field", "value"]
    values = {row["field"]: row["value"] for row in payload["records"]}
    assert values["provenance_status"] == "managed"
    assert values["event_count"] == 1


def test_api_provenance_reports_legacy_and_rejects_outside_root(tmp_path: Path) -> None:
    ref = _copy_campaign(tmp_path)

    legacy_response = _client(tmp_path).post("/campaign/provenance", json=ref)
    outside_response = _client(tmp_path).post(
        "/campaign/provenance",
        json={"config_path": "../outside.yaml", "log_path": ref["log_path"]},
    )

    assert legacy_response.status_code == 200
    legacy_values = {
        row["field"]: row["value"]
        for row in legacy_response.json()["provenance"]["records"]
    }
    assert legacy_values["provenance_status"] == "legacy"
    assert legacy_values["resume_status"] == "legacy"
    assert outside_response.status_code == 400
    assert outside_response.json()["error"]["code"] == "path_outside_root"


def test_api_provenance_rejects_missing_log(tmp_path: Path) -> None:
    ref = _copy_campaign(tmp_path)
    (tmp_path / ref["log_path"]).unlink()

    response = _client(tmp_path).post("/campaign/provenance", json=ref)

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "provenance_error"
    assert "provenance status is unknown" in response.json()["error"]["message"]


def test_api_provenance_diagnoses_mismatch_while_campaign_load_fails_closed(
    tmp_path: Path,
) -> None:
    ref = _copy_campaign(tmp_path)
    log_path = tmp_path / ref["log_path"]
    log_path.unlink()
    CampaignSession.initialize(tmp_path / ref["config_path"], log_path)
    log_path.write_bytes(log_path.read_bytes() + b"\n")
    before_log = log_path.read_bytes()

    api_client = _client(tmp_path)
    provenance_response = api_client.post("/campaign/provenance", json=ref)
    summary_response = api_client.post("/campaign/summary", json=ref)

    assert provenance_response.status_code == 200
    values = {
        row["field"]: row["value"]
        for row in provenance_response.json()["provenance"]["records"]
    }
    assert values["integrity_status"] == "mismatch"
    assert summary_response.status_code == 400
    assert summary_response.json()["error"]["code"] == "stale_log"
    assert log_path.read_bytes() == before_log


def test_api_provenance_structures_malformed_managed_config_failure(tmp_path: Path) -> None:
    ref = _copy_campaign(tmp_path)
    config_path = tmp_path / ref["config_path"]
    log_path = tmp_path / ref["log_path"]
    log_path.unlink()
    CampaignSession.initialize(config_path, log_path)
    config_path.write_text("campaign_name: [broken\n", encoding="utf-8")
    before_log = log_path.read_bytes()

    api_client = _client(tmp_path)
    provenance_response = api_client.post("/campaign/provenance", json=ref)
    summary_response = api_client.post("/campaign/summary", json=ref)

    assert provenance_response.status_code == 200
    values = {
        row["field"]: row["value"]
        for row in provenance_response.json()["provenance"]["records"]
    }
    assert values["reason_code"] == "config_semantics_changed"
    assert values["log_bytes_match"] is True
    assert summary_response.status_code == 400
    error = summary_response.json()["error"]
    assert error["code"] == "stale_log"
    assert error["reason_code"] == "config_semantics_changed"
    assert log_path.read_bytes() == before_log


def test_api_strict_loading_and_explicit_provenance_recovery(tmp_path: Path) -> None:
    ref = _copy_campaign(tmp_path)
    strict_ref = {**ref, "require_provenance": True}
    api_client = _client(tmp_path)

    legacy = api_client.post("/campaign/summary", json=strict_ref)
    assert legacy.status_code == 400
    legacy_error = legacy.json()["error"]
    assert legacy_error["code"] == "provenance_error"
    assert legacy_error["reason_code"] == "manifest_required"
    assert "manifest is required" in legacy_error["message"]

    config_path = tmp_path / ref["config_path"]
    log_path = tmp_path / ref["log_path"]
    log_path.unlink()
    CampaignSession.initialize(config_path, log_path)
    manifest_path = provenance_module.manifest_path_for_log(log_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pending = provenance_module._manifest_with_pending_transaction(
        manifest,
        config_file=config_path,
        operation="append_suggestions",
        affected_row_ids=["row_1"],
        metadata={"appended_row_count": 1},
        resulting_hash="1" * 64,
        resulting_row_count=1,
    )
    provenance_module._write_json_atomic(manifest_path, pending)
    before_log = log_path.read_bytes()

    blocked = api_client.post("/campaign/summary", json=strict_ref)
    assert blocked.status_code == 409
    blocked_error = blocked.json()["error"]
    assert blocked_error["code"] == "provenance_recovery_required"
    assert blocked_error["reason_code"] == "pending_previous_state"
    assert blocked_error["retryable"] is False
    assert blocked_error["suggested_action"] == (
        "Run provenance recovery, then reload the campaign and retry."
    )
    assert "cancel the pending transaction" in blocked_error["recovery_action"]
    recovered = api_client.post(
        "/campaign/provenance/recover",
        json={
            **strict_ref,
            "expected_log_fingerprint": file_fingerprint(log_path),
        },
    )
    assert recovered.status_code == 200
    values = {
        row["field"]: row["value"]
        for row in recovered.json()["provenance"]["records"]
    }
    assert values["resume_status"] == "ready"
    assert log_path.read_bytes() == before_log


def test_api_provenance_recovery_requires_log_fingerprint(tmp_path: Path) -> None:
    ref = _copy_campaign(tmp_path)

    response = _client(tmp_path).post("/campaign/provenance/recover", json=ref)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "request_validation"
