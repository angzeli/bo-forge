"""CLI, HTTP and Streamlit lifecycle adapters share backend preview/apply checks."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from bo_forge._campaign import provenance as io
from bo_forge.provenance import migrate_provenance
from bo_forge_api.api import create_app
from tests.test_provenance_lifecycle import apply, legacy, snapshot, v1

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("operation", ["adopt", "migrate", "accept-config", "fork"])
def test_api_lifecycle_preview_apply_and_stale_retry(tmp_path, operation):
    cfg, log = legacy(tmp_path) if operation == "adopt" else v1(tmp_path)
    if operation == "accept-config":
        apply(migrate_provenance, cfg, log)
        cfg.write_bytes(cfg.read_bytes() + b"\n# updated comment\n")
    client = TestClient(create_app(tmp_path))
    request = {"config_path": cfg.name, "log_path": log.name}
    if operation == "fork":
        request.update(destination="child", config_changes={"campaign_name": "new child"})
    before = snapshot(cfg, log)
    url = f"/campaign/provenance/{operation}"
    response = client.post(url, json=request)
    assert response.status_code == 200, response.text
    preview = response.json()
    assert str(tmp_path) not in response.text
    assert preview["applied"] is False
    assert snapshot(cfg, log) == before
    assert client.post(url, json={**request, "apply": True}).status_code == 422
    commit = {
        **request,
        "apply": True,
        "reason": "Reviewed",
        "expected_identities": preview["expected_identities"],
    }
    response = client.post(url, json=commit)
    assert response.status_code == 200, response.text
    assert response.json()["applied"]
    assert snapshot(cfg, log)[:2] == before[:2]
    failed = client.post(url, json=commit)
    assert failed.status_code == 400
    assert {"code", "message", "retryable", "suggested_action"} <= failed.json()["error"].keys()


def test_api_lifecycle_paths_and_archives_bounded(tmp_path):
    cfg, log = v1(tmp_path)
    client = TestClient(create_app(tmp_path))
    request = {"config_path": cfg.name, "log_path": log.name, "destination": "../outside"}
    response = client.post("/campaign/provenance/fork", json=request)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "path_outside_root"
    apply(migrate_provenance, cfg, log)
    manifest = io.load_manifest(log)
    archive = tmp_path / manifest["archives"][0]["path"]
    raw = archive.read_bytes()
    archive.unlink()
    outside = tmp_path / "outside.json"
    outside.write_bytes(raw)
    archive.symlink_to(outside)
    response = client.post(
        "/campaign/provenance/migrate", json={"config_path": cfg.name, "log_path": log.name}
    )
    assert response.json()["error"]["code"] == "path_outside_root"


def test_server_stage_stales_after_manifest_only_migration(tmp_path):
    cfg, log = v1(tmp_path)
    client = TestClient(create_app(tmp_path))
    request = {"config_path": cfg.name, "log_path": log.name, "batch_size": 1}
    dry_run = client.post("/campaign/suggestions/dry-run", json=request)
    assert dry_run.status_code == 200, dry_run.text
    stage_id = dry_run.json()["stage"]["stage_id"]
    apply(migrate_provenance, cfg, log)
    before = snapshot(cfg, log)
    append = client.post(f"/campaign/stages/{stage_id}/append")
    assert append.status_code == 409, append.text
    assert append.json()["error"]["code"] == "stage_stale"
    assert snapshot(cfg, log) == before


def test_cli_adopt_preview_apply_and_provenance(tmp_path):
    cfg, log = legacy(tmp_path)
    command = [
        sys.executable,
        "-m",
        "bo_forge",
        "provenance-adopt",
        "--config",
        str(cfg),
        "--log",
        str(log),
    ]
    result = subprocess.run(command, capture_output=True, text=True, cwd=ROOT, check=True)
    preview = tmp_path / "preview.json"
    preview.write_text(result.stdout)
    assert json.loads(result.stdout)["applied"] is False
    result = subprocess.run(
        command + ["--apply", "--reason", "Existing data", "--preview", str(preview)],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=True,
    )
    assert json.loads(result.stdout)["applied"] is True
    result = subprocess.run(
        [sys.executable, "-m", "bo_forge", "provenance", "--config", str(cfg), "--log", str(log)],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=True,
    )
    assert "earlier_history_unknown" in result.stdout


@pytest.mark.parametrize("operation", ["adopt", "migrate", "accept-config", "fork"])
def test_streamlit_lifecycle_confirm_reload(tmp_path, operation):
    from streamlit.testing.v1 import AppTest

    cfg, log = legacy(tmp_path) if operation == "adopt" else v1(tmp_path)
    if operation == "accept-config":
        apply(migrate_provenance, cfg, log)
        cfg.write_bytes(cfg.read_bytes() + b"\n# formatting\n")
    before = snapshot(cfg, log)
    app = AppTest.from_file(ROOT / "bo_forge_app" / "streamlit_app.py").run(timeout=15)
    next(w for w in app.text_input if w.label == "YAML config path").set_value(str(cfg))
    next(w for w in app.text_input if w.label == "CSV log path").set_value(str(log))
    next(w for w in app.button if w.label == "Load campaign").click().run(timeout=15)
    next(w for w in app.button if w.label == "Inspect lifecycle actions").click().run(timeout=15)
    next(w for w in app.selectbox if w.label == "Lifecycle action").set_value(operation).run(
        timeout=15
    )
    if operation == "fork":
        next(w for w in app.text_input if w.label == "New campaign directory").set_value(
            str(tmp_path / "child")
        ).run(timeout=15)
    next(w for w in app.button if w.label == "Preview lifecycle action").click().run(timeout=15)
    assert not app.exception
    assert next(w for w in app.button if w.label == "Apply lifecycle action").disabled
    assert snapshot(cfg, log) == before
    next(w for w in app.text_input if w.label == "Lifecycle reason").set_value("Start tracking")
    next(
        w for w in app.checkbox if w.label == "Confirm this provenance lifecycle change"
    ).check().run(timeout=15)
    next(w for w in app.button if w.label == "Apply lifecycle action").click().run(timeout=15)
    assert not app.exception
    assert snapshot(cfg, log)[:2] == before[:2]
    if operation == "fork":
        assert io.load_manifest(tmp_path / "child" / "campaign.csv")["origin"]["kind"] == "fork"
    else:
        assert io.load_manifest(log)["schema_version"] == 2
    from bo_forge_app.streamlit_helpers import SESSION_KEY

    assert app.session_state[SESSION_KEY].provenance_policy == "compatible"
