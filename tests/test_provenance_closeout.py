"""Regression tests for rollback ownership and fail-closed resume diagnostics."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from bo_forge import CampaignSession, ProvenanceError, provenance_summary
from bo_forge._campaign import provenance as io
from bo_forge.cli import run
from bo_forge.provenance import accept_provenance_config, migrate_provenance
from bo_forge_api.api import create_app
from tests._provenance_support import _downgrade, apply, legacy, snapshot
from tests._session_support import write_config


def test_initialize_rollback_preserves_intervening_committed_writer(tmp_path, monkeypatch):
    cfg = write_config(tmp_path / "campaign.yaml")
    log = tmp_path / "campaign.csv"
    initialize = io._initialize_campaign_files
    load = CampaignSession.from_files
    committed = {}

    def write():
        writer = load(cfg, log)
        row = dict.fromkeys(writer.df.columns, "")
        row.update(row_id="other_writer", iteration=0, status="suggested", source="sobol", x=0.25)
        writer.append_suggestions(pd.DataFrame([row], columns=writer.df.columns))
        committed["snapshot"] = snapshot(cfg, log)

    def publish_then_write(*args, **kwargs):
        result = initialize(*args, **kwargs)
        # Schedule a real writer after publication releases the lock, before session loading.
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(write).result(timeout=15)
        return result

    def fail_load(*args, **kwargs):
        raise OSError("Injected session load failure")

    monkeypatch.setattr(io, "_initialize_campaign_files", publish_then_write)
    monkeypatch.setattr(CampaignSession, "from_files", classmethod(fail_load))
    with pytest.raises(ProvenanceError, match="rollback was incomplete"):
        CampaignSession.initialize(cfg, log)
    assert snapshot(cfg, log) == committed["snapshot"]
    loaded = load(cfg, log)
    loaded.validate()
    assert loaded.df["row_id"].tolist() == ["other_writer"]
    assert [event["operation"] for event in io.load_manifest(log)["events"]] == [
        "initialize", "append_suggestions",
    ]


@pytest.mark.parametrize("schema", [1, 2])
@pytest.mark.parametrize("field,value", [
    ("environment_id", []), ("environment_id", {}), ("environment_id", None),
    ("environment_id", 1), ("sequence", True), ("sequence", 1.0), ("sequence", "1"),
])
def test_malformed_event_types_are_structured_and_non_mutating(
    tmp_path, capsys, schema, field, value,
):
    cfg = write_config(tmp_path / "campaign.yaml")
    log = tmp_path / "campaign.csv"
    CampaignSession.initialize(cfg, log)
    if schema == 1:
        _downgrade(log)
    manifest = io.load_manifest(log)
    manifest["events"][0][field] = value
    io._write_json_atomic(io.manifest_path_for_log(log), manifest)
    before = snapshot(cfg, log)
    with pytest.raises(ProvenanceError) as error:
        CampaignSession.from_files(cfg, log)
    assert error.value.reason_code == "manifest_invalid"
    assert run(["provenance", "--config", str(cfg), "--log", str(log)]) == 1
    assert "Traceback" not in capsys.readouterr().err
    client = TestClient(create_app(tmp_path), raise_server_exceptions=False)
    response = client.post("/campaign/provenance", json={
        "config_path": cfg.name, "log_path": log.name,
    })
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "provenance_error"
    assert error["reason_code"] == "manifest_invalid"
    assert error["retryable"] is False
    assert error["suggested_action"]
    assert snapshot(cfg, log) == before


@pytest.mark.parametrize("schema,pending", [(1, False), (2, False), (1, True), (2, True)])
def test_formatting_guidance_matches_available_lifecycle(tmp_path, capsys, schema, pending):
    cfg = write_config(tmp_path / "campaign.yaml")
    log = tmp_path / "campaign.csv"
    CampaignSession.initialize(cfg, log)
    if schema == 1:
        _downgrade(log)
    if pending:
        manifest = io.load_manifest(log)
        interrupted = io._manifest_with_pending_transaction(
            manifest, config_file=cfg, operation="append_suggestions", affected_row_ids=[],
            metadata={"appended_row_count": 0}, resulting_hash=manifest["log"]["sha256"],
            resulting_row_count=0,
        )
        io._write_json_atomic(io.manifest_path_for_log(log), interrupted)
    cfg.write_bytes(cfg.read_bytes() + b"\n# formatting only\n")
    before = snapshot(cfg, log)
    values = dict(provenance_summary(cfg, log).values)
    assert values["reason_code"] == "config_bytes_changed_semantics_same"
    action = values["recovery_action"]
    if pending:
        assert "provenance-recover" in action
        assert "provenance-accept-config" not in action
    elif schema == 1:
        assert "Restore" in action
        assert "provenance-migrate" in action
        assert "provenance-accept-config" in action
    else:
        assert "provenance-accept-config" in action
        assert "initialize" not in action
        assert accept_provenance_config(cfg, log)["no_op"] is False
    assert run(["provenance", "--config", str(cfg), "--log", str(log)]) == 1
    assert action in capsys.readouterr().err
    response = TestClient(create_app(tmp_path)).post("/campaign/provenance", json={
        "config_path": cfg.name, "log_path": log.name,
    })
    assert response.status_code == 200
    rows = response.json()["provenance"]["records"]
    assert {row["field"]: row["value"] for row in rows}["recovery_action"] == action
    assert snapshot(cfg, log) == before


def test_required_legacy_guidance_offers_explicit_adoption(tmp_path, capsys):
    cfg, log = legacy(tmp_path)
    before = snapshot(cfg, log)
    with pytest.raises(ProvenanceError) as error:
        CampaignSession.from_files(cfg, log, provenance_policy="required")
    assert "provenance-adopt" in error.value.recovery_action
    assert "compatible" in error.value.recovery_action
    assert run([
        "provenance", "--config", str(cfg), "--log", str(log), "--require-provenance",
    ]) == 1
    assert "provenance-adopt" in capsys.readouterr().err
    assert snapshot(cfg, log) == before


def test_original_v310_writer_fixture_remains_mutable_and_migratable(tmp_path):
    fixture = json.loads((Path(__file__).parent / "fixtures/provenance_v1.json").read_text())
    assert fixture["writer_version"] == "3.1.0"
    cfg, log = tmp_path / "campaign.yaml", tmp_path / "campaign.csv"
    cfg.write_bytes(fixture["config"].encode("utf-8"))
    log.write_bytes(fixture["csv"].encode("utf-8"))
    manifest_path = io.manifest_path_for_log(log)
    manifest_path.write_bytes(fixture["manifest"].encode("utf-8"))
    before = snapshot(cfg, log)
    campaign = CampaignSession.from_files(cfg, log, provenance_policy="required")
    campaign.validate()
    campaign.report()
    assert snapshot(cfg, log) == before
    original = io.load_manifest(log)
    assert original["schema_version"] == 1
    assert len(original["events"]) == 3
    batch = campaign.suggest_next(batch_size=1)
    campaign.append_suggestions(batch)
    campaign.mark_observed(str(batch.iloc[0]["row_id"]), 2.5)
    updated = io.load_manifest(log)
    assert updated["schema_version"] == 1
    assert updated["events"][:3] == original["events"]
    previous_bytes = manifest_path.read_bytes()
    apply(migrate_provenance, cfg, log)
    migrated = io.load_manifest(log)
    assert migrated["campaign_id"] == original["campaign_id"]
    assert migrated["events"][:-1] == updated["events"]
    reference = migrated["events"][-1]["metadata"]["previous_manifest"]
    assert (tmp_path / reference["path"]).read_bytes() == previous_bytes
    CampaignSession.from_files(cfg, log, provenance_policy="required").validate()
