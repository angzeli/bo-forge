"""Regression coverage for lifecycle publication, refresh, and validation boundaries."""

from __future__ import annotations

import json
import multiprocessing
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from bo_forge import CampaignSession
from bo_forge._campaign import provenance as io
from bo_forge.cli import run
from bo_forge.config import parse_campaign_config
from bo_forge.errors import LogConflictError, ProvenanceError
from bo_forge.provenance import (
    accept_provenance_config,
    adopt_provenance,
    fork_campaign,
    migrate_provenance,
)
from bo_forge_api.api import create_app
from tests._session_support import write_cost_review_config
from tests.test_provenance_lifecycle import apply, legacy, snapshot, v1


def _review_campaign(tmp_path, operation):
    cfg = write_cost_review_config(tmp_path / "campaign.yaml")
    log = tmp_path / "campaign.csv"
    campaign = CampaignSession.initialize(cfg, log)
    batch = campaign.suggest_next(batch_size=1)
    if operation != "append":
        campaign.append_suggestions(batch)
        if operation == "observe":
            campaign.review_suggestion(str(batch.iloc[0]["row_id"]), "accept")
    return campaign, batch


def _mutate(campaign, operation, batch):
    if operation == "append":
        campaign.append_suggestions(batch)
    elif operation == "review":
        campaign.review_suggestion(str(batch.iloc[0]["row_id"]), "accept")
    else:
        campaign.mark_observed(str(batch.iloc[0]["row_id"]), 1.0, actual_cost=0.5)


@pytest.mark.parametrize("schema", [1, 2])
@pytest.mark.parametrize("operation", ["append", "review", "observe"])
def test_reload_refreshes_ordinary_managed_mutations(tmp_path, schema, operation):
    writer, batch = _review_campaign(tmp_path, operation)
    cfg, log = writer.config_path, writer.log_path
    if schema == 1:
        _downgrade(log)
        writer = CampaignSession.from_files(cfg, log)
    reader = CampaignSession.from_files(cfg, log, provenance_policy="required")
    _mutate(writer, operation, batch)
    before = snapshot(cfg, log)
    with pytest.raises(LogConflictError):
        reader.append_suggestions(batch)
    reader.reload()
    reader.validate()
    assert reader.df.equals(writer.df)
    assert reader._manifest_fingerprint == io.manifest_fingerprint(log)
    assert reader._provenance_policy == "required"
    assert snapshot(cfg, log) == before


def test_adoption_success_survives_temporary_cleanup_failure(tmp_path, monkeypatch):
    cfg, log = legacy(tmp_path)
    before = snapshot(cfg, log)
    original = Path.unlink

    def fail_temporary_cleanup(path, *args, **kwargs):
        if path.name.endswith(".tmp") and io.manifest_path_for_log(log).exists():
            raise PermissionError("temporary cleanup unavailable")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_temporary_cleanup)
    assert apply(adopt_provenance, cfg, log)["applied"]
    assert snapshot(cfg, log)[:2] == before[:2]
    CampaignSession.from_files(cfg, log).validate()
    assert len(io.load_manifest(log)["events"]) == 1


def test_adoption_link_failure_leaves_campaign_legacy(tmp_path, monkeypatch):
    from bo_forge._campaign import provenance_lifecycle

    cfg, log = legacy(tmp_path)
    before = snapshot(cfg, log)

    def fail_link(*args):
        raise OSError("publication unavailable")

    monkeypatch.setattr(provenance_lifecycle.os, "link", fail_link)
    with pytest.raises(ProvenanceError, match="publication unavailable"):
        apply(adopt_provenance, cfg, log)
    assert snapshot(cfg, log) == before


@pytest.mark.parametrize("metadata", [{}, {"row_count": None}, {"row_count": -1},
                                      {"row_count": True}, {"row_count": "0"}])
def test_migration_rejects_invalid_baseline_before_preview_or_archiving(tmp_path, metadata):
    cfg, log = v1(tmp_path)
    preview = migrate_provenance(cfg, log)
    manifest = io.load_manifest(log)
    manifest["events"][0]["metadata"] = metadata
    io._write_json_atomic(io.manifest_path_for_log(log), manifest)
    before = snapshot(cfg, log)
    with pytest.raises(ProvenanceError, match="initialization metadata"):
        migrate_provenance(cfg, log)
    with pytest.raises(LogConflictError):
        migrate_provenance(cfg, log, apply=True, reason="Old preview",
                           expected_identities=preview["expected_identities"])
    client = TestClient(create_app(tmp_path), raise_server_exceptions=False)
    response = client.post("/campaign/provenance/migrate",
                           json={"config_path": cfg.name, "log_path": log.name})
    assert response.status_code == 400
    assert {"code", "message", "retryable", "suggested_action"} <= response.json()["error"].keys()
    expected = {**preview["expected_identities"], "config_path": cfg.name,
                "log_path": log.name, "manifest_sha256": io.manifest_fingerprint(log)}
    response = client.post("/campaign/provenance/migrate", json={
        "config_path": cfg.name, "log_path": log.name, "apply": True, "reason": "Reviewed",
        "expected_identities": expected,
    })
    assert response.status_code == 400
    assert "initialization metadata" in response.json()["error"]["message"]
    assert snapshot(cfg, log) == before
    assert not list(tmp_path.glob("*.archive.*"))


@pytest.mark.parametrize("value", [[], None, 1, {}, {"expected_identities": []}])
def test_cli_invalid_preview_shape_is_controlled(tmp_path, capsys, value):
    cfg, log = legacy(tmp_path)
    before = snapshot(cfg, log)
    preview = tmp_path / "preview.json"
    preview.write_text(json.dumps(value))
    assert run(["provenance-adopt", "--config", str(cfg), "--log", str(log),
                "--apply", "--reason", "Reviewed", "--preview", str(preview)]) == 1
    assert "Preview must be an object" in capsys.readouterr().err
    assert snapshot(cfg, log) == before


@pytest.mark.parametrize("data", [b"", b'row_id,status\n"unterminated', b"\xff"])
def test_invalid_csv_is_a_controlled_cli_and_api_failure(tmp_path, capsys, data):
    cfg, log = legacy(tmp_path)
    log.write_bytes(data)
    before = snapshot(cfg, log)
    assert run(["provenance-adopt", "--config", str(cfg), "--log", str(log)]) == 1
    assert "Cannot read campaign CSV" in capsys.readouterr().err
    client = TestClient(create_app(tmp_path), raise_server_exceptions=False)
    response = client.post("/campaign/provenance/adopt",
                           json={"config_path": cfg.name, "log_path": log.name})
    assert response.status_code == 400
    assert "error" in response.json()
    assert snapshot(cfg, log) == before


@pytest.mark.parametrize("changes", [[], None, 1, "not a mapping"])
def test_cli_rejects_nonmapping_fork_changes(tmp_path, capsys, changes):
    cfg, log = v1(tmp_path)
    before = snapshot(cfg, log)
    assert run(["provenance-fork", "--config", str(cfg), "--log", str(log),
                "--destination", str(tmp_path / "child"),
                "--config-changes", json.dumps(changes)]) == 1
    assert "Config changes must be a JSON object" in capsys.readouterr().err
    assert snapshot(cfg, log) == before
    assert not (tmp_path / "child").exists()


@pytest.mark.parametrize("changes", [
    {}, {"model": {"previous": "invented", "resulting": "x"}},
    {"campaign_name": {"previous": "wrong", "resulting": "child"}},
])
def test_child_differences_must_match_captured_configurations(tmp_path, changes):
    cfg, log = v1(tmp_path)
    child = tmp_path / "child"
    apply(fork_campaign, cfg, log, child, config_changes={"campaign_name": "child"})
    path = child / "campaign.csv"
    manifest = io.load_manifest(path)
    manifest["origin"]["parent"]["config_differences"] = changes
    io._write_json_atomic(io.manifest_path_for_log(path), manifest)
    before = snapshot(child / "campaign.yaml", path)
    with pytest.raises(ProvenanceError, match="differences do not match"):
        CampaignSession.from_files(child / "campaign.yaml", path)
    assert snapshot(child / "campaign.yaml", path) == before


def test_child_formatting_acceptance_preserves_semantic_lineage(tmp_path):
    cfg, log = legacy(tmp_path)
    cfg.write_text(
        cfg.read_text() + '\nconstraints:\n  - name: bound\n    expression: "x <= 0.9"\n'
    )
    apply(adopt_provenance, cfg, log)
    child = tmp_path / "child"
    apply(fork_campaign, cfg, log, child, config_changes={"campaign_name": "child"})
    child_cfg, child_log = child / "campaign.yaml", child / "campaign.csv"
    child_cfg.write_text(child_cfg.read_text().replace("x <= 0.9", "x<=0.9"))
    apply(accept_provenance_config, child_cfg, child_log)
    CampaignSession.from_files(child_cfg, child_log).validate()


def test_child_snapshot_cannot_reinterpret_protected_definitions(tmp_path):
    cfg, log = v1(tmp_path)
    child = tmp_path / "child"
    apply(fork_campaign, cfg, log, child)
    child_cfg, child_log = child / "campaign.yaml", child / "campaign.csv"
    manifest = io.load_manifest(child_log)
    raw = yaml.safe_load(manifest["config"]["snapshot"])
    raw["variables"][0]["upper"] = 2
    content = yaml.safe_dump(raw).encode()
    child_cfg.write_bytes(content)
    manifest["config"] = {
        "snapshot": content.decode(), "byte_sha256": io._sha256_bytes(content),
        "semantic_sha256": io.config_semantic_sha256(parse_campaign_config(raw)),
    }
    io._write_json_atomic(io.manifest_path_for_log(child_log), manifest)
    before = snapshot(child_cfg, child_log)
    with pytest.raises(ProvenanceError, match="protected campaign definitions"):
        CampaignSession.from_files(child_cfg, child_log)
    assert snapshot(child_cfg, child_log) == before


def _downgrade(log):
    manifest = io.load_manifest(log)
    manifest["schema_version"] = 1
    manifest.pop("origin")
    manifest.pop("archives")
    io._write_json_atomic(io.manifest_path_for_log(log), manifest)


def _process_race(cfg, log, operation, batch, preview, barrier, results):
    campaign = CampaignSession.from_files(cfg, log)
    barrier.wait(timeout=30)
    try:
        if operation == "migrate":
            migrate_provenance(cfg, log, apply=True, reason="Process race",
                               expected_identities=preview["expected_identities"])
        else:
            _mutate(campaign, operation, batch)
        results.put("committed")
    except LogConflictError:
        results.put("blocked")


@pytest.mark.parametrize("operation", ["append", "review", "observe"])
def test_lifecycle_and_mutation_processes_do_not_lose_events(tmp_path, operation):
    campaign, batch = _review_campaign(tmp_path, operation)
    cfg, log = campaign.config_path, campaign.log_path
    _downgrade(log)
    preview = migrate_provenance(cfg, log)
    count = len(io.load_manifest(log)["events"])
    context = multiprocessing.get_context("spawn")
    barrier, results = context.Barrier(3), context.Queue()
    processes = [context.Process(target=_process_race,
                                 args=(cfg, log, op, batch, preview, barrier, results))
                 for op in ("migrate", operation)]
    try:
        for process in processes:
            process.start()
        barrier.wait(timeout=30)
        assert sorted(results.get(timeout=30) for _ in processes) == ["blocked", "committed"]
        for process in processes:
            process.join(timeout=30)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        results.close()
    CampaignSession.from_files(cfg, log).validate()
    assert len(io.load_manifest(log)["events"]) == count + 1
