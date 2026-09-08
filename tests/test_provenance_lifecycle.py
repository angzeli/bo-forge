"""Lifecycle transactions preserve campaign bytes and reject stale intent."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from bo_forge import CampaignSession
from bo_forge._campaign import provenance as io
from bo_forge.application import CampaignAppService, staged_bundle_invalidation_reason
from bo_forge.errors import (
    BOForgeError,
    LogConflictError,
    ProvenanceError,
    ProvenanceRecoveryRequired,
)
from bo_forge.provenance import (
    accept_provenance_config,
    adopt_provenance,
    fork_campaign,
    migrate_provenance,
    provenance_summary,
    recover_provenance,
)
from tests._provenance_support import apply, legacy, snapshot, v1


def test_service_direct_session_retains_staged_manifest_identity(tmp_path):
    cfg, log = v1(tmp_path)
    loaded = CampaignSession.from_files(cfg, log)
    direct = CampaignSession(cfg, log, loaded.config, loaded.df)
    result = CampaignAppService.from_session(direct).suggest_dry_run(1)
    assert result.bundle["manifest_fingerprint"] == io.manifest_fingerprint(log)
    assert staged_bundle_invalidation_reason(result.bundle, cfg, log) is None


@pytest.mark.parametrize("corruption", ["archive_kind", "migration_version"])
def test_v2_malformed_archive_and_migration_types_fail_closed(tmp_path, corruption):
    cfg, log = v1(tmp_path)
    apply(migrate_provenance, cfg, log)
    manifest = io.load_manifest(log)
    if corruption == "archive_kind":
        manifest["archives"][0]["kind"] = []
    else:
        manifest["events"][-1]["metadata"]["from_schema"] = True
    io._write_json_atomic(io.manifest_path_for_log(log), manifest)
    before = snapshot(cfg, log)
    with pytest.raises(ProvenanceError):
        CampaignSession.from_files(cfg, log)
    assert snapshot(cfg, log) == before


@pytest.mark.parametrize("populated", [False, True])
def test_adoption_unknown_history_and_source_bytes(tmp_path, populated):
    cfg, log = legacy(tmp_path)
    if populated:
        session = CampaignSession.from_files(cfg, log)
        session.append_suggestions(session.suggest_next(batch_size=1))
        session.mark_observed(session.df.iloc[0]["row_id"], 1.5)
    before = snapshot(cfg, log)
    preview = adopt_provenance(cfg, log)
    assert preview["history"] == "earlier_history_unknown"
    assert snapshot(cfg, log) == before
    result = apply(adopt_provenance, cfg, log)
    assert result["applied"]
    assert snapshot(cfg, log)[:2] == before[:2]
    manifest = io.load_manifest(log)
    assert manifest["schema_version"] == 2
    assert manifest["origin"]["history"] == "earlier_history_unknown"
    assert [e["operation"] for e in manifest["events"]] == ["adopt"]
    assert manifest["events"][0]["affected_row_ids"] == []
    assert manifest["origin"]["baseline_row_count"] == int(populated)
    CampaignSession.from_files(cfg, log, provenance_policy="required").validate()


@pytest.mark.parametrize("content", [b"{}", b"invalid json"])
def test_adoption_refuses_any_existing_manifest(tmp_path, content):
    cfg, log = legacy(tmp_path)
    io.manifest_path_for_log(log).write_bytes(content)
    before = snapshot(cfg, log)
    with pytest.raises(ProvenanceError, match="existing manifest"):
        adopt_provenance(cfg, log)
    assert snapshot(cfg, log) == before


def test_concurrent_adoption_has_one_winner(tmp_path):
    cfg, log = legacy(tmp_path)
    preview = adopt_provenance(cfg, log)

    def attempt():
        try:
            return adopt_provenance(
                cfg,
                log,
                apply=True,
                reason="Adopt",
                expected_identities=preview["expected_identities"],
            )["applied"]
        except LogConflictError:
            return False

    with ThreadPoolExecutor(2) as pool:
        assert sorted(pool.map(lambda _: attempt(), range(2))) == [False, True]
    assert len(io.load_manifest(log)["events"]) == 1


def test_migration_archives_exact_v1_and_preserves_history(tmp_path):
    cfg, log = v1(tmp_path)
    before = snapshot(cfg, log)
    old = io.load_manifest(log)
    apply(migrate_provenance, cfg, log)
    new = io.load_manifest(log)
    assert new["campaign_id"] == old["campaign_id"]
    assert new["created_at"] == old["created_at"]
    assert new["events"][:-1] == old["events"]
    assert new["events"][-1]["operation"] == "migrate"
    archive = log.parent / new["archives"][0]["path"]
    assert archive.read_bytes() == before[2]
    assert archive.stat().st_mode & 0o222 == 0
    assert snapshot(cfg, log)[:2] == before[:2]
    after = snapshot(cfg, log)
    assert apply(migrate_provenance, cfg, log)["no_op"]
    assert snapshot(cfg, log) == after


def test_v1_mutation_does_not_upgrade(tmp_path):
    cfg, log = v1(tmp_path)
    session = CampaignSession.from_files(cfg, log)
    session.append_suggestions(session.suggest_next(batch_size=1))
    assert io.load_manifest(log)["schema_version"] == 1


def test_format_acceptance_archives_old_config_and_rejects_semantics(tmp_path):
    cfg, log = v1(tmp_path)
    cfg.write_bytes(cfg.read_bytes() + b"\n# formatting\n")
    with pytest.raises(ProvenanceError, match="v2 migration"):
        accept_provenance_config(cfg, log)
    cfg.write_bytes(cfg.read_bytes().removesuffix(b"\n# formatting\n"))
    apply(migrate_provenance, cfg, log)
    old = snapshot(cfg, log)
    cfg.write_bytes(old[0] + b"\n# reviewed formatting\n")
    preview = accept_provenance_config(cfg, log)
    accept_provenance_config(
        cfg,
        log,
        apply=True,
        reason="Comment update",
        expected_identities=preview["expected_identities"],
    )
    new = io.load_manifest(log)
    event = new["events"][-1]
    assert event["operation"] == "accept_config"
    assert (log.parent / event["metadata"]["previous_config"]["path"]).read_bytes() == old[0]
    assert (log.parent / event["metadata"]["previous_manifest"]["path"]).read_bytes() == old[2]
    assert log.read_bytes() == old[1]
    CampaignSession.from_files(cfg, log).validate()
    with pytest.raises(LogConflictError, match="stale"):
        accept_provenance_config(
            cfg,
            log,
            apply=True,
            reason="Replay",
            expected_identities=preview["expected_identities"],
        )
    cfg.write_text(cfg.read_text().replace("upper: 1", "upper: 2"))
    before = snapshot(cfg, log)
    with pytest.raises(LogConflictError, match="semantics"):
        accept_provenance_config(cfg, log)
    assert snapshot(cfg, log) == before


@pytest.mark.parametrize("edit", ["config", "log", "manifest"])
def test_stale_preview_never_applies(tmp_path, edit):
    cfg, log = legacy(tmp_path)
    preview = adopt_provenance(cfg, log)
    target = {"config": cfg, "log": log, "manifest": io.manifest_path_for_log(log)}[edit]
    target.write_bytes((target.read_bytes() if target.exists() else b"") + b"\n")
    before = snapshot(cfg, log)
    with pytest.raises(LogConflictError, match="stale"):
        adopt_provenance(
            cfg, log, apply=True, reason="Stale", expected_identities=preview["expected_identities"]
        )
    assert snapshot(cfg, log) == before


@pytest.mark.parametrize("version", [1, 2])
def test_fork_inherits_csv_with_portable_parent_snapshot(tmp_path, version):
    cfg, log = v1(tmp_path)
    session = CampaignSession.from_files(cfg, log)
    session.append_suggestions(session.suggest_next(batch_size=1))
    session.mark_observed(session.df.iloc[0]["row_id"], 2.0)
    if version == 2:
        apply(migrate_provenance, cfg, log)
    before = snapshot(cfg, log)
    destination = tmp_path / "child"
    result = apply(
        fork_campaign,
        cfg,
        log,
        destination,
        config_changes={"campaign_name": "child", "bo": {"random_seed": 42}},
    )
    assert result["applied"]
    assert snapshot(cfg, log) == before
    child = CampaignSession.from_files(destination / "campaign.yaml", destination / "campaign.csv")
    assert child.config.campaign_name == "child"
    assert child.log_path.read_bytes() == before[1]
    manifest = io.load_manifest(child.log_path)
    assert manifest["campaign_id"] != io.load_manifest(log)["campaign_id"]
    assert [e["operation"] for e in manifest["events"]] == ["fork"]
    parent = manifest["origin"]["parent"]
    assert (destination / parent["manifest"]["path"]).read_bytes() == before[2]
    cfg.rename(tmp_path / "moved.yaml")
    log.rename(tmp_path / "moved.csv")
    assert (
        dict(provenance_summary(child.config_path, child.log_path).values)["parent_campaign_id"]
        == parent["campaign_id"]
    )


@pytest.mark.parametrize(
    "changes", [{"variables": []}, {"objective": {}}, {"bo": {"acquisition": "invalid"}}]
)
def test_invalid_fork_definitions_leave_parent_unchanged(tmp_path, changes):
    cfg, log = v1(tmp_path)
    before = snapshot(cfg, log)
    with pytest.raises(BOForgeError):
        fork_campaign(cfg, log, tmp_path / "child", config_changes=changes)
    assert snapshot(cfg, log) == before
    assert not (tmp_path / "child").exists()


def test_fork_pending_queue_and_destination_collision(tmp_path):
    cfg, log = v1(tmp_path)
    session = CampaignSession.from_files(cfg, log)
    session.append_suggestions(session.suggest_next(batch_size=1))
    with pytest.raises(ProvenanceError, match="observation queue"):
        fork_campaign(cfg, log, tmp_path / "child")
    session.mark_observed(session.df.iloc[0]["row_id"], 1.0)
    preview = fork_campaign(cfg, log, tmp_path / "child")
    (tmp_path / "child").mkdir()
    with pytest.raises(ProvenanceError, match="already exists"):
        fork_campaign(
            cfg,
            log,
            tmp_path / "child",
            apply=True,
            reason="Child",
            expected_identities=preview["expected_identities"],
        )


def test_interrupted_manifest_publish_preserves_active_state(tmp_path, monkeypatch):
    cfg, log = v1(tmp_path)
    before = snapshot(cfg, log)
    monkeypatch.setattr(
        io, "_write_json_atomic", lambda *a, **k: (_ for _ in ()).throw(OSError("injected"))
    )
    with pytest.raises(ProvenanceError, match="injected"):
        apply(migrate_provenance, cfg, log)
    assert snapshot(cfg, log) == before


def test_pending_migration_requires_explicit_recovery(tmp_path):
    from tests.test_provenance_resume import _pending_previous

    cfg, log = v1(tmp_path)
    _pending_previous(cfg, log)
    before = snapshot(cfg, log)
    with pytest.raises(ProvenanceRecoveryRequired):
        migrate_provenance(cfg, log)
    assert snapshot(cfg, log) == before
    recover_provenance(cfg, log)
    apply(migrate_provenance, cfg, log)


def test_migration_invalidates_session_and_staged_bundle(tmp_path):
    cfg, log = v1(tmp_path)
    service = CampaignAppService.load(cfg, log)
    staged = service.suggest_dry_run(1)
    apply(migrate_provenance, cfg, log)
    before = snapshot(cfg, log)
    assert "manifest changed" in staged_bundle_invalidation_reason(staged.bundle, cfg, log)
    with pytest.raises(LogConflictError, match="manifest changed"):
        service.session.append_suggestions(staged.suggestions)
    with pytest.raises(LogConflictError, match="manifest changed"):
        service.session.reload()
    with pytest.raises(ValueError, match="manifest changed"):
        service.append_staged(staged.bundle)
    assert snapshot(cfg, log) == before


def test_archive_escape_and_tampering_are_rejected(tmp_path):
    cfg, log = v1(tmp_path)
    apply(migrate_provenance, cfg, log)
    manifest = io.load_manifest(log)
    archive = tmp_path / manifest["archives"][0]["path"]
    archive.chmod(0o644)
    archive.write_bytes(b"tampered")
    with pytest.raises(ProvenanceError, match="archive hash"):
        CampaignSession.from_files(cfg, log)
    manifest["archives"][0]["path"] = "../escape"
    io._write_json_atomic(io.manifest_path_for_log(log), manifest)
    with pytest.raises(ProvenanceError, match="sibling"):
        io.load_manifest(log)


def test_new_v2_ledgers_continue_after_lifecycle_events(tmp_path):
    cfg, log = legacy(tmp_path)
    apply(adopt_provenance, cfg, log)
    campaign = CampaignSession.from_files(cfg, log)
    campaign.append_suggestions(campaign.suggest_next(batch_size=1))
    campaign.mark_observed(campaign.df.iloc[0]["row_id"], 2.5)
    assert [e["operation"] for e in io.load_manifest(log)["events"]] == [
        "adopt",
        "append_suggestions",
        "mark_observed",
    ]


def test_invalid_log_and_missing_reason_fail_without_writes(tmp_path):
    cfg, log = legacy(tmp_path)
    preview = adopt_provenance(cfg, log)
    with pytest.raises(ProvenanceError, match="nonempty reason"):
        adopt_provenance(cfg, log, apply=True, expected_identities=preview["expected_identities"])
    log.write_text("invalid,column\n1,2\n")
    before = snapshot(cfg, log)
    with pytest.raises(BOForgeError):
        adopt_provenance(cfg, log)
    assert snapshot(cfg, log) == before


def test_error_after_manifest_replacement_rolls_back(tmp_path, monkeypatch):
    cfg, log = v1(tmp_path)
    before = snapshot(cfg, log)
    original = io._write_json_atomic

    def write_then_fail(path, payload):
        original(path, payload)
        raise OSError("failure after replacement")

    monkeypatch.setattr(io, "_write_json_atomic", write_then_fail)
    with pytest.raises(ProvenanceError, match="failure after replacement"):
        apply(migrate_provenance, cfg, log)
    assert snapshot(cfg, log) == before


def test_fork_publication_failure_leaves_no_visible_child(tmp_path, monkeypatch):
    from bo_forge._campaign import provenance_fork

    cfg, log = v1(tmp_path)
    before = snapshot(cfg, log)

    def fail(*args):
        raise OSError("publication failed")

    monkeypatch.setattr(provenance_fork, "_rename_directory_exclusive", fail)
    with pytest.raises(ProvenanceError, match="publication failed"):
        apply(fork_campaign, cfg, log, tmp_path / "child")
    assert not (tmp_path / "child").exists()
    assert snapshot(cfg, log) == before


def test_destination_created_during_preparation_is_never_overwritten(tmp_path, monkeypatch):
    from bo_forge._campaign import provenance_fork

    cfg, log = v1(tmp_path)
    original = provenance_fork._rename_directory_exclusive

    def competing_directory(source, destination):
        destination.mkdir()
        original(source, destination)

    monkeypatch.setattr(provenance_fork, "_rename_directory_exclusive", competing_directory)
    with pytest.raises(ProvenanceError, match="already exists"):
        apply(fork_campaign, cfg, log, tmp_path / "child")
    assert list((tmp_path / "child").iterdir()) == []


@pytest.mark.parametrize("operation", ["append", "review", "observe"])
def test_lifecycle_races_with_mutation_are_serialized(tmp_path, operation):
    import threading

    from tests._session_support import write_cost_review_config

    cfg = write_cost_review_config(tmp_path / "campaign.yaml")
    log = tmp_path / "campaign.csv"
    campaign = CampaignSession.initialize(cfg, log)
    batch = campaign.suggest_next(batch_size=1)
    if operation != "append":
        campaign.append_suggestions(batch)
        if operation == "observe":
            campaign.review_suggestion(str(batch.iloc[0]["row_id"]), "accept")
    path = io.manifest_path_for_log(log)
    manifest = io.load_manifest(log)
    manifest.update(schema_version=1)
    manifest.pop("origin")
    manifest.pop("archives")
    io._write_json_atomic(path, manifest)
    campaign = CampaignSession.from_files(cfg, log)
    preview = migrate_provenance(cfg, log)
    barrier = threading.Barrier(2)

    def migrate():
        barrier.wait(timeout=5)
        try:
            migrate_provenance(
                cfg,
                log,
                apply=True,
                reason="Migration",
                expected_identities=preview["expected_identities"],
            )
            return "migrated"
        except LogConflictError:
            return "blocked"

    def mutate():
        barrier.wait(timeout=5)
        try:
            if operation == "append":
                campaign.append_suggestions(batch)
            elif operation == "review":
                campaign.review_suggestion(str(batch.iloc[0]["row_id"]), "accept")
            else:
                campaign.mark_observed(str(batch.iloc[0]["row_id"]), 1.0, actual_cost=0.5)
            return "mutated"
        except LogConflictError:
            return "blocked"

    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(migrate), pool.submit(mutate)]
        outcomes = [future.result(timeout=10) for future in futures]
    assert outcomes.count("blocked") == 1
    CampaignSession.from_files(cfg, log).validate()
    ledger = io.load_manifest(log)
    assert len(ledger["events"]) == len(manifest["events"]) + 1


def test_format_acceptance_preserves_crlf_snapshot_exactly(tmp_path):
    cfg, log = v1(tmp_path)
    apply(migrate_provenance, cfg, log)
    cfg.write_bytes(cfg.read_bytes().replace(b"\n", b"\r\n"))
    apply(accept_provenance_config, cfg, log)
    assert io.load_manifest(log)["config"]["snapshot"].encode() == cfg.read_bytes()


def test_lifecycle_canonicalizes_log_aliases(tmp_path):
    cfg, log = legacy(tmp_path)
    alias = tmp_path / "alias.csv"
    alias.symlink_to(log)
    preview = adopt_provenance(cfg, alias)
    adopt_provenance(
        cfg,
        log,
        apply=True,
        reason="Canonical campaign",
        expected_identities=preview["expected_identities"],
    )
    assert io.manifest_path_for_log(alias) == io.manifest_path_for_log(log)
    assert not (tmp_path / "alias.csv.manifest.json").exists()


@pytest.mark.parametrize("decision", ["reject", "defer"])
def test_fork_retains_rejected_and_deferred_history(tmp_path, decision):
    from tests._session_support import write_cost_review_config

    cfg = write_cost_review_config(tmp_path / "campaign.yaml")
    campaign = CampaignSession.initialize(cfg, tmp_path / "campaign.csv")
    batch = campaign.suggest_next(batch_size=1)
    campaign.append_suggestions(batch)
    campaign.review_suggestion(str(batch.iloc[0]["row_id"]), decision)
    apply(fork_campaign, cfg, campaign.log_path, tmp_path / "child")
    assert (tmp_path / "child" / "campaign.csv").read_bytes() == campaign.log_path.read_bytes()


@pytest.mark.parametrize("corruption", ["unknown_origin", "extra_metadata", "bad_parent"])
def test_strict_v2_rejects_lifecycle_metadata_corruption(tmp_path, corruption):
    cfg, log = v1(tmp_path)
    apply(migrate_provenance, cfg, log)
    manifest = io.load_manifest(log)
    if corruption == "unknown_origin":
        manifest["origin"]["kind"] = []
    elif corruption == "extra_metadata":
        manifest["events"][-1]["metadata"]["force"] = True
    else:
        manifest["origin"]["parent"] = {"campaign_id": "invented"}
    io._write_json_atomic(io.manifest_path_for_log(log), manifest)
    with pytest.raises(ProvenanceError):
        io.load_manifest(log)
