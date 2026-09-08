"""Bounded lifecycle acceptance without fitting an optimization model."""

from __future__ import annotations

import shutil

import pytest

from bo_forge import CampaignSession, ProvenanceRecoveryRequired, recover_provenance
from bo_forge._campaign import provenance as io
from bo_forge.provenance import (
    accept_provenance_config,
    adopt_provenance,
    fork_campaign,
    migrate_provenance,
)
from tests._provenance_support import _downgrade, apply, snapshot
from tests._session_support import write_cost_review_config

KINDS = ("legacy", "v1", "v2", "adopted", "migrated", "forked")


def _campaign(root, kind):
    root.mkdir()
    cfg = write_cost_review_config(root / "campaign.yaml", initial_design_size=8)
    log = root / "campaign.csv"
    campaign = CampaignSession.initialize(cfg, log)
    if kind in {"legacy", "adopted"}:
        io.manifest_path_for_log(log).unlink()
        if kind == "adopted":
            apply(adopt_provenance, cfg, log)
    elif kind in {"v1", "migrated"}:
        _downgrade(log)
        if kind == "migrated":
            apply(migrate_provenance, cfg, log)
    elif kind == "forked":
        apply(fork_campaign, cfg, log, root / "child")
        cfg, log = root / "child/campaign.yaml", root / "child/campaign.csv"
    policy = "compatible" if kind == "legacy" else "required"
    campaign = CampaignSession.from_files(cfg, log, provenance_policy=policy)
    return campaign


@pytest.mark.parametrize("kind", KINDS)
def test_complete_campaign_acceptance_sequence(tmp_path, kind):
    campaign = _campaign(tmp_path / "source", kind)
    cfg, log = campaign.config_path, campaign.log_path
    before = snapshot(cfg, log)
    suggestions = campaign.suggest_next(batch_size=1)
    campaign.validate()
    campaign.report()
    campaign.provenance_summary()
    assert snapshot(cfg, log) == before
    previous = io.load_manifest(log) if kind != "legacy" else None
    row_id = str(suggestions.iloc[0]["row_id"])
    campaign.append_suggestions(suggestions)
    campaign.review_suggestion(row_id, "accept")
    campaign.mark_observed(row_id, 1.25, actual_cost=0.5)
    campaign.reload()
    campaign.validate()
    assert campaign._provenance_policy == ("compatible" if kind == "legacy" else "required")
    assert campaign.df.iloc[-1]["status"] == "observed"
    assert cfg.read_bytes() == before[0]
    if previous is None:
        assert not io.manifest_path_for_log(log).exists()
        assert "provenance" not in campaign.report()
        return
    assert dict(campaign.report()["provenance"].values)["resume_status"] == "ready"
    current = io.load_manifest(log)
    assert current["schema_version"] == previous["schema_version"]
    assert current["campaign_id"] == previous["campaign_id"]
    assert current["events"][:-3] == previous["events"]
    assert [event["operation"] for event in current["events"][-3:]] == [
        "append_suggestions", "review_suggestion", "mark_observed",
    ]
    assert all(event["affected_row_ids"] == [row_id] for event in current["events"][-3:])
    assert [event["sequence"] for event in current["events"]] == list(
        range(1, len(current["events"]) + 1)
    )
    assert current["log"]["sha256"] == io._sha256_file(log)
    assert current["log"]["row_count"] == len(campaign.df)


@pytest.mark.parametrize("kind", KINDS[1:])
@pytest.mark.parametrize("resulting", [False, True])
def test_recovery_then_ordinary_mutation_preserves_origin(tmp_path, kind, resulting):
    campaign = _campaign(tmp_path / "source", kind)
    cfg, log = campaign.config_path, campaign.log_path
    batch = campaign.suggest_next(batch_size=1)
    manifest = io.load_manifest(log)
    candidate = batch.to_csv(index=False).encode("utf-8")
    row_id = str(batch.iloc[0]["row_id"])
    pending = io._manifest_with_pending_transaction(
        manifest, config_file=cfg, operation="append_suggestions",
        affected_row_ids=[row_id], metadata={"appended_row_count": 1},
        resulting_hash=io._sha256_bytes(candidate), resulting_row_count=1,
    )
    io._write_json_atomic(io.manifest_path_for_log(log), pending)
    if resulting:
        log.write_bytes(candidate)
    before = snapshot(cfg, log)
    with pytest.raises(ProvenanceRecoveryRequired):
        CampaignSession.from_files(cfg, log, provenance_policy="required")
    with pytest.raises(ProvenanceRecoveryRequired):
        campaign.append_suggestions(batch)
    assert snapshot(cfg, log) == before
    recover_provenance(cfg, log, expected_log_fingerprint=io._sha256_file(log))
    after = snapshot(cfg, log)
    assert after[:2] == before[:2]
    recover_provenance(cfg, log)
    assert snapshot(cfg, log) == after
    recovered = io.load_manifest(log)
    assert len(recovered["events"]) == len(manifest["events"]) + int(resulting)
    if resulting:
        assert recovered["events"][-1] == pending["pending_transaction"]["event"]
    campaign = CampaignSession.from_files(cfg, log, provenance_policy="required")
    if not resulting:
        campaign.append_suggestions(batch)
    campaign.review_suggestion(row_id, "accept")
    campaign.mark_observed(row_id, 2.5, actual_cost=0.25)
    campaign.validate()
    final = io.load_manifest(log)
    assert final.get("origin") == manifest.get("origin")
    assert final["schema_version"] == manifest["schema_version"]
    assert len(final["events"]) == len(manifest["events"]) + 3


def test_migrated_archives_and_child_are_portable_and_independent(tmp_path):
    parent = _campaign(tmp_path / "original", "migrated")
    cfg, log = parent.config_path, parent.log_path
    old_config = cfg.read_bytes()
    cfg.write_bytes(old_config + b"\n# portable formatting\n")
    apply(accept_provenance_config, cfg, log)
    parent = CampaignSession.from_files(cfg, log, provenance_policy="required")
    batch = parent.suggest_next(batch_size=1)
    parent.append_suggestions(batch)
    row_id = str(batch.iloc[0]["row_id"])
    parent.review_suggestion(row_id, "accept")
    parent.mark_observed(row_id, 1.5, actual_cost=0.5)
    before = snapshot(cfg, log)
    shutil.copytree(cfg.parent, tmp_path / "relocated")
    cfg.parent.rename(tmp_path / "parent-moved")
    cfg, log = tmp_path / "relocated/campaign.yaml", tmp_path / "relocated/campaign.csv"
    CampaignSession.from_files(cfg, log, provenance_policy="required").validate()
    assert snapshot(cfg, log) == before
    manifest = io.load_manifest(log)
    archived_bytes = {
        reference["path"]: (log.parent / reference["path"]).read_bytes()
        for reference in manifest["archives"]
    }
    assert old_config in archived_bytes.values()
    apply(fork_campaign, cfg, log, tmp_path / "child", config_changes={"campaign_name": "child"})
    shutil.copytree(tmp_path / "child", tmp_path / "independent-child")
    cfg.parent.rename(tmp_path / "parent-moved-again")
    child = CampaignSession.from_files(
        tmp_path / "independent-child/campaign.yaml", tmp_path / "independent-child/campaign.csv",
        provenance_policy="required",
    )
    assert child.log_path.read_bytes() == before[1]
    next_batch = child.suggest_next(batch_size=1)
    child.append_suggestions(next_batch)
    next_id = str(next_batch.iloc[0]["row_id"])
    child.review_suggestion(next_id, "accept")
    child.mark_observed(next_id, 3.5, actual_cost=0.5)
    child.reload()
    child.validate()
    assert child.df.iloc[0].equals(parent.df.iloc[0])
    assert dict(child.provenance_summary().values)["parent_campaign_id"] == manifest["campaign_id"]
    moved = tmp_path / "parent-moved-again"
    assert snapshot(moved / "campaign.yaml", moved / "campaign.csv") == before
    assert all((moved / name).read_bytes() == data for name, data in archived_bytes.items())
