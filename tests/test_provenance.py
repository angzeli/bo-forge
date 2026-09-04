"""Managed campaign provenance, transaction, and compatibility tests."""

from __future__ import annotations

import json
import multiprocessing
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

import bo_forge._campaign.provenance as provenance_module
import bo_forge._campaign.provenance_environment as provenance_environment
from bo_forge import CampaignSession, ProvenanceError, provenance_summary
from bo_forge.config import CampaignConfig
from bo_forge.errors import LogConflictError, LogWriteError
from bo_forge.logs import append_suggestions
from bo_forge.validation import canonical_columns
from tests._session_support import config, write_config, write_log


def _suggestion(cfg: CampaignConfig, row_id: str, x: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "row_id": row_id,
                "iteration": 0,
                "status": "suggested",
                "source": "sobol",
                "x": x,
                "score": "",
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            }
        ],
        columns=canonical_columns(cfg),
    )


def _manifest(log_path: Path) -> tuple[Path, dict[str, object]]:
    path = provenance_module.manifest_path_for_log(log_path)
    return path, json.loads(path.read_text(encoding="utf-8"))


def _append_in_process(
    config_path: str,
    log_path: str,
    row_id: str,
    x: float,
    barrier: object,
    results: object,
) -> None:
    cfg = CampaignConfig.from_yaml(config_path)
    try:
        barrier.wait(timeout=10)
        append_suggestions(log_path, _suggestion(cfg, row_id, x), config=cfg)
    except Exception as exc:
        results.put((row_id, type(exc).__name__, str(exc)))
    else:
        results.put((row_id, "ok", ""))


def test_initialize_writes_deterministic_schema_v1_manifest(tmp_path: Path) -> None:
    (tmp_path / "config").mkdir()
    config_path = write_config(tmp_path / "config" / "campaign.yaml")
    log_path = tmp_path / "data" / "campaign.csv"

    campaign = CampaignSession.initialize(config_path, log_path)

    manifest_path, manifest = _manifest(log_path)
    raw = manifest_path.read_text(encoding="utf-8")
    assert raw.endswith("\n")
    assert raw == json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    assert set(manifest) == {
        "campaign_id",
        "config",
        "created_at",
        "environments",
        "events",
        "log",
        "optimization",
        "paths",
        "pending_transaction",
        "schema_version",
        "updated_at",
    }
    assert manifest["schema_version"] == 1
    uuid.UUID(str(manifest["campaign_id"]))
    assert datetime.fromisoformat(str(manifest["created_at"]).replace("Z", "+00:00"))
    assert manifest["created_at"] == manifest["updated_at"]
    assert manifest["paths"] == {
        "config": "../config/campaign.yaml",
        "log": "campaign.csv",
    }
    assert "/Users/" not in raw
    assert manifest["config"]["snapshot"] == config_path.read_text(encoding="utf-8")
    assert set(manifest["config"]) == {
        "byte_sha256",
        "semantic_sha256",
        "snapshot",
    }
    assert manifest["log"]["row_count"] == 0
    event = manifest["events"][0]
    assert set(event) == {
        "affected_row_ids",
        "environment_id",
        "event_id",
        "metadata",
        "operation",
        "previous_log_sha256",
        "resulting_log_sha256",
        "sequence",
        "timestamp",
    }
    assert event["operation"] == "initialize"
    environment = manifest["environments"][0]
    assert environment["bo_forge"] == "3.1.0"
    assert event["environment_id"] == environment["environment_id"]
    assert manifest["pending_transaction"] is None
    assert campaign.is_provenance_managed is True


def test_initialize_round_trips_discrete_fidelity_optimization_identity(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "fidelity.yaml"
    config_path.write_text(
        """
campaign_name: provenance_fidelity
objective: {name: score, direction: maximize}
variables:
  - {name: x, type: continuous, lower: 0, upper: 1}
  - {name: fidelity, type: continuous, lower: 0, upper: 1}
fidelity:
  variable: fidelity
  levels: [0.0, 0.5, 1.0]
  target: 1.0
bo: {acquisition: qmf_kg, batch_size: 2, initial_design_size: 4}
""",
        encoding="utf-8",
    )

    campaign = CampaignSession.initialize(config_path, tmp_path / "fidelity.csv")
    reloaded = CampaignSession.from_files(config_path, campaign.log_path)
    _, manifest = _manifest(campaign.log_path)

    assert reloaded.is_provenance_managed is True
    assert manifest["optimization"]["fidelity"]["levels"] == [0.0, 0.5, 1.0]


def test_semantic_hash_ignores_yaml_formatting_but_tracks_real_change(
    tmp_path: Path,
) -> None:
    original = write_config(tmp_path / "original.yaml")
    formatted = tmp_path / "formatted.yaml"
    formatted.write_text(
        "# same parsed campaign with comments\n" + original.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    reordered = tmp_path / "reordered.yaml"
    reordered.write_text(
        """
bo:
  random_seed: 5
  acquisition: log_ei
  initial_design_size: 2
  batch_size: 1
  num_restarts: 2
  raw_samples: 16
  mc_samples: 16
variables:
  - upper: 1
    lower: 0
    type: continuous
    name: x
objective: {direction: maximize, name: score}
campaign_name: session_test
""",
        encoding="utf-8",
    )
    changed = write_config(tmp_path / "changed.yaml", initial_design_size=3)

    original_config = CampaignConfig.from_yaml(original)
    formatted_config = CampaignConfig.from_yaml(formatted)
    reordered_config = CampaignConfig.from_yaml(reordered)
    changed_config = CampaignConfig.from_yaml(changed)

    assert original.read_bytes() != formatted.read_bytes()
    assert provenance_module.config_semantic_sha256(
        original_config
    ) == provenance_module.config_semantic_sha256(formatted_config)
    assert original.read_bytes() != reordered.read_bytes()
    assert provenance_module.config_semantic_sha256(
        original_config
    ) == provenance_module.config_semantic_sha256(reordered_config)
    assert provenance_module.config_semantic_sha256(
        original_config
    ) != provenance_module.config_semantic_sha256(changed_config)


def test_initialize_refuses_to_overwrite_log_or_manifest(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    log_path = tmp_path / "campaign.csv"
    log_path.write_text("existing", encoding="utf-8")

    with pytest.raises(ProvenanceError, match="already exists"):
        CampaignSession.initialize(config_path, log_path)

    assert log_path.read_text(encoding="utf-8") == "existing"
    log_path.unlink()
    manifest_path = provenance_module.manifest_path_for_log(log_path)
    manifest_path.write_text("existing manifest", encoding="utf-8")
    with pytest.raises(ProvenanceError, match="already exists"):
        CampaignSession.initialize(config_path, log_path)
    assert not log_path.exists()
    assert manifest_path.read_text(encoding="utf-8") == "existing manifest"


def test_initialize_uses_home_relative_reference_without_author_path_exposure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    author_home = tmp_path / "home" / "author"
    author_home.mkdir(parents=True)
    config_path = write_config(author_home / "campaign.yaml")
    log_path = tmp_path / "shared" / "campaign.csv"
    monkeypatch.setattr(provenance_module, "_author_home", lambda: author_home)

    campaign = CampaignSession.initialize(config_path, log_path)
    manifest_path, manifest = _manifest(log_path)

    assert campaign.is_provenance_managed is True
    assert manifest["paths"]["config"] == "~/campaign.yaml"
    assert str(author_home) not in manifest_path.read_text(encoding="utf-8")
    assert list(log_path.parent.glob(".*.tmp")) == []


def test_initialize_failure_before_log_link_leaves_fail_closed_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SimulatedCrash(BaseException):
        pass

    config_path = write_config(tmp_path / "campaign.yaml")
    log_path = tmp_path / "campaign.csv"
    real_link = os.link
    link_calls = 0

    def crash_on_log_link(source: Path, destination: Path) -> None:
        nonlocal link_calls
        link_calls += 1
        if link_calls == 2:
            raise SimulatedCrash
        real_link(source, destination)

    monkeypatch.setattr(provenance_module.os, "link", crash_on_log_link)
    with pytest.raises(SimulatedCrash):
        provenance_module.initialize_campaign(config_path, log_path)

    assert not log_path.exists()
    assert provenance_module.manifest_path_for_log(log_path).exists()
    with pytest.raises(LogConflictError, match="does not match its provenance manifest"):
        CampaignSession.from_files(config_path, log_path)


def test_initialize_rolls_back_exact_new_artifacts_when_session_load_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    log_path = tmp_path / "campaign.csv"

    def fail_load(
        _cls: type[CampaignSession],
        _config_path: str | Path,
        _log_path: str | Path,
    ) -> CampaignSession:
        raise RuntimeError("load failed")

    monkeypatch.setattr(CampaignSession, "from_files", classmethod(fail_load))
    with pytest.raises(RuntimeError, match="load failed"):
        CampaignSession.initialize(config_path, log_path)

    assert not log_path.exists()
    assert not provenance_module.manifest_path_for_log(log_path).exists()


def test_managed_append_review_and_observation_write_ordered_events(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "campaign.yaml"
    config_path.write_text(
        """
campaign_name: provenance_review
objective: {name: score, direction: maximize}
variables:
  - {name: x, type: continuous, lower: 0, upper: 1}
review: {enabled: true}
bo: {batch_size: 1, initial_design_size: 2, random_seed: 5}
""",
        encoding="utf-8",
    )
    campaign = CampaignSession.initialize(config_path, tmp_path / "campaign.csv")
    suggestion = _suggestion(campaign.config, "row_1", 0.3)
    suggestion["review_status"] = "pending"
    suggestion["review_note"] = ""

    campaign.append_suggestions(suggestion)
    campaign.review_suggestion("row_1", "accept", "approved")
    campaign.mark_observed("row_1", 1.25)

    _, manifest = _manifest(campaign.log_path)
    assert [event["sequence"] for event in manifest["events"]] == [1, 2, 3, 4]
    assert [event["operation"] for event in manifest["events"]] == [
        "initialize",
        "append_suggestions",
        "review_suggestion",
        "mark_observed",
    ]
    assert manifest["events"][1]["affected_row_ids"] == ["row_1"]
    assert manifest["events"][2]["metadata"] == {"decision": "accepted"}
    assert manifest["events"][3]["metadata"]["actual_cost_recorded"] is False
    assert manifest["log"]["sha256"] == campaign.log_fingerprint
    assert manifest["log"]["row_count"] == 1
    assert len(manifest["environments"]) == 1
    assert manifest["pending_transaction"] is None
    report = campaign.report()
    assert "provenance" in report
    assert "Provenance" in campaign.export_report(tmp_path / "report.txt").read_text(
        encoding="utf-8"
    )


def test_managed_invalid_review_and_observation_leave_both_files_unchanged(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "campaign.yaml"
    config_path.write_text(
        """
campaign_name: provenance_failures
objective: {name: score, direction: maximize}
variables:
  - {name: x, type: continuous, lower: 0, upper: 1}
review: {enabled: true}
bo: {batch_size: 1, initial_design_size: 2, random_seed: 5}
""",
        encoding="utf-8",
    )
    campaign = CampaignSession.initialize(config_path, tmp_path / "campaign.csv")
    suggestion = _suggestion(campaign.config, "row_1", 0.3)
    suggestion["review_status"] = "pending"
    suggestion["review_note"] = ""
    campaign.append_suggestions(suggestion)
    manifest_path = provenance_module.manifest_path_for_log(campaign.log_path)
    before_log = campaign.log_path.read_bytes()
    before_manifest = manifest_path.read_bytes()

    with pytest.raises(LogWriteError, match="Invalid review decision"):
        campaign.review_suggestion("row_1", "invalid")
    assert campaign.log_path.read_bytes() == before_log
    assert manifest_path.read_bytes() == before_manifest

    campaign.review_suggestion("row_1", "accept")
    before_log = campaign.log_path.read_bytes()
    before_manifest = manifest_path.read_bytes()
    with pytest.raises(LogWriteError, match="must be finite"):
        campaign.mark_observed("row_1", float("nan"))
    assert campaign.log_path.read_bytes() == before_log
    assert manifest_path.read_bytes() == before_manifest


def test_reads_and_dry_run_do_not_add_provenance_events(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    campaign = CampaignSession.initialize(config_path, tmp_path / "campaign.csv")
    _, before = _manifest(campaign.log_path)

    campaign.validate()
    campaign.summary()
    campaign.report()
    campaign.suggest_next(batch_size=1)
    campaign.provenance_summary()

    _, after = _manifest(campaign.log_path)
    assert after == before


def test_managed_config_or_log_mismatch_fails_without_mutation(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    campaign = CampaignSession.initialize(config_path, tmp_path / "campaign.csv")
    suggestion = _suggestion(campaign.config, "row_1", 0.3)
    manifest_path = provenance_module.manifest_path_for_log(campaign.log_path)

    config_path.write_text(
        config_path.read_text(encoding="utf-8") + "# changed bytes\n",
        encoding="utf-8",
    )
    before_log = campaign.log_path.read_bytes()
    before_manifest = manifest_path.read_bytes()
    with pytest.raises(LogConflictError, match="config changed"):
        append_suggestions(campaign.log_path, suggestion, config=campaign.config)
    assert campaign.log_path.read_bytes() == before_log
    assert manifest_path.read_bytes() == before_manifest


def test_managed_external_log_change_fails_without_mutation(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    campaign = CampaignSession.initialize(config_path, tmp_path / "campaign.csv")
    manifest_path = provenance_module.manifest_path_for_log(campaign.log_path)
    _suggestion(campaign.config, "external", 0.2).to_csv(campaign.log_path, index=False)
    before_log = campaign.log_path.read_bytes()
    before_manifest = manifest_path.read_bytes()

    with pytest.raises(LogConflictError, match="does not match its provenance manifest"):
        append_suggestions(
            campaign.log_path,
            _suggestion(campaign.config, "new", 0.8),
            config=campaign.config,
        )

    assert campaign.log_path.read_bytes() == before_log
    assert manifest_path.read_bytes() == before_manifest


@pytest.mark.parametrize("changed_file", ["config", "log"])
def test_managed_campaign_load_fails_closed_on_current_file_mismatch(
    tmp_path: Path,
    changed_file: str,
) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    campaign = CampaignSession.initialize(config_path, tmp_path / "campaign.csv")
    if changed_file == "config":
        config_path.write_text(
            config_path.read_text(encoding="utf-8") + "# changed bytes\n",
            encoding="utf-8",
        )
    else:
        campaign.log_path.write_bytes(campaign.log_path.read_bytes() + b"\n")
    before_config = config_path.read_bytes()
    before_log = campaign.log_path.read_bytes()
    manifest_path = provenance_module.manifest_path_for_log(campaign.log_path)
    before_manifest = manifest_path.read_bytes()

    with pytest.raises(LogConflictError, match="does not match its provenance manifest"):
        CampaignSession.from_files(config_path, campaign.log_path)
    with pytest.raises(LogConflictError, match="does not match its provenance manifest"):
        campaign.reload()

    values = provenance_summary(config_path, campaign.log_path).set_index("field")["value"]
    assert values["integrity_status"] == "mismatch"
    assert config_path.read_bytes() == before_config
    assert campaign.log_path.read_bytes() == before_log
    assert manifest_path.read_bytes() == before_manifest


def test_provenance_summary_rejects_missing_legacy_log(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    log_path = tmp_path / "missing.csv"

    with pytest.raises(ProvenanceError, match="does not exist"):
        provenance_summary(config_path, log_path)


def test_managed_stale_session_rejects_write_without_losing_events(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "campaign.yaml", initial_design_size=3)
    first = CampaignSession.initialize(config_path, tmp_path / "campaign.csv")
    stale = CampaignSession.from_files(config_path, first.log_path)
    first.append_suggestions(_suggestion(first.config, "row_1", 0.2))
    manifest_path = provenance_module.manifest_path_for_log(first.log_path)
    before_log = first.log_path.read_bytes()
    before_manifest = manifest_path.read_bytes()

    with pytest.raises(LogConflictError, match="changed after it was loaded"):
        stale.append_suggestions(_suggestion(stale.config, "row_2", 0.8))

    assert first.log_path.read_bytes() == before_log
    assert manifest_path.read_bytes() == before_manifest


def test_managed_session_rejects_manifest_removal_without_mutating_log(
    tmp_path: Path,
) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    campaign = CampaignSession.initialize(config_path, tmp_path / "campaign.csv")
    manifest_path = provenance_module.manifest_path_for_log(campaign.log_path)
    manifest_path.unlink()
    before_log = campaign.log_path.read_bytes()

    with pytest.raises(LogConflictError, match="provenance state changed"):
        campaign.append_suggestions(_suggestion(campaign.config, "row_1", 0.3))

    assert campaign.log_path.read_bytes() == before_log
    assert not manifest_path.exists()


def test_legacy_session_rejects_manifest_appearance_without_mutating_files(
    tmp_path: Path,
) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    cfg = config()
    log_path = write_log(tmp_path / "campaign.csv", cfg)
    campaign = CampaignSession.from_files(config_path, log_path)
    manifest_path = provenance_module.manifest_path_for_log(log_path)
    manifest_path.write_text("{}\n", encoding="utf-8")
    before_log = log_path.read_bytes()
    before_manifest = manifest_path.read_bytes()

    with pytest.raises(LogConflictError, match="provenance state changed"):
        campaign.append_suggestions(_suggestion(cfg, "row_1", 0.3))

    assert log_path.read_bytes() == before_log
    assert manifest_path.read_bytes() == before_manifest


def test_managed_pending_write_failure_rolls_back_both_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    campaign = CampaignSession.initialize(config_path, tmp_path / "campaign.csv")
    manifest_path = provenance_module.manifest_path_for_log(campaign.log_path)
    before_log = campaign.log_path.read_bytes()
    before_manifest = manifest_path.read_bytes()

    def fail_pending(_path: Path, _payload: dict[str, object]) -> None:
        raise OSError("pending write failed")

    monkeypatch.setattr(provenance_module, "_write_json_atomic", fail_pending)
    with pytest.raises(LogWriteError, match="pending write failed"):
        campaign.append_suggestions(_suggestion(campaign.config, "row_1", 0.3))

    assert campaign.log_path.read_bytes() == before_log
    assert manifest_path.read_bytes() == before_manifest


def test_managed_preflight_failure_cleans_candidate_temp_and_preserves_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    campaign = CampaignSession.initialize(config_path, tmp_path / "campaign.csv")
    manifest_path = provenance_module.manifest_path_for_log(campaign.log_path)
    before_log = campaign.log_path.read_bytes()
    before_manifest = manifest_path.read_bytes()

    def fail_backup(_path: Path) -> Path:
        raise OSError("backup unavailable")

    monkeypatch.setattr(provenance_module, "_copy_backup", fail_backup)
    with pytest.raises(LogWriteError, match="Could not prepare managed campaign mutation"):
        campaign.append_suggestions(_suggestion(campaign.config, "row_1", 0.3))

    assert campaign.log_path.read_bytes() == before_log
    assert manifest_path.read_bytes() == before_manifest
    assert list(tmp_path.glob(".campaign.csv.*.tmp")) == []
    assert list(tmp_path.glob(".campaign.csv.*.bak")) == []


def test_candidate_csv_serialization_failure_is_typed_and_cleans_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    campaign = CampaignSession.initialize(config_path, tmp_path / "campaign.csv")
    manifest_path = provenance_module.manifest_path_for_log(campaign.log_path)
    before_log = campaign.log_path.read_bytes()
    before_manifest = manifest_path.read_bytes()

    def fail_serialization(*_args: object, **_kwargs: object) -> None:
        raise OSError("CSV serialization failed")

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_serialization)
    with pytest.raises(LogWriteError, match="Could not prepare candidate campaign log"):
        campaign.append_suggestions(_suggestion(campaign.config, "row_1", 0.3))

    assert campaign.log_path.read_bytes() == before_log
    assert manifest_path.read_bytes() == before_manifest
    assert list(tmp_path.glob(".campaign.csv.*.tmp")) == []


def test_manifest_temp_serialization_failure_is_typed_and_cleans_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    log_path = tmp_path / "campaign.csv"

    def fail_serialization(_payload: dict[str, object]) -> bytes:
        raise TypeError("not serializable")

    monkeypatch.setattr(provenance_module, "_manifest_bytes", fail_serialization)
    with pytest.raises(ProvenanceError, match="Could not prepare provenance manifest"):
        CampaignSession.initialize(config_path, log_path)

    assert not log_path.exists()
    assert not provenance_module.manifest_path_for_log(log_path).exists()
    assert list(tmp_path.glob(".*.tmp")) == []


def test_managed_finalize_failure_rolls_back_log_and_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    campaign = CampaignSession.initialize(config_path, tmp_path / "campaign.csv")
    manifest_path = provenance_module.manifest_path_for_log(campaign.log_path)
    before_log = campaign.log_path.read_bytes()
    before_manifest = manifest_path.read_bytes()
    real_write = provenance_module._write_json_atomic
    calls = 0

    def fail_finalize(path: Path, payload: dict[str, object]) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("finalize failed")
        real_write(path, payload)

    monkeypatch.setattr(provenance_module, "_write_json_atomic", fail_finalize)
    with pytest.raises(LogWriteError, match="finalize failed"):
        campaign.append_suggestions(_suggestion(campaign.config, "row_1", 0.3))

    assert campaign.log_path.read_bytes() == before_log
    assert manifest_path.read_bytes() == before_manifest


def test_rollback_failure_retains_recovery_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    campaign = CampaignSession.initialize(config_path, tmp_path / "campaign.csv")
    manifest_path = provenance_module.manifest_path_for_log(campaign.log_path)
    before_log = campaign.log_path.read_bytes()
    before_manifest = manifest_path.read_bytes()
    real_write = provenance_module._write_json_atomic
    real_replace = Path.replace
    write_calls = 0

    def fail_finalize(path: Path, payload: dict[str, object]) -> None:
        nonlocal write_calls
        write_calls += 1
        if write_calls == 2:
            raise OSError("finalize failed")
        real_write(path, payload)

    def fail_log_restore(source: Path, target: str | Path) -> Path:
        if source.suffix == ".bak" and Path(target) == campaign.log_path:
            raise OSError("restore failed")
        return real_replace(source, target)

    monkeypatch.setattr(provenance_module, "_write_json_atomic", fail_finalize)
    monkeypatch.setattr(Path, "replace", fail_log_restore)
    with pytest.raises(LogWriteError, match="Recovery CSV retained"):
        campaign.append_suggestions(_suggestion(campaign.config, "row_1", 0.3))

    backups = list(tmp_path.glob(".campaign.csv.*.bak"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == before_log
    assert manifest_path.read_bytes() == before_manifest


def test_interrupted_transaction_is_reported_then_finalized_on_next_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SimulatedCrash(BaseException):
        pass

    config_path = write_config(tmp_path / "campaign.yaml", initial_design_size=3)
    campaign = CampaignSession.initialize(config_path, tmp_path / "campaign.csv")
    real_write = provenance_module._write_json_atomic
    calls = 0

    def crash_before_finalize(path: Path, payload: dict[str, object]) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise SimulatedCrash
        real_write(path, payload)

    monkeypatch.setattr(provenance_module, "_write_json_atomic", crash_before_finalize)
    with pytest.raises(SimulatedCrash):
        append_suggestions(
            campaign.log_path,
            _suggestion(campaign.config, "row_1", 0.2),
            config=campaign.config,
        )
    monkeypatch.setattr(provenance_module, "_write_json_atomic", real_write)

    summary = provenance_summary(config_path, campaign.log_path)
    assert bool(summary.set_index("field").loc["pending_transaction", "value"])
    reloaded = CampaignSession.from_files(config_path, campaign.log_path)
    assert reloaded.is_provenance_managed is True
    append_suggestions(
        campaign.log_path,
        _suggestion(campaign.config, "row_2", 0.8),
        config=campaign.config,
    )

    _, manifest = _manifest(campaign.log_path)
    assert [event["sequence"] for event in manifest["events"]] == [1, 2, 3]
    assert [event["affected_row_ids"] for event in manifest["events"][1:]] == [
        ["row_1"],
        ["row_2"],
    ]
    assert manifest["pending_transaction"] is None


def test_pending_transaction_is_cancelled_when_log_kept_previous_hash(
    tmp_path: Path,
) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    campaign = CampaignSession.initialize(config_path, tmp_path / "campaign.csv")
    manifest_path, manifest = _manifest(campaign.log_path)
    pending = provenance_module._manifest_with_pending_transaction(
        manifest,
        config_file=config_path,
        operation="append_suggestions",
        affected_row_ids=["cancelled"],
        metadata={"appended_row_count": 1},
        resulting_hash="1" * 64,
        resulting_row_count=1,
    )
    provenance_module._write_json_atomic(manifest_path, pending)

    reloaded = CampaignSession.from_files(config_path, campaign.log_path)
    assert reloaded.is_provenance_managed is True

    append_suggestions(
        campaign.log_path,
        _suggestion(campaign.config, "row_1", 0.3),
        config=campaign.config,
    )

    _, recovered = _manifest(campaign.log_path)
    assert [event["affected_row_ids"] for event in recovered["events"]] == [[], ["row_1"]]
    assert recovered["pending_transaction"] is None


def test_pending_transaction_with_unknown_log_hash_fails_closed(
    tmp_path: Path,
) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    campaign = CampaignSession.initialize(config_path, tmp_path / "campaign.csv")
    manifest_path, manifest = _manifest(campaign.log_path)
    pending = provenance_module._manifest_with_pending_transaction(
        manifest,
        config_file=config_path,
        operation="append_suggestions",
        affected_row_ids=["interrupted"],
        metadata={"appended_row_count": 1},
        resulting_hash="1" * 64,
        resulting_row_count=1,
    )
    provenance_module._write_json_atomic(manifest_path, pending)
    _suggestion(campaign.config, "external", 0.2).to_csv(campaign.log_path, index=False)
    before_log = campaign.log_path.read_bytes()
    before_manifest = manifest_path.read_bytes()

    with pytest.raises(LogConflictError, match="matches neither"):
        append_suggestions(
            campaign.log_path,
            _suggestion(campaign.config, "new", 0.8),
            config=campaign.config,
        )

    assert campaign.log_path.read_bytes() == before_log
    assert manifest_path.read_bytes() == before_manifest


def test_managed_symlink_path_uses_canonical_manifest_and_preserves_modes(
    tmp_path: Path,
) -> None:
    if os.name == "nt":
        pytest.skip("POSIX symlink and mode behavior is platform-specific")
    config_path = write_config(tmp_path / "campaign.yaml")
    campaign = CampaignSession.initialize(config_path, tmp_path / "campaign.csv")
    linked_log = tmp_path / "linked.csv"
    linked_log.symlink_to(campaign.log_path)
    manifest_path = provenance_module.manifest_path_for_log(campaign.log_path)
    campaign.log_path.chmod(0o640)
    manifest_path.chmod(0o600)

    linked = CampaignSession.from_files(config_path, linked_log)
    linked.append_suggestions(_suggestion(linked.config, "row_1", 0.3))

    assert provenance_module.manifest_path_for_log(linked_log) == manifest_path
    assert campaign.log_path.stat().st_mode & 0o777 == 0o640
    assert manifest_path.stat().st_mode & 0o777 == 0o600


def test_managed_threaded_appends_preserve_rows_and_event_sequences(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "campaign.yaml", initial_design_size=3)
    campaign = CampaignSession.initialize(config_path, tmp_path / "campaign.csv")
    barrier = threading.Barrier(2)

    def append(row_id: str, x: float) -> None:
        barrier.wait(timeout=5)
        append_suggestions(
            campaign.log_path,
            _suggestion(campaign.config, row_id, x),
            config=campaign.config,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(append, "thread_1", 0.2),
            executor.submit(append, "thread_2", 0.8),
        ]
        for future in futures:
            future.result(timeout=15)

    _, manifest = _manifest(campaign.log_path)
    assert sorted(pd.read_csv(campaign.log_path)["row_id"].tolist()) == [
        "thread_1",
        "thread_2",
    ]
    assert [event["sequence"] for event in manifest["events"]] == [1, 2, 3]


def test_managed_process_appends_preserve_rows_and_event_sequences(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "campaign.yaml", initial_design_size=3)
    campaign = CampaignSession.initialize(config_path, tmp_path / "campaign.csv")
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    results = context.Queue()
    processes = [
        context.Process(
            target=_append_in_process,
            args=(str(config_path), str(campaign.log_path), row_id, x, barrier, results),
        )
        for row_id, x in (("process_1", 0.2), ("process_2", 0.8))
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0

    outcomes = sorted(results.get(timeout=2) for _ in processes)
    assert outcomes == [("process_1", "ok", ""), ("process_2", "ok", "")]
    _, manifest = _manifest(campaign.log_path)
    assert sorted(pd.read_csv(campaign.log_path)["row_id"].tolist()) == [
        "process_1",
        "process_2",
    ]
    assert [event["sequence"] for event in manifest["events"]] == [1, 2, 3]


def test_legacy_campaign_remains_manifest_free_and_report_compatible(
    tmp_path: Path,
) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    cfg = config()
    log_path = write_log(tmp_path / "campaign.csv", cfg)
    campaign = CampaignSession.from_files(config_path, log_path)
    before_report_keys = list(campaign.report())

    campaign.append_suggestions(_suggestion(cfg, "row_1", 0.5))

    assert not provenance_module.manifest_path_for_log(log_path).exists()
    assert campaign.provenance_summary().to_dict("records") == [
        {"field": "provenance_status", "value": "legacy"}
    ]
    assert list(campaign.report()) == before_report_keys


def test_unknown_manifest_schema_fails_clearly(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    campaign = CampaignSession.initialize(config_path, tmp_path / "campaign.csv")
    manifest_path, manifest = _manifest(campaign.log_path)
    manifest["schema_version"] = 99
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ProvenanceError, match="Unsupported provenance schema_version"):
        CampaignSession.from_files(config_path, campaign.log_path)


def test_malformed_manifest_snapshot_fails_as_provenance_error(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    campaign = CampaignSession.initialize(config_path, tmp_path / "campaign.csv")
    manifest_path, manifest = _manifest(campaign.log_path)
    snapshot = "campaign_name: [unterminated\n"
    manifest["config"]["snapshot"] = snapshot
    manifest["config"]["byte_sha256"] = provenance_module._sha256_bytes(
        snapshot.encode("utf-8")
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ProvenanceError, match="invalid config snapshot"):
        CampaignSession.from_files(config_path, campaign.log_path)


def test_truncated_manifest_event_fails_as_provenance_error(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    campaign = CampaignSession.initialize(config_path, tmp_path / "campaign.csv")
    manifest_path, manifest = _manifest(campaign.log_path)
    del manifest["events"][0]["environment_id"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ProvenanceError, match="invalid event record"):
        CampaignSession.from_files(config_path, campaign.log_path)


def test_duplicate_event_ids_fail_as_provenance_error(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    campaign = CampaignSession.initialize(config_path, tmp_path / "campaign.csv")
    campaign.append_suggestions(_suggestion(campaign.config, "row_1", 0.3))
    manifest_path, manifest = _manifest(campaign.log_path)
    manifest["events"][1]["event_id"] = manifest["events"][0]["event_id"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ProvenanceError, match="duplicate event IDs"):
        CampaignSession.from_files(config_path, campaign.log_path)


def test_non_monotonic_event_timestamps_fail_as_provenance_error(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    campaign = CampaignSession.initialize(config_path, tmp_path / "campaign.csv")
    campaign.append_suggestions(_suggestion(campaign.config, "row_1", 0.3))
    manifest_path, manifest = _manifest(campaign.log_path)
    manifest["events"][1]["timestamp"] = "2000-01-01T00:00:00Z"
    manifest["updated_at"] = manifest["events"][1]["timestamp"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ProvenanceError, match="non-monotonic event timestamps"):
        CampaignSession.from_files(config_path, campaign.log_path)


def test_new_event_timestamp_does_not_move_backward_with_system_clock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    campaign = CampaignSession.initialize(config_path, tmp_path / "campaign.csv")
    _, before = _manifest(campaign.log_path)
    monkeypatch.setattr(provenance_module, "_utc_now", lambda: "2000-01-01T00:00:00Z")

    campaign.append_suggestions(_suggestion(campaign.config, "row_1", 0.3))

    reloaded = CampaignSession.from_files(config_path, campaign.log_path)
    _, after = _manifest(campaign.log_path)
    assert reloaded.is_provenance_managed is True
    assert after["events"][1]["timestamp"] == before["updated_at"]


def test_tampered_environment_identity_fails_as_provenance_error(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    campaign = CampaignSession.initialize(config_path, tmp_path / "campaign.csv")
    manifest_path, manifest = _manifest(campaign.log_path)
    manifest["environments"][0]["python"] = "tampered"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ProvenanceError, match="invalid environment identity"):
        CampaignSession.from_files(config_path, campaign.log_path)


def test_git_identity_uses_one_bounded_status_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    class Result:
        stdout = "# branch.oid abc123\n1 .M N... tracked.py\n"

    def fake_run(command: list[str], **_kwargs: object) -> Result:
        calls.append(command)
        return Result()

    monkeypatch.setattr(provenance_environment.subprocess, "run", fake_run)

    assert provenance_environment._git_identity(tmp_path) == {
        "commit": "abc123",
        "dirty": True,
    }
    assert calls == [["git", "status", "--porcelain=v2", "--branch"]]
