"""Fail-closed provenance loading and explicit recovery tests."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pytest

import bo_forge._campaign.provenance as provenance_module
import bo_forge._campaign.provenance_resume as resume_module
import bo_forge.logs as logs_module
from bo_forge import (
    CampaignSession,
    ProvenanceError,
    ProvenanceRecoveryRequired,
    provenance_summary,
    recover_provenance,
)
from bo_forge.errors import LogBusyError, LogConflictError
from bo_forge.logs import append_suggestions
from bo_forge.validation import canonical_columns
from tests._session_support import config, write_config, write_cost_review_config, write_log


def _manifest(log_path: Path) -> tuple[Path, dict[str, object]]:
    path = provenance_module.manifest_path_for_log(log_path)
    return path, json.loads(path.read_text(encoding="utf-8"))


def _suggestion(row_id: str = "row_1") -> pd.DataFrame:
    cfg = config()
    row = {column: "" for column in canonical_columns(cfg)}
    row.update(
        {
            "row_id": row_id,
            "iteration": 0,
            "status": "suggested",
            "source": "sobol",
            "x": 0.4,
        }
    )
    return pd.DataFrame([row], columns=canonical_columns(cfg))


def _pending_previous(config_path: Path, log_path: Path) -> Path:
    manifest_path, manifest = _manifest(log_path)
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
    return manifest_path


def _pending_resulting(config_path: Path, log_path: Path) -> Path:
    manifest_path, manifest = _manifest(log_path)
    _suggestion().to_csv(log_path, index=False)
    resulting_hash = provenance_module._sha256_file(log_path)
    pending = provenance_module._manifest_with_pending_transaction(
        manifest,
        config_file=config_path,
        operation="append_suggestions",
        affected_row_ids=["row_1"],
        metadata={"appended_row_count": 1},
        resulting_hash=resulting_hash,
        resulting_row_count=1,
    )
    provenance_module._write_json_atomic(manifest_path, pending)
    return manifest_path


def test_provenance_policy_compatible_and_required_for_legacy_campaign(
    tmp_path: Path,
) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    log_path = write_log(tmp_path / "campaign.csv", config())

    compatible = CampaignSession.from_files(config_path, log_path)
    assert compatible.is_provenance_managed is False
    with pytest.raises(ProvenanceError) as error:
        CampaignSession.from_files(
            config_path,
            log_path,
            provenance_policy="required",
        )
    assert error.value.reason_code == "manifest_required"
    assert log_path.read_bytes() == write_log(tmp_path / "expected.csv", config()).read_bytes()


def test_managed_initialize_and_reload_keep_required_policy(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    campaign = CampaignSession.initialize(config_path, tmp_path / "campaign.csv")

    assert campaign._provenance_policy == "required"
    campaign.reload()
    provenance_module.manifest_path_for_log(campaign.log_path).unlink()
    with pytest.raises(ProvenanceError) as error:
        campaign.reload()
    assert error.value.reason_code == "manifest_required"


def test_unknown_provenance_policy_fails_before_loading(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="provenance_policy"):
        CampaignSession.from_files(
            tmp_path / "missing.yaml",
            tmp_path / "missing.csv",
            provenance_policy="ignore",
        )


def test_manifest_invalid_and_path_mismatch_reason_codes(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    invalid = CampaignSession.initialize(config_path, tmp_path / "invalid.csv")
    invalid_manifest = provenance_module.manifest_path_for_log(invalid.log_path)
    invalid_manifest.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ProvenanceError) as invalid_error:
        CampaignSession.from_files(config_path, invalid.log_path)
    assert invalid_error.value.reason_code == "manifest_invalid"

    mismatched = CampaignSession.initialize(config_path, tmp_path / "mismatch.csv")
    manifest_path, manifest = _manifest(mismatched.log_path)
    manifest["paths"]["config"] = "other.yaml"
    provenance_module._write_json_atomic(manifest_path, manifest)
    with pytest.raises(ProvenanceError) as path_error:
        CampaignSession.from_files(config_path, mismatched.log_path)
    assert path_error.value.reason_code == "manifest_path_mismatch"


def test_missing_managed_log_has_stable_blocked_reason(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    campaign = CampaignSession.initialize(config_path, tmp_path / "campaign.csv")
    campaign.log_path.unlink()

    values = dict(provenance_summary(config_path, campaign.log_path).itertuples(False, None))
    assert values["resume_status"] == "blocked"
    assert values["reason_code"] == "log_missing"
    with pytest.raises(LogConflictError) as error:
        CampaignSession.from_files(config_path, campaign.log_path)
    assert error.value.reason_code == "log_missing"


def test_provenance_summary_row_order_and_ready_state(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    campaign = CampaignSession.initialize(config_path, tmp_path / "campaign.csv")

    summary = provenance_summary(config_path, campaign.log_path)
    assert summary["field"].tolist()[:9] == [
        "provenance_status",
        "integrity_status",
        "resume_status",
        "reason_code",
        "recovery_action",
        "config_semantic_match",
        "current_environment_id",
        "environment_match",
        "environment_changes",
    ]
    values = dict(summary.itertuples(index=False, name=None))
    assert values["resume_status"] == "ready"
    assert values["reason_code"] is None
    assert values["config_semantic_match"] is True


@pytest.mark.parametrize(
    ("edit", "reason_code"),
    [
        (lambda text: text + "# formatting only\n", "config_bytes_changed_semantics_same"),
        (
            lambda text: text.replace("campaign_name: session_test", "campaign_name: changed"),
            "config_semantics_changed",
        ),
    ],
)
def test_config_mismatch_reason_codes(
    tmp_path: Path,
    edit: object,
    reason_code: str,
) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    campaign = CampaignSession.initialize(config_path, tmp_path / "campaign.csv")
    before_log = campaign.log_path.read_bytes()
    manifest_path = provenance_module.manifest_path_for_log(campaign.log_path)
    before_manifest = manifest_path.read_bytes()
    transform = edit
    assert callable(transform)
    config_path.write_text(transform(config_path.read_text(encoding="utf-8")), encoding="utf-8")

    values = dict(provenance_summary(config_path, campaign.log_path).itertuples(False, None))
    assert values["resume_status"] == "blocked"
    assert values["reason_code"] == reason_code
    assert values["log_bytes_match"] is True
    with pytest.raises(LogConflictError) as error:
        CampaignSession.from_files(config_path, campaign.log_path)
    assert error.value.reason_code == reason_code
    assert campaign.log_path.read_bytes() == before_log
    assert manifest_path.read_bytes() == before_manifest


def test_malformed_managed_config_is_classified_without_parser_leakage(
    tmp_path: Path,
) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    campaign = CampaignSession.initialize(config_path, tmp_path / "campaign.csv")
    before_log = campaign.log_path.read_bytes()
    before_manifest = provenance_module.manifest_path_for_log(campaign.log_path).read_bytes()
    config_path.write_text("campaign_name: [broken\n", encoding="utf-8")

    values = dict(provenance_summary(config_path, campaign.log_path).itertuples(False, None))
    assert values["reason_code"] == "config_semantics_changed"
    assert values["log_bytes_match"] is True
    with pytest.raises(LogConflictError) as error:
        CampaignSession.from_files(config_path, campaign.log_path)
    assert error.value.reason_code == "config_semantics_changed"
    assert campaign.log_path.read_bytes() == before_log
    assert provenance_module.manifest_path_for_log(campaign.log_path).read_bytes() == (
        before_manifest
    )


def test_log_hash_and_row_count_reason_codes(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    first = CampaignSession.initialize(config_path, tmp_path / "hash.csv")
    first.log_path.write_bytes(first.log_path.read_bytes() + b"\n")
    hash_values = dict(provenance_summary(config_path, first.log_path).itertuples(False, None))
    assert hash_values["reason_code"] == "log_hash_changed"

    second = CampaignSession.initialize(config_path, tmp_path / "rows.csv")
    manifest_path, manifest = _manifest(second.log_path)
    manifest["log"]["row_count"] = 1
    provenance_module._write_json_atomic(manifest_path, manifest)
    row_values = dict(provenance_summary(config_path, second.log_path).itertuples(False, None))
    assert row_values["reason_code"] == "log_row_count_changed"


@pytest.mark.parametrize(
    ("prepare", "reason_code", "event_count"),
    [
        (_pending_previous, "pending_previous_state", 1),
        (_pending_resulting, "pending_resulting_state", 2),
    ],
)
def test_explicit_recovery_changes_only_manifest_and_is_idempotent(
    tmp_path: Path,
    prepare: object,
    reason_code: str,
    event_count: int,
) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    campaign = CampaignSession.initialize(config_path, tmp_path / "campaign.csv")
    setup = prepare
    assert callable(setup)
    manifest_path = setup(config_path, campaign.log_path)
    before_config = config_path.read_bytes()
    before_log = campaign.log_path.read_bytes()

    values = dict(provenance_summary(config_path, campaign.log_path).itertuples(False, None))
    assert values["reason_code"] == reason_code
    with pytest.raises(ProvenanceRecoveryRequired) as error:
        CampaignSession.from_files(config_path, campaign.log_path)
    assert error.value.reason_code == reason_code

    recovered = recover_provenance(config_path, campaign.log_path)
    recovered_manifest = manifest_path.read_bytes()
    recovered_values = dict(recovered.itertuples(False, None))
    assert recovered_values["resume_status"] == "ready"
    assert recovered_values["event_count"] == event_count
    assert config_path.read_bytes() == before_config
    assert campaign.log_path.read_bytes() == before_log

    recover_provenance(config_path, campaign.log_path)
    assert manifest_path.read_bytes() == recovered_manifest


def test_recovery_rejects_unknown_or_stale_log_without_writes(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    campaign = CampaignSession.initialize(config_path, tmp_path / "campaign.csv")
    manifest_path = _pending_previous(config_path, campaign.log_path)
    _suggestion("external").to_csv(campaign.log_path, index=False)
    before_log = campaign.log_path.read_bytes()
    before_manifest = manifest_path.read_bytes()

    with pytest.raises(LogConflictError) as error:
        recover_provenance(config_path, campaign.log_path)
    assert error.value.reason_code == "pending_unknown_state"
    assert campaign.log_path.read_bytes() == before_log
    assert manifest_path.read_bytes() == before_manifest


def test_recovery_rejects_stale_expected_fingerprint_without_writes(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    campaign = CampaignSession.initialize(config_path, tmp_path / "campaign.csv")
    manifest_path = _pending_previous(config_path, campaign.log_path)
    before_log = campaign.log_path.read_bytes()
    before_manifest = manifest_path.read_bytes()

    with pytest.raises(LogConflictError, match="changed after it was loaded"):
        recover_provenance(
            config_path,
            campaign.log_path,
            expected_log_fingerprint="stale",
        )
    assert campaign.log_path.read_bytes() == before_log
    assert manifest_path.read_bytes() == before_manifest


def test_existing_session_suggestion_and_report_require_explicit_recovery(
    tmp_path: Path,
) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    campaign = CampaignSession.initialize(config_path, tmp_path / "campaign.csv")
    manifest_path = _pending_previous(config_path, campaign.log_path)
    before_log = campaign.log_path.read_bytes()
    before_manifest = manifest_path.read_bytes()

    for operation in (campaign.suggest_next, campaign.report):
        with pytest.raises(ProvenanceRecoveryRequired):
            operation()
    assert campaign.log_path.read_bytes() == before_log
    assert manifest_path.read_bytes() == before_manifest


def test_pending_transaction_blocks_observation_without_mutation(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    campaign = CampaignSession.initialize(config_path, tmp_path / "campaign.csv")
    campaign.append_suggestions(_suggestion())
    manifest_path = _pending_previous(config_path, campaign.log_path)
    before_log = campaign.log_path.read_bytes()
    before_manifest = manifest_path.read_bytes()

    with pytest.raises(ProvenanceRecoveryRequired):
        campaign.mark_observed("row_1", 0.7)
    assert campaign.log_path.read_bytes() == before_log
    assert manifest_path.read_bytes() == before_manifest


def test_pending_transaction_blocks_review_without_mutation(tmp_path: Path) -> None:
    config_path = write_cost_review_config(tmp_path / "campaign.yaml")
    campaign = CampaignSession.initialize(config_path, tmp_path / "campaign.csv")
    suggestion = campaign.suggest_next(batch_size=1)
    campaign.append_suggestions(suggestion)
    row_id = str(suggestion["row_id"].iloc[0])
    manifest_path = _pending_previous(config_path, campaign.log_path)
    before_log = campaign.log_path.read_bytes()
    before_manifest = manifest_path.read_bytes()

    with pytest.raises(ProvenanceRecoveryRequired):
        campaign.review_suggestion(row_id, "accept")
    assert campaign.log_path.read_bytes() == before_log
    assert manifest_path.read_bytes() == before_manifest


@pytest.mark.parametrize("operation", ["append", "review", "observe"])
def test_managed_mutation_cannot_fall_back_if_manifest_disappears_after_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    config_path = (
        write_cost_review_config(tmp_path / "campaign.yaml")
        if operation == "review"
        else write_config(tmp_path / "campaign.yaml")
    )
    campaign = CampaignSession.initialize(config_path, tmp_path / "campaign.csv")
    if operation == "append":
        def mutate() -> None:
            campaign.append_suggestions(_suggestion())
    else:
        suggestion = campaign.suggest_next(batch_size=1)
        campaign.append_suggestions(suggestion)
        row_id = str(suggestion["row_id"].iloc[0])

        def mutate() -> None:
            if operation == "review":
                campaign.review_suggestion(row_id, "accept")
            else:
                campaign.mark_observed(row_id, 0.7)
    before_log = campaign.log_path.read_bytes()
    manifest_path = provenance_module.manifest_path_for_log(campaign.log_path)
    original_check = logs_module._assert_expected_log_fingerprint

    def remove_manifest_after_check(path: Path, expected: str | None) -> bool | None:
        expected_managed = original_check(path, expected)
        manifest_path.unlink()
        return expected_managed

    monkeypatch.setattr(
        logs_module,
        "_assert_expected_log_fingerprint",
        remove_manifest_after_check,
    )

    with pytest.raises(LogConflictError, match="provenance state changed during mutation"):
        mutate()
    assert campaign.log_path.read_bytes() == before_log
    assert not manifest_path.exists()


def test_legacy_mutation_rejects_manifest_appearance_after_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    log_path = write_log(tmp_path / "campaign.csv", config())
    campaign = CampaignSession.from_files(config_path, log_path)
    before_log = log_path.read_bytes()
    manifest_path = provenance_module.manifest_path_for_log(log_path)
    original_check = logs_module._assert_expected_log_fingerprint

    def add_manifest_after_check(path: Path, expected: str | None) -> bool | None:
        expected_managed = original_check(path, expected)
        manifest_path.write_text("{}\n", encoding="utf-8")
        return expected_managed

    monkeypatch.setattr(
        logs_module,
        "_assert_expected_log_fingerprint",
        add_manifest_after_check,
    )

    with pytest.raises(LogConflictError, match="provenance state changed during mutation"):
        campaign.append_suggestions(_suggestion())
    assert log_path.read_bytes() == before_log


def test_recovery_reports_log_lock_contention_without_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    campaign = CampaignSession.initialize(config_path, tmp_path / "campaign.csv")
    manifest_path = _pending_previous(config_path, campaign.log_path)
    before_log = campaign.log_path.read_bytes()
    before_manifest = manifest_path.read_bytes()
    monkeypatch.setattr(logs_module, "LOG_LOCK_TIMEOUT_SECONDS", 0.01)

    with logs_module._campaign_log_lock(campaign.log_path):
        with pytest.raises(LogBusyError):
            recover_provenance(config_path, campaign.log_path)
    assert campaign.log_path.read_bytes() == before_log
    assert manifest_path.read_bytes() == before_manifest


def test_concurrent_recovery_and_append_preserve_coherent_ledger(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "campaign.yaml", initial_design_size=3)
    campaign = CampaignSession.initialize(config_path, tmp_path / "campaign.csv")
    _pending_previous(config_path, campaign.log_path)
    barrier = threading.Barrier(2)

    def recover() -> str:
        barrier.wait(timeout=5)
        recover_provenance(config_path, campaign.log_path)
        return "recovered"

    def append() -> str:
        barrier.wait(timeout=5)
        try:
            append_suggestions(campaign.log_path, _suggestion(), config=campaign.config)
        except ProvenanceRecoveryRequired:
            return "blocked"
        return "appended"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = {executor.submit(recover), executor.submit(append)}
        outcomes = {future.result(timeout=10) for future in results}

    assert "recovered" in outcomes
    assert outcomes <= {"recovered", "blocked", "appended"}
    manifest_path, manifest = _manifest(campaign.log_path)
    assert manifest["pending_transaction"] is None
    log = pd.read_csv(campaign.log_path, keep_default_na=False)
    assert len(manifest["events"]) == len(log) + 1
    assert manifest_path.exists()


@pytest.mark.parametrize("operation", ["review", "observe"])
def test_concurrent_recovery_and_row_mutation_preserve_coherent_ledger(
    tmp_path: Path,
    operation: str,
) -> None:
    config_path = (
        write_cost_review_config(tmp_path / "campaign.yaml")
        if operation == "review"
        else write_config(tmp_path / "campaign.yaml")
    )
    campaign = CampaignSession.initialize(config_path, tmp_path / "campaign.csv")
    suggestion = campaign.suggest_next(batch_size=1)
    campaign.append_suggestions(suggestion)
    row_id = str(suggestion["row_id"].iloc[0])
    _pending_previous(config_path, campaign.log_path)
    barrier = threading.Barrier(2)

    def recover() -> str:
        barrier.wait(timeout=5)
        recover_provenance(config_path, campaign.log_path)
        return "recovered"

    def mutate() -> str:
        barrier.wait(timeout=5)
        try:
            if operation == "review":
                campaign.review_suggestion(row_id, "accept")
            else:
                campaign.mark_observed(row_id, 0.7)
        except ProvenanceRecoveryRequired:
            return "blocked"
        return "mutated"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = {
            future.result(timeout=10)
            for future in {executor.submit(recover), executor.submit(mutate)}
        }

    assert "recovered" in outcomes
    assert outcomes <= {"recovered", "blocked", "mutated"}
    _, manifest = _manifest(campaign.log_path)
    assert manifest["pending_transaction"] is None
    log = pd.read_csv(campaign.log_path, keep_default_na=False)
    mutation_recorded = "mutated" in outcomes
    if operation == "review":
        assert bool(log.loc[0, "review_status"] == "accepted") is mutation_recorded
        assert (manifest["events"][-1]["operation"] == "review_suggestion") is (
            mutation_recorded
        )
    else:
        assert bool(log.loc[0, "status"] == "observed") is mutation_recorded
        assert (manifest["events"][-1]["operation"] == "mark_observed") is (
            mutation_recorded
        )


def test_environment_drift_is_informational_and_non_mutating(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    campaign = CampaignSession.initialize(config_path, tmp_path / "campaign.csv")
    manifest_path, manifest = _manifest(campaign.log_path)
    current = dict(manifest["environments"][0])
    current["environment_id"] = "f" * 64
    current["python"] = "different"
    monkeypatch.setattr(resume_module, "capture_environment", lambda _path: current)
    before_manifest = manifest_path.read_bytes()

    summary = provenance_summary(config_path, campaign.log_path)
    values = dict(summary.itertuples(False, None))
    assert values["resume_status"] == "ready"
    assert values["environment_match"] is False
    assert values["environment_changes"] == "python"
    CampaignSession.from_files(config_path, campaign.log_path)
    assert manifest_path.read_bytes() == before_manifest


def test_normal_load_and_validation_skip_environment_recapture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    campaign = CampaignSession.initialize(config_path, tmp_path / "campaign.csv")
    monkeypatch.setattr(
        resume_module,
        "capture_environment",
        lambda _path: pytest.fail("resume enforcement must not recapture environment"),
    )

    loaded = CampaignSession.from_files(config_path, campaign.log_path)
    loaded.validate()


@pytest.mark.parametrize(
    "method_name",
    [
        "plot_progress",
        "plot_diagnostics",
        "plot_cost_progress",
        "plot_replicates",
        "plot_pareto",
        "plot_pareto_parallel",
        "plot_hypervolume",
        "plot_stage_diagnostics",
        "plot_fidelity_diagnostics",
        "plot_fidelity_progress",
        "plot_context_diagnostics",
        "plot_model_diagnostics",
        "plot_model_comparison",
        "plot_qlog_nei_diagnostics",
    ],
)
def test_session_plot_methods_enforce_provenance_resume_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method_name: str,
) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    campaign = CampaignSession.initialize(config_path, tmp_path / "campaign.csv")
    error = ProvenanceRecoveryRequired(
        "recovery required",
        reason_code="pending_previous_state",
        recovery_action="Recover provenance.",
    )

    def block() -> None:
        raise error

    monkeypatch.setattr(campaign, "_assert_provenance_resumable", block)
    with pytest.raises(ProvenanceRecoveryRequired):
        getattr(campaign, method_name)()
