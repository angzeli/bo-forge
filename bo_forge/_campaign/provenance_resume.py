"""Fail-closed provenance inspection and explicit schema-v1 recovery."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from bo_forge._campaign import provenance as manifest_io
from bo_forge._campaign.provenance_environment import capture_environment
from bo_forge.config import CampaignConfig
from bo_forge.errors import (
    ConfigError,
    LogConflictError,
    ProvenanceError,
    ProvenanceRecoveryRequired,
)

PROVENANCE_POLICIES = {"compatible", "required"}

_RECOVERY_ACTIONS = {
    "manifest_required": "Initialize a managed campaign or use compatible loading.",
    "manifest_invalid": "Repair or restore the schema-v1 manifest before resuming.",
    "manifest_path_mismatch": "Use the config and log referenced by this manifest.",
    "config_bytes_changed_semantics_same": (
        "Restore the exact recorded YAML bytes or initialize a new managed campaign."
    ),
    "config_semantics_changed": (
        "Restore the recorded YAML semantics or initialize a new managed campaign."
    ),
    "log_missing": "Restore the managed CSV at its recorded path.",
    "log_hash_changed": "Restore a CSV state recorded by the provenance ledger.",
    "log_row_count_changed": "Restore the CSV whose row count matches its recorded hash.",
    "pending_previous_state": "Run provenance recovery to cancel the pending transaction.",
    "pending_resulting_state": "Run provenance recovery to finalize the pending transaction.",
    "pending_unknown_state": (
        "Restore the CSV to the recorded previous or intended resulting state."
    ),
}

_REASON_MESSAGES = {
    "config_bytes_changed_semantics_same": (
        "Managed campaign config changed and does not match its provenance manifest, "
        "although parsed semantics are unchanged."
    ),
    "config_semantics_changed": (
        "Managed campaign config semantics do not match its provenance manifest."
    ),
    "log_missing": (
        "Managed campaign log is missing and does not match its provenance manifest."
    ),
    "log_hash_changed": "Managed campaign log does not match its provenance manifest.",
    "log_row_count_changed": "Managed campaign row count does not match its manifest.",
    "pending_previous_state": (
        "Managed campaign has an interrupted transaction in its previous log state."
    ),
    "pending_resulting_state": (
        "Managed campaign has an interrupted transaction in its resulting log state."
    ),
    "pending_unknown_state": (
        "Interrupted managed mutation cannot be recovered because the campaign log "
        "matches neither the previous nor intended hash."
    ),
}


@dataclass(frozen=True)
class ProvenanceInspection:
    """One read-only classification of current campaign provenance state."""

    config_file: Path
    log_file: Path
    manifest: dict[str, Any] | None
    config: CampaignConfig | None
    provenance_status: str
    integrity_status: str
    resume_status: str
    reason_code: str | None
    recovery_action: str | None
    config_bytes_match: bool | None
    config_semantic_match: bool | None
    current_log_sha256: str | None
    log_bytes_match: bool | None
    current_environment_id: str | None
    environment_match: bool | None
    environment_changes: str | None

    def to_frame(self) -> pd.DataFrame:
        """Return stable ordered summary rows for this inspection."""
        rows = [
            ("provenance_status", self.provenance_status),
            ("integrity_status", self.integrity_status),
            ("resume_status", self.resume_status),
            ("reason_code", self.reason_code),
            ("recovery_action", self.recovery_action),
            ("config_semantic_match", self.config_semantic_match),
            ("current_environment_id", self.current_environment_id),
            ("environment_match", self.environment_match),
            ("environment_changes", self.environment_changes),
        ]
        if self.manifest is not None:
            rows.extend(_managed_detail_rows(self))
        return pd.DataFrame(rows, columns=["field", "value"])


def normalize_provenance_policy(value: str) -> str:
    """Validate and normalize a campaign provenance loading policy."""
    if value not in PROVENANCE_POLICIES:
        allowed = ", ".join(sorted(PROVENANCE_POLICIES))
        raise ValueError(f"provenance_policy must be one of: {allowed}.")
    return value


def load_campaign_session(
    session_type: Any,
    config_path: str | Path,
    log_path: str | Path,
    *,
    provenance_policy: str = "compatible",
) -> Any:
    """Load one session with a stable config/log/provenance snapshot."""
    from bo_forge.logs import _load_campaign_log_snapshot, _log_file_fingerprint

    policy = normalize_provenance_policy(provenance_policy)
    parsed_config_path = Path(config_path)
    parsed_log_path = Path(log_path)
    config_fingerprint = _log_file_fingerprint(parsed_config_path)
    try:
        config = CampaignConfig.from_yaml(parsed_config_path)
    except ConfigError as exc:
        if manifest_io.validate_manifest_references(parsed_config_path, parsed_log_path) is None:
            raise
        action = _RECOVERY_ACTIONS["config_semantics_changed"]
        error = LogConflictError(f"{_REASON_MESSAGES['config_semantics_changed']} {action}")
        error.reason_code = "config_semantics_changed"
        error.recovery_action = action
        raise error from exc
    if _log_file_fingerprint(parsed_config_path) != config_fingerprint:
        raise LogConflictError(
            "Campaign config changed while it was being loaded. Reload the campaign."
        )
    df, fingerprint = _load_campaign_log_snapshot(parsed_log_path, config)
    if _log_file_fingerprint(parsed_config_path) != config_fingerprint:
        raise LogConflictError(
            "Campaign config changed while it was being loaded. Reload the campaign."
        )
    manifest = manifest_io.validate_manifest_for_load(
        parsed_config_path,
        parsed_log_path,
        config=config,
        log_row_count=len(df),
        provenance_policy=policy,
    )
    if manifest is not None and _log_file_fingerprint(parsed_log_path) != fingerprint:
        raise LogConflictError(
            "Campaign log changed while it was being loaded. Reload the campaign."
        )
    session = session_type(
        config_path=parsed_config_path,
        log_path=parsed_log_path,
        config=config,
        df=df,
        log_fingerprint=fingerprint,
        config_fingerprint=config_fingerprint,
    )
    session._provenance_managed = manifest is not None
    session._provenance_policy = policy
    return session


def inspect_provenance(
    config_path: str | Path,
    log_path: str | Path,
    *,
    provenance_policy: str = "compatible",
    config: CampaignConfig | None = None,
    log_row_count: int | None = None,
    include_environment: bool = True,
) -> ProvenanceInspection:
    """Classify resume safety without writing or repairing campaign files."""
    policy = normalize_provenance_policy(provenance_policy)
    config_file = Path(config_path).expanduser().resolve(strict=False)
    log_file = Path(log_path).expanduser().resolve(strict=False)
    manifest = manifest_io.validate_manifest_references(config_file, log_file)
    if manifest is None:
        if policy == "required":
            raise ProvenanceError(
                f"Campaign provenance manifest is required for log '{log_file}'.",
                reason_code="manifest_required",
                recovery_action=_RECOVERY_ACTIONS["manifest_required"],
            )
        return _legacy_inspection(config_file, log_file)
    return inspect_loaded_manifest(
        config_file,
        log_file,
        manifest,
        config=config,
        log_row_count=log_row_count,
        include_environment=include_environment,
    )


def inspect_loaded_manifest(
    config_file: Path,
    log_file: Path,
    manifest: dict[str, Any],
    *,
    config: CampaignConfig | None = None,
    log_row_count: int | None = None,
    include_environment: bool = True,
) -> ProvenanceInspection:
    """Classify an already validated manifest against current files."""
    config_bytes = manifest_io._read_config_bytes(config_file)
    byte_match = (
        manifest_io._sha256_bytes(config_bytes) == manifest["config"]["byte_sha256"]
    )
    try:
        current_config = (
            config
            if config is not None and byte_match
            else CampaignConfig.from_yaml(config_file)
        )
    except ConfigError:
        current_config = None
        semantic_match = False
    else:
        semantic_match = (
            manifest_io.config_semantic_sha256(current_config)
            == manifest["config"]["semantic_sha256"]
        )
    environment = (
        _environment_state(config_file, manifest)
        if include_environment
        else (None, None, None)
    )
    state = _managed_file_state(
        log_file,
        manifest,
        byte_match=byte_match,
        semantic_match=semantic_match,
        log_row_count=log_row_count,
    )
    return ProvenanceInspection(
        config_file=config_file,
        log_file=log_file,
        manifest=manifest,
        config=current_config,
        provenance_status="managed",
        integrity_status=state[0],
        resume_status=state[1],
        reason_code=state[2],
        recovery_action=_RECOVERY_ACTIONS.get(state[2]),
        config_bytes_match=byte_match,
        config_semantic_match=semantic_match,
        current_log_sha256=state[3],
        log_bytes_match=state[4],
        current_environment_id=environment[0],
        environment_match=environment[1],
        environment_changes=environment[2],
    )


def enforce_resumable(inspection: ProvenanceInspection) -> None:
    """Raise the typed failure represented by a non-resumable inspection."""
    if inspection.resume_status in {"ready", "legacy"}:
        return
    reason = inspection.reason_code or "manifest_invalid"
    message = _REASON_MESSAGES.get(reason, "Managed campaign cannot be resumed safely.")
    action = inspection.recovery_action or _RECOVERY_ACTIONS["manifest_invalid"]
    if inspection.resume_status == "recovery_required":
        raise ProvenanceRecoveryRequired(
            f"{message} {action}",
            reason_code=reason,
            recovery_action=action,
        )
    error = LogConflictError(f"{message} {action}")
    error.reason_code = reason
    error.recovery_action = action
    raise error


def recover_provenance(
    config_path: str | Path,
    log_path: str | Path,
    *,
    expected_log_fingerprint: str | None = None,
) -> pd.DataFrame:
    """Explicitly resolve a recoverable pending transaction under the log lock."""
    from bo_forge.logs import _assert_expected_log_fingerprint, _campaign_log_lock

    config_file = Path(config_path).expanduser().resolve(strict=False)
    log_file = Path(log_path).expanduser().resolve(strict=False)
    with _campaign_log_lock(log_file):
        _assert_expected_log_fingerprint(log_file, expected_log_fingerprint)
        inspection = inspect_provenance(
            config_file,
            log_file,
            provenance_policy="required",
            include_environment=False,
        )
        if inspection.resume_status == "recovery_required":
            assert inspection.manifest is not None
            manifest_io._recover_pending_transaction(
                manifest_io.manifest_path_for_log(log_file),
                log_file,
                inspection.manifest,
            )
        else:
            enforce_resumable(inspection)
    return provenance_summary(config_file, log_file)


def provenance_summary(config_path: str | Path, log_path: str | Path) -> pd.DataFrame:
    """Return ordered provenance diagnostics without repairing campaign files."""
    inspection = inspect_provenance(config_path, log_path)
    if inspection.provenance_status == "legacy" and not inspection.log_file.exists():
        raise ProvenanceError(
            f"Campaign log '{inspection.log_file}' does not exist; provenance status is unknown.",
            reason_code="log_missing",
            recovery_action=_RECOVERY_ACTIONS["log_missing"],
        )
    return inspection.to_frame()


def _managed_file_state(
    log_file: Path,
    manifest: dict[str, Any],
    *,
    byte_match: bool,
    semantic_match: bool,
    log_row_count: int | None,
) -> tuple[str, str, str | None, str | None, bool]:
    current_hash = manifest_io._sha256_file_or_none(log_file)
    accepted_hashes = {manifest["log"]["sha256"]}
    pending = manifest.get("pending_transaction")
    if pending is not None:
        accepted_hashes.update(
            {pending["previous_log_sha256"], pending["resulting_log_sha256"]}
        )
    log_bytes_match = current_hash is not None and current_hash in accepted_hashes
    if not byte_match:
        reason = (
            "config_bytes_changed_semantics_same"
            if semantic_match
            else "config_semantics_changed"
        )
        return "mismatch", "blocked", reason, current_hash, log_bytes_match
    if not semantic_match:
        return (
            "mismatch",
            "blocked",
            "config_semantics_changed",
            current_hash,
            log_bytes_match,
        )
    if current_hash is None:
        return "mismatch", "blocked", "log_missing", None, False
    if not log_bytes_match:
        reason = "pending_unknown_state" if pending is not None else "log_hash_changed"
        return "mismatch", "blocked", reason, current_hash, False
    expected_rows = _expected_row_count(manifest, current_hash)
    actual_rows = log_row_count if log_row_count is not None else _read_log_row_count(log_file)
    if actual_rows != expected_rows:
        return "mismatch", "blocked", "log_row_count_changed", current_hash, True
    if pending is not None:
        reason = (
            "pending_resulting_state"
            if current_hash == pending["resulting_log_sha256"]
            else "pending_previous_state"
        )
        return "pending_recovery", "recovery_required", reason, current_hash, True
    return "valid", "ready", None, current_hash, True


def _expected_row_count(manifest: dict[str, Any], current_hash: str) -> int:
    pending = manifest.get("pending_transaction")
    if pending is not None and current_hash == pending["resulting_log_sha256"]:
        return int(pending["resulting_log_row_count"])
    return int(manifest["log"]["row_count"])


def _read_log_row_count(path: Path) -> int:
    try:
        return len(pd.read_csv(path, keep_default_na=False))
    except (OSError, pd.errors.ParserError) as exc:
        raise ProvenanceError(
            f"Could not inspect managed campaign log '{path}': {exc}"
        ) from exc


def _environment_state(
    config_file: Path,
    manifest: dict[str, Any],
) -> tuple[str, bool, str | None]:
    current = capture_environment(config_file)
    last_event = manifest["events"][-1] if manifest["events"] else None
    recorded_id = None if last_event is None else last_event["environment_id"]
    recorded = next(
        (
            item
            for item in manifest["environments"]
            if item["environment_id"] == recorded_id
        ),
        None,
    )
    matches = recorded_id == current["environment_id"]
    changes = None if recorded is None else _environment_changes(recorded, current)
    return str(current["environment_id"]), matches, changes


def _environment_changes(recorded: dict[str, Any], current: dict[str, Any]) -> str | None:
    ignored = {"captured_at", "environment_id"}
    changed = [
        key
        for key in sorted(set(recorded) | set(current))
        if key not in ignored and recorded.get(key) != current.get(key)
    ]
    return ", ".join(changed) or None


def _legacy_inspection(config_file: Path, log_file: Path) -> ProvenanceInspection:
    return ProvenanceInspection(
        config_file=config_file,
        log_file=log_file,
        manifest=None,
        config=None,
        provenance_status="legacy",
        integrity_status="not_managed",
        resume_status="legacy",
        reason_code=None,
        recovery_action=None,
        config_bytes_match=None,
        config_semantic_match=None,
        current_log_sha256=manifest_io._sha256_file_or_none(log_file),
        log_bytes_match=None,
        current_environment_id=None,
        environment_match=None,
        environment_changes=None,
    )


def _managed_detail_rows(inspection: ProvenanceInspection) -> list[tuple[str, object]]:
    assert inspection.manifest is not None
    manifest = inspection.manifest
    events = manifest["events"]
    last_event = events[-1] if events else None
    return [
        ("campaign_id", manifest["campaign_id"]),
        ("schema_version", manifest["schema_version"]),
        ("config_byte_sha256", manifest["config"]["byte_sha256"]),
        ("config_semantic_sha256", manifest["config"]["semantic_sha256"]),
        ("config_bytes_match", inspection.config_bytes_match),
        ("log_sha256", manifest["log"]["sha256"]),
        ("current_log_sha256", inspection.current_log_sha256),
        ("log_bytes_match", inspection.log_bytes_match),
        ("log_row_count", manifest["log"]["row_count"]),
        ("environment_count", len(manifest["environments"])),
        ("event_count", len(events)),
        ("last_event_sequence", None if last_event is None else last_event["sequence"]),
        ("last_mutation", None if last_event is None else last_event["operation"]),
        ("updated_at", manifest["updated_at"]),
        ("pending_transaction", manifest.get("pending_transaction") is not None),
    ]
