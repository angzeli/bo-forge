"""Schema-v1 structural validation for campaign provenance manifests."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from bo_forge.config import CampaignConfig, parse_campaign_config
from bo_forge.errors import BOForgeError, LogConflictError, ProvenanceError

_HASH_LENGTH = 64
_MISSING_LOG_HASH = "<missing-log>"
_OPERATIONS = {
    "initialize",
    "append_suggestions",
    "review_suggestion",
    "mark_observed",
}
_DEPENDENCIES = {
    "numpy",
    "pandas",
    "torch",
    "botorch",
    "gpytorch",
    "matplotlib",
    "PyYAML",
    "filelock",
}


def validate_manifest_payload(
    payload: object,
    path: Path,
    *,
    semantic_hash: Callable[[CampaignConfig], str],
    optimization_identity: Callable[[CampaignConfig], dict[str, Any]],
) -> None:
    """Validate the complete schema-v1 manifest contract."""
    if not isinstance(payload, dict):
        raise ProvenanceError(f"Provenance manifest '{path}' must contain a JSON object.")
    required = {
        "schema_version",
        "campaign_id",
        "created_at",
        "updated_at",
        "paths",
        "config",
        "log",
        "optimization",
        "environments",
        "events",
        "pending_transaction",
    }
    if payload.get("schema_version") != 1:
        raise ProvenanceError(
            f"Unsupported provenance schema_version in '{path}': "
            f"{payload.get('schema_version')!r}."
        )
    if set(payload) != required:
        raise ProvenanceError(f"Provenance manifest '{path}' has invalid top-level fields.")

    _validate_uuid(payload["campaign_id"], "campaign_id", path)
    created_at = _validate_timestamp(payload["created_at"], "created_at", path)
    updated_at = _validate_timestamp(payload["updated_at"], "updated_at", path)
    _validate_paths(payload["paths"], path)
    parsed_config = _validate_config(payload["config"], path, semantic_hash)
    if payload["optimization"] != optimization_identity(parsed_config):
        raise ProvenanceError(f"Provenance manifest '{path}' has invalid optimization identity.")
    _validate_log(payload["log"], path)
    environment_ids = _validate_environments(payload["environments"], path)
    events = _validate_events(payload["events"], environment_ids, path)
    _validate_ledger_identity(payload, events, created_at, updated_at, path)
    _validate_pending(payload["pending_transaction"], payload, environment_ids, path)


def validate_current_manifest_state(
    manifest: dict[str, Any],
    *,
    config_byte_sha256: str,
    config_semantic_sha256: str,
    log_sha256: str | None,
    log_row_count: int,
) -> None:
    """Validate current config and log identity against a loaded manifest."""
    if config_byte_sha256 != manifest["config"]["byte_sha256"]:
        raise LogConflictError(
            "Managed campaign config does not match its provenance manifest. "
            "Restore the recorded config before loading the campaign."
        )
    if config_semantic_sha256 != manifest["config"]["semantic_sha256"]:
        raise LogConflictError("Managed campaign semantic config identity no longer matches.")

    expected_row_count = manifest["log"]["row_count"]
    accepted_hashes = {manifest["log"]["sha256"]}
    pending = manifest.get("pending_transaction")
    if pending is not None:
        accepted_hashes.update(
            {pending["previous_log_sha256"], pending["resulting_log_sha256"]}
        )
        if log_sha256 == pending["resulting_log_sha256"]:
            expected_row_count = pending["resulting_log_row_count"]
    if log_sha256 not in accepted_hashes:
        raise LogConflictError(
            "Managed campaign log does not match its provenance manifest. "
            "Restore a coherent campaign state before loading it."
        )
    if log_row_count != expected_row_count:
        raise LogConflictError("Managed campaign row count does not match its manifest.")


def _validate_paths(value: object, path: Path) -> None:
    if not isinstance(value, dict) or set(value) != {"config", "log"}:
        raise ProvenanceError(f"Provenance manifest '{path}' has invalid path references.")
    for reference in value.values():
        if not isinstance(reference, str) or not reference or Path(reference).is_absolute():
            raise ProvenanceError(f"Provenance manifest '{path}' must use relative paths.")


def _validate_config(
    value: object,
    path: Path,
    semantic_hash: Callable[[CampaignConfig], str],
) -> CampaignConfig:
    required = {"snapshot", "byte_sha256", "semantic_sha256"}
    if not isinstance(value, dict) or set(value) != required:
        raise ProvenanceError(f"Provenance manifest '{path}' has an invalid config snapshot.")
    snapshot = value["snapshot"]
    if not isinstance(snapshot, str):
        raise ProvenanceError(f"Provenance manifest '{path}' config snapshot must be text.")
    _validate_hash(value["byte_sha256"], "config byte hash", path)
    _validate_hash(value["semantic_sha256"], "semantic config hash", path)
    if _sha256(snapshot.encode("utf-8")) != value["byte_sha256"]:
        raise ProvenanceError(f"Provenance manifest '{path}' config snapshot hash is invalid.")
    try:
        parsed = parse_campaign_config(yaml.safe_load(snapshot))
    except (BOForgeError, yaml.YAMLError) as exc:
        raise ProvenanceError(
            f"Provenance manifest '{path}' contains an invalid config snapshot: {exc}"
        ) from exc
    if semantic_hash(parsed) != value["semantic_sha256"]:
        raise ProvenanceError(f"Provenance manifest '{path}' semantic config hash is invalid.")
    return parsed


def _validate_log(value: object, path: Path) -> None:
    if not isinstance(value, dict) or set(value) != {"sha256", "row_count"}:
        raise ProvenanceError(f"Provenance manifest '{path}' has invalid log metadata.")
    _validate_hash(value["sha256"], "log hash", path)
    if not _is_nonnegative_int(value["row_count"]):
        raise ProvenanceError(f"Provenance manifest '{path}' has invalid log row_count.")


def _validate_environments(value: object, path: Path) -> set[str]:
    if not isinstance(value, list) or not value:
        raise ProvenanceError(f"Provenance manifest '{path}' has invalid environments.")
    identifiers: set[str] = set()
    required = {
        "environment_id",
        "captured_at",
        "bo_forge",
        "python",
        "platform",
        "dependencies",
        "git",
    }
    for environment in value:
        if not isinstance(environment, dict) or set(environment) != required:
            raise ProvenanceError(f"Provenance manifest '{path}' has invalid environment data.")
        identifier = environment["environment_id"]
        _validate_hash(identifier, "environment_id", path)
        _validate_timestamp(environment["captured_at"], "environment captured_at", path)
        if identifier in identifiers or identifier != _environment_hash(environment):
            raise ProvenanceError(f"Provenance manifest '{path}' has invalid environment identity.")
        if not isinstance(environment["bo_forge"], str) or not isinstance(
            environment["python"], str
        ):
            raise ProvenanceError(f"Provenance manifest '{path}' has invalid environment data.")
        _validate_environment_details(environment, path)
        identifiers.add(identifier)
    return identifiers


def _validate_environment_details(environment: dict[str, Any], path: Path) -> None:
    platform = environment["platform"]
    dependencies = environment["dependencies"]
    git = environment["git"]
    if not isinstance(platform, dict) or set(platform) != {"system", "release", "machine"}:
        raise ProvenanceError(f"Provenance manifest '{path}' has invalid platform data.")
    if not all(isinstance(item, str) for item in platform.values()):
        raise ProvenanceError(f"Provenance manifest '{path}' has invalid platform data.")
    if not isinstance(dependencies, dict) or set(dependencies) != _DEPENDENCIES:
        raise ProvenanceError(f"Provenance manifest '{path}' has invalid dependencies.")
    if not all(isinstance(item, str) for item in dependencies.values()):
        raise ProvenanceError(f"Provenance manifest '{path}' has invalid dependencies.")
    if not isinstance(git, dict) or set(git) != {"commit", "dirty"}:
        raise ProvenanceError(f"Provenance manifest '{path}' has invalid Git identity.")
    if not isinstance(git["commit"], str) or not (
        isinstance(git["dirty"], bool) or git["dirty"] == "unknown"
    ):
        raise ProvenanceError(f"Provenance manifest '{path}' has invalid Git identity.")


def _validate_events(value: object, environment_ids: set[str], path: Path) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ProvenanceError(f"Provenance manifest '{path}' has invalid event ledger.")
    events: list[dict[str, Any]] = []
    previous_hash = _MISSING_LOG_HASH
    previous_timestamp: datetime | None = None
    event_ids: set[str] = set()
    for sequence, event in enumerate(value, start=1):
        _validate_event(event, sequence, environment_ids, path)
        event_id = str(event["event_id"])
        if event_id in event_ids:
            raise ProvenanceError(f"Provenance manifest '{path}' has duplicate event IDs.")
        event_ids.add(event_id)
        timestamp = _validate_timestamp(event["timestamp"], "event timestamp", path)
        if previous_timestamp is not None and timestamp < previous_timestamp:
            raise ProvenanceError(
                f"Provenance manifest '{path}' has non-monotonic event timestamps."
            )
        previous_timestamp = timestamp
        if event["previous_log_sha256"] != previous_hash:
            raise ProvenanceError(f"Provenance manifest '{path}' has a broken event hash chain.")
        previous_hash = event["resulting_log_sha256"]
        events.append(event)
    if events[0]["operation"] != "initialize":
        raise ProvenanceError(f"Provenance manifest '{path}' must start with initialize.")
    return events


def _validate_event(
    event: object,
    sequence: int,
    environment_ids: set[str],
    path: Path,
) -> None:
    required = {
        "sequence",
        "event_id",
        "timestamp",
        "operation",
        "affected_row_ids",
        "previous_log_sha256",
        "resulting_log_sha256",
        "environment_id",
        "metadata",
    }
    if not isinstance(event, dict) or set(event) != required or event["sequence"] != sequence:
        raise ProvenanceError(f"Provenance manifest '{path}' has an invalid event record.")
    _validate_uuid(event["event_id"], "event_id", path)
    _validate_timestamp(event["timestamp"], "event timestamp", path)
    if event["operation"] not in _OPERATIONS:
        raise ProvenanceError(f"Provenance manifest '{path}' has an unsupported operation.")
    if not isinstance(event["affected_row_ids"], list) or not all(
        isinstance(row_id, str) for row_id in event["affected_row_ids"]
    ):
        raise ProvenanceError(f"Provenance manifest '{path}' has invalid affected row IDs.")
    _validate_hash(event["resulting_log_sha256"], "event resulting hash", path)
    if event["previous_log_sha256"] != _MISSING_LOG_HASH:
        _validate_hash(event["previous_log_sha256"], "event previous hash", path)
    if event["environment_id"] not in environment_ids or not isinstance(event["metadata"], dict):
        raise ProvenanceError(f"Provenance manifest '{path}' has invalid event metadata.")


def _validate_ledger_identity(
    payload: dict[str, Any],
    events: list[dict[str, Any]],
    created_at: datetime,
    updated_at: datetime,
    path: Path,
) -> None:
    if payload["log"]["sha256"] != events[-1]["resulting_log_sha256"]:
        raise ProvenanceError(f"Provenance manifest '{path}' log hash is not ledger-derived.")
    first = _validate_timestamp(events[0]["timestamp"], "event timestamp", path)
    last = _validate_timestamp(events[-1]["timestamp"], "event timestamp", path)
    if created_at != first or updated_at != last:
        raise ProvenanceError(f"Provenance manifest '{path}' has inconsistent ledger timestamps.")


def _validate_pending(
    pending: object,
    payload: dict[str, Any],
    environment_ids: set[str],
    path: Path,
) -> None:
    if pending is None:
        return
    required = {
        "event",
        "previous_log_sha256",
        "resulting_log_sha256",
        "resulting_log_row_count",
    }
    if not isinstance(pending, dict) or set(pending) != required:
        raise ProvenanceError(f"Provenance manifest '{path}' has an invalid pending transaction.")
    event = pending["event"]
    _validate_event(event, len(payload["events"]) + 1, environment_ids, path)
    if any(item["event_id"] == event["event_id"] for item in payload["events"]):
        raise ProvenanceError(f"Provenance manifest '{path}' has duplicate event IDs.")
    pending_timestamp = _validate_timestamp(event["timestamp"], "event timestamp", path)
    last_timestamp = _validate_timestamp(
        payload["events"][-1]["timestamp"],
        "event timestamp",
        path,
    )
    if pending_timestamp < last_timestamp:
        raise ProvenanceError(
            f"Provenance manifest '{path}' has a non-monotonic pending timestamp."
        )
    if event["operation"] == "initialize" or not _is_nonnegative_int(
        pending["resulting_log_row_count"]
    ):
        raise ProvenanceError(f"Provenance manifest '{path}' has an invalid pending transaction.")
    if not (
        pending["previous_log_sha256"]
        == event["previous_log_sha256"]
        == payload["log"]["sha256"]
    ) or pending["resulting_log_sha256"] != event["resulting_log_sha256"]:
        raise ProvenanceError(f"Provenance manifest '{path}' has invalid pending hashes.")


def _validate_hash(value: object, field: str, path: Path) -> None:
    if not isinstance(value, str) or len(value) != _HASH_LENGTH:
        raise ProvenanceError(f"Provenance manifest '{path}' has invalid {field}.")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ProvenanceError(f"Provenance manifest '{path}' has invalid {field}.") from exc


def _validate_uuid(value: object, field: str, path: Path) -> None:
    try:
        uuid.UUID(str(value))
    except (ValueError, TypeError, AttributeError) as exc:
        raise ProvenanceError(
            f"Provenance manifest '{path}' has invalid {field}: {value!r}."
        ) from exc


def _validate_timestamp(value: object, field: str, path: Path) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ProvenanceError(
            f"Provenance manifest '{path}' has invalid {field}: {value!r}."
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ProvenanceError(f"Provenance manifest '{path}' {field} must be UTC.")
    return parsed


def _environment_hash(environment: dict[str, Any]) -> str:
    details = {
        key: value
        for key, value in environment.items()
        if key not in {"environment_id", "captured_at"}
    }
    encoded = json.dumps(
        details,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return _sha256(encoded)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0
