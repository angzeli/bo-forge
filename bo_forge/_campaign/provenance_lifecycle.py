"""Explicit preview/apply transactions for adoption, migration, formatting and forks."""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from bo_forge._campaign import provenance as io
from bo_forge._campaign.provenance_resume import enforce_resumable, inspect_provenance
from bo_forge._campaign.provenance_schema import validate_manifest_payload
from bo_forge._campaign.provenance_v2 import archive_bytes, verify_archives
from bo_forge.config import CampaignConfig, parse_campaign_config
from bo_forge.errors import LogConflictError, ProvenanceError
from bo_forge.logs import _campaign_log_lock
from bo_forge.validation import validate_campaign_data


def adopt_provenance(config_path, log_path, *, apply=False, reason="", expected_identities=None):
    """Preview legacy adoption; apply only with a reason and unchanged preview identities."""
    return lifecycle(
        "adopt",
        config_path,
        log_path,
        apply=apply,
        reason=reason,
        expected_identities=expected_identities,
    )


def migrate_provenance(config_path, log_path, *, apply=False, reason="", expected_identities=None):
    """Preview explicit v1-to-v2 migration, preserving campaign identity and old events."""
    return lifecycle(
        "migrate",
        config_path,
        log_path,
        apply=apply,
        reason=reason,
        expected_identities=expected_identities,
    )


def accept_provenance_config(
    config_path,
    log_path,
    *,
    apply=False,
    reason="",
    expected_identities=None,
):
    """Preview acknowledgment of byte-only YAML changes for a schema-v2 campaign."""
    return lifecycle(
        "accept-config",
        config_path,
        log_path,
        apply=apply,
        reason=reason,
        expected_identities=expected_identities,
    )


def fork_campaign(
    config_path,
    log_path,
    destination,
    *,
    config_changes=None,
    apply=False,
    reason="",
    expected_identities=None,
):
    """Preview a child with inherited CSV bytes and restricted configuration changes."""
    return lifecycle(
        "fork",
        config_path,
        log_path,
        destination=destination,
        config_changes=config_changes,
        apply=apply,
        reason=reason,
        expected_identities=expected_identities,
    )


def source_identities(config_file: Path, log_file: Path) -> dict[str, Any]:
    manifest_path = io.manifest_path_for_log(log_file)
    return {
        "config_path": str(config_file),
        "log_path": str(log_file),
        "config_sha256": io._sha256_file(config_file),
        "log_sha256": io._sha256_file(log_file),
        "manifest_sha256": io._sha256_file_or_none(manifest_path),
        "manifest_present": manifest_path.exists() or manifest_path.is_symlink(),
    }


def lifecycle(
    operation: str,
    config_path,
    log_path,
    *,
    apply=False,
    reason="",
    expected_identities=None,
    destination=None,
    config_changes=None,
) -> dict[str, Any]:
    """Validate and publish one lifecycle operation under the existing campaign lock."""
    config_file = Path(config_path).expanduser().resolve()
    log_file = Path(log_path).expanduser().resolve()
    destination = None if destination is None else Path(destination).expanduser().absolute()
    if operation not in {"adopt", "migrate", "accept-config", "fork"}:
        raise ProvenanceError("Unknown provenance lifecycle operation.")
    if operation != "fork" and (destination is not None or config_changes):
        raise ProvenanceError("Only fork operations accept a destination or config changes.")
    if apply and (not isinstance(reason, str) or not reason.strip() or expected_identities is None):
        raise ProvenanceError(
            "Applying requires a nonempty reason and preview expected_identities."
        )
    try:
        with _campaign_log_lock(log_file):
            identities = source_identities(config_file, log_file)
            identities["request_sha256"] = io._sha256_bytes(
                io._canonical_json_bytes(
                    {
                        "operation": operation,
                        "destination": str(destination) if destination else None,
                        "config_changes": config_changes or {},
                    }
                )
            )
            if apply and identities != expected_identities:
                raise LogConflictError("Lifecycle preview is stale. Preview the campaign again.")
            config, df, manifest = _validate_operation(operation, config_file, log_file)
            child = _prepare_child_config(config_file, config, df, destination, config_changes)
            if operation == "fork" and child is None:
                raise ProvenanceError(
                    "Fork requires a previously nonexistent destination directory."
                )
            result = _preview(operation, identities, manifest, df, destination, child)
            if not apply or result["no_op"]:
                return result
            _publish(
                operation,
                config_file,
                log_file,
                config,
                df,
                manifest,
                reason.strip(),
                identities,
                destination,
                child,
            )
            result["applied"] = True
            return result
    except OSError as exc:
        raise ProvenanceError(f"Could not complete provenance lifecycle operation: {exc}") from exc


def _validate_operation(operation, config_file, log_file):
    path = io.manifest_path_for_log(log_file)
    if path.is_symlink():
        raise ProvenanceError("Lifecycle operations refuse symlinked manifests.")
    if operation == "adopt" and (path.exists() or path.is_symlink()):
        raise ProvenanceError("Cannot adopt a campaign with an existing manifest.")
    config = CampaignConfig.from_yaml(config_file)
    try:
        df = pd.read_csv(log_file, keep_default_na=False)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError) as exc:
        raise ProvenanceError(f"Cannot read campaign CSV: {exc}") from exc
    validate_campaign_data(config, df)
    inspection = inspect_provenance(config_file, log_file, include_environment=False)
    manifest = inspection.manifest
    if operation != "adopt" and manifest is None:
        raise ProvenanceError(
            "This operation requires a managed campaign; adopt legacy data first."
        )
    if operation == "accept-config":
        assert manifest is not None
        if manifest["schema_version"] != 2:
            raise ProvenanceError(
                "Explicit v2 migration is required before accepting config formatting."
            )
        if manifest["pending_transaction"] is not None:
            raise ProvenanceError(
                "Recover the pending transaction before accepting config formatting."
            )
        if not inspection.config_semantic_match:
            raise LogConflictError(
                "Config semantics changed; formatting acceptance is not allowed."
            )
        io._assert_manifest_log_state(manifest, log_file)
    else:
        enforce_resumable(inspection)
    if manifest is not None:
        verify_archives(manifest, path)
    if operation == "migrate":
        _validate_migration_baseline(manifest)
    return config, df, manifest


def _validate_migration_baseline(manifest):
    if manifest["schema_version"] != 1:
        return
    row_count = manifest["events"][0]["metadata"].get("row_count")
    if type(row_count) is not int or row_count < 0:
        raise ProvenanceError(
            "Cannot migrate: initialization metadata must record a nonnegative integer row_count."
        )


def _prepare_child_config(config_file, config, df, destination, changes):
    if destination is None:
        if changes:
            raise ProvenanceError("Configuration changes are only supported for forks.")
        return None
    if destination.exists() or destination.is_symlink():
        raise ProvenanceError("Fork destination already exists.")
    if not destination.parent.is_dir():
        raise ProvenanceError("Fork destination parent must be an existing directory.")
    unresolved = df["status"] == "suggested"
    if config.review.enabled:
        unresolved &= df["review_status"].isin(["pending", "accepted"])
    if unresolved.any():
        raise ProvenanceError("Resolve the observation queue before forking a campaign.")
    changes = changes or {}
    if not isinstance(changes, dict) or set(changes) - {"campaign_name", "bo", "model"}:
        raise ProvenanceError("Fork changes are limited to campaign_name, bo, and model.")
    raw = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    for key, value in changes.items():
        if key in {"bo", "model"}:
            if not isinstance(value, dict):
                raise ProvenanceError(f"Fork {key} changes must be a mapping.")
            raw[key] = {**(raw.get(key) or {}), **value}
        else:
            raw[key] = value
    parsed = parse_campaign_config(raw)
    before, after = asdict(config), asdict(parsed)
    differences = {
        key: {"previous": before[key], "resulting": after[key]}
        for key in before
        if before[key] != after[key]
    }
    if set(differences) - {"campaign_name", "bo", "model"}:
        raise ProvenanceError("Fork would change protected campaign definitions.")
    validate_campaign_data(parsed, df)
    return parsed, yaml.safe_dump(raw, sort_keys=False).encode("utf-8"), differences


def _preview(operation, identities, manifest, df, destination, child):
    no_op = operation == "migrate" and manifest["schema_version"] == 2
    if operation == "accept-config":
        no_op = identities["config_sha256"] == manifest["config"]["byte_sha256"]
    return {
        "operation": operation,
        "applied": False,
        "no_op": no_op,
        "expected_identities": identities,
        "row_count": len(df),
        "campaign_id": None if manifest is None else manifest["campaign_id"],
        "destination": str(destination) if destination is not None else None,
        "config_differences": {} if child is None else child[2],
        "history": "earlier_history_unknown"
        if operation == "adopt"
        else ("inherited_parent_data" if operation == "fork" else "preserved"),
    }


def _publish(
    operation, config_file, log_file, config, df, manifest, reason, identities, destination, child
):
    path = io.manifest_path_for_log(log_file)
    if operation == "fork":
        _publish_fork(config_file, log_file, manifest, reason, identities, destination, child)
        return
    if operation == "adopt":
        updated = io._initial_manifest(
            config_file=config_file,
            log_file=log_file,
            config=config,
            config_bytes=config_file.read_bytes(),
            log_hash=identities["log_sha256"],
            row_count=len(df),
        )
        updated["origin"].update(kind="adopt", history="earlier_history_unknown")
        updated["events"][0].update(operation="adopt", metadata={"reason": reason})
    else:
        updated = io._deep_copy(manifest)
        previous = archive_bytes(path, path.read_bytes(), "manifest")
        if operation == "migrate":
            updated.update(
                schema_version=2,
                archives=[],
                origin={
                    "kind": "initialize",
                    "history": "tracked_from_initialization",
                    "baseline_log_sha256": manifest["events"][0]["resulting_log_sha256"],
                    "baseline_row_count": manifest["events"][0]["metadata"]["row_count"],
                    "parent": None,
                },
            )
            metadata = {
                "reason": reason,
                "previous_manifest": previous,
                "from_schema": 1,
                "to_schema": 2,
            }
        else:
            old_config = archive_bytes(
                path, manifest["config"]["snapshot"].encode("utf-8"), "config"
            )
            _add_archive(updated, old_config)
            metadata = {
                "reason": reason,
                "previous_manifest": previous,
                "previous_config": old_config,
                "old_byte_sha256": manifest["config"]["byte_sha256"],
                "new_byte_sha256": identities["config_sha256"],
            }
            updated["config"].update(
                snapshot=config_file.read_bytes().decode("utf-8"),
                byte_sha256=identities["config_sha256"],
            )
        _add_archive(updated, previous)
        _lifecycle_event(updated, config_file, operation.replace("-", "_"), metadata)
    _validate_prepared(updated, path)
    _recheck(config_file, log_file, identities)
    if operation == "adopt":
        temporary = io._prepare_json_temp(path, updated)
        try:
            os.link(temporary, path)
        finally:
            # Publication is the commit point; leftover temporary links are harmless.
            io._remove_file(temporary)
    else:
        _replace_manifest(path, updated)


def _replace_manifest(path, updated):
    previous = path.read_bytes()
    mode = path.stat().st_mode & 0o777
    try:
        io._write_json_atomic(path, updated)
    except Exception:
        # Rollback boundary: restore only the exact candidate published by this action.
        if path.read_bytes() == io._manifest_bytes(updated):
            io._write_bytes_atomic(path, previous, mode=mode)
        raise


def _lifecycle_event(manifest, config_file, operation, metadata):
    environment_id = io._merge_environment(manifest, io.capture_environment(config_file))
    timestamp = max(io._utc_now(), manifest["updated_at"])
    event = io._new_event(
        sequence=len(manifest["events"]) + 1,
        timestamp=timestamp,
        operation=operation,
        row_ids=[],
        previous_hash=manifest["log"]["sha256"],
        resulting_hash=manifest["log"]["sha256"],
        environment_id=environment_id,
        metadata=metadata,
    )
    manifest["events"].append(event)
    manifest["updated_at"] = timestamp


def _add_archive(manifest, reference):
    if reference not in manifest["archives"]:
        manifest["archives"].append(reference)


def _recheck(config_file, log_file, identities):
    current = source_identities(config_file, log_file)
    if any(current[key] != identities[key] for key in current):
        raise LogConflictError("Campaign changed during lifecycle preparation; preview again.")


def _validate_prepared(manifest, path):
    validate_manifest_payload(
        manifest,
        path,
        semantic_hash=io.config_semantic_sha256,
        optimization_identity=io._optimization_identity,
    )
    verify_archives(manifest, path)


def _publish_fork(config_file, log_file, parent, reason, identities, destination, child):
    from bo_forge._campaign.provenance_fork import publish_fork

    publish_fork(config_file, log_file, parent, reason, identities, destination, child)
