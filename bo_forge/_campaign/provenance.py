"""Versioned campaign provenance manifests and managed mutation transactions."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import stat
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import pandas as pd

from bo_forge._campaign.provenance_environment import capture_environment
from bo_forge._campaign.provenance_schema import (
    validate_manifest_payload,
)
from bo_forge.config import CampaignConfig
from bo_forge.errors import BOForgeError, LogConflictError, LogWriteError, ProvenanceError
from bo_forge.validation import validate_campaign_data

MANIFEST_SCHEMA_VERSION = 1
_MISSING_LOG_HASH = "<missing-log>"


def manifest_path_for_log(log_path: str | Path) -> Path:
    """Return the canonical sidecar path for a campaign log."""
    canonical = Path(log_path).expanduser().resolve(strict=False)
    return canonical.with_name(f"{canonical.name}.manifest.json")


def config_semantic_sha256(config: CampaignConfig) -> str:
    """Return the normalized semantic identity for a parsed campaign config."""
    normalized = asdict(config)
    normalized["constraints"] = [
        {
            "name": item.name,
            "expression_ast": _canonical_expression_ast(item.expression),
        }
        for item in config.constraints
    ]
    if config.cost is not None:
        normalized["cost"]["expression_ast"] = _canonical_expression_ast(
            config.cost.expression
        )
        normalized["cost"].pop("expression", None)
    return _sha256_bytes(_canonical_json_bytes(normalized))


def load_manifest(log_path: str | Path) -> dict[str, Any] | None:
    """Load and structurally validate a campaign manifest when one exists."""
    path = manifest_path_for_log(log_path)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ProvenanceError(f"Could not read provenance manifest '{path}': {exc}") from exc
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ProvenanceError(
            f"Provenance manifest '{path}' is not valid UTF-8 JSON: {exc}"
        ) from exc
    validate_manifest_payload(
        payload,
        path,
        semantic_hash=config_semantic_sha256,
        optimization_identity=_optimization_identity,
    )
    return payload


def validate_manifest_references(
    config_path: str | Path,
    log_path: str | Path,
) -> dict[str, Any] | None:
    """Validate that a manifest belongs to the supplied config and log paths."""
    manifest = load_manifest(log_path)
    if manifest is None:
        return None
    manifest_path = manifest_path_for_log(log_path)
    referenced_config, referenced_log = _resolved_manifest_paths(manifest_path, manifest)
    expected_config = Path(config_path).expanduser().resolve(strict=False)
    expected_log = Path(log_path).expanduser().resolve(strict=False)
    if referenced_config != expected_config or referenced_log != expected_log:
        raise ProvenanceError(
            "Provenance manifest path references do not match the requested campaign: "
            f"manifest='{manifest_path}'.",
            reason_code="manifest_path_mismatch",
            recovery_action="Use the config and log referenced by this manifest.",
        )
    return manifest


def validate_manifest_for_load(
    config_path: str | Path,
    log_path: str | Path,
    *,
    config: CampaignConfig,
    log_row_count: int,
    provenance_policy: str = "compatible",
) -> dict[str, Any] | None:
    """Fail closed when a present manifest disagrees with current campaign files."""
    from bo_forge._campaign.provenance_resume import enforce_resumable, inspect_provenance

    inspection = inspect_provenance(
        config_path,
        log_path,
        provenance_policy=provenance_policy,
        config=config,
        log_row_count=log_row_count,
        include_environment=False,
    )
    enforce_resumable(inspection)
    return inspection.manifest


def initialize_campaign(config_path: str | Path, log_path: str | Path) -> tuple[Path, Path]:
    """Create an empty canonical log and schema-v1 manifest without overwriting files."""
    from bo_forge.io import empty_campaign_log
    from bo_forge.logs import _campaign_log_lock
    config_file = Path(config_path).expanduser().resolve(strict=False)
    log_file = Path(log_path).expanduser().resolve(strict=False)
    manifest_file = manifest_path_for_log(log_file)
    config = CampaignConfig.from_yaml(config_file)
    config_bytes = _read_config_bytes(config_file)
    empty_log = empty_campaign_log(config)

    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ProvenanceError(
            f"Could not prepare campaign log directory '{log_file.parent}': {exc}"
        ) from exc
    with _campaign_log_lock(log_file):
        if log_file.exists() or manifest_file.exists():
            existing = log_file if log_file.exists() else manifest_file
            raise ProvenanceError(
                f"Cannot initialize campaign because '{existing}' already exists."
            )
        log_temp = _prepare_csv_temp(log_file, empty_log, config)
        try:
            log_hash = _sha256_file(log_temp)
            manifest = _initial_manifest(
                config_file=config_file,
                log_file=log_file,
                config=config,
                config_bytes=config_bytes,
                log_hash=log_hash,
                row_count=len(empty_log),
            )
            manifest_temp = _prepare_json_temp(manifest_file, manifest)
            _link_initialized_files(log_temp, log_file, manifest_temp, manifest_file)
        finally:
            _remove_file(log_temp)
            if "manifest_temp" in locals():
                _remove_file(manifest_temp)
    return log_file, manifest_file


def initialize_campaign_session(
    session_type: Any,
    config_path: str | Path,
    log_path: str | Path,
) -> Any:
    """Initialize and load a session, removing only unchanged artifacts on failure."""
    from bo_forge.logs import _campaign_log_lock
    initialized_log, initialized_manifest = initialize_campaign(config_path, log_path)
    try:
        log_bytes = initialized_log.read_bytes()
        manifest_bytes = initialized_manifest.read_bytes()
        return session_type.from_files(
            config_path,
            initialized_log,
            provenance_policy="required",
        )
    except Exception as exc:
        # This broad boundary protects the exact files created before session loading.
        try:
            with _campaign_log_lock(initialized_log):
                unchanged = (
                    "log_bytes" in locals()
                    and "manifest_bytes" in locals()
                    and initialized_log.exists()
                    and initialized_manifest.exists()
                    and initialized_log.read_bytes() == log_bytes
                    and initialized_manifest.read_bytes() == manifest_bytes
                )
                if unchanged:
                    initialized_log.unlink()
                    initialized_manifest.unlink()
                else:
                    raise OSError("created files changed or could not be verified")
        except OSError as cleanup_exc:
            raise ProvenanceError(
                "Campaign initialization failed and rollback was incomplete; inspect "
                f"'{initialized_log}' and '{initialized_manifest}': {cleanup_exc}."
            ) from exc
        raise


def provenance_summary(
    config_path: str | Path,
    log_path: str | Path,
) -> pd.DataFrame:
    """Return ordered campaign provenance and integrity fields without repairing files."""
    from bo_forge._campaign.provenance_resume import provenance_summary as inspect_summary

    return inspect_summary(config_path, log_path)


def write_managed_campaign_log(
    path: Path,
    df: pd.DataFrame,
    *,
    config: CampaignConfig | None,
    operation: str,
    affected_row_ids: list[str],
    metadata: dict[str, object],
    expected_managed: bool | None = None,
) -> bool:
    """Write a managed campaign mutation, returning False for legacy campaigns."""
    manifest_file = manifest_path_for_log(path)
    manifest_exists = manifest_file.exists()
    if expected_managed is not None and manifest_exists != expected_managed:
        raise LogConflictError(
            "Campaign provenance state changed during mutation. Reload the campaign "
            f"before retrying: log='{path}'."
        )
    if not manifest_exists:
        return False
    temp_path: Path | None = None
    backup_path: Path | None = None
    preserve_backup = False
    try:
        manifest, managed_config, config_file = _prepare_managed_state(
            manifest_file,
            path,
            config,
        )
        temp_path = _prepare_csv_temp(path, df, managed_config)
        previous_manifest_bytes = manifest_file.read_bytes()
        previous_manifest_mode = stat.S_IMODE(manifest_file.stat().st_mode)
        backup_path = _copy_backup(path)
    except BOForgeError:
        _remove_file(temp_path)
        _remove_file(backup_path)
        raise
    except OSError as exc:
        _remove_file(temp_path)
        _remove_file(backup_path)
        raise LogWriteError(f"Could not prepare managed campaign mutation: {exc}") from exc

    assert temp_path is not None
    assert backup_path is not None
    log_replaced = False
    try:
        resulting_hash = _sha256_file(temp_path)
        pending_manifest = _manifest_with_pending_transaction(
            manifest,
            config_file=config_file,
            operation=operation,
            affected_row_ids=affected_row_ids,
            metadata=metadata,
            resulting_hash=resulting_hash,
            resulting_row_count=len(df),
        )
        _write_json_atomic(manifest_file, pending_manifest)
        temp_path.chmod(stat.S_IMODE(path.stat().st_mode))
        temp_path.replace(path)
        log_replaced = True
        _validate_written_log(path, managed_config)
        finalized = _finalize_pending_manifest(pending_manifest)
        _write_json_atomic(manifest_file, finalized)
    except Exception as exc:
        # This is the rollback boundary for a two-file mutation.
        rollback_errors, recovery_backup = _rollback_managed_write(
            path=path,
            backup_path=backup_path,
            log_replaced=log_replaced,
            manifest_path=manifest_file,
            manifest_bytes=previous_manifest_bytes,
            manifest_mode=previous_manifest_mode,
        )
        preserve_backup = recovery_backup is not None
        if rollback_errors:
            recovery = (
                f" Recovery CSV retained at '{recovery_backup}'."
                if recovery_backup is not None
                else ""
            )
            raise LogWriteError(
                "Managed campaign mutation failed and rollback was incomplete: "
                f"error={exc}; rollback_errors={rollback_errors}.{recovery}"
            ) from exc
        if isinstance(exc, BOForgeError):
            raise
        raise LogWriteError(f"Managed campaign mutation failed: {exc}") from exc
    finally:
        _remove_file(temp_path)
        if not preserve_backup:
            _remove_file(backup_path)
    return True


def _initial_manifest(
    *,
    config_file: Path,
    log_file: Path,
    config: CampaignConfig,
    config_bytes: bytes,
    log_hash: str,
    row_count: int,
) -> dict[str, Any]:
    timestamp = _utc_now()
    environment = capture_environment(config_file)
    event = _new_event(
        sequence=1,
        timestamp=timestamp,
        operation="initialize",
        row_ids=[],
        previous_hash=_MISSING_LOG_HASH,
        resulting_hash=log_hash,
        environment_id=environment["environment_id"],
        metadata={"row_count": row_count},
    )
    manifest_parent = manifest_path_for_log(log_file).parent
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "campaign_id": str(uuid.uuid4()),
        "created_at": timestamp,
        "updated_at": timestamp,
        "paths": {
            "config": _relative_path(manifest_parent, config_file),
            "log": _relative_path(manifest_parent, log_file),
        },
        "config": {
            "snapshot": config_bytes.decode("utf-8"),
            "byte_sha256": _sha256_bytes(config_bytes),
            "semantic_sha256": config_semantic_sha256(config),
        },
        "log": {"sha256": log_hash, "row_count": row_count},
        "optimization": _optimization_identity(config),
        "environments": [environment],
        "events": [event],
        "pending_transaction": None,
    }


def _prepare_managed_state(
    manifest_path: Path,
    log_path: Path,
    provided_config: CampaignConfig | None,
) -> tuple[dict[str, Any], CampaignConfig, Path]:
    from bo_forge._campaign.provenance_resume import (
        enforce_resumable,
        inspect_loaded_manifest,
    )

    manifest = load_manifest(log_path)
    if manifest is None:
        raise ProvenanceError(f"Managed manifest disappeared during mutation: '{manifest_path}'.")
    config_file, referenced_log = _resolved_manifest_paths(manifest_path, manifest)
    if referenced_log != log_path.resolve(strict=False):
        raise ProvenanceError(
            f"Provenance manifest '{manifest_path}' references a different log."
        )
    inspection = inspect_loaded_manifest(
        config_file,
        log_path,
        manifest,
        include_environment=False,
    )
    enforce_resumable(inspection)
    assert inspection.config is not None
    current_config = inspection.config
    if provided_config is not None and (
        config_semantic_sha256(provided_config) != config_semantic_sha256(current_config)
    ):
        raise LogConflictError("Mutation config does not match the managed campaign config.")
    return manifest, current_config, config_file


def _manifest_with_pending_transaction(
    manifest: dict[str, Any],
    *,
    config_file: Path,
    operation: str,
    affected_row_ids: list[str],
    metadata: dict[str, object],
    resulting_hash: str,
    resulting_row_count: int,
) -> dict[str, Any]:
    updated = _deep_copy(manifest)
    environment = capture_environment(config_file)
    environment_id = _merge_environment(updated, environment)
    sequence = int(updated["events"][-1]["sequence"]) + 1
    timestamp = max(_utc_now(), str(updated["updated_at"]))
    event = _new_event(
        sequence=sequence,
        timestamp=timestamp,
        operation=operation,
        row_ids=affected_row_ids,
        previous_hash=updated["log"]["sha256"],
        resulting_hash=resulting_hash,
        environment_id=environment_id,
        metadata=metadata,
    )
    updated["pending_transaction"] = {
        "event": event,
        "previous_log_sha256": updated["log"]["sha256"],
        "resulting_log_sha256": resulting_hash,
        "resulting_log_row_count": resulting_row_count,
    }
    return updated


def _recover_pending_transaction(
    manifest_path: Path,
    log_path: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    pending = manifest["pending_transaction"]
    current_hash = _sha256_file_or_none(log_path)
    if current_hash == pending["resulting_log_sha256"]:
        recovered = _finalize_pending_manifest(manifest)
    elif current_hash == pending["previous_log_sha256"]:
        recovered = _deep_copy(manifest)
        recovered["pending_transaction"] = None
    else:
        raise LogConflictError(
            "Interrupted managed mutation cannot be recovered because the campaign log "
            "matches neither the previous nor intended hash."
        )
    _write_json_atomic(manifest_path, recovered)
    return recovered


def _finalize_pending_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    updated = _deep_copy(manifest)
    pending = updated["pending_transaction"]
    event = pending["event"]
    updated["events"].append(event)
    updated["log"] = {
        "sha256": pending["resulting_log_sha256"],
        "row_count": pending["resulting_log_row_count"],
    }
    updated["updated_at"] = event["timestamp"]
    updated["pending_transaction"] = None
    return updated


def _assert_manifest_log_state(manifest: dict[str, Any], log_path: Path) -> None:
    current_hash = _sha256_file_or_none(log_path)
    if current_hash != manifest["log"]["sha256"]:
        raise LogConflictError(
            "Managed campaign log does not match its provenance manifest. "
            "Restore a coherent campaign state before retrying."
        )
    try:
        row_count = len(pd.read_csv(log_path, keep_default_na=False))
    except (OSError, pd.errors.ParserError) as exc:
        raise LogWriteError(f"Could not inspect managed campaign log '{log_path}': {exc}") from exc
    if row_count != manifest["log"]["row_count"]:
        raise LogConflictError("Managed campaign row count does not match its manifest.")


def _resolved_manifest_paths(
    manifest_path: Path,
    manifest: dict[str, Any],
) -> tuple[Path, Path]:
    parent = manifest_path.parent
    return (
        _resolve_manifest_reference(parent, manifest["paths"]["config"]),
        _resolve_manifest_reference(parent, manifest["paths"]["log"]),
    )


def _resolve_manifest_reference(parent: Path, reference: str) -> Path:
    if reference == "~":
        return _author_home()
    if reference.startswith("~/"):
        return (_author_home() / reference[2:]).resolve(strict=False)
    return (parent / reference).resolve(strict=False)


def _optimization_identity(config: CampaignConfig) -> dict[str, Any]:
    bo = config.bo
    identity: dict[str, Any] = {
        "acquisition": bo.acquisition,
        "model_profile": config.model.profile,
        "random_seed": bo.random_seed,
        "initial_design_method": bo.initial_design_method,
        "initial_design_size": bo.initial_design_size,
        "batch_size": bo.batch_size,
        "raw_samples": bo.raw_samples,
        "num_restarts": bo.num_restarts,
        "mc_samples": bo.mc_samples,
        "min_normalized_distance": bo.min_normalized_distance,
    }
    if config.fidelity is not None:
        identity["fidelity"] = _deep_copy(asdict(config.fidelity))
    return identity


def _merge_environment(manifest: dict[str, Any], environment: dict[str, Any]) -> str:
    environment_id = str(environment["environment_id"])
    if not any(item.get("environment_id") == environment_id for item in manifest["environments"]):
        manifest["environments"].append(environment)
    return environment_id


def _new_event(
    *,
    sequence: int,
    timestamp: str,
    operation: str,
    row_ids: list[str],
    previous_hash: str,
    resulting_hash: str,
    environment_id: str,
    metadata: dict[str, object],
) -> dict[str, Any]:
    return {
        "sequence": sequence,
        "event_id": str(uuid.uuid4()),
        "timestamp": timestamp,
        "operation": operation,
        "affected_row_ids": list(row_ids),
        "previous_log_sha256": previous_hash,
        "resulting_log_sha256": resulting_hash,
        "environment_id": environment_id,
        "metadata": dict(metadata),
    }


def _prepare_csv_temp(path: Path, df: pd.DataFrame, config: CampaignConfig) -> Path:
    temp_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            df.to_csv(handle, index=False, float_format="%.17g")
            handle.flush()
            os.fsync(handle.fileno())
        _validate_written_log(temp_path, config)
    except BOForgeError:
        _remove_file(temp_path)
        raise
    except (OSError, TypeError, ValueError, UnicodeError) as exc:
        _remove_file(temp_path)
        raise LogWriteError(f"Could not prepare candidate campaign log '{path}': {exc}") from exc
    assert temp_path is not None
    return temp_path


def _validate_written_log(path: Path, config: CampaignConfig) -> None:
    try:
        df = pd.read_csv(path, keep_default_na=False)
    except (OSError, pd.errors.ParserError) as exc:
        raise LogWriteError(f"Could not read candidate campaign log '{path}': {exc}") from exc
    validate_campaign_data(config, df)


def _prepare_json_temp(path: Path, payload: dict[str, Any]) -> Path:
    temp_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(_manifest_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
    except (OSError, TypeError, ValueError) as exc:
        _remove_file(temp_path)
        raise ProvenanceError(f"Could not prepare provenance manifest '{path}': {exc}") from exc
    assert temp_path is not None
    return temp_path


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else None
    _write_bytes_atomic(path, _manifest_bytes(payload), mode=mode)


def _write_bytes_atomic(path: Path, data: bytes, *, mode: int | None = None) -> None:
    temp_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if mode is not None:
            temp_path.chmod(mode)
        temp_path.replace(path)
    finally:
        _remove_file(temp_path)


def _link_initialized_files(
    log_temp: Path,
    log_path: Path,
    manifest_temp: Path,
    manifest_path: Path,
) -> None:
    try:
        os.link(manifest_temp, manifest_path)
        try:
            os.link(log_temp, log_path)
        except OSError:
            manifest_path.unlink(missing_ok=True)
            raise
    except OSError as exc:
        raise ProvenanceError(f"Could not initialize managed campaign: {exc}") from exc


def _copy_backup(path: Path) -> Path:
    with NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".bak",
        delete=False,
    ) as handle:
        backup = Path(handle.name)
    try:
        shutil.copy2(path, backup)
    except OSError:
        _remove_file(backup)
        raise
    return backup


def _rollback_managed_write(
    *,
    path: Path,
    backup_path: Path,
    log_replaced: bool,
    manifest_path: Path,
    manifest_bytes: bytes,
    manifest_mode: int,
) -> tuple[list[str], Path | None]:
    errors: list[str] = []
    recovery_backup: Path | None = None
    if log_replaced:
        try:
            backup_path.replace(path)
        except OSError as exc:
            errors.append(f"log={exc}")
            if backup_path.exists():
                recovery_backup = backup_path
    try:
        _write_bytes_atomic(manifest_path, manifest_bytes, mode=manifest_mode)
    except OSError as exc:
        errors.append(f"manifest={exc}")
    return errors, recovery_backup


def _read_config_bytes(path: Path) -> bytes:
    try:
        data = path.read_bytes()
        data.decode("utf-8")
    except OSError as exc:
        raise ProvenanceError(f"Could not read campaign config '{path}': {exc}") from exc
    except UnicodeError as exc:
        raise ProvenanceError(f"Campaign config '{path}' must be UTF-8.") from exc
    return data


def _canonical_expression_ast(expression: str) -> str:
    parsed = ast.parse(expression, mode="eval")
    return ast.dump(parsed, annotate_fields=True, include_attributes=False)


def _relative_path(parent: Path, path: Path) -> str:
    author_home = _author_home()
    resolved_parent = parent.expanduser().resolve(strict=False)
    resolved_path = path.expanduser().resolve(strict=False)
    if resolved_path.is_relative_to(author_home) and not resolved_parent.is_relative_to(
        author_home
    ):
        return f"~/{resolved_path.relative_to(author_home).as_posix()}"
    return Path(os.path.relpath(resolved_path, resolved_parent)).as_posix()


def _author_home() -> Path:
    return Path.home().expanduser().resolve(strict=False)


def _manifest_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False).encode("utf-8") + b"\n"


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _deep_copy(payload: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(payload))


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ProvenanceError(f"Could not hash provenance file '{path}': {exc}") from exc
    return digest.hexdigest()


def _sha256_file_or_none(path: Path) -> str | None:
    return _sha256_file(path) if path.exists() else None


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _remove_file(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass
