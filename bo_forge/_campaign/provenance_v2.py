"""Schema-v2 lifecycle metadata and portable, immutable snapshot archives."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import yaml

from bo_forge._campaign import provenance as io
from bo_forge._campaign.provenance_schema import (
    _is_nonnegative_int,
    _validate_hash,
    _validate_uuid,
)
from bo_forge.config import parse_campaign_config
from bo_forge.errors import ProvenanceError


def validate_lifecycle_metadata(payload: dict[str, Any], path: Path) -> None:
    origin = payload["origin"]
    required = {"kind", "history", "baseline_log_sha256", "baseline_row_count", "parent"}
    if not isinstance(origin, dict) or set(origin) != required:
        raise ProvenanceError("Invalid provenance origin metadata.")
    histories = {
        "initialize": "tracked_from_initialization",
        "adopt": "earlier_history_unknown",
        "fork": "inherited_parent_data",
    }
    if not isinstance(origin["kind"], str) or origin["kind"] not in histories:
        raise ProvenanceError("Invalid provenance origin kind.")
    if origin["history"] != histories[origin["kind"]]:
        raise ProvenanceError("Invalid provenance origin history.")
    first = payload["events"][0]
    if first["operation"] != origin["kind"]:
        raise ProvenanceError("Origin does not match the first provenance event.")
    _validate_hash(origin["baseline_log_sha256"], "origin baseline hash", path)
    if origin["baseline_log_sha256"] != first["resulting_log_sha256"] or not _is_nonnegative_int(
        origin["baseline_row_count"]
    ):
        raise ProvenanceError("Invalid provenance origin baseline.")
    archives = payload["archives"]
    if not isinstance(archives, list):
        raise ProvenanceError("Invalid provenance archives.")
    names: set[str] = set()
    for archive in archives:
        validate_archive_reference(archive, path)
        if archive["path"] in names:
            raise ProvenanceError("Duplicate provenance archive reference.")
        names.add(archive["path"])
    _validate_parent(origin, archives, path)
    for event in payload["events"]:
        _validate_lifecycle_event(event, archives)


def validate_archive_reference(reference: object, path: Path) -> None:
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256", "kind"}:
        raise ProvenanceError("Invalid provenance archive reference.")
    name = reference["path"]
    if not isinstance(name, str) or Path(name).name != name or name in {"", ".", ".."}:
        raise ProvenanceError("Provenance archives must use sibling filenames.")
    _validate_hash(reference["sha256"], "archive hash", path)
    if not isinstance(reference["kind"], str) or reference["kind"] not in {
        "manifest", "config", "parent_manifest",
    }:
        raise ProvenanceError("Invalid provenance archive kind.")
    suffix = "yaml" if reference["kind"] == "config" else "json"
    if not name.endswith(f".{reference['sha256']}.archive.{suffix}"):
        raise ProvenanceError("Provenance archive filename must include its content hash.")


def _validate_parent(origin: dict, archives: list, path: Path) -> None:
    parent = origin["parent"]
    if origin["kind"] != "fork":
        if parent is not None:
            raise ProvenanceError("Only fork origins may reference a parent.")
        return
    fields = {"campaign_id", "manifest", "baseline_log_sha256", "config_differences"}
    if not isinstance(parent, dict) or set(parent) != fields:
        raise ProvenanceError("Invalid parent lineage metadata.")
    _validate_uuid(parent["campaign_id"], "parent campaign ID", path)
    if parent["manifest"] not in archives or parent["manifest"]["kind"] != "parent_manifest":
        raise ProvenanceError("Parent manifest snapshot is not archived.")
    if parent["baseline_log_sha256"] != origin["baseline_log_sha256"]:
        raise ProvenanceError("Parent and child baseline hashes differ.")
    changes = parent["config_differences"]
    if not isinstance(changes, dict) or set(changes) - {"campaign_name", "bo", "model"}:
        raise ProvenanceError("Invalid child configuration differences.")


def _validate_lifecycle_event(event: dict, archives: list) -> None:
    operation = event["operation"]
    if operation not in {"adopt", "migrate", "accept_config", "fork"}:
        return
    metadata = event["metadata"]
    fields = {
        "adopt": {"reason"},
        "fork": {"reason"},
        "migrate": {"reason", "previous_manifest", "from_schema", "to_schema"},
        "accept_config": {
            "reason",
            "previous_manifest",
            "previous_config",
            "old_byte_sha256",
            "new_byte_sha256",
        },
    }
    if set(metadata) != fields[operation]:
        raise ProvenanceError("Invalid lifecycle event metadata fields.")
    reason = metadata.get("reason")
    if not isinstance(reason, str) or not reason.strip() or event["affected_row_ids"]:
        raise ProvenanceError("Lifecycle events require a reason and no fabricated row events.")
    if operation in {"migrate", "accept_config"}:
        if event["previous_log_sha256"] != event["resulting_log_sha256"]:
            raise ProvenanceError("Lifecycle metadata changes cannot change the log hash.")
        if metadata.get("previous_manifest") not in archives:
            raise ProvenanceError("Lifecycle event requires an archived previous manifest.")
    if operation == "migrate" and (
        type(metadata.get("from_schema")) is not int or metadata["from_schema"] != 1
        or type(metadata.get("to_schema")) is not int or metadata["to_schema"] != 2
    ):
        raise ProvenanceError("Invalid migration versions.")
    if operation == "accept_config":
        if metadata.get("previous_config") not in archives:
            raise ProvenanceError("Formatting acceptance requires the old config snapshot.")
        for key in ("old_byte_sha256", "new_byte_sha256"):
            _validate_hash(metadata.get(key), key, Path("manifest"))


def verify_archives(manifest: dict, manifest_path: Path) -> None:
    """Verify captured bytes locally; never follow a mutable parent campaign path."""
    for reference in manifest.get("archives", []):
        path = manifest_path.parent / reference["path"]
        if path.is_symlink() or not path.is_file():
            raise ProvenanceError(f"Missing or unsafe provenance archive: '{path.name}'.")
        if io._sha256_file(path) != reference["sha256"]:
            raise ProvenanceError(f"Provenance archive hash mismatch: '{path.name}'.")
    if manifest.get("origin", {}).get("parent") is not None:
        _verify_parent_snapshot(manifest, manifest_path)


def _verify_parent_snapshot(manifest: dict, manifest_path: Path) -> None:
    parent = manifest["origin"]["parent"]
    try:
        snapshot = json.loads((manifest_path.parent / parent["manifest"]["path"]).read_bytes())
    except (ValueError, UnicodeError) as exc:
        raise ProvenanceError("Invalid captured parent manifest.") from exc
    from bo_forge._campaign.provenance_schema import validate_manifest_payload

    validate_manifest_payload(
        snapshot,
        manifest_path,
        semantic_hash=io.config_semantic_sha256,
        optimization_identity=io._optimization_identity,
    )
    if snapshot["campaign_id"] != parent["campaign_id"] or (
        snapshot["log"]["sha256"] != parent["baseline_log_sha256"]
        or snapshot["log"]["row_count"] != manifest["origin"]["baseline_row_count"]
        or snapshot["pending_transaction"] is not None
    ):
        raise ProvenanceError("Captured parent manifest does not match child lineage.")
    _verify_configuration_lineage(manifest, snapshot)


def _verify_configuration_lineage(manifest: dict, snapshot: dict) -> None:
    parent_config = parse_campaign_config(yaml.safe_load(snapshot["config"]["snapshot"]))
    child_config = parse_campaign_config(yaml.safe_load(manifest["config"]["snapshot"]))
    before, after = asdict(parent_config), asdict(child_config)
    permitted = ("campaign_name", "bo", "model")
    differences = {
        key: {"previous": before[key], "resulting": after[key]}
        for key in permitted if before[key] != after[key]
    }
    if manifest["origin"]["parent"]["config_differences"] != differences:
        raise ProvenanceError("Child configuration differences do not match captured snapshots.")
    projected = replace(parent_config, **{key: getattr(child_config, key) for key in permitted})
    if io.config_semantic_sha256(projected) != manifest["config"]["semantic_sha256"]:
        raise ProvenanceError("Child snapshot changes protected campaign definitions.")


def archive_bytes(manifest_path: Path, data: bytes, kind: str) -> dict[str, str]:
    """Publish a read-only content-addressed snapshot without overwriting a file."""
    digest = io._sha256_bytes(data)
    suffix = "yaml" if kind == "config" else "json"
    path = manifest_path.with_name(f"{manifest_path.name}.{digest}.archive.{suffix}")
    if path.is_symlink():
        raise ProvenanceError("Refusing a symlinked provenance archive.")
    temporary = None
    try:
        with NamedTemporaryFile(dir=path.parent, prefix=".archive-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o444)
        os.link(temporary, path)
    except FileExistsError:
        if path.is_symlink() or path.read_bytes() != data:
            raise ProvenanceError("Existing provenance archive differs from its hash.") from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"path": path.name, "sha256": digest, "kind": kind}
