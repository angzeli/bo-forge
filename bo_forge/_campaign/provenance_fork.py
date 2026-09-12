"""Prepare and publish a self-contained child campaign without altering its parent."""

from __future__ import annotations

from pathlib import Path
from tempfile import mkdtemp

from bo_forge._campaign import provenance as io
from bo_forge._campaign.provenance_v2 import archive_bytes
from bo_forge._filesystem import _DirectoryPublicationUnavailable, rename_directory_exclusive
from bo_forge.errors import ProvenanceError


def publish_fork(config_file, log_file, parent, reason, identities, destination, child):
    from bo_forge._campaign.provenance_lifecycle import _recheck, _validate_prepared
    from bo_forge.logs import _campaign_log_lock

    destination = Path(destination)
    with _campaign_log_lock(destination / "campaign.csv"):
        if destination.exists() or destination.is_symlink():
            raise ProvenanceError("Fork destination already exists.")
        temporary = Path(mkdtemp(prefix=f".{destination.name}.preparing-", dir=destination.parent))
        config, config_bytes, differences = child
        child_config, child_log = temporary / "campaign.yaml", temporary / "campaign.csv"
        child_config.write_bytes(config_bytes)
        child_log.write_bytes(log_file.read_bytes())
        manifest_path = io.manifest_path_for_log(child_log)
        manifest = io._initial_manifest(
            config_file=child_config,
            log_file=child_log,
            config=config,
            config_bytes=config_bytes,
            log_hash=identities["log_sha256"],
            row_count=parent["log"]["row_count"],
        )
        snapshot = archive_bytes(
            manifest_path, io.manifest_path_for_log(log_file).read_bytes(), "parent_manifest"
        )
        manifest["archives"].append(snapshot)
        manifest["origin"].update(
            kind="fork",
            history="inherited_parent_data",
            parent={
                "campaign_id": parent["campaign_id"],
                "manifest": snapshot,
                "baseline_log_sha256": identities["log_sha256"],
                "config_differences": differences,
            },
        )
        manifest["events"][0].update(operation="fork", metadata={"reason": reason})
        _validate_prepared(manifest, manifest_path)
        io._validate_written_log(child_log, config)
        io._write_json_atomic(manifest_path, manifest)
        _recheck(config_file, log_file, identities)
        _rename_directory_exclusive(temporary, destination)


def _rename_directory_exclusive(source: Path, destination: Path) -> None:
    """Use OS no-replace rename, including when a competing empty directory appears."""
    try:
        rename_directory_exclusive(source, destination)
    except FileExistsError as exc:
        raise ProvenanceError("Fork destination already exists.") from exc
    except _DirectoryPublicationUnavailable as exc:
        raise ProvenanceError(exc.strerror) from exc
