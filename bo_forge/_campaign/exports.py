"""Keep campaign-owned files separate from user-selected artifact destinations."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path


def _validate_export_destination(
    path: str | Path, config_path: str | Path, log_path: str | Path,
) -> None:
    """Reject source aliases; callers must match their writer's home expansion."""
    try:
        destination = Path(path).resolve()
    except RuntimeError as exc:
        raise OSError(f"Could not resolve export destination '{path}': {exc}") from exc
    for source, reserved_name in _campaign_source_paths(config_path, log_path):
        collision = destination == source.resolve() or (
            destination.exists() and source.exists() and destination.samefile(source)
        )
        # Reserve sidecar case variants portably, including before the file exists.
        if reserved_name and destination.name.casefold() == source.name.casefold():
            collision = collision or destination.parent == source.parent or (
                destination.parent.exists() and source.parent.exists()
                and destination.parent.samefile(source.parent)
            )
        if collision:
            raise OSError(
                f"Export destination '{path}' conflicts with campaign source '{source}'. "
                "Choose a separate artifact output path; campaign files were not changed."
            )


def _campaign_source_paths(
    config_path: str | Path, log_path: str | Path,
) -> Iterator[tuple[Path, bool]]:
    from bo_forge._campaign.provenance import load_manifest, manifest_path_for_log

    yield Path(config_path).expanduser().resolve(), False
    yield Path(log_path).expanduser().resolve(), False
    manifest_path = manifest_path_for_log(log_path)
    # Reserve the sidecar name even for legacy campaigns; exports must not adopt them.
    yield manifest_path, True
    manifest = load_manifest(log_path)
    if manifest is not None:
        for archive in manifest.get("archives", []):
            yield manifest_path.parent / archive["path"], False
