"""Keep campaign-owned files separate from user-selected artifact destinations."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextvars import ContextVar
from pathlib import Path
from typing import Any

_PLOT_SOURCES: ContextVar[tuple[Path, Path] | None] = ContextVar("plot_sources", default=None)


def _validate_export_destination(
    path: str | Path, config_path: str | Path, log_path: str | Path,
) -> None:
    """Reject source aliases and reserved parents; match the writer's home expansion."""
    try:
        destination = Path(path).resolve()
    except RuntimeError as exc:
        raise OSError(f"Could not resolve export destination '{path}': {exc}") from exc
    for source, reserved_name in _campaign_source_paths(config_path, log_path):
        if any(_source_alias(part, source, reserved_name)
               for part in (destination, *destination.parents)):
            raise OSError(
                f"Export destination '{path}' conflicts with campaign source '{source}'. "
                "Choose a separate artifact output path; campaign files were not changed."
            )


def _source_alias(destination: Path, source: Path, reserved_name: bool) -> bool:
    if destination == source.resolve() or (
        destination.exists() and source.exists() and destination.samefile(source)
    ):
        return True
    # Reserve sidecar case variants portably, including before the file exists.
    return reserved_name and destination.name.casefold() == source.name.casefold() and (
        destination.parent == source.parent or (
            destination.parent.exists() and source.parent.exists()
            and destination.parent.samefile(source.parent)
        )
    )


def _guard_campaign_plot(
    plotter: Callable, sources: tuple[Path, Path] | None, *args: Any, **kwargs: Any,
) -> Any:
    """Check before rendering and scope the same guard to the final figure write."""
    if sources is None:
        return plotter(*args, **kwargs)
    token = _PLOT_SOURCES.set(sources)
    try:
        path, filename = kwargs.get("save_path"), kwargs.get("filename")
        if path is not None and filename is not None:
            raise ValueError("Pass either filename or save_path, not both.")
        if filename is not None:
            path = Path(kwargs.get("fig_folder", "figures")) / filename
        if path is not None:
            from matplotlib import rcParams

            _validate_plot_export_path(path, rcParams["savefig.format"])
        return plotter(*args, **kwargs)
    finally:
        _PLOT_SOURCES.reset(token)


def _validate_plot_export_path(path: str | Path, default_format: str) -> None:
    sources = _PLOT_SOURCES.get()
    if sources is not None:
        _validate_export_destination(path, *sources)
        if not Path(path).suffix:
            # Matplotlib appends its default format to extensionless filenames.
            _validate_export_destination(str(path).rstrip(".") + "." + default_format, *sources)


def _session_plot(session: Any, plotter: Callable, kwargs: dict[str, Any]) -> Any:
    session._assert_provenance_resumable()
    return _guard_campaign_plot(
        plotter, (session.config_path, session.log_path), session.config, session.df, **kwargs,
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
