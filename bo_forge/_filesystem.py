"""Small filesystem primitives shared by artifact and campaign publication."""

from __future__ import annotations

import ctypes
import errno
import os
import sys
from pathlib import Path


class _DirectoryPublicationUnavailable(OSError):
    """Distinguish an unsupported OS from a filesystem rejecting the rename."""


def fsync_directory(directory: Path) -> None:
    """Best-effort flush of a directory entry change, such as a completed rename."""
    if os.name == "nt":
        return
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass  # Some filesystems reject directory fsync; the rename already succeeded.
    finally:
        os.close(descriptor)


def rename_directory_exclusive(source: Path, destination: Path) -> None:
    """Atomically publish a sibling directory without replacing any destination."""
    if sys.platform == "darwin":
        rename = _libc_function("renamex_np")
        result = rename(os.fsencode(source), os.fsencode(destination), 4)  # RENAME_EXCL
    elif sys.platform.startswith("linux"):
        rename = _libc_function("renameat2")
        # AT_FDCWD for both directory descriptors; flag 1 is RENAME_NOREPLACE.
        result = rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    else:
        raise _DirectoryPublicationUnavailable(
            errno.ENOTSUP,
            "Atomic no-overwrite directory publication requires macOS or Linux.",
        )
    if result:
        code = ctypes.get_errno()
        if code in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), str(destination))
        raise OSError(code, os.strerror(code), str(destination))


def _libc_function(name: str):
    """Return a libc symbol, or report that this C library cannot publish exclusively."""
    try:
        return getattr(ctypes.CDLL(None, use_errno=True), name)
    except (AttributeError, OSError) as exc:
        # musl and glibc before 2.28 lack renameat2; CDLL(None) itself can fail.
        raise _DirectoryPublicationUnavailable(
            errno.ENOTSUP,
            f"Atomic no-overwrite directory publication requires libc '{name}'.",
        ) from exc
