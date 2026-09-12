"""Small filesystem primitives shared by artifact and campaign publication."""

from __future__ import annotations

import ctypes
import errno
import os
import sys
from pathlib import Path


class _DirectoryPublicationUnavailable(OSError):
    """Distinguish an unsupported OS from a filesystem rejecting the rename."""


def rename_directory_exclusive(source: Path, destination: Path) -> None:
    """Atomically publish a sibling directory without replacing any destination."""
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        rename = libc.renamex_np
        result = rename(os.fsencode(source), os.fsencode(destination), 4)  # RENAME_EXCL
    elif sys.platform.startswith("linux"):
        rename = libc.renameat2
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
