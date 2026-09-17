"""Bounded run metadata and crash-resistant benchmark evidence writes."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tempfile import NamedTemporaryFile

TERMINAL = {"complete", "failed", "timeout", "interrupted"}


def utc_now():
    return datetime.now(UTC).isoformat()


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    """Replace only a tool-owned metadata file; never campaign source data."""
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    path = Path(path)
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                prefix=".benchmark-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def append_trace(path, row):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_trace(path):
    """Keep completed evidence before an interrupted final JSONL record."""
    if not Path(path).exists():
        return [], None
    lines = Path(path).read_bytes().splitlines(keepends=True)
    result = []
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            if index == len(lines) - 1 and not line.endswith(b"\n"):
                return result, "Interrupted final trace record; partial evidence retained."
            raise ValueError(f"Malformed evaluation trace at line {index + 1}: {path}") from None
        if not isinstance(row, dict):
            raise ValueError(f"Trace rows must be objects: {path}")
        result.append(row)
    return result, None


def environment():
    import bo_forge

    packages = {"bo-forge": bo_forge.__version__}
    for name in ("torch", "botorch", "gpytorch", "numpy", "pandas", "scipy", "pyyaml"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "unknown"
    return {"python": sys.version, "platform": platform.platform(), "packages": packages,
            "git": _git_identity(), "cpu_threads": 1}


def _git_identity():
    root = Path(__file__).resolve().parents[1]
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True,
                                  capture_output=True, timeout=3, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=normal"],
                               cwd=root, text=True, capture_output=True,
                               timeout=3, check=True).stdout
        return {"revision": revision, "dirty": bool(dirty)}
    except (OSError, subprocess.SubprocessError):
        return {"revision": "unknown", "dirty": "unknown"}
