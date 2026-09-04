"""Bounded environment snapshots for campaign provenance manifests."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_SCIENTIFIC_DISTRIBUTIONS = (
    "numpy",
    "pandas",
    "torch",
    "botorch",
    "gpytorch",
    "matplotlib",
    "PyYAML",
    "filelock",
)


def capture_environment(config_path: Path) -> dict[str, Any]:
    """Capture dependency and local source identity without importing dependencies."""
    details = {
        "bo_forge": _bo_forge_version(),
        "python": platform.python_version(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "dependencies": {
            name: _distribution_version(name) for name in _SCIENTIFIC_DISTRIBUTIONS
        },
        "git": _git_identity(config_path.parent),
    }
    encoded = json.dumps(
        details,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return {
        "environment_id": hashlib.sha256(encoded).hexdigest(),
        "captured_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        **details,
    }


def _git_identity(cwd: Path) -> dict[str, object]:
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain=v2", "--branch"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.splitlines()
    except (OSError, subprocess.SubprocessError):
        return {"commit": "unknown", "dirty": "unknown"}
    commit = next(
        (
            line.removeprefix("# branch.oid ")
            for line in status
            if line.startswith("# branch.oid ")
        ),
        "unknown",
    )
    if commit == "(initial)":
        commit = "unknown"
    dirty = any(not line.startswith("# ") for line in status)
    return {"commit": commit or "unknown", "dirty": dirty}


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _bo_forge_version() -> str:
    from bo_forge import __version__

    return __version__
