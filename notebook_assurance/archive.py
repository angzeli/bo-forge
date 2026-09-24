"""Strict source extraction and identity checks for trusted tutorial execution."""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def extract(sdist, destination):
    """Reject links, traversal, duplicate members, and non-regular archive content."""
    destination = Path(destination)
    with tarfile.open(sdist, "r:gz") as archive:
        members = archive.getmembers()
        seen = set()
        roots = set()
        for member in members:
            path = PurePosixPath(member.name)
            if (path.is_absolute() or ".." in path.parts or not path.parts
                    or "\\" in member.name or not (member.isfile() or member.isdir())
                    or path.as_posix() in seen):
                raise ValueError(f"Unsafe or duplicate archive member: {member.name}")
            seen.add(path.as_posix())
            roots.add(path.parts[0])
        if len(roots) != 1:
            raise ValueError("Expected one source-distribution root.")
        destination.mkdir(parents=True, exist_ok=False)
        for member in members:
            target = destination / member.name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open("xb") as output:
                    output.write(source.read())
    root = destination / roots.pop()
    if not (root / "pyproject.toml").is_file():
        raise ValueError("Source distribution has no pyproject.toml.")
    return root


def protected_inputs(root):
    return {
        str(path.resolve()): digest(path)
        for pattern in ("notebooks/*.ipynb", "configs/*.yaml", "examples/*_campaign_log.csv",
                        "benchmarks/specs/*.yaml")
        for path in sorted(Path(root).glob(pattern))
    }


def check_protected(identities):
    for filename, expected in identities.items():
        path = Path(filename)
        if not path.is_file() or path.is_symlink() or digest(path) != expected:
            raise ValueError(f"Packaged input changed: {filename}")
