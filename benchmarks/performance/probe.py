"""Standalone installed-environment probe, invoked with Python isolated mode."""

import hashlib
import importlib.metadata as metadata
import json
import platform
import sys
from pathlib import Path


def identity():
    import bo_forge

    package = Path(bo_forge.__file__).resolve()
    if not package.is_relative_to(Path(sys.prefix).resolve()):
        raise ValueError(f"BO Forge import escaped installation: {package}")
    distribution = metadata.distribution("bo-forge")
    files = {}
    for entry in distribution.files:
        name = str(entry)
        if name.endswith((".py", "METADATA", "RECORD")):
            files[name] = hashlib.sha256(distribution.locate_file(entry).read_bytes()).hexdigest()
    versions = {dist.metadata["Name"].lower().replace("_", "-"): dist.version
                for dist in metadata.distributions()}
    if bo_forge.__version__ != distribution.version:
        raise ValueError("Imported version differs from installed metadata.")
    return {"python": platform.python_version(), "executable": sys.executable,
            "prefix": sys.prefix, "import_path": str(package),
            "version": distribution.version, "packages": versions, "files": files}


if __name__ == "__main__":
    Path(sys.argv[1]).write_text(json.dumps(identity(), indent=2, sort_keys=True) + "\n")
