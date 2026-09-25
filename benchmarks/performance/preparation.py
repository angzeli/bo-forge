"""Prepare matched, non-editable core installs outside either source archive."""

import importlib.metadata as metadata
import json
import platform
import shutil
import sys
import time
import tomllib
from pathlib import Path

from benchmarks.performance.execution import execute
from benchmarks.storage import write_json
from notebook_assurance.archive import digest, extract


def checked(command, cwd, log, deadline):
    result = execute(command, cwd, log, min(600, deadline - time.monotonic()))
    if result["status"] != "complete":
        raise RuntimeError(f"Preparation failed: {command!r}; {result}; see {log}")


def frozen_constraints(path):
    versions = {dist.metadata["Name"].lower().replace("_", "-"): dist.version
                for dist in metadata.distributions()}
    versions.pop("bo-forge", None)
    path.write_text("".join(f"{name}=={version}\n" for name, version in sorted(versions.items())))
    return versions


def prepare(archive, directory, constraints, deadline):
    directory.mkdir()
    archive_copy = directory / "source.tar.gz"
    shutil.copyfile(archive, archive_copy)
    source = extract(archive_copy, directory / "extracted")
    project = tomllib.loads((source / "pyproject.toml").read_text())["project"]
    wheels = directory / "wheels"
    checked([sys.executable, "-m", "build", "--wheel", "--no-isolation", str(source),
             "--outdir", str(wheels)], directory, directory / "build.log", deadline)
    wheel, = wheels.glob("*.whl")
    venv = directory / "venv"
    checked([sys.executable, "-m", "venv", str(venv)], directory,
            directory / "venv.log", deadline)
    python = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    index = (["--extra-index-url", "https://download.pytorch.org/whl/cpu"]
             if "+cpu" in metadata.version("torch") else [])
    checked([str(python), "-I", "-m", "pip", "install", "--constraint", str(constraints),
             *index, str(wheel)], directory, directory / "install.log", deadline)
    checked([str(python), "-I", "-m", "pip", "check"], directory,
            directory / "pip-check.log", deadline)
    identity = probe(python, directory, deadline)
    if identity["version"] != project["version"]:
        raise ValueError("Installed package version differs from source archive.")
    identity.update(archive_sha256=digest(archive_copy), wheel_sha256=digest(wheel),
                    archive="source.tar.gz", wheel=str(wheel.relative_to(directory)),
                    requirements=project["dependencies"])
    write_json(directory / "identity.json", identity)
    return identity


def probe(python, directory, deadline, name="probe"):
    result_path = directory / f"{name}.json"
    script = Path(__file__).with_name("probe.py")
    checked([str(python), "-I", str(script), str(result_path)], directory,
            directory / f"{name}.log", deadline)
    return json.loads(result_path.read_text())


def match_environments(identities):
    left, right = (identities[key] for key in ("baseline", "candidate"))
    for field in ("python", "requirements"):
        if left[field] != right[field]:
            raise ValueError(f"Environment mismatch: {field}")
    dependencies = [{key: value for key, value in identity["packages"].items()
                     if key != "bo-forge"} for identity in (left, right)]
    if dependencies[0] != dependencies[1]:
        raise ValueError("Environment mismatch: installed dependency versions")


def machine():
    import psutil

    return {"os": platform.platform(), "cpu": platform.processor() or platform.machine(),
            "machine": platform.machine(), "logical_cpus": psutil.cpu_count(),
            "physical_cpus": psutil.cpu_count(logical=False), "python": platform.python_version(),
            "note": "Same host for this pair; runner labels do not identify fixed hardware."}
