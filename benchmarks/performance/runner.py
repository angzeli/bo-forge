"""Sequential paired startup and suggestion measurements on one allocated host."""

import json
import shutil
import sys
import time
from pathlib import Path

from benchmarks.performance import HARNESS_VERSION, OVERALL_TIMEOUT, PROCESS_TIMEOUT, schedule
from benchmarks.performance.execution import execute, thread_settings
from benchmarks.performance.preparation import (
    frozen_constraints,
    machine,
    match_environments,
    prepare,
    probe,
)
from benchmarks.runner import _cancellation_handler
from benchmarks.storage import utc_now, write_json
from notebook_assurance.archive import digest


def run(baseline, candidate, output):
    from benchmarks.performance.workloads import materialize

    archives = {key: Path(value).resolve(strict=True)
                for key, value in (("baseline", baseline), ("candidate", candidate))}
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    deadline = start + OVERALL_TIMEOUT
    metadata = {"schema_version": 1, "harness_version": HARNESS_VERSION,
                "created_at": utc_now(), "machine": machine(), "threads": thread_settings(),
                "schedule": schedule(), "status": "preparing", "installations": {},
                "process_timeout_seconds": PROCESS_TIMEOUT,
                "overall_timeout_seconds": OVERALL_TIMEOUT}
    _retain_harness(root, metadata)
    _initialize_samples(root, metadata)
    print("13 cases, 156 processes (including 26 retained warm-ups); "
          "600 seconds per process; 90 minutes overall.", flush=True)
    try:
        with _cancellation_handler():
            constraints = root / "constraints.txt"
            frozen_constraints(constraints)
            metadata["constraints_sha256"] = digest(constraints)
            for version, archive in archives.items():
                metadata["installations"][version] = prepare(
                    archive, root / version, constraints, deadline)
                write_json(root / "run.json", metadata)
            match_environments(metadata["installations"])
            metadata["inputs"] = materialize(root / "inputs")
            metadata["preparation_seconds"] = time.monotonic() - start
            metadata["status"] = "running"
            write_json(root / "run.json", metadata)
            for index, item in enumerate(metadata["schedule"]):
                print(f"[{index + 1}/156] {item['sample_id']}", flush=True)
                _sample(root, item, metadata, deadline)
            _postflight(root, metadata, deadline)
            metadata["status"] = "complete"
    except BaseException as exc:
        # Evidence boundary: preserve failures and cancellation, never call them acceptance.
        metadata.update(status="failed" if isinstance(exc, Exception) else "interrupted",
                        message=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        metadata["elapsed_seconds"] = time.monotonic() - start
        metadata["finished_at"] = utc_now()
        _checkpoint(root / "run.json", metadata)
    return root


def _checkpoint(path, value):
    original = sys.exception()
    try:
        write_json(path, value)
    except BaseException as exc:
        # Evidence boundary: a persistence failure must not replace the interruption.
        if original is None:
            raise
        original.add_note(f"Evidence checkpoint failed: {type(exc).__name__}: {exc}")


def _retain_harness(root, metadata):
    snapshot = root / "harness"
    snapshot.mkdir()
    metadata["harness_files"] = {}
    for path in sorted(Path(__file__).parent.glob("*.py")):
        shutil.copyfile(path, snapshot / path.name)
        metadata["harness_files"][path.name] = digest(path)
    repository = Path(__file__).resolve().parents[2]
    metadata["shared_harness_files"] = {}
    shared = list((repository / "benchmarks").glob("*.py")) + [
        repository / "notebook_assurance/archive.py",
        repository / "notebook_assurance/processes.py"]
    for path in shared:
        name = path.relative_to(repository).as_posix()
        target = snapshot / "shared" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        metadata["shared_harness_files"][name] = digest(path)


def _initialize_samples(root, metadata):
    for item in metadata["schedule"]:
        directory = root / "samples" / item["sample_id"]
        directory.mkdir(parents=True)
        write_json(directory / "status.json", {**item, "status": "not_run",
                   "process_seconds": None, "call_seconds": None, "peak_rss_bytes": None})
    write_json(root / "run.json", metadata)


def _request(root, item, metadata):
    directory = root / "samples" / item["sample_id"]
    route = item.get("route", "qmf_kg")
    inputs = metadata["inputs"][route]
    request = dict(item, installation_root=metadata["installations"][item["version"]]["prefix"],
                   expected_source=inputs["expected_source"],
                   result_path=str(directory / "result.json"),
                   suggestions_path=str(directory / "suggestions.csv"))
    for key in ("config", "log"):
        source = root / "inputs" / inputs[f"{key}_path"]
        target = directory / source.name
        shutil.copyfile(source, target)
        request[f"{key}_path"] = str(target)
        request[f"{key}_sha256"] = digest(target)
    write_json(directory / "request.json", request)
    return request


def _command(python, item, request, directory):
    prefix = [python, "-I"]
    if item["kind"] == "import":
        return prefix + ["-c", "import bo_forge"]
    if item["kind"] in ("version", "help"):
        return prefix + ["-m", "bo_forge", f"--{item['kind']}"]
    if item["kind"] == "validate":
        return prefix + ["-m", "bo_forge", "validate", "--config", request["config_path"],
                         "--log", request["log_path"]]
    return prefix + [str(directory.parents[1] / "harness" / "worker.py"),
                     str(directory / "request.json")]


def _sample(root, item, metadata, deadline):
    directory = root / "samples" / item["sample_id"]
    request = _request(root, item, metadata)
    python = metadata["installations"][item["version"]]["executable"]
    status = json.loads((directory / "status.json").read_text())
    command = _command(python, item, request, directory)
    try:
        result = execute(command, directory, directory / "process.log",
                         min(PROCESS_TIMEOUT, deadline - time.monotonic()))
        status.update(result)
        if result["status"] == "complete":
            for key in ("config", "log"):
                if digest(Path(request[f"{key}_path"])) != request[f"{key}_sha256"]:
                    raise ValueError(f"{item['sample_id']}: {key} input changed")
            if item["kind"] == "suggest":
                status["worker"] = json.loads((directory / "result.json").read_text())
                status.update({key: status["worker"][key]
                               for key in ("call_seconds", "peak_rss_bytes")})
            status["artifact_hashes"] = {
                path.name: digest(path) for path in directory.iterdir()
                if path.is_file() and path.name != "status.json"}
    except BaseException as exc:
        # Evidence boundary: checkpoint the active sample before propagating cancellation.
        status.update(getattr(exc, "performance_result", {}))
        status.update(status="failed" if isinstance(exc, Exception) else "interrupted",
                      message=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        _checkpoint(directory / "status.json", status)


def _postflight(root, metadata, deadline):
    for version, identity in metadata["installations"].items():
        after = probe(identity["executable"], root / version, deadline, "postflight")
        for key in ("files", "packages", "python", "import_path"):
            if after[key] != identity[key]:
                raise ValueError(f"{version}: installed {key} changed during measurement")
    failed = [item["sample_id"] for item in metadata["schedule"]
              if json.loads((root / "samples" / item["sample_id"] / "status.json").read_text())[
                  "status"] != "complete"]
    if failed:
        raise ValueError(f"Incomplete measurements: {', '.join(failed)}")
