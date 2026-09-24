"""Fail-closed aggregation against the full profile and one archive identity."""

import json
import sys
from pathlib import Path

import nbformat

from notebook_assurance.archive import digest, extract, protected_inputs, write_json
from notebook_assurance.contracts import CONTRACTS, select, validate


def reject_constant(value):
    raise ValueError(f"Invalid JSON constant: {value}")


def verify_environment(directory, source):
    environment = json.loads((directory / "environment.json").read_text())
    final = json.loads((directory / "environment-final.json").read_text())
    if environment != final:
        raise ValueError("Kernel environment or imports changed during execution.")
    # Absolute import locations are historical execution evidence, not portable load paths.
    expected_suffix = f"/source/{source.name}/"
    imports = environment.get("imports", {})
    if (not environment.get("executable") or set(imports) != {"bo_forge", "benchmarks"}
            or any(expected_suffix not in location for location in imports.values())):
        raise ValueError("Invalid kernel import-location evidence.")


def verify_sources(original, retained):
    for filename, expected in protected_inputs(original).items():
        counterpart = retained / Path(filename).relative_to(original.resolve())
        if not counterpart.is_file() or counterpart.is_symlink() or digest(counterpart) != expected:
            raise ValueError(f"Retained packaged input differs from archive: {counterpart}")


def verify_passed(record, bundle, root):
    if record["status"] != "passed":
        return
    directory = (bundle / record["directory"]).resolve()
    if bundle.resolve() not in directory.parents:
        raise ValueError("Notebook evidence path escapes its bundle.")
    original = root / "notebooks" / record["notebook"]
    if record.get("notebook_sha256") != digest(original):
        raise ValueError(f"Notebook identity mismatch: {record['notebook']}")
    source = nbformat.read(original, as_version=4)
    executed = nbformat.read(directory / "executed.ipynb", as_version=4)
    if [(c.id, c.source, c.cell_type) for c in source.cells] != [
            (c.id, c.source, c.cell_type) for c in executed.cells]:
        raise ValueError("Executed notebook differs from the packaged cells.")
    cells = [c for c in executed.cells if c.cell_type == "code" and c.source.strip()]
    if (record.get("completed_cells") != len(cells) or not record.get("completion_checks")
            or any(c.execution_count is None or any(o.output_type == "error" for o in c.outputs)
                   for c in cells)):
        raise ValueError(f"Incomplete execution evidence: {record['notebook']}")
    rendered = sum("image/png" in o.get("data", {}) for cell in cells for o in cell.outputs)
    if (rendered != record.get("rendered_figures")
            or rendered < CONTRACTS.get(record["notebook"], {}).get("inline_figures", 0)):
        raise ValueError("Rendered-figure evidence is incomplete.")
    for name in ("environment.json", "environment-final.json", "status.json", "worker.log"):
        if not (directory / name).is_file():
            raise ValueError(f"Missing notebook evidence: {directory / name}")
    status = json.loads((directory / "status.json").read_text())
    if {k: v for k, v in record.items() if k != "directory"} != status:
        raise ValueError("Notebook status contradicts the run summary.")
    sources = list((directory / "source").iterdir())
    if len(sources) != 1:
        raise ValueError("Missing or ambiguous executed source workspace.")
    verify_sources(root, sources[0])
    verify_environment(directory, sources[0])
    checks = validate(sources[0], directory / "captures", record["notebook"])
    if checks != record["completion_checks"]:
        raise ValueError("Retained notebook outcome differs from its completion checks.")


def _read_bundle(path, expected):
    bundle = json.loads(path.read_text(), parse_constant=reject_constant)
    if not isinstance(bundle, dict):
        raise ValueError(f"Invalid run summary: {path}")
    if any(bundle.get(key) != expected[key] for key in ("archive_sha256", "profile", "scheduled")):
        raise ValueError(f"Mismatched archive or schedule: {path}")
    if not isinstance(bundle.get("notebooks"), list):
        raise ValueError(f"Invalid notebook records: {path}")
    for record in bundle["notebooks"]:
        if not isinstance(record, dict) or not isinstance(record.get("notebook"), str):
            raise ValueError(f"Invalid notebook record: {path}")
        if record.get("status") not in ("passed", "failed", "timeout", "interrupted", "not_run"):
            raise ValueError(f"Invalid notebook status: {record['notebook']}")
    if bundle.get("status") != "passed" and all(
            r["status"] == "passed" for r in bundle["notebooks"]):
        raise ValueError(f"Run status does not attest successful execution: {path}")
    return bundle["notebooks"]


def aggregate(sdist, profile, evidence, output):
    output = Path(output).absolute()
    output.mkdir(parents=True, exist_ok=False)
    result = {"schema_version": 1, "archive_sha256": None, "scheduled": [],
              "profile": profile, "notebooks": [], "status": "failed", "errors": []}
    records = {}
    path, name, record = Path(sdist).absolute(), None, None
    try:
        root = extract(sdist, output / "discovery")
        result["archive_sha256"] = digest(sdist)
        scheduled = result["scheduled"] = select(root, profile)
        if not scheduled:
            raise ValueError("Empty notebook schedule.")
        path = Path(evidence).absolute()
        if not path.is_dir():
            raise FileNotFoundError(f"Missing evidence directory: {path}")
        paths = sorted(path.rglob("result.json"))
        for path in paths:
            name, record = None, None
            for record in _read_bundle(path, result):
                name = record["notebook"]
                if name not in scheduled or name in records:
                    raise ValueError(f"Unexpected or duplicate notebook evidence: {name}")
                verify_passed(record, path.parent, root)
                records[name] = {**record, "evidence_path": str(
                    path.parent / record.get("directory", "."))}
    except BaseException as exc:
        # Aggregation boundary: persist diagnostics, including cancellation, before reraising.
        error = {"exception_type": type(exc).__name__, "message": str(exc),
                 "evidence_path": str(path), "notebook": name}
        result["errors"].append(error)
        if name in result["scheduled"]:
            records[name] = {"notebook": name, "status": "failed", "elapsed_seconds": None,
                             "evidence_path": str(path),
                             "failing_cell_index": record.get("failing_cell_index"),
                             "failing_cell_id": record.get("failing_cell_id")}
        raise
    finally:
        original = sys.exception()
        result["notebooks"] = [
            records.get(name, {"notebook": name, "status": "not_run", "elapsed_seconds": None})
            for name in result["scheduled"]]
        if (not result["errors"] and result["notebooks"]
                and all(r["status"] == "passed" for r in result["notebooks"])):
            result["status"] = "passed"
        try:
            write_json(output / "aggregate.json", result)
        except Exception as exc:
            # Persistence boundary: do not replace the failure that caused this report.
            if original is None:
                raise
            original.add_note(f"Aggregate report persistence failed: {type(exc).__name__}: {exc}")
    return result
