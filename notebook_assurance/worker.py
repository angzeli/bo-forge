"""Execute unchanged cells and checkpoint evidence after each completed cell."""

from __future__ import annotations

import json
import shutil
import sys
import time
import traceback
from pathlib import Path

import nbformat
from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient
from nbclient.exceptions import CellTimeoutError

from notebook_assurance.archive import check_protected, digest, write_json
from notebook_assurance.contracts import validate


def check_cells(notebook):
    for index, cell in enumerate(notebook.cells):
        tags = cell.metadata.get("tags", [])
        if any("skip" in tag.lower() or tag in {"raises-exception", "allow-errors"}
               for tag in tags):
            raise ValueError(f"Execution skip/error-allowing tag at cell {index}: {tags}")
        if cell.cell_type == "code" and (cell.outputs or cell.execution_count is not None):
            raise ValueError(f"Packaged notebook is not output-free: cell {index}")


def capture_exports(runtime, captures):
    for prefix in ("bo-forge-predictive-", "bo-forge-benchmark-notebook-"):
        for source in (runtime / "tmp").glob(prefix + "*"):
            if source.is_dir():
                shutil.copytree(source, captures / source.name, dirs_exist_ok=True)


def bootstrap(root, output, *, setup=False):
    # Instrumentation is separate from the tutorial's unchanged code cells.
    threading = ""
    if setup:
        threading = (
            "_torch.set_num_threads(1)\n_torch.set_num_interop_threads(1)\n"
            "get_ipython().run_line_magic('matplotlib', 'inline')\n"
        )
    return nbformat.v4.new_code_cell(f"""
import importlib.metadata as _metadata, json as _json, platform as _platform, sys as _sys
from pathlib import Path as _Path
import torch as _torch
{threading}
assert _torch.get_num_threads() == _torch.get_num_interop_threads() == 1
import bo_forge as _bo_forge, benchmarks as _benchmarks
_root = _Path({str(root)!r}).resolve()
_locations = {{'bo_forge': _bo_forge.__file__, 'benchmarks': _benchmarks.__file__}}
assert all(_root in _Path(p).resolve().parents for p in _locations.values()), _locations
assert all(_root in _Path(m.__file__).resolve().parents
           for n, m in list(_sys.modules.items()) if n.startswith(('bo_forge.', 'benchmarks.'))
           and getattr(m, '__file__', None)), 'Import escaped extracted source'
_versions = {{name: _metadata.version(name) for name in
             ['botorch', 'torch', 'gpytorch', 'numpy', 'pandas', 'matplotlib',
              'nbclient', 'nbformat', 'ipykernel']}}
_Path({str(output)!r}).write_text(_json.dumps({{
    'python': _sys.version, 'executable': _sys.executable, 'platform': _platform.platform(),
    'torch_threads': _torch.get_num_threads(),
    'torch_interop_threads': _torch.get_num_interop_threads(),
    'bo_forge': _bo_forge.__version__, 'packages': _versions, 'imports': _locations}}, indent=2))
""")


def run_probe(client, cell):
    original = client.nb
    timeout = client.timeout
    client.timeout = 60  # Bounded kernel instrumentation; tutorial cells retain their own limit.
    client.nb = nbformat.v4.new_notebook(cells=[cell])
    try:
        client.execute_cell(cell, 0)
    finally:
        client.nb = original
        client.timeout = timeout


def verify_execution(notebook, completed, settings):
    expected = sum(c.cell_type == "code" and bool(c.source.strip()) for c in notebook.cells)
    if completed != expected:
        raise ValueError(f"Incomplete cell execution: {completed}/{expected}")
    rendered = sum("image/png" in item.get("data", {})
                   for cell in notebook.cells for item in cell.get("outputs", []))
    if rendered < settings.get("inline_figures", 0):
        raise ValueError("Missing rendered inline figures.")
    return rendered


def checkpoint_notebook(notebook, path):
    temporary = path.with_suffix(".ipynb.tmp")
    nbformat.write(notebook, temporary)
    temporary.replace(path)


def execute(settings):
    directory, root = Path(settings["directory"]), Path(settings["root"])
    runtime = directory / "runtime"
    notebook_path = root / "notebooks" / settings["notebook"]
    notebook = nbformat.read(notebook_path, as_version=4)
    state = {"notebook": settings["notebook"], "status": "running", "completed_cells": 0,
             "notebook_sha256": digest(notebook_path), "failing_cell_index": None,
             "failing_cell_id": None, "warnings": [], "completion_checks": {},
             "artifacts": {"notebook": "executed.ipynb", "log": "worker.log",
                           "workspace": "source", "captures": "captures"}}
    started = time.monotonic()
    identities = json.loads((runtime / "protected.json").read_text())

    def checkpoint():
        state["elapsed_seconds"] = time.monotonic() - started
        checkpoint_notebook(notebook, directory / "executed.ipynb")
        write_json(directory / "status.json", state)

    def cell_start(cell, cell_index):
        if cell.cell_type == "code" and cell.source.strip():
            state.update(failing_cell_index=cell_index, failing_cell_id=cell.id)
            checkpoint()

    def cell_done(cell, cell_index, execute_reply):
        if execute_reply["content"]["status"] == "ok":
            state["completed_cells"] += 1
        for item in cell.outputs:
            if item.output_type == "stream" and item.name == "stderr":
                state["warnings"].append({"cell_id": cell.id, "text": item.text})
        capture_exports(runtime, directory / "captures")
        check_protected(identities)
        checkpoint()

    checkpoint()
    try:
        check_cells(notebook)
        manager = AsyncKernelManager(kernel_name="assurance", kernel_spec_manager=KernelSpecManager(
            kernel_dirs=[str(runtime / "kernels")]))
        client = NotebookClient(notebook, km=manager, timeout=settings["cell_timeout"],
                                allow_errors=False, force_raise_errors=True,
                                shutdown_kernel="immediate",
                                resources={"metadata": {"path": str(root)}})
        with client.setup_kernel(cwd=str(root), cleanup_kc=True):
            run_probe(client, bootstrap(root, directory / "environment.json", setup=True))
            client.on_cell_start = cell_start
            client.on_cell_executed = cell_done
            for index, cell in enumerate(notebook.cells):
                client.execute_cell(cell, index)
            client.on_cell_start = None
            client.on_cell_executed = None
            run_probe(client, bootstrap(root, directory / "environment-final.json"))
        state["rendered_figures"] = verify_execution(notebook, state["completed_cells"], settings)
        check_protected(identities)
        state.update(failing_cell_index=None, failing_cell_id=None, phase="completion_checks")
        prefix = settings.get("capture_prefix")
        if prefix and list((runtime / "tmp").glob(prefix + "*")):
            raise ValueError("Notebook did not clean up its temporary workspace.")
        state["completion_checks"] = validate(root, directory / "captures", settings["notebook"])
        state.update(status="passed", failing_cell_index=None, failing_cell_id=None)
    except BaseException as exc:
        # Worker boundary: retain partial scientific evidence before propagating failure.
        state.update(status="timeout" if isinstance(exc, CellTimeoutError) else "failed",
                     exception_type=type(exc).__name__, message=str(exc),
                     traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        checkpoint()
    return 0 if state["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(execute(json.loads(Path(sys.argv[1]).read_text())))
