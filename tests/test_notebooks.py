from pathlib import Path

import nbformat
import pytest
from nbformat.validator import validate

NOTEBOOKS = sorted(Path("notebooks").glob("*.ipynb"))

assert NOTEBOOKS, "No notebooks found under notebooks/*.ipynb"


@pytest.mark.parametrize("notebook_path", NOTEBOOKS)
def test_notebook_metadata_is_valid(notebook_path: Path) -> None:
    notebook = nbformat.read(notebook_path, as_version=4)
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        assert not cell.outputs, (
            f"{notebook_path} code cell {index} has committed outputs. "
            "Clear notebook outputs before committing."
        )
        assert cell.execution_count is None, (
            f"{notebook_path} code cell {index} has execution_count="
            f"{cell.execution_count!r}. Reset execution counts before committing."
        )
    validate(notebook)
