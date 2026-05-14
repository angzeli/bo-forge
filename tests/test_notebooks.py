from pathlib import Path

import nbformat
import pytest
from nbformat.validator import validate

NOTEBOOKS = sorted(Path("notebooks").glob("*.ipynb"))

assert NOTEBOOKS, "No notebooks found under notebooks/*.ipynb"


@pytest.mark.parametrize("notebook_path", NOTEBOOKS)
def test_notebook_metadata_is_valid(notebook_path: Path) -> None:
    notebook = nbformat.read(notebook_path, as_version=4)
    validate(notebook)
