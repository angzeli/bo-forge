from pathlib import Path

import nbformat
import pytest
from nbformat.validator import validate

NOTEBOOKS = sorted(Path("notebooks").glob("*.ipynb"))
API_NOTEBOOKS = [
    notebook_path
    for notebook_path in NOTEBOOKS
    if notebook_path.name != "04_cli_four_variable_campaign.ipynb"
]
CLI_NOTEBOOK = Path("notebooks/04_cli_four_variable_campaign.ipynb")
REPLICATE_NOTEBOOK = Path("notebooks/08_replicate_aware_campaign.ipynb")

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


def notebook_source(notebook_path: Path) -> str:
    notebook = nbformat.read(notebook_path, as_version=4)
    return "\n".join(cell.source for cell in notebook.cells)


@pytest.mark.parametrize("notebook_path", NOTEBOOKS)
def test_notebook_defines_15_step_target(notebook_path: Path) -> None:
    source = notebook_source(notebook_path)
    if notebook_path == REPLICATE_NOTEBOOK:
        assert "TARGET_REPLICATE_GROUPS = 15" in source
    else:
        assert "TARGET_OBSERVED_ROWS = 15" in source


@pytest.mark.parametrize("notebook_path", API_NOTEBOOKS)
def test_api_notebooks_use_campaign_session(notebook_path: Path) -> None:
    source = notebook_source(notebook_path)

    assert "CampaignSession.from_files" in source


def test_cli_notebook_uses_package_module_invocation() -> None:
    source = notebook_source(CLI_NOTEBOOK)

    assert '"-m",' in source
    assert '"bo_forge"' in source


@pytest.mark.parametrize("notebook_path", NOTEBOOKS)
def test_notebooks_write_to_ignored_working_artifacts(notebook_path: Path) -> None:
    source = notebook_source(notebook_path)

    assert "_working_log.csv" in source
    assert "_latest_suggestions.csv" in source
    assert 'PROJECT_ROOT / "reports"' in source or 'Path("reports/' in source
