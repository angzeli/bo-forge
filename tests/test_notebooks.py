import math
import shutil
from pathlib import Path

import nbformat
import pytest
from nbformat.validator import validate

from bo_forge import CampaignSession

NOTEBOOKS = sorted(Path("notebooks").glob("*.ipynb"))
API_NOTEBOOKS = [
    notebook_path
    for notebook_path in NOTEBOOKS
    if notebook_path.name != "04_cli_four_variable_campaign.ipynb"
]
CLI_NOTEBOOK = Path("notebooks/04_cli_four_variable_campaign.ipynb")
REPLICATE_NOTEBOOK = Path("notebooks/08_replicate_aware_campaign.ipynb")
FOUR_OBJECTIVE_NOTEBOOK = Path("notebooks/11_four_objective_qlogehvi_campaign.ipynb")
MULTI_FIDELITY_NOTEBOOK = Path("notebooks/15_multi_fidelity_qmfkg_campaign.ipynb")

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
    elif notebook_path == FOUR_OBJECTIVE_NOTEBOOK:
        assert "TARGET_OBSERVED_ROWS = 50" in source
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


def test_multi_fidelity_notebook_uses_existing_qmfkg_assets() -> None:
    source = notebook_source(MULTI_FIDELITY_NOTEBOOK)

    assert "fidelity_summary()" in source
    assert "plot_fidelity_diagnostics" in source
    assert "configs\" / \"15_multi_fidelity_qmfkg.yaml" in source
    assert "examples\" / \"15_multi_fidelity_qmfkg_campaign_log.csv" in source
    assert "TARGET_OBSERVED_ROWS = 15" in source
    assert "CampaignSession.from_files" in source
    assert "CampaignSession.from_files(CONFIG_PATH, WORKING_LOG_PATH)" in source
    assert "WORKING_CONFIG_PATH" not in source
    assert "write_working_config" not in source
    assert "text.replace(\"random_seed: 15\"" not in source


def test_multi_fidelity_tutorial_loop_reaches_target_with_committed_config(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "15_multi_fidelity_qmfkg_working_log.csv"
    shutil.copyfile("examples/15_multi_fidelity_qmfkg_campaign_log.csv", log_path)
    campaign = CampaignSession.from_files(
        "configs/15_multi_fidelity_qmfkg.yaml",
        log_path,
    )

    def simulate_activity(row: object) -> float:
        loading = float(row["catalyst_loading"])
        fidelity = float(row["fidelity"])
        target_activity = 1.08 + 1.35 * loading - 2.0 * (loading - 0.38) ** 2
        fidelity_bias = -0.80 * (1.0 - fidelity)
        smooth_variation = 0.06 * math.sin(8.0 * loading + 3.0 * fidelity)
        return round(target_activity + fidelity_bias + smooth_variation, 6)

    while len(campaign.observed_data()) < 15:
        suggestions = campaign.suggest_next(batch_size=1)
        campaign.append_suggestions(suggestions)
        for row_id in suggestions["row_id"]:
            row = campaign.df.loc[campaign.df["row_id"] == row_id].iloc[0]
            campaign.mark_observed(
                row_id=row_id,
                objective_value=simulate_activity(row),
            )

    assert len(campaign.observed_data()) == 15
