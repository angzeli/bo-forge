import math
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
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
CONTEXTUAL_NOTEBOOK = Path("notebooks/16_contextual_logei_campaign.ipynb")
MODEL_PROFILE_NOTEBOOK = Path("notebooks/17_model_profile_logei_campaign.ipynb")
QLOG_NEI_NOTEBOOK = Path("notebooks/18_noisy_pending_qlognei_campaign.ipynb")

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


def test_contextual_notebook_uses_existing_logei_assets() -> None:
    source = notebook_source(CONTEXTUAL_NOTEBOOK)

    assert "context_summary()" in source
    assert "plot_context_diagnostics" in source
    assert "configs\" / \"16_contextual_logei.yaml" in source
    assert "examples\" / \"16_contextual_logei_campaign_log.csv" in source
    assert "TARGET_OBSERVED_ROWS = 15" in source
    assert "CampaignSession.from_files" in source
    assert "CampaignSession.from_files(CONFIG_PATH, WORKING_LOG_PATH)" in source


def test_contextual_tutorial_loop_reaches_target_with_committed_config(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "16_contextual_logei_working_log.csv"
    shutil.copyfile("examples/16_contextual_logei_campaign_log.csv", log_path)
    campaign = CampaignSession.from_files(
        "configs/16_contextual_logei.yaml",
        log_path,
    )

    def simulate_yield(row: object) -> float:
        loading = float(row["catalyst_loading"])
        temperature = float(row["reaction_temperature"])
        acidity = float(row["feedstock_acidity"])
        solvent_bonus = {"MeCN": 0.08, "EtOH": 0.03, "Water": -0.04}[str(row["solvent"])]
        loading_term = 0.95 + 0.75 * loading - 1.10 * (loading - 0.55) ** 2
        temperature_term = -0.00009 * (temperature - 92.0) ** 2
        context_term = 0.22 * acidity - 0.35 * (acidity - 0.55) ** 2
        smooth_variation = 0.025 * math.sin(
            7.0 * loading + 0.04 * temperature + 2.0 * acidity
        )
        return round(
            loading_term + temperature_term + context_term + solvent_bonus + smooth_variation,
            6,
        )

    context_cycle = [0.25, 0.5, 0.75]
    generated = 0
    while len(campaign.observed_data()) < 15:
        suggestions = campaign.suggest_next(
            context_values={"feedstock_acidity": context_cycle[generated % 3]}
        )
        campaign.append_suggestions(suggestions)
        for row_id in suggestions["row_id"]:
            row = campaign.df.loc[campaign.df["row_id"] == row_id].iloc[0]
            campaign.mark_observed(row_id=row_id, objective_value=simulate_yield(row))
            generated += 1

    assert len(campaign.observed_data()) == 15


def test_model_profile_notebook_uses_existing_logei_assets() -> None:
    source = notebook_source(MODEL_PROFILE_NOTEBOOK)

    assert "model_summary()" in source
    assert "plot_model_diagnostics" in source
    assert "configs\" / \"17_model_profile_logei.yaml" in source
    assert "examples\" / \"17_model_profile_campaign_log.csv" in source
    assert "TARGET_OBSERVED_ROWS = 15" in source
    assert "CampaignSession.from_files" in source
    assert "CampaignSession.from_files(CONFIG_PATH, WORKING_LOG_PATH)" in source


def test_qlog_nei_notebook_uses_existing_pending_aware_assets() -> None:
    source = notebook_source(QLOG_NEI_NOTEBOOK)

    assert "qlog_nei_summary()" in source
    assert "plot_qlog_nei_diagnostics" in source
    assert "configs\" / \"18_noisy_pending_qlognei.yaml" in source
    assert "examples\" / \"18_noisy_pending_qlognei_campaign_log.csv" in source
    assert "TARGET_OBSERVED_ROWS = 15" in source
    assert "CampaignSession.from_files" in source
    assert "CampaignSession.from_files(CONFIG_PATH, WORKING_LOG_PATH)" in source
    assert "X_pending" in source


def test_model_profile_tutorial_workflow_smoke(tmp_path: Path) -> None:
    log_path = tmp_path / "17_model_profile_logei_working_log.csv"
    latest_path = tmp_path / "17_model_profile_logei_latest_suggestions.csv"
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    shutil.copyfile("examples/17_model_profile_campaign_log.csv", log_path)
    campaign = CampaignSession.from_files(
        "configs/17_model_profile_logei.yaml",
        log_path,
    )

    def simulate_activity(row: object) -> float:
        loading = float(row["catalyst_loading"])
        temperature = float(row["reaction_temperature"])
        loading_term = 0.62 + 0.85 * loading - 0.92 * (loading - 0.68) ** 2
        temperature_term = -0.00008 * (temperature - 108.0) ** 2
        smooth_variation = 0.035 * math.sin(9.0 * loading + 0.035 * temperature)
        return round(loading_term + temperature_term + smooth_variation, 6)

    campaign.validate()
    model_summary = campaign.model_summary()
    model_values = dict(
        zip(model_summary["field"], model_summary["value"], strict=True)
    )
    assert model_values["model_profile"] == "smooth"
    before = log_path.read_bytes()
    suggestions = campaign.suggest_next(batch_size=1)
    assert log_path.read_bytes() == before
    suggestions.to_csv(latest_path, index=False)
    campaign.append_suggestions(suggestions)
    for row_id in suggestions["row_id"]:
        row = campaign.df.loc[campaign.df["row_id"] == row_id].iloc[0]
        campaign.mark_observed(row_id=row_id, objective_value=simulate_activity(row))

    campaign = CampaignSession.from_files("configs/17_model_profile_logei.yaml", log_path)
    campaign.plot_model_diagnostics(save_path=report_dir / "17_model_profile_diagnostics.png")
    campaign.export_report(report_dir / "17_model_profile_report.md")

    assert latest_path.exists()
    assert (report_dir / "17_model_profile_diagnostics.png").exists()
    assert (report_dir / "17_model_profile_report.md").exists()
    plt.close("all")
