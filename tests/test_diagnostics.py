from pathlib import Path

import matplotlib
import pandas as pd

from bo_forge.config import BOConfig, CampaignConfig, ObjectiveConfig, VariableConfig
from bo_forge.diagnostics import plot_diagnostics, plot_progress
from bo_forge.validation import canonical_columns

matplotlib.use("Agg")


def config() -> CampaignConfig:
    return CampaignConfig(
        campaign_name="diagnostics",
        objective=ObjectiveConfig(name="activity", direction="maximize"),
        variables=(
            VariableConfig("x", "continuous", 0.0, 1.0),
            VariableConfig("temperature", "continuous", 300.0, 800.0),
        ),
        bo=BOConfig(),
    )


def observed_log() -> pd.DataFrame:
    cfg = config()
    return pd.DataFrame(
        [
            {
                "row_id": "obs_0",
                "iteration": 0,
                "status": "observed",
                "source": "manual",
                "x": 0.2,
                "temperature": 500.0,
                "activity": 1.0,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            },
            {
                "row_id": "obs_1",
                "iteration": 1,
                "status": "observed",
                "source": "manual",
                "x": 0.7,
                "temperature": 650.0,
                "activity": 1.8,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            },
        ],
        columns=canonical_columns(cfg),
    )


def test_plot_progress_uses_report_ready_style_and_can_save(tmp_path: Path) -> None:
    output_path = tmp_path / "progress.png"

    fig, ax = plot_progress(config(), observed_log(), filename=output_path)

    assert output_path.exists()
    assert fig.get_facecolor() == (1.0, 1.0, 1.0, 1.0)
    assert ax.get_facecolor() == (1.0, 1.0, 1.0, 1.0)
    assert ax.spines["left"].get_linewidth() == 1.8
    assert ax.xaxis.label.get_fontweight() == "bold"
    assert ax.yaxis.label.get_color() == "black"


def test_plot_diagnostics_styles_colorbar_for_2d_observations() -> None:
    fig, ax = plot_diagnostics(config(), observed_log())

    assert fig.get_facecolor() == (1.0, 1.0, 1.0, 1.0)
    assert ax.spines["bottom"].get_linewidth() == 1.8
    assert len(fig.axes) == 2
    colorbar_ax = fig.axes[1]
    assert colorbar_ax.yaxis.label.get_text() == "activity"
    assert colorbar_ax.yaxis.label.get_fontweight() == "bold"

