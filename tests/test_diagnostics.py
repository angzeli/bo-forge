from pathlib import Path

import matplotlib
import pandas as pd
import pytest
from matplotlib import pyplot as plt

from bo_forge.config import (
    BOConfig,
    CampaignConfig,
    FidelityConfig,
    ObjectiveConfig,
    ReplicateConfig,
    StageConfig,
    VariableConfig,
)
from bo_forge.diagnostics import (
    _directional_best_so_far,
    _normalised_variable_coverage,
    plot_diagnostics,
    plot_fidelity_diagnostics,
    plot_progress,
    plot_replicates,
    plot_stage_diagnostics,
)
from bo_forge.logs import load_campaign_log
from bo_forge.validation import canonical_columns

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def close_matplotlib_figures() -> None:
    yield
    plt.close("all")


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


def config_3d(direction: str = "maximize") -> CampaignConfig:
    return CampaignConfig(
        campaign_name="diagnostics_3d",
        objective=ObjectiveConfig(name="activity", direction=direction),
        variables=(
            VariableConfig("x", "continuous", 0.0, 1.0),
            VariableConfig("temperature", "continuous", 300.0, 800.0),
            VariableConfig("concentration", "continuous", 0.1, 2.0),
        ),
        bo=BOConfig(),
    )


def observed_log_3d(direction: str = "maximize") -> pd.DataFrame:
    cfg = config_3d(direction)
    return pd.DataFrame(
        [
            {
                "row_id": "obs_0",
                "iteration": 0,
                "status": "observed",
                "source": "manual",
                "x": 0.25,
                "temperature": 400.0,
                "concentration": 0.575,
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
                "x": 0.75,
                "temperature": 650.0,
                "concentration": 1.525,
                "activity": 0.5,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            },
            {
                "row_id": "obs_2",
                "iteration": 2,
                "status": "observed",
                "source": "manual",
                "x": 0.5,
                "temperature": 550.0,
                "concentration": 1.05,
                "activity": 2.0,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            },
        ],
        columns=canonical_columns(cfg),
    )


def replicate_config() -> CampaignConfig:
    cfg = config()
    return CampaignConfig(
        campaign_name="replicate_diagnostics",
        objective=cfg.objective,
        variables=(VariableConfig("x", "continuous", 0.0, 1.0),),
        bo=cfg.bo,
        replicates=ReplicateConfig(enabled=True),
    )


def fidelity_config() -> CampaignConfig:
    return CampaignConfig(
        campaign_name="fidelity_diagnostics",
        objective=ObjectiveConfig(name="activity", direction="maximize"),
        variables=(
            VariableConfig("x", "continuous", 0.0, 1.0),
            VariableConfig("fidelity", "continuous", 0.2, 1.0),
        ),
        bo=BOConfig(batch_size=1, initial_design_size=2, acquisition="qmf_kg"),
        fidelity=FidelityConfig(variable="fidelity", target=1.0),
    )


def fidelity_log() -> pd.DataFrame:
    cfg = fidelity_config()
    return pd.DataFrame(
        [
            {
                "row_id": "mf_0",
                "iteration": 0,
                "status": "observed",
                "source": "manual",
                "x": 0.2,
                "fidelity": 0.4,
                "activity": 0.8,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            },
            {
                "row_id": "mf_1",
                "iteration": 1,
                "status": "observed",
                "source": "manual",
                "x": 0.7,
                "fidelity": 1.0,
                "activity": 1.4,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            },
        ],
        columns=canonical_columns(cfg),
    )


def replicate_log() -> pd.DataFrame:
    cfg = replicate_config()
    return pd.DataFrame(
        [
            {
                "row_id": "rep_0a",
                "iteration": 0,
                "status": "observed",
                "source": "manual",
                "replicate_group": "group_0",
                "replicate_index": 0,
                "x": 0.2,
                "activity": 1.0,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            },
            {
                "row_id": "rep_0b",
                "iteration": 0,
                "status": "observed",
                "source": "manual",
                "replicate_group": "group_0",
                "replicate_index": 1,
                "x": 0.2,
                "activity": 1.4,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            },
            {
                "row_id": "rep_1a",
                "iteration": 1,
                "status": "observed",
                "source": "manual",
                "replicate_group": "group_1",
                "replicate_index": 0,
                "x": 0.7,
                "activity": 1.3,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            },
        ],
        columns=canonical_columns(cfg),
    )


def structured_config() -> CampaignConfig:
    cfg = config()
    return CampaignConfig(
        campaign_name="structured_diagnostics",
        objective=cfg.objective,
        variables=cfg.variables,
        bo=cfg.bo,
        stages=(
            StageConfig("screen", ("x",)),
            StageConfig("refine", ("x", "temperature")),
        ),
    )


def structured_log() -> pd.DataFrame:
    cfg = structured_config()
    return pd.DataFrame(
        [
            {
                "row_id": "screen_0",
                "iteration": 0,
                "status": "observed",
                "source": "manual",
                "stage": "screen",
                "x": 0.3,
                "temperature": "",
                "activity": 1.0,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            },
            {
                "row_id": "refine_0",
                "iteration": 1,
                "status": "suggested",
                "source": "manual",
                "stage": "refine",
                "x": 0.5,
                "temperature": 650.0,
                "activity": "",
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


def test_plot_progress_save_path_writes_exact_nested_path(tmp_path: Path) -> None:
    save_path = tmp_path / "reports" / "progress.png"

    fig, ax = plot_progress(config(), observed_log(), save_path=save_path)

    assert save_path.exists()
    assert fig is ax.figure


def test_plot_progress_rejects_filename_and_save_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="filename or save_path"):
        plot_progress(
            config(),
            observed_log(),
            filename="progress.png",
            save_path=tmp_path / "progress.png",
        )


def test_plot_diagnostics_styles_colorbar_for_2d_observations() -> None:
    fig, ax = plot_diagnostics(config(), observed_log())

    assert fig.get_facecolor() == (1.0, 1.0, 1.0, 1.0)
    assert ax.spines["bottom"].get_linewidth() == 1.8
    assert len(fig.axes) == 2
    colorbar_ax = fig.axes[1]
    assert colorbar_ax.yaxis.label.get_text() == "activity"
    assert colorbar_ax.yaxis.label.get_fontweight() == "bold"


def test_plot_diagnostics_3d_uses_high_dimensional_heatmap_without_mutating_df() -> None:
    cfg = config_3d()
    df = observed_log_3d()
    before = df.copy(deep=True)

    fig, axes = plot_diagnostics(cfg, df)

    pd.testing.assert_frame_equal(df, before)
    assert len(axes) == 2
    assert len(fig.axes) == 3
    assert axes[0].get_title() == "Objective progress"
    assert axes[1].get_title() == "Variable coverage"
    assert axes[1].get_position().x0 > axes[0].get_position().x0
    assert len(axes[1].images) == 1
    assert [label.get_text() for label in axes[1].get_xticklabels()] == [
        "x",
        "temperature",
        "concentration",
    ]
    assert fig.axes[-1].yaxis.label.get_text() == "Normalised value"


def test_plot_diagnostics_3d_text_stays_inside_figure() -> None:
    cfg = CampaignConfig.from_yaml("configs/03_simple_3d_maximise_logei.yaml")
    df = load_campaign_log("examples/03_simple_3d_maximise_logei_campaign_log.csv", cfg)

    fig, axes = plot_diagnostics(cfg, df)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    figure_bbox = fig.bbox
    margin = 2

    text_artists = [
        axes[0].title,
        axes[0].xaxis.label,
        axes[0].yaxis.label,
        axes[1].title,
        axes[1].xaxis.label,
        axes[1].yaxis.label,
        fig.axes[-1].yaxis.label,
        *axes[0].get_xticklabels(),
        *axes[0].get_yticklabels(),
        *axes[1].get_xticklabels(),
        *axes[1].get_yticklabels(),
        *fig.axes[-1].get_yticklabels(),
    ]
    for text in text_artists:
        if not text.get_text():
            continue
        bbox = text.get_window_extent(renderer)
        assert bbox.x0 >= figure_bbox.x0 - margin
        assert bbox.y0 >= figure_bbox.y0 - margin
        assert bbox.x1 <= figure_bbox.x1 + margin
        assert bbox.y1 <= figure_bbox.y1 + margin

    title_0 = axes[0].title.get_window_extent(renderer)
    title_1 = axes[1].title.get_window_extent(renderer)
    assert not title_0.overlaps(title_1)

    x_tick_bboxes = [
        label.get_window_extent(renderer)
        for label in axes[1].get_xticklabels()
        if label.get_text()
    ]
    for left, right in zip(x_tick_bboxes, x_tick_bboxes[1:], strict=False):
        assert not left.overlaps(right)


def test_plot_diagnostics_save_path_writes_exact_nested_path(tmp_path: Path) -> None:
    save_path = tmp_path / "reports" / "diagnostics.png"

    fig, axes = plot_diagnostics(config_3d(), observed_log_3d(), save_path=save_path)

    assert save_path.exists()
    assert len(axes) == 2
    assert fig.axes[0] is axes[0]


def test_plot_diagnostics_rejects_filename_and_save_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="filename or save_path"):
        plot_diagnostics(
            config(),
            observed_log(),
            filename="diagnostics.png",
            save_path=tmp_path / "diagnostics.png",
        )


def test_plot_replicates_writes_nested_output_without_mutating_df(tmp_path: Path) -> None:
    cfg = replicate_config()
    df = replicate_log()
    before = df.copy(deep=True)
    save_path = tmp_path / "reports" / "replicates.png"

    fig, ax = plot_replicates(cfg, df, save_path=save_path)

    pd.testing.assert_frame_equal(df, before)
    assert save_path.exists()
    assert fig is ax.figure
    assert ax.get_title() == "replicate_diagnostics: replicate summary"
    assert ax.get_xlabel() == "Replicate group index"
    assert [label.get_text() for label in ax.get_xticklabels()] == ["1", "2"]


def test_plot_stage_diagnostics_writes_stage_figure(tmp_path: Path) -> None:
    cfg = structured_config()
    df = structured_log()
    before = df.copy(deep=True)
    save_path = tmp_path / "reports" / "stage_diagnostics.png"

    fig, axes = plot_stage_diagnostics(cfg, df, save_path=save_path)

    assert save_path.exists()
    assert hasattr(fig, "savefig")
    assert len(axes) == 2
    assert axes[0].get_title() == "Rows by stage"
    assert axes[1].get_title() == "Active variable map"
    assert axes[1].images[0].get_array().tolist() == [[1.0, 0.0], [1.0, 1.0]]
    pd.testing.assert_frame_equal(df, before)


def test_plot_fidelity_diagnostics_writes_figure_and_labels(tmp_path: Path) -> None:
    cfg = fidelity_config()
    save_path = tmp_path / "reports" / "fidelity.png"

    fig, axes = plot_fidelity_diagnostics(cfg, fidelity_log(), save_path=save_path)

    assert save_path.exists()
    assert axes[0].get_xlabel() == "fidelity"
    assert axes[0].get_ylabel() == "activity"
    assert axes[1].get_xlabel() == "fidelity"
    assert axes[1].get_ylabel() == "Observed rows"
    assert "target fidelity = 1" in axes[0].get_legend_handles_labels()[1]
    plt.close(fig)


def test_plot_fidelity_diagnostics_handles_empty_observed_log() -> None:
    cfg = fidelity_config()

    fig, axes = plot_fidelity_diagnostics(cfg, pd.DataFrame(columns=canonical_columns(cfg)))

    assert "No observed fidelity data yet." in axes[0].texts[0].get_text()
    assert axes[0].get_xlabel() == "fidelity"
    assert axes[1].get_ylabel() == "Observed rows"
    plt.close(fig)


def test_plot_fidelity_diagnostics_rejects_non_fidelity_config() -> None:
    with pytest.raises(ValueError, match="requires a config with fidelity"):
        plot_fidelity_diagnostics(config(), observed_log())


def test_directional_best_so_far_uses_campaign_direction() -> None:
    values = pd.Series([1.0, 0.5, 2.0])

    assert _directional_best_so_far(config_3d("maximize"), values).tolist() == [1.0, 1.0, 2.0]
    assert _directional_best_so_far(config_3d("minimize"), values).tolist() == [1.0, 0.5, 0.5]


def test_normalised_variable_coverage_uses_config_bounds_not_observed_minmax() -> None:
    coverage = _normalised_variable_coverage(config_3d(), observed_log_3d())

    assert coverage["x"].tolist() == pytest.approx([0.25, 0.75, 0.5])
    assert coverage["temperature"].tolist() == pytest.approx([0.2, 0.7, 0.5])
    assert coverage["concentration"].tolist() == pytest.approx([0.25, 0.75, 0.5])


def test_normalised_variable_coverage_clips_for_plotting() -> None:
    cfg = config_3d()
    df = observed_log_3d()
    df.loc[0, "x"] = -0.5
    df.loc[1, "temperature"] = 900.0

    coverage = _normalised_variable_coverage(cfg, df)

    assert coverage.loc[0, "x"] == 0.0
    assert coverage.loc[1, "temperature"] == 1.0
