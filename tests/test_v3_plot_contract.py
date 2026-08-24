"""Scientific plotting contract for the v3 visual reset."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import pytest
from matplotlib import pyplot as plt
from PIL import Image

import bo_forge.diagnostics as diagnostics
from bo_forge.plot_style import (
    AXIS_LABEL_SIZE,
    HIGHLIGHT_COLOR,
    OBSERVED_COLOR,
    SPINE_WIDTH,
)
from tests.test_diagnostics import config, observed_log


def test_every_public_diagnostic_plot_uses_scoped_style() -> None:
    for name in diagnostics.__all__:
        assert hasattr(getattr(diagnostics, name), "__wrapped__"), name


def test_plot_style_is_scoped_and_applied_to_late_artists() -> None:
    before = {key: mpl.rcParams[key] for key in ("lines.linewidth", "lines.markersize")}
    with mpl.rc_context({"lines.linewidth": 9.0, "lines.markersize": 13.0}):
        fig, ax = diagnostics.plot_progress(config(), observed_log())

        assert mpl.rcParams["lines.linewidth"] == 9.0
        assert mpl.rcParams["lines.markersize"] == 13.0
        assert [line.get_color() for line in ax.lines] == [OBSERVED_COLOR, HIGHLIGHT_COLOR]
        assert all(line.get_linewidth() == 2.0 for line in ax.lines)
        assert all(line.get_markersize() == 5.5 for line in ax.lines)

    assert {key: mpl.rcParams[key] for key in before} == before
    plt.close(fig)


def test_progress_plot_preserves_campaign_data_coordinates() -> None:
    fig, ax = diagnostics.plot_progress(config(), observed_log())

    assert [line.get_label() for line in ax.lines] == ["observed", "best so far"]
    assert [list(line.get_xdata()) for line in ax.lines] == [[1, 2], [1, 2]]
    assert [list(line.get_ydata()) for line in ax.lines] == [[1.0, 1.8], [1.0, 1.8]]
    plt.close(fig)


def test_plot_artist_and_png_export_contract(tmp_path: Path) -> None:
    output = tmp_path / "progress.png"
    fig, ax = diagnostics.plot_progress(config(), observed_log(), save_path=output)

    assert output.is_file()
    assert [path.name for path in tmp_path.iterdir()] == ["progress.png"]
    assert fig.get_facecolor() == (1.0, 1.0, 1.0, 1.0)
    assert ax.get_facecolor() == (1.0, 1.0, 1.0, 1.0)
    assert all(spine.get_visible() for spine in ax.spines.values())
    assert all(spine.get_linewidth() == SPINE_WIDTH for spine in ax.spines.values())
    assert not any(line.get_visible() for line in ax.get_xgridlines() + ax.get_ygridlines())
    assert ax.xaxis.label.get_fontsize() == AXIS_LABEL_SIZE
    assert ax.yaxis.label.get_fontsize() == AXIS_LABEL_SIZE
    assert ax.xaxis.label.get_fontweight() == "bold"
    assert ax.yaxis.label.get_fontweight() == "bold"
    assert ax.get_legend() is not None
    assert ax.get_legend().get_frame().get_alpha() == 1.0
    with Image.open(output) as image:
        dpi_x, dpi_y = image.info["dpi"]
    assert dpi_x == pytest.approx(600, rel=1e-3)
    assert dpi_y == pytest.approx(600, rel=1e-3)
    plt.close(fig)
