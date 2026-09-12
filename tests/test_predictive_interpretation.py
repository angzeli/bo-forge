"""Hand-worked interpretation examples, not empirical calibration fixtures."""

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from bo_forge.predictive import _aggregate_metrics, _prediction_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(params=[
    "docs/PREDICTIVE_EVALUATION.md", "notebooks/23_predictive_diagnostics.ipynb",
])
def interpretation_text(request):
    path = PROJECT_ROOT / request.param
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".ipynb":
        text = "\n".join("".join(cell["source"]) for cell in json.loads(text)["cells"]
                         if cell["cell_type"] == "markdown")
    return text


@pytest.mark.parametrize("path", [
    "docs/PREDICTIVE_EVALUATION.md", "notebooks/23_predictive_diagnostics.ipynb",
])
def test_interpretation_fixture_works_outside_checkout(tmp_path, monkeypatch, path):
    monkeypatch.chdir(tmp_path)
    text = interpretation_text.__wrapped__(SimpleNamespace(param=path))
    assert "19/20" in text


def test_hand_worked_uncertainty_tradeoff_matches_documented_table(interpretation_text):
    table = [line for line in interpretation_text.splitlines()
             if line.startswith(("| A |", "| B |"))]
    assert len(table) == 2
    rows = []
    for line, std in zip(table, (0.5, 2.0), strict=True):
        _, displayed_std, z, covered, width, nlpd = [part.strip() for part in line.split("|")[1:-1]]
        assert float(displayed_std) == std
        row = _prediction_metrics(12.0, 10.0, std**2)
        summary = _aggregate_metrics(pd.DataFrame([row]))
        assert summary["rmse"] == summary["mae"] == 2.0
        assert row["standardized_residual"] == float(z)
        assert row["interval_covered"] is (covered == "yes")
        assert summary["mean_interval_width"] == pytest.approx(float(width), abs=5e-5)
        expected_nlpd = 0.5 * math.log(2 * math.pi * std**2) + 2.0 / std**2
        assert row["negative_log_predictive_density"] == pytest.approx(expected_nlpd)
        assert expected_nlpd == pytest.approx(float(nlpd), abs=5e-5)
        rows.append(row)
    assert not rows[0]["interval_covered"] and rows[1]["interval_covered"]
    assert rows[0]["negative_log_predictive_density"] > rows[1]["negative_log_predictive_density"]


@pytest.mark.parametrize("factor", [0.1, 10.0])
def test_unit_rescaling_changes_density_score_not_standardized_error(factor):
    original = _prediction_metrics(12.0, 10.0, 4.0)
    scaled = _prediction_metrics(12.0 * factor, 10.0 * factor, 4.0 * factor**2)
    assert scaled["standardized_residual"] == pytest.approx(original["standardized_residual"])
    assert scaled["interval_covered"] is original["interval_covered"]
    assert scaled["negative_log_predictive_density"] == pytest.approx(
        original["negative_log_predictive_density"] + math.log(factor),
    )
    before = _aggregate_metrics(pd.DataFrame([original]))
    after = _aggregate_metrics(pd.DataFrame([scaled]))
    for name in ("rmse", "mae", "mean_interval_width"):
        assert after[name] == pytest.approx(factor * before[name])


def test_twenty_observations_have_five_percentage_point_coverage_steps(interpretation_text):
    covered = _prediction_metrics(12.0, 10.0, 4.0)
    missed = _prediction_metrics(12.0, 10.0, 0.25)
    coverage = [_aggregate_metrics(pd.DataFrame([covered] * n + [missed] * (20 - n)))[
        "interval_coverage"
    ] for n in (18, 19, 20)]
    assert coverage == pytest.approx([0.9, 0.95, 1.0])
    assert coverage[1] - coverage[0] == pytest.approx(0.05)
    assert "19/20" in interpretation_text
    assert "2.3026" in interpretation_text
    assert "independent-binomial" in interpretation_text
