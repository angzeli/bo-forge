"""Complete and incomplete evaluations retain the same evidence across adapters."""

import json
import sys

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import bo_forge.models as models
from bo_forge import CampaignSession, cli
from bo_forge._campaign.provenance import manifest_path_for_log
from bo_forge.application import CampaignAppService
from bo_forge.predictive import PredictiveEvaluationResult
from bo_forge_app.streamlit_helpers import CONFIG_PATH_KEY, LOG_PATH_KEY
from bo_forge_app.views.predictive_evaluation import EVALUATION_CACHE_KEY
from tests.test_predictive_evaluation import controlled_fit
from tests.test_predictive_hardening import METRICS, _file_campaign
from tests.test_streamlit_predictive_evaluation import _click, _evaluation_app


def _assert_bundle(path, result):
    assert {item.name for item in path.iterdir()} == {
        "summary.csv", "predictions.csv", "fold_outcomes.csv", "metadata.json",
    }
    for name in ("summary", "predictions", "fold_outcomes"):
        pd.testing.assert_frame_equal(
            pd.read_csv(path / f"{name}.csv").fillna(""),
            getattr(result, name).fillna(""), check_dtype=False,
        )
    assert json.loads((path / "metadata.json").read_text()) == result.metadata


def _assert_evidence(result, incomplete):
    assert result.summary.model_profile.tolist() == ["default", "smooth"]
    assert result.summary.fit_status.tolist() == [
        "incomplete" if incomplete else "complete", "complete",
    ]
    assert result.summary.loc[1, METRICS].notna().all()
    if incomplete:
        assert result.summary.loc[0, METRICS].isna().all()
        assert result.summary.loc[0, "completed_folds"] == 2
        failed = result.fold_outcomes.loc[result.fold_outcomes.fit_status.eq("failed")]
        assert len(failed) == 1
        assert failed.fit_message.iloc[0] == "known acceptance fold failure"
        rows = result.predictions.loc[result.predictions.fit_status.eq("failed")]
        assert len(rows) == failed.held_out_rows.iloc[0]
        assert rows.fit_message.eq("known acceptance fold failure").all()
        assert rows.predicted_mean.isna().all()
        assert result.predictions.fit_status.eq("complete").any()


@pytest.mark.parametrize("incomplete", [False, True])
def test_evaluation_evidence_survives_session_service_cli_and_streamlit(
    tmp_path, monkeypatch, capsys, incomplete,
):
    # AppTest replaces __main__; subsequent spawn-based tests must retain pytest's module.
    monkeypatch.setitem(sys.modules, "__main__", sys.modules["__main__"])
    config, log = _file_campaign(tmp_path, True)
    sources = (config, log, manifest_path_for_log(log))
    before = {path: path.read_bytes() for path in sources}
    session = CampaignSession.from_files(config, log, provenance_policy="required")
    service = CampaignAppService.from_session(session)
    metadata_before = session.model_summary().copy(deep=True)
    calls = []
    good = controlled_fit([])

    def fit(cfg, training):
        calls.append((cfg.model.profile, training.row_id.tolist()))
        if incomplete and cfg.model.profile == "default" and "row_000" not in set(training.row_id):
            raise RuntimeError("known acceptance fold failure")
        return good(cfg, training)

    monkeypatch.setattr(models, "fit_gp_model", fit)
    options = dict(profiles=["default", "smooth"], folds=3, seed=7)
    result = session.model_predictive_evaluation(**options)
    _assert_evidence(result, incomplete)
    equivalent = service.model_predictive_evaluation(**options)
    for name in ("summary", "predictions", "fold_outcomes"):
        pd.testing.assert_frame_equal(getattr(result, name), getattr(equivalent, name))
    assert result.metadata == equivalent.metadata
    assert len(calls) == 12
    _assert_bundle(result.export(tmp_path / "python-export"), result)
    cli_output = tmp_path / "cli-export"
    assert cli.run([
        "model-evaluate", "--config", str(config), "--log", str(log), "--require-provenance",
        "--profile", "default", "--profile", "smooth", "--folds", "3", "--seed", "7",
        "--output-dir", str(cli_output),
    ]) == int(incomplete)
    assert ("incomplete" in capsys.readouterr().out) == incomplete
    _assert_bundle(cli_output, result)
    assert len(calls) == 18
    app = AppTest.from_function(_evaluation_app)
    app.session_state["test_campaign"] = service
    app.session_state[CONFIG_PATH_KEY] = str(config)
    app.session_state[LOG_PATH_KEY] = str(log)
    app.run(timeout=10)
    assert not app.exception and not app.dataframe and len(calls) == 18
    app.multiselect(key="evaluation_profiles").set_value(options["profiles"])
    app.number_input(key="evaluation_folds").set_value(3)
    app.number_input(key="evaluation_seed").set_value(7)
    _click(app, "Run predictive evaluation")
    assert len(calls) == 24
    cached = app.session_state[EVALUATION_CACHE_KEY]
    _assert_evidence(cached["result"], incomplete)
    for name, table in zip(("summary", "predictions", "fold_outcomes"), app.dataframe, strict=True):
        pd.testing.assert_frame_equal(getattr(result, name), table.value)
    assert any(item.label == "Interpret these results" for item in app.expander)
    assert any("Incomplete predictive evaluation" in warning.value for warning in app.warning) == (
        incomplete
    )
    app.checkbox[0].check().run(timeout=10)
    assert app.session_state[EVALUATION_CACHE_KEY] is cached
    monkeypatch.setattr(models, "fit_gp_model", lambda *_: pytest.fail("must not refit"))
    figures = []

    def inspect_plot(method):
        original = getattr(PredictiveEvaluationResult, method)

        def plot(owner, **kwargs):
            figure, axes = original(owner, **kwargs)
            figures.append(figure)
            text = " ".join(item.get_text() for item in axes[0].texts)
            assert ("aggregate metrics withheld" in text) == incomplete
            assert ("95% interval coverage:" in text) != incomplete
            return figure, axes

        monkeypatch.setattr(PredictiveEvaluationResult, method, plot)

    for method in ("plot_predictions", "plot_residuals"):
        inspect_plot(method)
    try:
        _click(app, "Show evaluation predictions plot")
        app.selectbox(key="evaluation_plot_kind").select("Residuals").run(timeout=10)
        _click(app, "Show evaluation residuals plot")
        assert len(figures) == 2
        destination = tmp_path / "ui-export"
        app.text_input(key="evaluation_output_dir").set_value(str(destination))
        _click(app, "Export predictive evaluation")
        _assert_bundle(destination, result)
        assert any(("incomplete" if incomplete else "Wrote predictive evaluation") in item.value
                   for item in app.success)
        _click(app, "Export predictive evaluation")
        assert any("already exists" in item.value for item in app.error)
        assert app.session_state[EVALUATION_CACHE_KEY] is cached
        _assert_bundle(destination, result)
    finally:
        for figure in figures:
            plt.close(figure)
    assert len(calls) == 24
    assert {path: path.read_bytes() for path in sources} == before
    assert session.config.model.profile == "default"
    pd.testing.assert_frame_equal(session.model_summary(), metadata_before)
