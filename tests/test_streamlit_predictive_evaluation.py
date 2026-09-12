"""Explicit/lazy evaluation, identity-bound retention, and non-mutating UI tests."""

import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
from matplotlib.figure import Figure
from streamlit.testing.v1 import AppTest

from bo_forge._campaign.provenance import manifest_path_for_log
from bo_forge.application import CampaignAppService
from bo_forge.errors import ConfigError
from bo_forge.session import CampaignSession
from bo_forge_app.streamlit_helpers import CONFIG_PATH_KEY, LOG_PATH_KEY, SESSION_KEY
from bo_forge_app.ui.state import ACTIVE_PANEL_KEY, PROVENANCE_POLICY_KEY
from bo_forge_app.views.predictive_evaluation import EVALUATION_CACHE_KEY


def _evaluation_app():
    import streamlit as st

    from bo_forge_app.views.predictive_evaluation import render_predictive_evaluation

    st.checkbox("Unrelated display option")
    render_predictive_evaluation(st, st.session_state["test_campaign"])


def _comparison_app():
    import streamlit as st

    from bo_forge_app.views.analyze import _render_model_comparison_action

    _render_model_comparison_action(st, st.session_state["test_campaign"])


def _plot_result(**_kwargs):
    figure = Figure()
    return figure, figure.subplots()


@pytest.fixture
def evaluation_ui(tmp_path, monkeypatch):
    config = tmp_path / "campaign.yaml"
    log = tmp_path / "campaign.csv"
    config.write_bytes(Path("configs/17_model_profile_logei.yaml").read_bytes())
    log.write_bytes(Path("examples/17_model_profile_campaign_log.csv").read_bytes())
    service = CampaignAppService.load(config, log)
    result = SimpleNamespace(
        summary=pd.DataFrame([{
            "model_profile": "default", "rmse": 0.2, "fit_status": "complete", "fit_message": "",
        }]),
        predictions=pd.DataFrame([{"observed": 1.0, "predicted": 0.8}]),
        fold_outcomes=pd.DataFrame([{
            "fold": 1, "fit_status": "complete", "fit_warning_count": 0, "fit_message": "",
        }]),
        metadata={"folds": 5, "seed": 0}, export=Mock(),
        plot_predictions=Mock(side_effect=_plot_result),
        plot_residuals=Mock(side_effect=_plot_result),
    )
    evaluate = Mock(return_value=result)
    monkeypatch.setattr(CampaignAppService, "model_predictive_evaluation", evaluate, raising=False)
    app = AppTest.from_function(_evaluation_app)
    app.session_state["test_campaign"] = service
    app.session_state[CONFIG_PATH_KEY] = str(config)
    app.session_state[LOG_PATH_KEY] = str(log)
    app.run(timeout=10)
    return SimpleNamespace(app=app, service=service, result=result, evaluate=evaluate,
                           config=config, log=log, root=tmp_path)


def _click(app, label):
    next(button for button in app.button if button.label == label).click().run(timeout=10)
    assert not app.exception


def _run(ui):
    _click(ui.app, "Run predictive evaluation")


def _assert_no_result(ui):
    assert EVALUATION_CACHE_KEY not in ui.app.session_state
    assert not ui.app.dataframe
    assert not any(button.label == "Export predictive evaluation" for button in ui.app.button)
    ui.result.export.assert_not_called()
    ui.result.plot_predictions.assert_not_called()
    ui.result.plot_residuals.assert_not_called()


def test_evaluation_explicit_count_cache_and_no_writes(evaluation_ui):
    ui = evaluation_ui
    before = {path: path.read_bytes() for path in ui.root.iterdir()}
    assert not ui.app.exception
    assert ui.app.metric[0].label == "Requested fits"
    assert ui.app.metric[0].value == "5"
    ui.evaluate.assert_not_called()
    _assert_no_result(ui)
    _run(ui)
    ui.evaluate.assert_called_once_with(
        profiles=["smooth"], folds=5, seed=0,
    )
    assert len(ui.app.dataframe) == 3
    ui.app.checkbox[0].check().run(timeout=10)
    assert len(ui.app.dataframe) == 3
    assert ui.evaluate.call_count == 1
    ui.result.export.assert_not_called()
    ui.result.plot_predictions.assert_not_called()
    ui.result.plot_residuals.assert_not_called()
    assert {path: path.read_bytes() for path in ui.root.iterdir()} == before


def test_evaluation_plots_and_exports_reuse_result(evaluation_ui):
    ui = evaluation_ui
    _run(ui)
    _click(ui.app, "Show evaluation predictions plot")
    ui.result.plot_predictions.assert_called_once_with()
    ui.app.selectbox(key="evaluation_plot_kind").select("Residuals").run(timeout=10)
    ui.result.plot_residuals.assert_not_called()
    _click(ui.app, "Show evaluation residuals plot")
    ui.result.plot_residuals.assert_called_once_with()
    _click(ui.app, "Export evaluation residuals plot")
    assert ui.result.plot_residuals.call_args.kwargs["save_path"].suffix == ".png"
    output = ui.root / "artifacts"
    ui.app.text_input(key="evaluation_output_dir").set_value(str(output))
    _click(ui.app, "Export predictive evaluation")
    ui.result.export.assert_called_once_with(output)
    assert ui.evaluate.call_count == 1


@pytest.mark.parametrize("option", ["profiles", "folds", "seed"])
def test_evaluation_options_invalidate_and_count_before_run(evaluation_ui, option):
    ui = evaluation_ui
    _run(ui)
    if option == "profiles":
        ui.app.multiselect(key="evaluation_profiles").set_value(["robust", "rough"])
        expected_count = "10"
    else:
        ui.app.number_input(key=f"evaluation_{option}").set_value(3)
        expected_count = "3" if option == "folds" else "5"
    ui.app.run(timeout=10)
    assert ui.app.metric[0].value == expected_count
    _assert_no_result(ui)
    assert ui.evaluate.call_count == 1
    _run(ui)
    assert ui.evaluate.call_count == 2
    if option == "profiles":
        assert ui.evaluate.call_args.kwargs["profiles"] == ["robust", "rough"]


def test_profile_reordering_and_empty_selection(evaluation_ui):
    ui = evaluation_ui
    ui.app.multiselect(key="evaluation_profiles").set_value(
        ["default", "smooth", "rough", "robust"],
    ).run(timeout=10)
    _run(ui)
    ui.app.multiselect(key="evaluation_profiles").set_value(
        ["robust", "rough", "smooth", "default"],
    ).run(timeout=10)
    _assert_no_result(ui)
    assert ui.app.metric[0].value == "20"
    ui.app.multiselect(key="evaluation_profiles").set_value([]).run(timeout=10)
    assert ui.app.metric[0].value == "0"
    assert ui.app.button(key="evaluation_run").disabled
    assert ui.evaluate.call_count == 1


@pytest.mark.parametrize("source", ["config", "log", "manifest", "observations", "config_object",
                                    "config_path", "log_path", "policy"])
def test_evaluation_source_changes_drop_cache_without_writes(evaluation_ui, source):
    ui = evaluation_ui
    _run(ui)
    if source in {"config", "log"}:
        path = getattr(ui, source)
        stat = path.stat()
        data = path.read_bytes()
        changed = data.replace(b"1", b"2", 1)
        assert changed != data and len(changed) == len(data)
        path.write_bytes(changed)
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    elif source == "manifest":
        manifest_path_for_log(ui.log).write_text("{}")
    elif source == "observations":
        ui.service.df.loc[0, ui.service.config.objective.name] = 99.0
    elif source == "config_object":
        ui.service.session.config = replace(ui.service.config, campaign_name="changed")
    elif source == "policy":
        ui.app.session_state[PROVENANCE_POLICY_KEY] = True
    else:
        key = CONFIG_PATH_KEY if source == "config_path" else LOG_PATH_KEY
        ui.app.session_state[key] = str(ui.root / "other")
    before = {path: path.read_bytes() for path in ui.root.iterdir()}
    ui.app.run(timeout=10)
    assert not ui.app.exception
    _assert_no_result(ui)
    assert ui.evaluate.call_count == 1
    assert {path: path.read_bytes() for path in ui.root.iterdir()} == before


def test_source_change_during_fit_discards_result(evaluation_ui):
    ui = evaluation_ui

    def change_during_fit(**_):
        ui.log.write_bytes(ui.log.read_bytes() + b"\n")
        return ui.result

    ui.evaluate.side_effect = change_during_fit
    _run(ui)
    _assert_no_result(ui)
    assert ui.app.error


def test_evaluation_failure_clears_old_result(evaluation_ui):
    ui = evaluation_ui
    _run(ui)
    ui.evaluate.side_effect = ConfigError("not enough observations")
    _run(ui)
    _assert_no_result(ui)
    assert "not enough observations" in ui.app.error[0].value


def test_export_error_preserves_result_without_refitting(evaluation_ui):
    ui = evaluation_ui
    _run(ui)
    ui.result.export.side_effect = FileExistsError("Output directory already exists")
    _click(ui.app, "Export predictive evaluation")
    assert "already exists" in ui.app.error[0].value
    assert EVALUATION_CACHE_KEY in ui.app.session_state
    assert ui.evaluate.call_count == 1


@pytest.mark.parametrize("error", [OSError("disk full"), TypeError("invalid metadata"),
                                   ValueError("nonfinite metadata")])
def test_export_failure_can_retry_same_result_without_refitting(evaluation_ui, monkeypatch, error):
    import bo_forge.predictive as predictive

    ui = evaluation_ui
    tables = predictive.PredictiveEvaluationResult(
        ui.result.summary, ui.result.predictions, ui.result.fold_outcomes, ui.result.metadata,
    )
    ui.result.export.side_effect = tables.export
    _run(ui)
    output = ui.root / "evaluation"
    ui.app.text_input(key="evaluation_output_dir").set_value(str(output))
    before = {path: path.read_bytes() for path in (ui.config, ui.log)}
    def failed_export(path):
        with monkeypatch.context() as patch:
            patch.setattr(predictive.json, "dumps", Mock(side_effect=error))
            return tables.export(path)

    ui.result.export.side_effect = failed_export
    _click(ui.app, "Export predictive evaluation")
    assert not output.exists()
    assert "retained for retry" in ui.app.error[0].value
    assert EVALUATION_CACHE_KEY in ui.app.session_state
    ui.result.export.side_effect = tables.export
    _click(ui.app, "Export predictive evaluation")
    assert len(list(output.iterdir())) == 4
    assert ui.evaluate.call_count == 1
    assert {path: path.read_bytes() for path in before} == before


def test_incomplete_and_warning_notices_are_visible_and_export_is_labeled(evaluation_ui):
    ui = evaluation_ui
    ui.result.summary.loc[0, ["fit_status", "fit_message", "rmse"]] = [
        "incomplete", "Fold 1: did not converge", None,
    ]
    ui.result.fold_outcomes.loc[0, ["fit_status", "fit_message", "fit_warning_count"]] = [
        "failed", "did not converge", 2,
    ]
    _run(ui)
    notices = " ".join(item.value for item in ui.app.warning)
    assert "Incomplete predictive evaluation" in notices
    assert "default" in notices and "Captured 2 fit warning(s)" in notices
    assert len(ui.app.dataframe) == 3
    _click(ui.app, "Export predictive evaluation")
    assert any("Wrote incomplete predictive evaluation" in item.value for item in ui.app.success)
    assert ui.evaluate.call_count == 1


def test_comparison_labels_are_in_sample_and_keep_form_key(evaluation_ui, monkeypatch):
    ui = evaluation_ui
    comparison = Mock(return_value=pd.DataFrame([{"training_rmse": 0.1}]))
    monkeypatch.setattr(CampaignAppService, "model_profile_comparison", comparison, raising=False)
    app = AppTest.from_function(_comparison_app)
    app.session_state["test_campaign"] = ui.service
    app.run(timeout=10)
    comparison.assert_not_called()
    assert any("Training error is not predictive quality" in item.value for item in app.markdown)
    button = next(
        button for button in app.button if button.label == "Run in-sample model comparison"
    )
    assert "model_comparison_form" in button.key
    _click(app, "Run in-sample model comparison")
    comparison.assert_called_once_with()
    assert any("In-sample Model Profile Comparison" in item.value for item in app.subheader)


def test_required_manifest_guard_applies_to_cached_result(evaluation_ui):
    ui = evaluation_ui
    managed_log = ui.root / "managed.csv"
    session = CampaignSession.initialize(ui.config, managed_log)
    service = CampaignAppService.load(ui.config, managed_log, provenance_policy="required")
    ui.app.session_state["test_campaign"] = service
    ui.app.session_state[LOG_PATH_KEY] = str(managed_log)
    ui.app.session_state[PROVENANCE_POLICY_KEY] = True
    ui.app.run(timeout=10)
    _run(ui)
    manifest_path_for_log(session.log_path).unlink()
    before = managed_log.read_bytes()
    ui.app.run(timeout=10)
    _assert_no_result(ui)
    assert ui.app.warning
    assert ui.evaluate.call_count == 1
    assert managed_log.read_bytes() == before
    assert not manifest_path_for_log(managed_log).exists()


@pytest.mark.parametrize("unsupported", ["is_multi_objective", "is_contextual_campaign",
                                         "replicates", "is_structured_campaign", "fidelity"])
def test_unsupported_campaigns_hide_evaluation_and_drop_cache(evaluation_ui, unsupported):
    ui = evaluation_ui
    _run(ui)
    config = SimpleNamespace(
        is_multi_objective=False, is_contextual_campaign=False,
        replicates=SimpleNamespace(enabled=False), is_structured_campaign=False, fidelity=None,
    )
    setattr(config, unsupported, SimpleNamespace(enabled=True) if unsupported == "replicates"
            else True)
    ui.service.session.config = config
    ui.app.run(timeout=10)
    assert not ui.app.exception
    assert not ui.app.subheader
    _assert_no_result(ui)
    assert ui.evaluate.call_count == 1


@pytest.mark.parametrize("area", ["Campaign", "Run"])
def test_full_workbench_navigation_retains_evaluation_and_nondefault_options(evaluation_ui, area):
    ui = evaluation_ui
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "bo_forge_app/streamlit_app.py")
    app.session_state[SESSION_KEY] = ui.service
    app.session_state[CONFIG_PATH_KEY] = str(ui.config)
    app.session_state[LOG_PATH_KEY] = str(ui.log)
    app.session_state[ACTIVE_PANEL_KEY] = "Analyze"
    before = {path: path.read_bytes() for path in (ui.config, ui.log)}
    app.run(timeout=10)
    app.multiselect(key="evaluation_profiles").set_value(["rough", "default"])
    app.number_input(key="evaluation_folds").set_value(3)
    app.number_input(key="evaluation_seed").set_value(17)
    app.run(timeout=10)
    _click(app, "Run predictive evaluation")
    identity = app.session_state[EVALUATION_CACHE_KEY]["identity"]
    app.radio(key=ACTIVE_PANEL_KEY).set_value(area).run(timeout=10)
    assert "evaluation_folds" not in app.session_state
    app.radio(key=ACTIVE_PANEL_KEY).set_value("Analyze").run(timeout=10)
    assert not app.exception
    assert app.multiselect(key="evaluation_profiles").value == ["rough", "default"]
    assert app.number_input(key="evaluation_folds").value == 3
    assert app.number_input(key="evaluation_seed").value == 17
    assert app.session_state[EVALUATION_CACHE_KEY]["identity"] == identity
    assert any(button.label == "Export predictive evaluation" for button in app.button)
    ui.evaluate.assert_called_once_with(profiles=["rough", "default"], folds=3, seed=17)
    ui.result.plot_predictions.assert_not_called()
    ui.result.plot_residuals.assert_not_called()
    ui.result.export.assert_not_called()
    assert {path: path.read_bytes() for path in before} == before


def test_saved_evaluation_options_do_not_leak_into_a_new_campaign(evaluation_ui):
    ui = evaluation_ui
    ui.app.multiselect(key="evaluation_profiles").set_value(["robust", "default"])
    ui.app.number_input(key="evaluation_folds").set_value(3)
    ui.app.number_input(key="evaluation_seed").set_value(17)
    ui.app.run(timeout=10)
    _run(ui)
    second = ui.root / "second"
    second.mkdir()
    config, log = second / "campaign.yaml", second / "campaign.csv"
    config.write_text(ui.config.read_text().replace("profile: smooth", "profile: rough"))
    log.write_bytes(ui.log.read_bytes())
    ui.app.session_state["test_campaign"] = CampaignAppService.load(config, log)
    ui.app.session_state[CONFIG_PATH_KEY] = str(config)
    ui.app.session_state[LOG_PATH_KEY] = str(log)
    ui.app.run(timeout=10)
    assert not ui.app.exception
    _assert_no_result(ui)
    assert ui.app.multiselect(key="evaluation_profiles").value == ["rough"]
    assert ui.app.number_input(key="evaluation_folds").value == 5
    assert ui.app.number_input(key="evaluation_seed").value == 0
    assert ui.evaluate.call_count == 1
