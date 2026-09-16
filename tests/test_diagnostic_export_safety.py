"""Campaign-aware diagnostic exports must not damage source files or sidecars."""

import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from streamlit.testing.v1 import AppTest

import bo_forge.diagnostics as diagnostics
import bo_forge.models as models
from bo_forge import CampaignSession, accept_provenance_config, cli
from bo_forge._campaign.exports import _validate_export_destination
from bo_forge._campaign.provenance import load_manifest, manifest_path_for_log
from bo_forge.application import CampaignAppService
from bo_forge_app.streamlit_helpers import CONFIG_PATH_KEY, LOG_PATH_KEY
from bo_forge_app.views.predictive_evaluation import EVALUATION_CACHE_KEY
from tests._provenance_support import apply
from tests.test_predictive_evaluation import controlled_fit
from tests.test_predictive_hardening import _file_campaign
from tests.test_streamlit_predictive_evaluation import _click, _evaluation_app


@pytest.fixture
def campaign(tmp_path):
    config, log = _file_campaign(tmp_path, True)
    config.write_bytes(config.read_bytes() + b"\n# Formatting-only edit\n")
    apply(accept_provenance_config, config, log)
    manifest = manifest_path_for_log(log)
    archives = [log.parent / item["path"] for item in load_manifest(log)["archives"]]
    sources = [config, log, manifest, *archives]
    return SimpleNamespace(
        config=config, log=log, manifest=manifest, archive=archives[0],
        before={path: path.read_bytes() for path in sources},
        service=CampaignAppService.load(config, log),
    )


def _assert_unchanged(campaign):
    assert {path: path.read_bytes() for path in campaign.before} == campaign.before
    CampaignSession.from_files(campaign.config, campaign.log).validate()


def _png_alias(source, kind):
    output = source.parent / "diagnostic.png"
    if kind == "symlink":
        output.symlink_to(source)
    else:
        output.hardlink_to(source)
    return output


@pytest.mark.parametrize("surface", ["session", "service", "cli"])
@pytest.mark.parametrize("source", ["config", "log", "manifest", "archive"])
@pytest.mark.parametrize("alias", ["symlink", "hardlink"])
def test_plot_source_aliases_fail_before_render(campaign, monkeypatch, surface, source, alias):
    output = _png_alias(getattr(campaign, source), alias)
    render = Mock(side_effect=AssertionError("must reject before rendering"))
    monkeypatch.setattr(diagnostics, "plot_progress", render)
    if surface == "cli":
        assert cli.run([
            "plot", "--config", str(campaign.config), "--log", str(campaign.log),
            "--kind", "progress", "--output", str(output),
        ]) == 1
    else:
        with pytest.raises(OSError, match="conflicts with campaign source"):
            if surface == "session":
                campaign.service.session.plot_progress(save_path=output)
            else:
                campaign.service.plot("progress", save_path=output)
    render.assert_not_called()
    _assert_unchanged(campaign)


@pytest.mark.parametrize("name", [
    "progress", "diagnostics", "cost_progress", "replicates", "pareto", "pareto_parallel",
    "hypervolume", "stage_diagnostics", "fidelity_diagnostics", "fidelity_progress",
    "context_diagnostics", "model_diagnostics", "model_comparison", "qlog_nei_diagnostics",
])
def test_every_session_plot_guards_filename_exports(campaign, monkeypatch, name):
    output = _png_alias(campaign.log, "symlink")
    render = Mock(side_effect=AssertionError("must reject before rendering"))
    monkeypatch.setattr(diagnostics, f"plot_{name}", render)
    with pytest.raises(OSError, match="conflicts with campaign source"):
        getattr(campaign.service.session, f"plot_{name}")(
            filename=output.name, fig_folder=output.parent,
        )
    render.assert_not_called()
    _assert_unchanged(campaign)


@pytest.mark.parametrize("extension", [".png", ""])
def test_plot_rechecks_actual_destination_at_write(campaign, monkeypatch, extension):
    import bo_forge.plot_style as style

    output = campaign.log.parent / f"plot{extension}"
    written_path = output.with_suffix(".png")
    save = style._save_figure

    def redirect(fig, **kwargs):
        written_path.symlink_to(campaign.log)
        return save(fig, **kwargs)

    monkeypatch.setattr(style, "_save_figure", redirect)
    with pytest.raises(OSError, match="conflicts with campaign source"):
        campaign.service.session.plot_progress(save_path=output)
    _assert_unchanged(campaign)


@pytest.mark.parametrize("fail", [False, True])
def test_plot_guard_is_call_scoped_on_success_and_failure(campaign, fail):
    from bo_forge._campaign.exports import _PLOT_SOURCES, _guard_campaign_plot

    def render():
        assert _PLOT_SOURCES.get() == (campaign.config, campaign.log)
        if fail:
            raise ValueError("render failed")
        return "figure"

    assert _PLOT_SOURCES.get() is None
    if fail:
        with pytest.raises(ValueError, match="render failed"):
            _guard_campaign_plot(render, (campaign.config, campaign.log))
    else:
        assert _guard_campaign_plot(render, (campaign.config, campaign.log)) == "figure"
    assert _PLOT_SOURCES.get() is None


def test_plot_guards_do_not_share_campaign_identity_between_threads(campaign, tmp_path):
    from bo_forge._campaign.exports import _guard_campaign_plot, _validate_plot_export_path

    other = tmp_path / "other"
    other.mkdir()
    other_sources = _file_campaign(other, False)
    ready = Barrier(2)
    output = _png_alias(campaign.log, "symlink")

    def check():
        ready.wait(timeout=5)
        try:
            _validate_plot_export_path(output, "png")
        except OSError:
            return "blocked"
        return "unrelated"

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(_guard_campaign_plot, check, sources) for sources in
                   ((campaign.config, campaign.log), other_sources)]
        assert [future.result(timeout=10) for future in futures] == ["blocked", "unrelated"]
    _assert_unchanged(campaign)


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("upper", [False, True])
def test_legacy_sidecar_and_its_descendants_are_reserved(tmp_path, nested, upper):
    config, log = _file_campaign(tmp_path, False)
    manifest = manifest_path_for_log(log)
    output = manifest.with_name(manifest.name.upper()) if upper else manifest
    if nested:
        output = output / "evaluation"
    with pytest.raises(OSError, match="conflicts with campaign source"):
        _validate_export_destination(output, config, log)
    assert not manifest.exists()
    assert not output.exists()
    CampaignSession.from_files(config, log).validate()


@pytest.mark.parametrize("nested", [False, True])
def test_cli_evaluation_rejects_reserved_directory_before_fitting(
    tmp_path, monkeypatch, capsys, nested,
):
    config, log = _file_campaign(tmp_path, False)
    before = config.read_bytes(), log.read_bytes()
    manifest = manifest_path_for_log(log)
    output = manifest / "evaluation" if nested else manifest
    fit = Mock(side_effect=AssertionError("must reject before fitting"))
    monkeypatch.setattr(models, "fit_gp_model", fit)
    assert cli.run([
        "model-evaluate", "--config", str(config), "--log", str(log),
        "--folds", "2", "--output-dir", str(output),
    ]) == 1
    assert "conflicts with campaign source" in capsys.readouterr().err
    fit.assert_not_called()
    assert before == (config.read_bytes(), log.read_bytes())
    assert not manifest.exists()
    CampaignSession.from_files(config, log).validate()


@pytest.mark.parametrize("kind", ["predictions", "residuals"])
def test_session_evaluation_plots_keep_source_guards(campaign, monkeypatch, kind):
    monkeypatch.setattr(models, "fit_gp_model", controlled_fit([]))
    result = campaign.service.model_predictive_evaluation(folds=2)
    output = _png_alias(campaign.archive, "hardlink")
    with pytest.raises(OSError, match="conflicts with campaign source"):
        getattr(result, f"plot_{kind}")(save_path=output)
    _assert_unchanged(campaign)


@pytest.mark.parametrize("nested", [False, True])
def test_session_evaluation_directory_export_preserves_legacy_state(tmp_path, monkeypatch, nested):
    config, log = _file_campaign(tmp_path, False)
    before = config.read_bytes(), log.read_bytes()
    calls = []
    monkeypatch.setattr(models, "fit_gp_model", controlled_fit(calls))
    result = CampaignSession.from_files(config, log).model_predictive_evaluation(folds=2)
    manifest = manifest_path_for_log(log)
    output = manifest / "evaluation" if nested else manifest
    with pytest.raises(OSError, match="conflicts with campaign source"):
        result.export(output)
    assert not manifest.exists()
    destination = result.export(tmp_path / "evaluation")
    assert len(list(destination.iterdir())) == 4
    assert len(calls) == 2
    assert before == (config.read_bytes(), log.read_bytes())
    CampaignSession.from_files(config, log).validate()


@pytest.mark.parametrize("action", ["plot", "bundle", "nested-bundle"])
def test_streamlit_rejects_conflicts_and_retries_retained_result(tmp_path, monkeypatch, action):
    monkeypatch.setitem(sys.modules, "__main__", sys.modules["__main__"])
    config, log = _file_campaign(tmp_path, False)
    before = config.read_bytes(), log.read_bytes()
    calls = []
    monkeypatch.setattr(models, "fit_gp_model", controlled_fit(calls))
    app = AppTest.from_function(_evaluation_app)
    app.session_state["test_campaign"] = CampaignAppService.load(config, log)
    app.session_state[CONFIG_PATH_KEY] = str(config)
    app.session_state[LOG_PATH_KEY] = str(log)
    app.run(timeout=10)
    _click(app, "Run predictive evaluation")
    cached = app.session_state[EVALUATION_CACHE_KEY]
    fit_count = len(calls)
    manifest = manifest_path_for_log(log)
    if action == "plot":
        output = _png_alias(log, "symlink")
        key, label = "evaluation_predictions_export_path", "Export evaluation predictions plot"
        retry = tmp_path / "safe.png"
    else:
        output = manifest / "evaluation" if action == "nested-bundle" else manifest
        key, label = "evaluation_output_dir", "Export predictive evaluation"
        retry = tmp_path / "safe-evaluation"
    app.text_input(key=key).set_value(str(output))
    _click(app, label)
    assert any("conflicts with campaign source" in item.value for item in app.error)
    assert not app.success
    assert app.session_state[EVALUATION_CACHE_KEY] is cached
    assert not manifest.exists()
    app.text_input(key=key).set_value(str(retry))
    _click(app, label)
    assert app.success and retry.exists()
    assert len(calls) == fit_count
    assert before == (config.read_bytes(), log.read_bytes())
    CampaignSession.from_files(config, log).validate()
