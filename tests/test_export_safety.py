"""Artifact exports must never replace the campaign files they describe."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from bo_forge._campaign.exports import _validate_export_destination
from bo_forge._campaign.provenance import load_manifest, manifest_path_for_log
from bo_forge.application import CampaignAppService, make_staged_suggestion_bundle
from bo_forge.cli import run
from bo_forge.provenance import adopt_provenance, migrate_provenance
from bo_forge.session import CampaignSession
from tests._provenance_support import apply, v1

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _campaign(tmp_path, *, managed=True):
    config, log = tmp_path / "campaign.yaml", tmp_path / "campaign.csv"
    config.write_bytes((PROJECT_ROOT / "configs/01_simple_2d_maximise_logei.yaml").read_bytes())
    log.write_bytes((PROJECT_ROOT / "examples/01_simple_2d_maximise_logei_campaign_log.csv")
                    .read_bytes())
    if managed:
        apply(adopt_provenance, config, log)
    service = CampaignAppService.load(config, log)
    row = service.df.iloc[0].to_dict()
    row.update(row_id="candidate", iteration=1, status="suggested", source="sobol")
    row[service.config.objective.name] = ""
    suggestions = pd.DataFrame([row], columns=service.df.columns)
    bundle = make_staged_suggestion_bundle(suggestions, config, log)
    return SimpleNamespace(
        config=config, log=log, manifest=manifest_path_for_log(log), service=service,
        suggestions=suggestions, bundle=bundle,
    )


@pytest.fixture
def campaign(tmp_path):
    return _campaign(tmp_path)


def _snapshot(campaign):
    return {path: path.read_bytes() if path.exists() else None
            for path in (campaign.config, campaign.log, campaign.manifest)}


def _alias(source, kind):
    if kind == "direct":
        return source
    if kind == "normalized":
        return source.parent / "unused" / ".." / source.name
    path = source.parent / f"alias-{source.name}"
    if kind == "symlink":
        path.symlink_to(source)
    else:
        path.hardlink_to(source)
    return path


@pytest.mark.parametrize("source", ["config", "log", "manifest"])
@pytest.mark.parametrize("alias", ["direct", "normalized", "symlink", "hardlink"])
def test_export_guard_rejects_source_aliases_without_side_effects(campaign, source, alias):
    before = _snapshot(campaign)
    output = _alias(getattr(campaign, source), alias)
    with pytest.raises(OSError, match="Choose a separate artifact output path"):
        _validate_export_destination(output, campaign.config, campaign.log)
    assert _snapshot(campaign) == before
    assert not (campaign.log.parent / "unused").exists()


def test_legacy_manifest_name_is_reserved_for_provenance(tmp_path):
    campaign = _campaign(tmp_path, managed=False)
    before = _snapshot(campaign)
    with pytest.raises(OSError, match="conflicts with campaign source"):
        campaign.service.export_staged_suggestions(campaign.bundle, campaign.manifest)
    assert _snapshot(campaign) == before


@pytest.mark.parametrize("command", ["suggest", "report"])
def test_case_only_manifest_name_cannot_create_a_legacy_sidecar(tmp_path, command):
    campaign = _campaign(tmp_path, managed=False)
    output = campaign.manifest.with_name(campaign.manifest.name.upper())
    before = _snapshot(campaign)
    assert run([command, "--config", str(campaign.config), "--log", str(campaign.log),
                "--output", str(output)]) == 1
    assert _snapshot(campaign) == before
    assert not output.exists()


def test_report_literal_tilde_path_matches_the_writer(tmp_path, monkeypatch):
    literal = tmp_path / "~"
    literal.mkdir()
    campaign = _campaign(literal, managed=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    before = _snapshot(campaign)
    assert run(["report", "--config", str(campaign.config), "--log", str(campaign.log),
                "--output", "~/campaign.csv"]) == 1
    assert _snapshot(campaign) == before


@pytest.mark.parametrize("append", [False, True])
@pytest.mark.parametrize("source", ["config", "log", "manifest"])
def test_cli_pandas_expanded_tilde_cannot_overwrite_sources(
    campaign, monkeypatch, source, append,
):
    monkeypatch.setenv("HOME", str(campaign.log.parent))
    work = campaign.log.parent / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    before = _snapshot(campaign)
    args = ["suggest", "--config", str(campaign.config), "--log", str(campaign.log),
            "--output", f"~/{getattr(campaign, source).name}"]
    if append:
        args.append("--append")
    assert run(args) == 1
    assert _snapshot(campaign) == before
    assert not (work / "~").exists()


def test_cli_distinct_tilde_export_uses_pandas_home_expansion(campaign, monkeypatch):
    monkeypatch.setenv("HOME", str(campaign.log.parent))
    work = campaign.log.parent / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    before = _snapshot(campaign)
    assert run(["suggest", "--config", str(campaign.config), "--log", str(campaign.log),
                "--output", "~/reports/rows.csv"]) == 0
    assert (campaign.log.parent / "reports" / "rows.csv").is_file()
    assert _snapshot(campaign) == before
    assert not (work / "~").exists()


@pytest.mark.parametrize("collision", [False, True])
def test_cli_unknown_user_expansion_preserves_pandas_literal_path_behavior(
    tmp_path, monkeypatch, collision,
):
    from os.path import expanduser

    literal_name = f"~bo-forge-missing-user-{tmp_path.name}"
    assert expanduser(literal_name) == literal_name
    literal = tmp_path / literal_name
    literal.mkdir()
    campaign = _campaign(literal, managed=False)
    monkeypatch.chdir(tmp_path)
    before = _snapshot(campaign)
    name = "campaign.csv" if collision else "rows.csv"
    assert run(["suggest", "--config", str(campaign.config), "--log", str(campaign.log),
                "--output", str(Path(literal_name) / name)]) == int(collision)
    assert _snapshot(campaign) == before
    if not collision:
        assert (literal / name).is_file()


@pytest.mark.parametrize("command", ["suggest", "report"])
def test_cli_symlink_loop_is_a_controlled_export_error(campaign, command, capsys):
    output = campaign.log.parent / "loop.csv"
    output.symlink_to(output)
    before = _snapshot(campaign)
    assert run([command, "--config", str(campaign.config), "--log", str(campaign.log),
                "--output", str(output)]) == 1
    assert "Could not resolve export destination" in capsys.readouterr().err
    assert _snapshot(campaign) == before


def test_cli_checks_output_alias_again_after_suggestion(campaign, monkeypatch):
    output = campaign.log.parent / "rows.csv"
    before = _snapshot(campaign)

    def suggest(_session, **_kwargs):
        output.symlink_to(campaign.log)
        return campaign.suggestions

    monkeypatch.setattr(CampaignSession, "suggest_next", suggest)
    assert run(["suggest", "--config", str(campaign.config), "--log", str(campaign.log),
                "--output", str(output), "--append"]) == 1
    assert _snapshot(campaign) == before


def test_service_expanded_tilde_path_matches_the_csv_writer(campaign, monkeypatch):
    monkeypatch.setenv("HOME", str(campaign.log.parent))
    before = _snapshot(campaign)
    with pytest.raises(OSError, match="conflicts with campaign source"):
        campaign.service.export_staged_suggestions(campaign.bundle, "~/campaign.csv")
    assert _snapshot(campaign) == before


@pytest.mark.parametrize("alias", ["direct", "hardlink"])
def test_report_export_preserves_referenced_provenance_archives(tmp_path, alias):
    config, log = v1(tmp_path)
    apply(migrate_provenance, config, log)
    archive = log.parent / load_manifest(log)["archives"][0]["path"]
    before = {path: path.read_bytes() for path in
              (config, log, manifest_path_for_log(log), archive)}
    session = CampaignSession.from_files(config, log)
    with pytest.raises(OSError, match="conflicts with campaign source"):
        session.export_report(_alias(archive, alias))
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize("command", ["suggest", "suggest-append", "report"])
@pytest.mark.parametrize("source", ["config", "log", "manifest"])
def test_cli_export_collision_fails_before_computation_or_mutation(
    campaign, monkeypatch, capsys, command, source,
):
    before = _snapshot(campaign)
    compute = Mock(side_effect=AssertionError("Export collision must fail before computation"))
    method = "report" if command == "report" else "suggest_next"
    monkeypatch.setattr(CampaignSession, method, compute)
    args = [command.split("-")[0], "--config", str(campaign.config),
            "--log", str(campaign.log), "--output", str(getattr(campaign, source))]
    if command == "suggest-append":
        args.append("--append")
    assert run(args) == 1
    captured = capsys.readouterr()
    assert "Choose a separate artifact output path" in captured.err
    assert "Reload" not in captured.err
    assert "Wrote" not in captured.out
    compute.assert_not_called()
    assert _snapshot(campaign) == before


@pytest.mark.parametrize("source", ["config", "log", "manifest"])
@pytest.mark.parametrize("export", ["session-report", "service-report", "staged"])
def test_session_and_service_exports_preserve_sources(campaign, source, export):
    before = _snapshot(campaign)
    output = _alias(getattr(campaign, source), "symlink")
    with pytest.raises(OSError, match="conflicts with campaign source"):
        if export == "staged":
            campaign.service.export_staged_suggestions(campaign.bundle, output)
        else:
            owner = campaign.service.session if export == "session-report" else campaign.service
            owner.export_report(output)
    assert _snapshot(campaign) == before
    assert campaign.bundle["appended"] is False
    pd.testing.assert_frame_equal(campaign.bundle["suggestions"], campaign.suggestions)


@pytest.mark.parametrize("managed", [False, True])
@pytest.mark.parametrize("append", [False, True])
def test_distinct_cli_export_keeps_explicit_append_semantics(tmp_path, managed, append):
    campaign = _campaign(tmp_path, managed=managed)
    before = _snapshot(campaign)
    output = tmp_path / "reports" / "suggestions.csv"
    args = ["suggest", "--config", str(campaign.config), "--log", str(campaign.log),
            "--batch-size", "1", "--output", str(output)]
    if append:
        args.append("--append")
    assert run(args) == 0
    suggestions = pd.read_csv(output)
    assert len(suggestions) == 1
    assert suggestions.iloc[0]["source"] == "sobol"
    if append:
        reloaded = CampaignSession.from_files(campaign.config, campaign.log)
        assert reloaded.df.row_id.tolist() == (
            campaign.service.df.row_id.tolist() + suggestions.row_id.tolist()
        )
        assert campaign.config.read_bytes() == before[campaign.config]
    else:
        assert _snapshot(campaign) == before


def test_distinct_service_exports_keep_existing_artifact_overwrite_behavior(campaign, tmp_path):
    before = _snapshot(campaign)
    for filename, writer in [
        ("report.txt", campaign.service.export_report),
        ("suggestions.csv", lambda path: campaign.service.export_staged_suggestions(
            campaign.bundle, path,
        )),
    ]:
        output = tmp_path / "reports" / filename
        assert writer(output) == output
        expected = output.read_bytes()
        output.write_text("old artifact")
        assert writer(output) == output
        assert output.read_bytes() == expected
    assert _snapshot(campaign) == before


def _export_app():
    import streamlit as st

    from bo_forge_app.views.analyze import _render_report_text_actions
    from bo_forge_app.views.run import _render_staged_suggestion_export

    campaign = st.session_state["test_campaign"]
    if st.session_state["test_export"] == "report":
        _render_report_text_actions(st, campaign, campaign.log_path)
    else:
        bundle = st.session_state["test_bundle"]
        _render_staged_suggestion_export(
            st, campaign, bundle, bundle["suggestions"], campaign.log_path,
        )


@pytest.mark.parametrize("export", ["report", "staged-service", "staged-session"])
@pytest.mark.parametrize("source", ["config", "log", "manifest"])
def test_streamlit_rejects_export_collision_and_retains_staging(
    campaign, export, source, monkeypatch,
):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    # AppTest swaps __main__; restore it before subsequent multiprocessing tests.
    monkeypatch.setitem(sys.modules, "__main__", sys.modules["__main__"])
    app = AppTest.from_function(_export_app)
    app.session_state["test_campaign"] = (
        campaign.service.session if export == "staged-session" else campaign.service
    )
    app.session_state["test_bundle"] = campaign.bundle
    app.session_state["test_export"] = export
    before = _snapshot(campaign)
    app.run(timeout=10)
    output = _alias(getattr(campaign, source), "symlink")
    app.text_input[0].set_value(str(output))
    label = "Export report" if export == "report" else "Export staged suggestions CSV"
    next(button for button in app.button if button.label == label).click().run(timeout=10)
    assert not app.exception
    assert not app.success
    assert any("Choose a separate artifact output path" in error.value for error in app.error)
    assert _snapshot(campaign) == before
    retained = app.session_state["test_bundle"]
    assert retained["appended"] is False
    pd.testing.assert_frame_equal(retained["suggestions"], campaign.suggestions)
    output = campaign.log.parent / "reports" / ("report.txt" if export == "report" else "rows.csv")
    app.text_input[0].set_value(str(output))
    next(button for button in app.button if button.label == label).click().run(timeout=10)
    assert not app.exception
    assert app.success
    assert output.is_file()
    assert _snapshot(campaign) == before
