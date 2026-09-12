"""Predictive evaluation CLI forwarding, exports, and error boundaries."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from bo_forge import cli
from bo_forge.errors import ConfigError
from bo_forge.session import CampaignSession


@pytest.fixture
def evaluator(monkeypatch):
    result = SimpleNamespace(
        summary=pd.DataFrame([{"model_profile": "rough", "fit_status": "complete", "rmse": 0.25}]),
        fold_outcomes=pd.DataFrame([
            {"model_profile": "rough", "fold": 1, "fit_status": "complete"},
        ]),
        export=Mock(),
        plot_predictions=Mock(), plot_residuals=Mock(),
    )
    campaign = SimpleNamespace(model_predictive_evaluation=Mock(return_value=result))
    loader = Mock(return_value=campaign)
    monkeypatch.setattr(CampaignSession, "from_files", loader)
    return campaign, result, loader


def _args():
    return ["model-evaluate", "--config", "campaign.yaml", "--log", "campaign.csv"]


def test_model_evaluate_defaults_are_read_only(evaluator, capsys):
    campaign, result, loader = evaluator
    assert cli.run(_args()) == 0
    campaign.model_predictive_evaluation.assert_called_once_with(profiles=None, folds=5, seed=0)
    loader.assert_called_once_with(
        Path("campaign.yaml"), Path("campaign.csv"), provenance_policy="compatible",
    )
    output = capsys.readouterr().out
    assert "rmse" in output
    assert "Fold outcomes:" in output
    result.export.assert_not_called()
    result.plot_predictions.assert_not_called()
    result.plot_residuals.assert_not_called()


def test_model_evaluate_order_options_required_policy_and_export(evaluator, tmp_path, capsys):
    campaign, result, loader = evaluator
    output = tmp_path / "evaluation"
    assert cli.run([
        *_args(), "--profile", "rough", "--profile", "smooth", "--folds", "3",
        "--seed", "17", "--require-provenance", "--output-dir", str(output),
    ]) == 0
    campaign.model_predictive_evaluation.assert_called_once_with(
        profiles=["rough", "smooth"], folds=3, seed=17,
    )
    assert loader.call_args.kwargs == {"provenance_policy": "required"}
    result.export.assert_called_once_with(output)
    assert str(output) in capsys.readouterr().out
    result.plot_predictions.assert_not_called()
    result.plot_residuals.assert_not_called()


@pytest.mark.parametrize("error", [ConfigError("unsupported campaign"), ValueError("bad folds")])
def test_model_evaluate_backend_errors_do_not_export(evaluator, error, capsys, tmp_path):
    campaign, result, _ = evaluator
    campaign.model_predictive_evaluation.side_effect = error
    assert cli.run([*_args(), "--output-dir", str(tmp_path / "out")]) == 1
    assert str(error) in capsys.readouterr().err
    result.export.assert_not_called()
    assert list(tmp_path.iterdir()) == []


def test_model_evaluate_export_failure_is_cli_error(evaluator, capsys, tmp_path):
    campaign, result, _ = evaluator
    result.export.side_effect = OSError("read-only directory")
    assert cli.run([*_args(), "--output-dir", str(tmp_path / "new-output")]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Could not export predictive evaluation" in captured.err
    assert "read-only directory" in captured.err
    assert campaign.model_predictive_evaluation.call_count == 1


def test_model_evaluate_exports_tables_only_and_refuses_existing_directory(evaluator, tmp_path):
    from bo_forge.predictive import PredictiveEvaluationResult

    campaign, result, _ = evaluator
    tables = PredictiveEvaluationResult(
        result.summary, pd.DataFrame(), pd.DataFrame(), {"folds": 5},
    )
    result.export.side_effect = tables.export
    output = tmp_path / "evaluation"
    args = [*_args(), "--output-dir", str(output)]
    assert cli.run(args) == 0
    assert {path.name for path in output.iterdir()} == {
        "summary.csv", "predictions.csv", "fold_outcomes.csv", "metadata.json",
    }
    before = {path: path.read_bytes() for path in output.iterdir()}
    assert cli.run(args) == 1
    assert {path: path.read_bytes() for path in output.iterdir()} == before
    assert campaign.model_predictive_evaluation.call_count == 1
    result.plot_predictions.assert_not_called()
    result.plot_residuals.assert_not_called()


@pytest.mark.parametrize("option", [["--folds", "abc"], ["--profile", "unknown"]])
def test_model_evaluate_parser_rejects_bad_options(evaluator, option):
    _, _, loader = evaluator
    assert cli.run([*_args(), *option]) == 2
    loader.assert_not_called()


def test_model_evaluate_required_provenance_fails_before_evaluation(tmp_path, capsys):
    config = tmp_path / "campaign.yaml"
    log = tmp_path / "campaign.csv"
    config.write_bytes(Path("configs/17_model_profile_logei.yaml").read_bytes())
    log.write_bytes(Path("examples/17_model_profile_campaign_log.csv").read_bytes())
    before = log.read_bytes()
    assert cli.run([
        "model-evaluate", "--config", str(config), "--log", str(log),
        "--require-provenance", "--output-dir", str(tmp_path / "out"),
    ]) == 1
    assert "provenance manifest is required" in capsys.readouterr().err
    assert log.read_bytes() == before
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("kind", ["directory", "file", "broken_symlink"])
def test_existing_export_destination_fails_before_fitting(evaluator, tmp_path, capsys, kind):
    campaign, result, _ = evaluator
    output = tmp_path / "existing"
    if kind == "directory":
        output.mkdir()
    elif kind == "file":
        output.write_text("keep this file")
    else:
        output.symlink_to(tmp_path / "missing")
    assert cli.run([*_args(), "--output-dir", str(output)]) == 1
    assert "must not already exist" in capsys.readouterr().err
    campaign.model_predictive_evaluation.assert_not_called()
    result.export.assert_not_called()
    if kind == "file":
        assert output.read_text() == "keep this file"


def test_export_still_refuses_destination_created_during_fitting(evaluator, tmp_path):
    from bo_forge.predictive import PredictiveEvaluationResult

    campaign, result, _ = evaluator
    output = tmp_path / "evaluation"

    def evaluate(**_kwargs):
        output.mkdir()
        (output / "summary.csv").write_text("concurrent output")
        return PredictiveEvaluationResult(result.summary, pd.DataFrame(), result.fold_outcomes, {})

    campaign.model_predictive_evaluation.side_effect = evaluate
    assert cli.run([*_args(), "--output-dir", str(output)]) == 1
    assert (output / "summary.csv").read_text() == "concurrent output"
    assert sorted(path.name for path in output.iterdir()) == ["summary.csv"]


def test_export_preflight_permission_failure_is_actionable(
    evaluator, tmp_path, monkeypatch, capsys,
):
    campaign, result, _ = evaluator
    output = tmp_path / "inaccessible"
    exists = Path.exists

    def inaccessible(path):
        if path == output:
            raise PermissionError("permission denied")
        return exists(path)

    monkeypatch.setattr(Path, "exists", inaccessible)
    assert cli.run([*_args(), "--output-dir", str(output)]) == 1
    assert "Could not inspect predictive evaluation output" in capsys.readouterr().err
    campaign.model_predictive_evaluation.assert_not_called()
    result.export.assert_not_called()


@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("export", [False, True])
def test_failed_folds_are_visible_and_nonzero_without_losing_exports(
    evaluator, tmp_path, capsys, partial, export,
):
    from bo_forge.predictive import PredictiveEvaluationResult

    campaign, result, _ = evaluator
    failed = {"model_profile": "rough", "fit_status": "incomplete", "rmse": None}
    complete = {"model_profile": "default", "fit_status": "complete", "rmse": 0.25}
    summaries = [failed, complete] if partial else [failed]
    outcomes = pd.DataFrame([{
        "model_profile": "rough", "fold": 1, "fit_status": "failed", "fit_message": "fit failed",
    }])
    tables = PredictiveEvaluationResult(pd.DataFrame(summaries), pd.DataFrame(), outcomes, {})
    campaign.model_predictive_evaluation.return_value = tables
    output = tmp_path / "evaluation"
    args = [*_args(), *(["--output-dir", str(output)] if export else [])]
    assert cli.run(args) == 1
    captured = capsys.readouterr()
    assert "fit failed" in captured.out
    assert "Fold outcomes:" in captured.out
    assert "evaluation is incomplete" in captured.err
    if export:
        assert pd.read_csv(output / "fold_outcomes.csv").fit_message.tolist() == ["fit failed"]
        assert len(list(output.iterdir())) == 4
    else:
        assert not output.exists()
