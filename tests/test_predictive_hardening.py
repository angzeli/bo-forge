"""Snapshot consistency and bounded failure visibility for predictive diagnostics."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

import bo_forge.models as models
import bo_forge.predictive as predictive
from bo_forge import CampaignSession, cli
from bo_forge._campaign.provenance import manifest_path_for_log
from bo_forge.application import CampaignAppService
from bo_forge.errors import LogConflictError
from tests._session_support import write_config
from tests.test_predictive_evaluation import controlled_fit, evaluation_data

METRICS = ["rmse", "mae", "mean_nlpd", "interval_coverage", "mean_interval_width"]


@pytest.mark.parametrize("direction", ["maximize", "minimize"])
@pytest.mark.parametrize("residual", [0.0, -2.0, 2.0])
def test_interval_metrics_are_invariant_to_representable_objective_offsets(
    monkeypatch, direction, residual,
):
    config, df = evaluation_data(direction=direction)
    baseline = None
    sign = 1 if direction == "maximize" else -1
    for offset in (0.0, 1e16, -1e16):
        df["score"] = offset + residual
        before = df.copy(deep=True)
        monkeypatch.setattr(models, "fit_gp_model", lambda *args, offset=offset: SimpleNamespace(
            posterior=lambda x, **kwargs: SimpleNamespace(
                mean=torch.full((len(x), 1), sign * offset, dtype=torch.double),
                variance=torch.ones((len(x), 1), dtype=torch.double),
            ),
        ))
        result = predictive.model_predictive_evaluation(config, df, folds=2)
        summary = result.summary.iloc[0]
        assert summary.fit_status == "complete"
        assert summary.mean_interval_width == pytest.approx(2 * predictive._Z95)
        assert summary.interval_coverage == float(abs(residual) <= predictive._Z95)
        assert result.predictions.interval_covered.eq(abs(residual) <= predictive._Z95).all()
        if baseline is None:
            baseline = summary[METRICS]
        else:
            pd.testing.assert_series_equal(summary[METRICS], baseline, check_exact=True)
        pd.testing.assert_frame_equal(df, before)


@pytest.mark.parametrize("direction", ["maximize", "minimize"])
@pytest.mark.parametrize("offset,variance", [(1e20, 4.0), (1e200, 1e308)])
def test_rounded_interval_endpoints_do_not_zero_positive_width(
    monkeypatch, direction, offset, variance,
):
    config, df = evaluation_data(direction=direction)
    df["score"] = offset
    sign = 1 if direction == "maximize" else -1
    monkeypatch.setattr(models, "fit_gp_model", lambda *args: SimpleNamespace(
        posterior=lambda x, **kwargs: SimpleNamespace(
            mean=torch.full((len(x), 1), sign * offset, dtype=torch.double),
            variance=torch.full((len(x), 1), variance, dtype=torch.double),
        ),
    ))
    result = predictive.model_predictive_evaluation(config, df, folds=2)
    summary = result.summary.iloc[0]
    assert summary.fit_status == "complete"
    assert summary.mean_interval_width == pytest.approx(2 * predictive._Z95 * np.sqrt(variance))
    assert summary.mean_interval_width > 0
    assert summary.interval_coverage == 1.0
    assert result.predictions.interval_lower.eq(result.predictions.interval_upper).all()


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("distance,covered", [
    (np.nextafter(predictive._Z95, 0.0), True),
    (predictive._Z95, True),
    (np.nextafter(predictive._Z95, np.inf), False),
])
def test_interval_coverage_keeps_inclusive_standardized_boundary(sign, distance, covered):
    assert predictive._prediction_metrics(sign * distance, 0.0, 1.0)["interval_covered"] is covered


def test_summary_appends_message_and_preserves_successful_profile(monkeypatch):
    config, df = evaluation_data()
    good = controlled_fit([])

    def fit(cfg, training):
        if cfg.model.profile == "rough":
            raise RuntimeError("fold did not converge")
        return good(cfg, training)

    monkeypatch.setattr(models, "fit_gp_model", fit)
    result = predictive.model_predictive_evaluation(config, df, ["rough", "default"], folds=2)
    assert result.summary.columns.tolist() == [
        "model_profile", "evaluation_scope", "fit_status", "observed_rows",
        "completed_folds", "total_folds", *METRICS, "fit_message",
    ]
    assert result.summary.fit_message.tolist() == [
        "Fold 1: fold did not converge; Fold 2: fold did not converge", "",
    ]
    assert result.summary.loc[0, METRICS].isna().all()
    assert result.summary.loc[1, METRICS].notna().all()


@pytest.mark.parametrize("magnitude", [0.0, 1e200, 1e308])
def test_extreme_finite_predictions_have_finite_equivalent_aggregates(monkeypatch, magnitude):
    config, df = evaluation_data()
    df["score"] = magnitude
    monkeypatch.setattr(models, "fit_gp_model", lambda *args: SimpleNamespace(
        posterior=lambda x, **kwargs: SimpleNamespace(
            mean=torch.zeros((len(x), 1), dtype=torch.double),
            variance=torch.full((len(x), 1), 1e308, dtype=torch.double),
        ),
    ))
    result = predictive.model_predictive_evaluation(config, df, folds=2)
    summary = result.summary.iloc[0]
    assert summary.fit_status == "complete"
    assert summary.rmse == pytest.approx(magnitude)
    assert summary.mae == pytest.approx(magnitude)
    assert np.isfinite(summary[METRICS].to_numpy(dtype=float)).all()
    assert summary.mean_nlpd == pytest.approx(result.predictions.negative_log_predictive_density[0])
    assert summary.fit_message == ""


def test_aggregate_failure_keeps_successful_predictions_but_withholds_metrics(monkeypatch):
    config, df = evaluation_data()

    def fail_aggregation(_rows):
        raise FloatingPointError("synthetic aggregate overflow")

    monkeypatch.setattr(models, "fit_gp_model", controlled_fit([]))
    monkeypatch.setattr(predictive, "_aggregate_metrics", fail_aggregation)
    result = predictive.model_predictive_evaluation(config, df, folds=2)
    assert result.fold_outcomes.fit_status.eq("complete").all()
    assert result.predictions.fit_status.eq("complete").all()
    assert result.summary.completed_folds.tolist() == [2]
    assert result.summary.fit_status.tolist() == ["incomplete"]
    assert result.summary[METRICS].isna().all().all()
    assert result.summary.fit_message[0] == "Aggregate metrics failed: synthetic aggregate overflow"


def test_unrepresentable_row_metrics_fail_without_inventing_values(monkeypatch):
    config, df = evaluation_data()
    df["score"] = 1e308
    monkeypatch.setattr(models, "fit_gp_model", lambda *args: SimpleNamespace(
        posterior=lambda x, **kwargs: SimpleNamespace(
            mean=torch.full((len(x), 1), -1e308, dtype=torch.double),
            variance=torch.ones((len(x), 1), dtype=torch.double),
        ),
    ))
    with np.errstate(over="ignore", invalid="ignore"):
        result = predictive.model_predictive_evaluation(config, df, folds=2)
    assert result.fold_outcomes.fit_status.eq("failed").all()
    assert result.predictions.predicted_mean.isna().all()
    assert result.summary[METRICS].isna().all().all()
    assert "Non-finite" in result.summary.fit_message[0]


def test_in_memory_evaluation_owns_nested_config_and_observation_snapshot(monkeypatch):
    config, df = evaluation_data()
    original_config, original_df = deepcopy(config), df.copy(deep=True)
    calls = []
    good = controlled_fit(calls)

    def fit(cfg, training):
        df["score"] = 999.0
        df["row_id"] = "changed"
        # Inject an external edit even through the frozen dataclass boundary.
        object.__setattr__(config.objective, "name", "changed")
        assert cfg.objective.name == "score"
        assert cfg is not config and cfg.objective is not config.objective
        return good(cfg, training)

    monkeypatch.setattr(models, "fit_gp_model", fit)
    result = predictive.model_predictive_evaluation(config, df, folds=2)
    monkeypatch.setattr(models, "fit_gp_model", controlled_fit([]))
    baseline = predictive.model_predictive_evaluation(original_config, original_df, folds=2)
    pd.testing.assert_frame_equal(result.predictions, baseline.predictions)
    assert result.metadata == baseline.metadata
    assert len(calls) == 2


def _file_campaign(tmp_path, managed):
    config = tmp_path / "campaign.yaml"
    log = tmp_path / "campaign.csv"
    write_config(config)
    _, frame = evaluation_data(8)
    frame.to_csv(log, index=False)
    if managed:
        from bo_forge import adopt_provenance

        preview = adopt_provenance(config, log)
        adopt_provenance(config, log, apply=True, reason="test baseline",
                         expected_identities=preview["expected_identities"])
    return config, log


@pytest.mark.parametrize("managed", [False, True])
@pytest.mark.parametrize("change", ["config", "log", "manifest"])
@pytest.mark.parametrize("surface", ["session", "service", "cli"])
def test_source_change_during_evaluation_is_rejected_without_export(
    tmp_path, monkeypatch, capsys, managed, change, surface,
):
    config, log = _file_campaign(tmp_path, managed)
    session = CampaignSession.from_files(config, log)
    target = {"config": config, "log": log, "manifest": manifest_path_for_log(log)}[change]
    changed = (target.read_bytes() if target.exists() else b"{}") + b"\n"

    calls = []
    good = controlled_fit(calls)

    def fit(cfg, training):
        target.write_bytes(changed)
        return good(cfg, training)

    monkeypatch.setattr(models, "fit_gp_model", fit)
    if surface == "cli":
        assert cli.run([
            "model-evaluate", "--config", str(config), "--log", str(log),
            "--output-dir", str(tmp_path / "out"),
        ]) == 1
        assert "changed during predictive evaluation" in capsys.readouterr().err
    else:
        owner = session if surface == "session" else CampaignAppService.from_session(session)
        with pytest.raises(LogConflictError, match="changed during predictive evaluation"):
            owner.model_predictive_evaluation()
    assert target.read_bytes() == changed
    assert len(calls) == 5
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("change", ["config", "log"])
def test_legacy_changes_before_evaluation_fail_before_fitting(tmp_path, monkeypatch, change):
    config, log = _file_campaign(tmp_path, False)
    session = CampaignSession.from_files(config, log)
    path = config if change == "config" else log
    path.write_bytes(path.read_bytes() + b"\n")
    monkeypatch.setattr(models, "fit_gp_model", lambda *_: pytest.fail("must not fit"))
    with pytest.raises(LogConflictError, match="Reload before predictive evaluation"):
        session.model_predictive_evaluation()


@pytest.mark.parametrize("fail_all", [False, True])
def test_incomplete_plots_show_only_valid_rows_and_no_withheld_coverage(monkeypatch, fail_all):
    import matplotlib.pyplot as plt

    config, df = evaluation_data()
    calls = []
    good = controlled_fit(calls)

    def fit(cfg, training):
        if fail_all or len(training) == 5:
            raise RuntimeError("cannot fit fold")
        return good(cfg, training)

    monkeypatch.setattr(models, "fit_gp_model", fit)
    result = predictive.model_predictive_evaluation(config, df, folds=3)
    assert result.summary.fit_status.tolist() == ["incomplete"]
    monkeypatch.setattr(models, "fit_gp_model", lambda *_: pytest.fail("must not refit"))
    for plot in (result.plot_predictions, result.plot_residuals):
        fig, axes = plot()
        try:
            labels = " ".join(text.get_text() for text in axes[0].texts)
            assert "aggregate metrics withheld" in labels
            assert "95% interval coverage:" not in labels
            assert ("No valid held-out predictions" in labels) == fail_all
            count = sum(len(item.get_offsets()) for item in axes[0].collections)
            assert count == int(result.predictions.fit_status.eq("complete").sum())
        finally:
            plt.close(fig)
