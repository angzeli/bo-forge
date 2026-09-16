"""Held-out evaluation contracts with controlled predictions and bounded real GPs."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

import bo_forge.models as models
from bo_forge._fit_metadata import FitMetadata
from bo_forge.config import (
    BOConfig,
    CampaignConfig,
    ContextConfig,
    FidelityConfig,
    ModelConfig,
    ObjectiveConfig,
    ReplicateConfig,
    StageConfig,
    VariableConfig,
)
from bo_forge.errors import ConfigError, LogValidationError
from bo_forge.predictive import model_predictive_evaluation
from bo_forge.validation import canonical_columns


def evaluation_data(count=8, direction="maximize"):
    config = CampaignConfig(
        campaign_name="evaluation",
        objective=ObjectiveConfig("score", direction),
        variables=(VariableConfig("x", "continuous", 0.0, 1.0),),
        bo=BOConfig(initial_design_size=2),
    )
    frame = pd.DataFrame(
        [
            {
                "row_id": f"row_{i:03}",
                "iteration": i,
                "status": "observed",
                "source": "manual",
                "x": x,
                "score": 2 * x + 1,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            }
            for i, x in enumerate(np.linspace(0.05, 0.95, count))
        ],
        columns=canonical_columns(config),
    )
    return config, frame


def controlled_fit(calls, *, variance=4.0):
    def fit(config, training):
        calls.append((config.model.profile, training.copy(deep=True)))

        def posterior(x, *, observation_noise):
            assert observation_noise is True
            sign = 1 if config.objective.direction == "maximize" else -1
            return SimpleNamespace(
                mean=sign * (2 * x[:, :1] + 1.5),
                variance=torch.full((len(x), 1), variance, dtype=torch.double),
            )

        return SimpleNamespace(posterior=posterior)

    return fit


@pytest.mark.parametrize("direction", ["maximize", "minimize"])
def test_controlled_metrics_original_units_and_fold_isolation(monkeypatch, direction):
    config, frame = evaluation_data(direction=direction)
    original = frame.copy(deep=True)
    calls = []
    monkeypatch.setattr(models, "fit_gp_model", controlled_fit(calls))
    result = model_predictive_evaluation(config, frame, ["rough", "default"], folds=3, seed=7)
    assert result.summary.model_profile.tolist() == ["rough", "default"]
    assert result.summary.fit_status.tolist() == ["complete", "complete"]
    assert result.summary.rmse.tolist() == pytest.approx([0.5, 0.5])
    assert result.summary.mae.tolist() == pytest.approx([0.5, 0.5])
    expected_nlpd = 0.5 * (np.log(2 * np.pi * 4) + 0.5**2 / 4)
    assert result.summary.mean_nlpd.tolist() == pytest.approx([expected_nlpd] * 2)
    assert result.summary.interval_coverage.tolist() == [1.0, 1.0]
    assert result.summary.mean_interval_width.tolist() == pytest.approx(
        [2 * 1.959963984540054 * 2] * 2
    )
    assert result.predictions.standardized_residual.tolist() == pytest.approx([-0.25] * 16)
    for fold in result.metadata["fold_membership"]:
        train, test = set(fold["training_row_ids"]), set(fold["held_out_row_ids"])
        assert train.isdisjoint(test)
        assert train | test == set(frame.row_id)
        for profile in ("rough", "default"):
            called = calls[(fold["fold"] - 1) * 2 + (profile == "default")]
            assert called[0] == profile
            assert set(called[1].row_id) == train
    pd.testing.assert_frame_equal(frame, original)


def test_folds_stable_under_row_reordering_and_profiles(monkeypatch):
    config, frame = evaluation_data()
    monkeypatch.setattr(models, "fit_gp_model", controlled_fit([]))
    a = model_predictive_evaluation(config, frame, ["default"], seed=42)
    b = model_predictive_evaluation(config, frame.iloc[::-1], ["smooth", "default"], seed=42)
    assert a.metadata["fold_membership"] == b.metadata["fold_membership"]
    assert a.metadata["source_identity"] == b.metadata["source_identity"]
    assert sorted(a.predictions.row_id) == sorted(frame.row_id)
    c = model_predictive_evaluation(config, frame, seed=43)
    assert a.metadata["fold_membership"] != c.metadata["fold_membership"]


@pytest.mark.parametrize("variance", [0, -1, float("nan"), float("inf")])
def test_invalid_variance_withholds_aggregates_and_retains_rows(monkeypatch, variance):
    config, frame = evaluation_data()
    monkeypatch.setattr(models, "fit_gp_model", controlled_fit([], variance=variance))
    result = model_predictive_evaluation(config, frame)
    assert result.summary.fit_status.tolist() == ["incomplete"]
    assert (
        result.summary[["rmse", "mae", "mean_nlpd", "interval_coverage", "mean_interval_width"]]
        .isna()
        .all()
        .all()
    )
    assert len(result.predictions) == len(frame)
    assert result.fold_outcomes.fit_status.eq("failed").all()
    assert result.predictions.predicted_mean.isna().all()
    assert "variance" in result.fold_outcomes.fit_message.iloc[0]


def test_posterior_failure_retains_fit_warning_evidence(monkeypatch):
    config, frame = evaluation_data()

    def posterior(*args, **kwargs):
        raise RuntimeError("Posterior failed after fit")

    metadata = FitMetadata((("fit_warning_count", 2), ("fit_warnings", "fit warnings")))
    monkeypatch.setattr(models, "fit_gp_model", lambda *args: SimpleNamespace(
        posterior=posterior, _bo_forge_fit_metadata=metadata,
    ))
    result = model_predictive_evaluation(config, frame, folds=2)
    assert result.summary.fit_status.tolist() == ["incomplete"]
    assert result.fold_outcomes.fit_warning_count.tolist() == [2, 2]
    assert result.fold_outcomes.fit_warnings.tolist() == ["fit warnings", "fit warnings"]


def test_fit_rng_identity_is_recorded_separately_from_split_seed(monkeypatch):
    config, frame = evaluation_data(5)
    monkeypatch.setattr(models, "fit_gpytorch_mll", lambda mll: torch.rand(1))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        first = model_predictive_evaluation(config, frame, folds=2, seed=0)
        torch.manual_seed(22)
        second = model_predictive_evaluation(config, frame, folds=2, seed=0)
    assert first.metadata["fold_membership"] == second.metadata["fold_membership"]
    assert first.metadata["source_identity"] == second.metadata["source_identity"]
    assert first.metadata["fitting_rng_fingerprints"] != second.metadata["fitting_rng_fingerprints"]
    assert first.fold_outcomes.fitting_rng_fingerprint.str.len().eq(64).all()


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_mean_is_explicit_failure(monkeypatch, value):
    config, frame = evaluation_data()
    monkeypatch.setattr(models, "fit_gp_model", lambda *args: SimpleNamespace(
        posterior=lambda x, **kwargs: SimpleNamespace(
            mean=torch.full((len(x), 1), value), variance=torch.ones((len(x), 1)),
        ),
    ))
    result = model_predictive_evaluation(config, frame, folds=2)
    assert result.summary.fit_status.tolist() == ["incomplete"]
    assert result.fold_outcomes.fit_message.str.contains("finite means").all()


def test_failed_fold_preserves_other_folds_profiles_and_pending_exclusion(monkeypatch):
    config, frame = evaluation_data()
    pending = frame.iloc[:1].copy()
    pending.loc[:, "row_id"] = "pending"
    pending.loc[:, "x"] = 0.99
    pending.loc[:, "status"] = "suggested"
    pending["score"] = ""
    frame = pd.concat([frame, pending], ignore_index=True)
    calls = []
    good = controlled_fit(calls)
    count = 0

    def fit(cfg, training):
        nonlocal count
        count += 1
        assert "pending" not in set(training.row_id)
        if count == 1:
            raise RuntimeError("Synthetic fold failure")
        return good(cfg, training)

    monkeypatch.setattr(models, "fit_gp_model", fit)
    result = model_predictive_evaluation(config, frame, ["default", "smooth"], folds=3)
    assert result.summary.fit_status.tolist() == ["incomplete", "complete"]
    assert pd.isna(result.summary.rmse.iloc[0])
    assert result.summary.completed_folds.tolist() == [2, 3]
    assert len(result.predictions) == 16
    assert "pending" not in set(result.predictions.row_id)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"folds": 1},
        {"folds": 6},
        {"folds": True},
        {"folds": 2.5},
        {"seed": -1},
        {"seed": True},
        {"profiles": []},
        {"profiles": "smooth"},
        {"profiles": ["smooth", "smooth"]},
        {"profiles": ["invalid"]},
    ],
)
def test_invalid_options_fail_before_fitting(monkeypatch, kwargs):
    config, frame = evaluation_data()
    monkeypatch.setattr(models, "fit_gp_model", lambda *args: pytest.fail("fit must not run"))
    with pytest.raises(ConfigError):
        model_predictive_evaluation(config, frame, **kwargs)


@pytest.mark.parametrize("count", [0, 4, 201])
def test_size_bounds(count):
    config, frame = evaluation_data(count)
    with pytest.raises(ConfigError, match="5 to 200"):
        model_predictive_evaluation(config, frame)


@pytest.mark.parametrize("kind", ["context", "replicates", "stages", "fidelity", "objectives"])
def test_unsupported_modes(kind):
    config, frame = evaluation_data()
    updates = {
        "context": {"context": ContextConfig(variables=("x",))},
        "replicates": {"replicates": ReplicateConfig(enabled=True)},
        "stages": {"stages": (StageConfig("one", ("x",)),)},
        "fidelity": {"fidelity": FidelityConfig(variable="x", target=1.0)},
        "objectives": {
            "objectives": (
                ObjectiveConfig("score", "maximize"),
                ObjectiveConfig("other", "maximize"),
            )
        },
    }
    with pytest.raises(ConfigError, match="standard single-objective"):
        model_predictive_evaluation(replace(config, **updates[kind]), frame)


def test_repeated_design_rejected():
    config, frame = evaluation_data()
    frame.loc[1, "x"] = frame.loc[0, "x"]
    with pytest.raises(LogValidationError, match="Repeated design rows"):
        model_predictive_evaluation(config, frame)


def test_default_profile_constant_outcomes_and_exports(monkeypatch, tmp_path):
    config, frame = evaluation_data(5)
    config = replace(config, model=ModelConfig(profile="smooth"))
    frame["score"] = 3.0
    monkeypatch.setattr(models, "fit_gp_model", controlled_fit([]))
    result = model_predictive_evaluation(config, frame, folds=2)
    assert result.summary.model_profile.tolist() == ["smooth"]
    assert np.isfinite(result.summary.rmse.iloc[0])
    destination = result.export(tmp_path / "evaluation")
    assert {p.name for p in destination.iterdir()} == {
        "summary.csv",
        "predictions.csv",
        "fold_outcomes.csv",
        "metadata.json",
    }
    with pytest.raises(FileExistsError):
        result.export(destination)


def test_real_gp_training_only_transforms_and_observation_noise(monkeypatch):
    config, frame = evaluation_data(6, "minimize")
    monkeypatch.setattr(models, "fit_gpytorch_mll", lambda *args: None)
    original = models.fit_gp_model
    observed_calls = []

    def fit(cfg, training):
        model = original(cfg, training)
        assert model.outcome_transform.means.item() == pytest.approx(-training.score.mean())
        assert model.input_transform.bounds[0].item() == pytest.approx(training.x.min())
        assert model.input_transform.bounds[1].item() == pytest.approx(training.x.max())
        x = torch.tensor([[0.4]], dtype=torch.double)
        assert (
            model.posterior(x, observation_noise=True).variance
            > model.posterior(x, observation_noise=False).variance
        ).all()
        observed_calls.append(len(training))
        return model

    monkeypatch.setattr(models, "fit_gp_model", fit)
    result = model_predictive_evaluation(config, frame, folds=3)
    assert result.summary.fit_status.tolist() == ["complete"]
    assert observed_calls == [4, 4, 4]


def test_bounded_real_gp_fit():
    config, frame = evaluation_data(5)
    result = model_predictive_evaluation(config, frame, folds=2)
    assert result.summary.fit_status.tolist() == ["complete"]
    assert result.fold_outcomes.fit_status.tolist() == ["complete", "complete"]
    assert result.summary.completed_folds.tolist() == [2]
    assert sorted(result.predictions.row_id) == sorted(frame.row_id)
    assert np.isfinite(result.predictions.predicted_mean).all()
    assert np.isfinite(result.predictions.predicted_variance).all()
    assert (result.predictions.predicted_variance > 0).all()
    assert np.isfinite(result.summary[[
        "rmse", "mae", "mean_nlpd", "interval_coverage", "mean_interval_width",
    ]].to_numpy(dtype=float)).all()


def test_nonfinite_prediction_and_shape_are_fold_failures(monkeypatch):
    config, frame = evaluation_data()
    for bad_mean in (torch.tensor([[float("inf")]]), torch.zeros((1, 1))):
        monkeypatch.setattr(
            models,
            "fit_gp_model",
            lambda *args, bad_mean=bad_mean: SimpleNamespace(
                posterior=lambda x, **kwargs: SimpleNamespace(
                    mean=bad_mean,
                    variance=torch.ones((len(x), 1)),
                ),
            ),
        )
        result = model_predictive_evaluation(config, frame, folds=2)
        assert result.summary.fit_status.tolist() == ["incomplete"]
        assert result.summary.rmse.isna().all()


def test_maximum_observations_and_source_identity_precision(monkeypatch):
    config, frame = evaluation_data(200)
    monkeypatch.setattr(models, "fit_gp_model", controlled_fit([]))
    result = model_predictive_evaluation(config, frame)
    assert len(result.predictions) == 200
    assert result.summary.completed_folds.tolist() == [5]
    changed = frame.copy(deep=True)
    changed.loc[0, "score"] = np.nextafter(changed.loc[0, "score"], np.inf)
    newer = model_predictive_evaluation(config, changed)
    assert newer.metadata["source_identity"] != result.metadata["source_identity"]


def test_mixed_constraints_review_cost_evaluates_objective_only(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    config = CampaignConfig.from_yaml(root / "configs/07_cost_aware_human_review_logei.yaml")
    path = root / "examples/07_cost_aware_human_review_campaign_log.csv"
    original = path.read_bytes()
    frame = pd.read_csv(path, keep_default_na=False)
    additional = frame.iloc[:1].copy()
    additional["row_id"] = "extra_observed"
    additional["catalyst_loading"] = 0.07
    frame = pd.concat([frame, additional], ignore_index=True)
    seen = []

    def fit(cfg, training):
        assert cfg.cost is not None and cfg.review.enabled and cfg.constraints
        seen.extend(training.row_id.tolist())

        def posterior(x, *, observation_noise):
            assert x.shape[1] == 6 and observation_noise
            return SimpleNamespace(mean=torch.zeros((len(x), 1), dtype=torch.double),
                                   variance=torch.ones((len(x), 1), dtype=torch.double))
        return SimpleNamespace(posterior=posterior)

    monkeypatch.setattr(models, "fit_gp_model", fit)
    result = model_predictive_evaluation(config, frame, folds=2)
    assert result.summary.fit_status.tolist() == ["complete"]
    assert result.metadata["objective_name"] == "yield_score"
    assert set(seen) == set(frame.row_id)
    assert path.read_bytes() == original


def test_result_plots_do_not_fit_and_write_one_file(monkeypatch, tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config, frame = evaluation_data()
    monkeypatch.setattr(models, "fit_gp_model", controlled_fit([]))
    result = model_predictive_evaluation(config, frame, ["rough", "default"], folds=2)
    monkeypatch.setattr(models, "fit_gp_model", lambda *args: pytest.fail("plot refit"))
    for method in ("plot_predictions", "plot_residuals"):
        path = tmp_path / f"{method}.png"
        fig, axes = getattr(result, method)(save_path=path)
        try:
            assert path.is_file()
            assert len(axes) == 2
            assert all("held-out" in ax.get_title() for ax in axes)
            assert all("95% interval coverage" in ax.texts[0].get_text() for ax in axes)
            assert all(spine.get_linewidth() == 1.8 for ax in axes for spine in ax.spines.values())
        finally:
            plt.close(fig)
    assert len(list(tmp_path.iterdir())) == 2
