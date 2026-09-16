"""Known-distribution diagnostics, not calibration evidence for learned profiles."""

import math
from statistics import NormalDist
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

import bo_forge.models as models
from bo_forge import __version__, model_predictive_evaluation
from tests.test_predictive_evaluation import evaluation_data

SCENARIOS = {
    "matched": (1.0, 0.0),
    "too_narrow": (0.5, 0.0),
    "too_wide": (2.0, 0.0),
    "biased": (1.0, 2.0),
}
PROFILES = ["robust", "smooth"]


def _known_distribution(direction="maximize"):
    config, frame = evaluation_data(200, direction)
    # Unique designs with mean 3 + 2*x and observation variance 4 in objective units.
    quantiles = [NormalDist().inv_cdf((i + 0.5) / 200) for i in range(200)]
    frame["score"] = [3 + 2 * x + 2 * z for x, z in zip(frame.x, quantiles, strict=True)]
    return config, frame


def _known_fitter(calls, scale=1.0, bias=0.0):
    def fit(config, training):
        calls.append((config.model.profile, training.row_id.tolist()))

        def posterior(x, *, observation_noise):
            assert observation_noise is True
            assert x.dtype == torch.double
            sign = 1 if config.objective.direction == "maximize" else -1
            return SimpleNamespace(
                mean=sign * (3 + 2 * x[:, :1] + 2 * bias),
                variance=torch.full((len(x), 1), (2 * scale)**2, dtype=torch.double),
            )

        return SimpleNamespace(posterior=posterior)

    return fit


def _expected_rows(frame, scale, bias):
    critical = NormalDist().inv_cdf(0.975)
    std = 2 * scale
    rows = []
    for row in frame.itertuples():
        mean = 3 + 2 * row.x + 2 * bias
        residual = row.score - mean
        standardized = residual / std
        rows.append({
            "predicted_mean": mean, "predicted_variance": std**2, "predicted_std": std,
            "residual": residual, "standardized_residual": standardized,
            "negative_log_predictive_density": (
                math.log(std * math.sqrt(2 * math.pi)) + standardized**2 / 2
            ),
            "interval_lower": mean - critical * std, "interval_upper": mean + critical * std,
            "interval_covered": abs(standardized) <= critical,
        })
    return rows


def _assert_expected(result, frame, scale, bias):
    rows = _expected_rows(frame, scale, bias)
    count = len(rows)
    expected = {
        "rmse": math.sqrt(math.fsum(row["residual"]**2 for row in rows) / count),
        "mae": math.fsum(abs(row["residual"]) for row in rows) / count,
        "mean_nlpd": math.fsum(row["negative_log_predictive_density"] for row in rows) / count,
        "interval_coverage": sum(row["interval_covered"] for row in rows) / count,
        "mean_interval_width": 2 * NormalDist().inv_cdf(0.975) * 2 * scale,
    }
    for profile in PROFILES:
        summary = result.summary.set_index("model_profile").loc[profile]
        assert summary.fit_status == "complete"
        assert summary.completed_folds == summary.total_folds == 5
        for name, value in expected.items():
            assert summary[name] == pytest.approx(value, rel=1e-12, abs=1e-12)
        predictions = result.predictions.loc[
            result.predictions.model_profile.eq(profile)
        ].sort_values("row_id")
        assert predictions.row_id.tolist() == frame.row_id.tolist()
        assert predictions.fit_status.eq("complete").all()
        assert predictions.observed.tolist() == pytest.approx(frame.score.tolist())
        for name in rows[0]:
            values = [row[name] for row in rows]
            if name == "interval_covered":
                assert predictions[name].tolist() == values
            else:
                assert predictions[name].tolist() == pytest.approx(values, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("direction", ["maximize", "minimize"])
def test_known_gaussian_distributions_through_public_evaluator(monkeypatch, direction):
    config, frame = _known_distribution(direction)
    before = frame.copy(deep=True)
    results = {}
    for scenario, (scale, bias) in SCENARIOS.items():
        calls = []
        monkeypatch.setattr(models, "fit_gp_model", _known_fitter(calls, scale, bias))
        result = model_predictive_evaluation(config, frame, PROFILES, folds=5, seed=17)
        results[scenario] = result
        _assert_expected(result, frame, scale, bias)
        assert result.summary.model_profile.tolist() == PROFILES
        metadata = result.metadata
        assert metadata["profiles"] == PROFILES
        assert metadata["seed"] == 17 and metadata["folds"] == 5
        assert metadata["observed_rows"] == 200 and metadata["requested_fit_count"] == 10
        assert metadata["prediction_units"] == "original_objective"
        assert metadata["variance"] == "observation_inclusive"
        assert metadata["interval_probability"] == 0.95
        assert metadata["software_versions"]["bo-forge"] == __version__
        assert metadata["objective_direction"] == direction
        assert "not proof" in metadata["interpretation"]
        held_out = []
        for fold in metadata["fold_membership"]:
            train, test = fold["training_row_ids"], fold["held_out_row_ids"]
            assert set(train).isdisjoint(test)
            assert set(train) | set(test) == set(frame.row_id)
            held_out.extend(test)
            offset = (fold["fold"] - 1) * 2
            assert calls[offset:offset + 2] == [(profile, train) for profile in PROFILES]
            for profile in PROFILES:
                actual = result.predictions.loc[
                    result.predictions.model_profile.eq(profile)
                    & result.predictions.fold.eq(fold["fold"]), "row_id"
                ].tolist()
                assert actual == test
        assert len(calls) == 10
        assert sorted(held_out) == sorted(frame.row_id)
        pd.testing.assert_frame_equal(frame, before)
    reference = results["matched"].summary.iloc[0]
    assert reference.interval_coverage == 0.95
    narrow, wide, biased = [results[name].summary.iloc[0]
                            for name in ("too_narrow", "too_wide", "biased")]
    assert narrow.interval_coverage < reference.interval_coverage < wide.interval_coverage
    assert wide.interval_coverage == 1.0
    assert narrow.mean_interval_width < reference.mean_interval_width < wide.mean_interval_width
    assert biased.rmse > reference.rmse and biased.mae > reference.mae
    for scenario in ("too_narrow", "too_wide"):
        other = results[scenario]
        assert other.summary.rmse.tolist() == pytest.approx(results["matched"].summary.rmse)
        assert other.summary.mae.tolist() == pytest.approx(results["matched"].summary.mae)
        assert other.summary.mean_nlpd.iloc[0] != pytest.approx(reference.mean_nlpd)
        scale, _ = SCENARIOS[scenario]
        assert other.predictions.standardized_residual.tolist() == pytest.approx(
            results["matched"].predictions.standardized_residual / scale,
        )
        assert other.metadata == results["matched"].metadata


def test_direction_row_and_profile_order_do_not_reinterpret_diagnostic_values(monkeypatch):
    monkeypatch.setattr(models, "fit_gp_model", _known_fitter([]))
    config, frame = _known_distribution()
    baseline = model_predictive_evaluation(config, frame, PROFILES, seed=17)
    other_config, other_frame = _known_distribution("minimize")
    reordered = model_predictive_evaluation(other_config, other_frame.iloc[::-1],
                                            list(reversed(PROFILES)), seed=17)
    assert reordered.summary.model_profile.tolist() == list(reversed(PROFILES))
    assert baseline.metadata["fold_membership"] == reordered.metadata["fold_membership"]
    assert baseline.metadata["observation_identity"] == reordered.metadata["observation_identity"]
    assert baseline.metadata["config_identity"] != reordered.metadata["config_identity"]
    assert baseline.metadata["source_identity"] != reordered.metadata["source_identity"]
    for name, keys in (("summary", ["model_profile"]),
                       ("predictions", ["model_profile", "row_id"]),
                       ("fold_outcomes", ["model_profile", "fold"])):
        pd.testing.assert_frame_equal(
            getattr(baseline, name).sort_values(keys).reset_index(drop=True),
            getattr(reordered, name).sort_values(keys).reset_index(drop=True),
        )
