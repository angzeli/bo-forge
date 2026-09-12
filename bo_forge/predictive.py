"""Explicit retrospective held-out evaluation, separate from campaign suggestions."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import shutil
from copy import deepcopy
from dataclasses import dataclass, replace
from importlib.metadata import version
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import pandas as pd

from bo_forge._filesystem import rename_directory_exclusive
from bo_forge._fit_metadata import config_identity
from bo_forge.config import CampaignConfig, ModelConfig
from bo_forge.errors import ConfigError, LogConflictError
from bo_forge.validation import design_tuples, get_observed_data, validate_campaign_data

_PROFILES = ("default", "smooth", "rough", "robust")
_Z95 = 1.959963984540054
_METRICS = ("rmse", "mae", "mean_nlpd", "interval_coverage", "mean_interval_width")
_PREDICTIONS = (
    "predicted_mean",
    "predicted_variance",
    "predicted_std",
    "residual",
    "standardized_residual",
    "negative_log_predictive_density",
    "interval_lower",
    "interval_upper",
    "interval_covered",
)


@dataclass(frozen=True)
class PredictiveEvaluationResult:
    """Completed evaluation tables; plotting and export never fit another model.

    Predictions and intervals are in original objective units. Variance includes
    inferred observation noise, rather than only latent-function uncertainty.
    """

    summary: pd.DataFrame
    predictions: pd.DataFrame
    fold_outcomes: pd.DataFrame
    metadata: dict

    def export(self, output_dir: str | Path) -> Path:
        """Publish a complete table bundle atomically; refuse overwrites."""
        destination = Path(output_dir)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(
                f"Predictive evaluation destination already exists: {destination}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(mkdtemp(prefix=".evaluation.preparing-", dir=destination.parent))
        try:
            for name in ("summary", "predictions", "fold_outcomes"):
                getattr(self, name).to_csv(temporary / f"{name}.csv", index=False)
            (temporary / "metadata.json").write_text(
                json.dumps(self.metadata, sort_keys=True, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            rename_directory_exclusive(temporary, destination)
        except BaseException:
            # Rollback owns only the unpublished temporary directory, never the destination.
            try:
                shutil.rmtree(temporary)
            except OSError:
                pass  # Cleanup must not replace the serialization/publication failure.
            raise
        return destination

    def plot_predictions(self, *, save_path: str | Path | None = None):
        """Plot observed versus held-out predicted outcomes without refitting."""
        from bo_forge._diagnostics.predictive import plot_predictions

        return plot_predictions(self, save_path=save_path)

    def plot_residuals(self, *, save_path: str | Path | None = None):
        """Plot observation-standardized held-out residuals without refitting."""
        from bo_forge._diagnostics.predictive import plot_residuals

        return plot_residuals(self, save_path=save_path)


def model_predictive_evaluation(
    config: CampaignConfig,
    df: pd.DataFrame,
    profiles: list[str] | tuple[str, ...] | None = None,
    *,
    folds: int = 5,
    seed: int = 0,
) -> PredictiveEvaluationResult:
    """Evaluate shuffled K-fold predictions using fresh training-fold transforms.

    Supports 5..200 observed, independent designs and 2..5 folds. These are
    retrospective checks on adaptively collected data, not estimates of future
    optimization performance or evidence of calibrated uncertainty.
    """
    config = deepcopy(config)
    names, observed = _prepare_evaluation(config, df.copy(deep=True), profiles, folds, seed)
    groups = np.array_split(np.random.default_rng(seed).permutation(len(observed)), folds)
    membership = []
    predictions, outcomes = [], []
    for fold, test_indices in enumerate(groups, start=1):
        test_indices = np.sort(test_indices)
        train_indices = np.setdiff1d(np.arange(len(observed)), test_indices)
        training, held_out = observed.iloc[train_indices], observed.iloc[test_indices]
        membership.append(
            {
                "fold": fold,
                "training_row_ids": training.row_id.astype(str).tolist(),
                "held_out_row_ids": held_out.row_id.astype(str).tolist(),
            }
        )
        for profile in names:
            profile_config = replace(config, model=ModelConfig(profile=profile))
            rows, outcome = _evaluate_fold(profile_config, training, held_out, fold)
            predictions.extend(rows)
            outcomes.append(outcome)
    prediction_frame = _profile_order(pd.DataFrame(predictions), names)
    outcome_frame = _profile_order(pd.DataFrame(outcomes), names)
    summary = pd.DataFrame(
        [
            _profile_summary(profile, prediction_frame, outcome_frame, len(observed))
            for profile in names
        ]
    )
    metadata = _evaluation_metadata(config, observed, names, folds, seed, membership)
    metadata["fitting_rng_fingerprints"] = outcome_frame[
        ["model_profile", "fold", "fitting_rng_fingerprint"]
    ].to_dict(orient="records")
    return PredictiveEvaluationResult(summary, prediction_frame, outcome_frame, metadata)


def _session_predictive_evaluation(session, profiles, folds, seed):
    """Guard file-loaded evaluations without imposing filesystem state on standalone calls."""
    session._assert_provenance_resumable()
    before = _session_sources(session)
    if before is not None and before[-3:] != (
        session.config_fingerprint, session.log_fingerprint, session._manifest_fingerprint,
    ):
        raise LogConflictError(
            "Campaign changed after loading. Reload before predictive evaluation."
        )
    result = model_predictive_evaluation(
        session.config, session.df, profiles=profiles, folds=folds, seed=seed,
    )
    if _session_sources(session) != before:
        raise LogConflictError(
            "Campaign changed during predictive evaluation. Reload and run again."
        )
    session._assert_provenance_resumable()
    return result


def _session_sources(session):
    from bo_forge._campaign.provenance import manifest_fingerprint
    from bo_forge.logs import _MISSING_LOG_FINGERPRINT, _log_file_fingerprint

    if session._provenance_managed is None and session.config_fingerprint is None:
        return None
    paths = tuple(Path(path).expanduser().resolve() for path in (
        session.config_path, session.log_path,
    ))
    return (
        paths, session._provenance_policy,
        _log_file_fingerprint(paths[0]),
        _log_file_fingerprint(paths[1]) or _MISSING_LOG_FINGERPRINT,
        manifest_fingerprint(paths[1]),
    )


def _prepare_evaluation(config, df, profiles, folds, seed):
    unsupported = (
        config.is_multi_objective
        or config.is_structured_campaign
        or config.fidelity is not None
        or config.context is not None
        or config.replicates.enabled
    )
    if unsupported:
        raise ConfigError(
            "Predictive evaluation supports standard single-objective campaigns only; "
            "context, replicates, stages, fidelity, and multi-objective are unsupported."
        )
    if isinstance(folds, bool) or not isinstance(folds, int) or not 2 <= folds <= 5:
        raise ConfigError("Predictive evaluation requires an integer fold count from 2 to 5.")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ConfigError("Predictive evaluation seed must be a nonnegative integer.")
    names = _requested_profiles(config, profiles)
    validate_campaign_data(config, df)
    observed = get_observed_data(config, df).copy(deep=True)
    observed = observed.sort_values("row_id", key=lambda s: s.astype(str)).reset_index(drop=True)
    count = len(observed)
    if not 5 <= count <= 200:
        raise ConfigError("Predictive evaluation requires 5 to 200 observed rows; no subsampling.")
    if count - math.ceil(count / folds) < 2:
        raise ConfigError("Predictive evaluation requires at least two training rows per fold.")
    if len(design_tuples(config, observed)) != count:
        raise ConfigError("Predictive evaluation does not support repeated observed designs.")
    return names, observed


def _requested_profiles(config, profiles):
    if isinstance(profiles, str):
        raise ConfigError("profiles must be a sequence of profile names, not a string.")
    names = list([config.model.profile] if profiles is None else profiles)
    if not names or len(names) > 4:
        raise ConfigError("Request between one and four model profiles.")
    for index, profile in enumerate(names):
        if profile not in _PROFILES:
            raise ConfigError(f"Unknown model profile '{profile}'. Expected one of {_PROFILES}.")
        if profile in names[:index]:
            raise ConfigError(f"Duplicate model profile requested: {profile}.")
    return names


def _evaluate_fold(config, training, held_out, fold):
    from bo_forge.models import _comparison_fit_metadata, fit_gp_model

    profile = config.model.profile
    outcome = {
        "model_profile": profile,
        "fold": fold,
        "training_rows": len(training),
        "held_out_rows": len(held_out),
        "fit_status": "complete",
        "fit_message": "",
        "fit_warning_count": 0,
    }
    owner = None
    try:
        owner = fit_gp_model(config, training.copy(deep=True))
        rows = _held_out_predictions(config, owner, held_out, fold)
    except Exception as exc:
        # Diagnostic isolation: retain every failed fold without hiding other outcomes.
        if owner is None:
            owner = exc
        outcome.update(fit_status="failed", fit_message=str(exc) or type(exc).__name__)
        rows = [
            _failed_prediction(config, row, fold, outcome["fit_message"])
            for _, row in held_out.iterrows()
        ]
    evidence = _comparison_fit_metadata(owner)
    outcome["fit_warning_count"] = evidence.get("fit_warning_count", 0)
    outcome["fit_warnings"] = evidence.get("fit_warnings", "")
    outcome["fitting_rng_fingerprint"] = evidence.get("fitting_rng_fingerprint")
    return rows, outcome


def _held_out_predictions(config, model, held_out, fold):
    import torch

    from bo_forge.transforms import dataframe_to_unit_cube, objective_from_model_space

    # Encoding uses configured bounds/categories; learned transforms belong to this model.
    x = dataframe_to_unit_cube(config, held_out)
    with torch.no_grad():
        posterior = model.posterior(x, observation_noise=True)
        mean = objective_from_model_space(config, posterior.mean.detach()).cpu().numpy().reshape(-1)
        variance = posterior.variance.detach().cpu().numpy().reshape(-1)
    if len(mean) != len(held_out) or len(variance) != len(held_out):
        raise ValueError("Predictive posterior shape does not match the held-out rows.")
    if not np.isfinite(mean).all() or not np.isfinite(variance).all() or (variance <= 0).any():
        raise ValueError("Held-out predictions require finite means and positive finite variance.")
    rows = []
    for (_, row), predicted, var in zip(held_out.iterrows(), mean, variance, strict=True):
        record = _failed_prediction(config, row, fold, "")
        record.update(_prediction_metrics(float(row[config.objective.name]), predicted, var))
        record["fit_status"] = "complete"
        rows.append(record)
    if not np.isfinite(pd.DataFrame(rows)[list(_PREDICTIONS)].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite held-out residual or interval metric.")
    return rows


def _prediction_metrics(observed, predicted, variance):
    std = float(np.sqrt(variance))
    residual = float(observed - predicted)
    standardized = residual / std
    # Rounded original-unit endpoints can coincide even with positive uncertainty.
    lower, upper = float(predicted - _Z95 * std), float(predicted + _Z95 * std)
    return {
        "predicted_mean": float(predicted),
        "predicted_variance": float(variance),
        "predicted_std": std,
        "residual": residual,
        "standardized_residual": standardized,
        "negative_log_predictive_density": float(
            0.5 * (np.log(2 * np.pi) + np.log(variance)) + (standardized / np.sqrt(2.0))**2
        ),
        "interval_lower": lower,
        "interval_upper": upper,
        "interval_covered": bool(abs(standardized) <= _Z95),
    }


def _failed_prediction(config, row, fold, message):
    return {
        "model_profile": config.model.profile,
        "row_id": str(row["row_id"]),
        "fold": fold,
        "observed": float(row[config.objective.name]),
        **dict.fromkeys(_PREDICTIONS, None),
        "fit_status": "failed",
        "fit_message": message,
    }


def _profile_order(frame, names):
    order = {name: index for index, name in enumerate(names)}
    return frame.sort_values(
        "model_profile", key=lambda s: s.map(order), kind="stable"
    ).reset_index(drop=True)


def _profile_summary(profile, predictions, outcomes, count):
    rows = predictions.loc[predictions.model_profile == profile]
    folds = outcomes.loc[outcomes.model_profile == profile]
    complete = bool(folds.fit_status.eq("complete").all())
    metrics = dict.fromkeys(_METRICS, None)
    failed = folds.loc[folds.fit_status.ne("complete")]
    message = "; ".join(f"Fold {row.fold}: {row.fit_message}" for row in failed.itertuples())
    if complete:
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                metrics = _aggregate_metrics(rows)
        except (FloatingPointError, OverflowError, ValueError) as exc:
            complete, metrics = False, dict.fromkeys(_METRICS, None)
            message = f"Aggregate metrics failed: {exc}"
    return {
        "model_profile": profile,
        "evaluation_scope": "out_of_fold",
        "fit_status": "complete" if complete else "incomplete",
        "observed_rows": count,
        "completed_folds": int(folds.fit_status.eq("complete").sum()),
        "total_folds": len(folds),
        **metrics,
        "fit_message": message,
    }


def _finite_mean(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("non-finite values in aggregate inputs")
    scale = float(np.max(np.abs(values)))
    return float(np.mean(values / scale) * scale) if scale else 0.0


def _aggregate_metrics(rows):
    residual = rows.residual.to_numpy(dtype=float)
    scale = float(np.max(np.abs(residual)))
    # Scale before squaring/summing: finite large residuals can have a finite RMS.
    metrics = dict(
        rmse=float(scale * np.sqrt(_finite_mean((residual / scale)**2))) if scale else 0.0,
        mae=_finite_mean(np.abs(residual)),
        mean_nlpd=_finite_mean(rows.negative_log_predictive_density),
        interval_coverage=_finite_mean(rows.interval_covered.astype(float)),
        mean_interval_width=_finite_mean(2 * _Z95 * rows.predicted_std.to_numpy(dtype=float)),
    )
    if not all(np.isfinite(value) for value in metrics.values()):
        raise ValueError("non-finite aggregate metrics")
    return metrics


def _evaluation_metadata(config, observed, profiles, folds, seed, membership):
    from bo_forge import __version__

    fields = ["row_id", *config.variable_names, config.objective.name]
    serialized = json.dumps(
        observed[fields].to_dict(orient="records"), sort_keys=True, allow_nan=False
    )
    observation_id = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    configuration_id = config_identity(config)
    return {
        "evaluation_scope": "out_of_fold",
        "method": "shuffled_k_fold",
        "campaign_name": config.campaign_name,
        "objective_name": config.objective.name,
        "objective_direction": config.objective.direction,
        "profiles": profiles,
        "folds": folds,
        "seed": seed,
        "requested_fit_count": folds * len(profiles),
        "observed_rows": len(observed),
        "fold_membership": membership,
        "config_identity": configuration_id,
        "observation_identity": observation_id,
        "source_identity": hashlib.sha256((configuration_id + observation_id).encode()).hexdigest(),
        "prediction_units": "original_objective",
        "variance": "observation_inclusive",
        "interval_probability": 0.95,
        "software_versions": {
            "bo-forge": __version__,
            "python": platform.python_version(),
            **{name: version(name) for name in (
                "torch", "botorch", "gpytorch", "numpy", "pandas", "scipy",
            )},
        },
        "interpretation": (
            "Retrospective checks on adaptively collected data; not proof of future "
            "optimization performance or calibrated uncertainty."
        ),
    }
