"""Model fitting helpers."""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass, replace

import pandas as pd
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

from bo_forge.config import CampaignConfig, ModelConfig
from bo_forge.errors import ConfigError
from bo_forge.multi_objective import objectives_to_model_space
from bo_forge.multifidelity import fidelity_feature_index
from bo_forge.replicates import modeling_observed_data_with_variance
from bo_forge.transforms import (
    dataframe_to_unit_cube,
    encoded_dimension,
    objective_to_model_space,
)
from bo_forge.validation import get_observed_data, validate_campaign_data

_LAST_FIT_METADATA: dict[str, object] = {}
_DEFAULT_COMPARISON_PROFILES = ("default", "smooth", "rough", "robust")
_MODEL_PROFILE_COMPARISON_COLUMNS = [
    "model_profile",
    "model_class",
    "covariance_profile",
    "fit_status",
    "fit_message",
    "fit_warning_count",
    "observed_rows_used_for_fitting",
    "encoded_dimension",
    "train_yvar_used",
    "rmse_model_space",
    "mae_model_space",
    "mean_predicted_std",
]


@dataclass(frozen=True)
class TrainingTensors:
    train_x: torch.Tensor
    train_y: torch.Tensor
    train_yvar: torch.Tensor | None = None


def dataframe_to_tensors(
    config: CampaignConfig, observed_df: pd.DataFrame
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert observed campaign rows to model-space tensors."""
    tensors = dataframe_to_training_tensors(config, observed_df)
    return tensors.train_x, tensors.train_y


def dataframe_to_training_tensors(
    config: CampaignConfig,
    observed_df: pd.DataFrame,
) -> TrainingTensors:
    """Convert observed rows to model-space tensors plus optional observation variance."""
    model_df, yvar_df = modeling_observed_data_with_variance(config, observed_df)
    if config.is_multi_objective:
        y_user = torch.tensor(
            model_df[config.objective_names].astype(float).to_numpy(),
            dtype=torch.double,
        )
        train_y = objectives_to_model_space(config, y_user)
        train_yvar = _yvar_tensor(yvar_df, config.objective_names)
        return TrainingTensors(dataframe_to_unit_cube(config, model_df), train_y, train_yvar)
    y_user = torch.tensor(
        model_df[[config.objective.name]].astype(float).to_numpy(),
        dtype=torch.double,
    )
    train_y = objective_to_model_space(config, y_user)
    train_yvar = _yvar_tensor(yvar_df, [config.objective.name])
    return TrainingTensors(dataframe_to_unit_cube(config, model_df), train_y, train_yvar)


def fit_gp_model(config: CampaignConfig, observed_df: pd.DataFrame) -> SingleTaskGP:
    """Fit a standard BoTorch SingleTaskGP to observed campaign data."""
    training = dataframe_to_training_tensors(config, observed_df)
    kwargs = {}
    if training.train_yvar is not None:
        kwargs["train_Yvar"] = training.train_yvar
    covar_module = _covar_module_for_profile(config, training.train_x.shape[-1])
    if covar_module is not None:
        kwargs["covar_module"] = covar_module
    model = SingleTaskGP(
        training.train_x,
        training.train_y,
        input_transform=Normalize(d=training.train_x.shape[-1]),
        outcome_transform=Standardize(m=training.train_y.shape[-1]),
        **kwargs,
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    _fit_mll_with_profile_metadata(config, mll, training, model_class="SingleTaskGP")
    return model


def fit_multi_fidelity_gp_model(
    config: CampaignConfig,
    observed_df: pd.DataFrame,
) -> SingleTaskMultiFidelityGP:
    """Fit BoTorch's single-task multi-fidelity GP to observed campaign data."""
    training = dataframe_to_training_tensors(config, observed_df)
    kwargs = {}
    if training.train_yvar is not None:
        kwargs["train_Yvar"] = training.train_yvar
    model = SingleTaskMultiFidelityGP(
        training.train_x,
        training.train_y,
        data_fidelities=[fidelity_feature_index(config)],
        outcome_transform=Standardize(m=training.train_y.shape[-1]),
        **kwargs,
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    _record_fit_metadata(
        config,
        training,
        model_class="SingleTaskMultiFidelityGP",
        covariance_profile="multi_fidelity",
        fit_status="ok",
        fit_warnings=[],
        fallback_status="not_needed",
    )
    return model


def model_summary(config: CampaignConfig, df: pd.DataFrame) -> pd.DataFrame:
    """Return read-only model-profile and fitting-input summary fields."""
    validate_campaign_data(config, df)
    observed = get_observed_data(config, df)
    model_class = _model_class_name(config)
    covariance_profile = _covariance_profile_name(config)
    encoded_dim = encoded_dimension(config)
    objective_count = len(config.objective_names)
    observed_rows_used = 0
    train_yvar_used = False
    training_fingerprint = None
    if not observed.empty:
        if config.is_structured_campaign:
            observed_rows_used = int(len(observed))
        else:
            training = dataframe_to_training_tensors(config, observed)
            observed_rows_used = int(training.train_x.shape[0])
            train_yvar_used = training.train_yvar is not None
            training_fingerprint = _training_fingerprint(training)

    metadata = _matching_last_fit_metadata(
        config,
        {
            "model_class": model_class,
            "covariance_profile": covariance_profile,
            "encoded_dimension": encoded_dim,
            "observed_rows_used_for_fitting": observed_rows_used,
            "objective_count": objective_count,
            "train_yvar_used": train_yvar_used,
            "training_fingerprint": training_fingerprint,
        },
    )
    rows = [
        ("model_profile", config.model.profile),
        ("model_class", model_class),
        ("covariance_profile", covariance_profile),
        ("encoded_dimension", encoded_dim),
        ("observed_rows_used_for_fitting", observed_rows_used),
        ("objective_count", objective_count),
        ("train_yvar_used", train_yvar_used),
        ("last_fit_status", metadata.get("fit_status", "not_recorded")),
        ("last_fit_warning_count", metadata.get("fit_warning_count", 0)),
        ("last_fit_warnings", metadata.get("fit_warnings", "")),
        ("fallback_status", metadata.get("fallback_status", "not_recorded")),
    ]
    return pd.DataFrame(rows, columns=["field", "value"])


def model_profile_comparison(
    config: CampaignConfig,
    df: pd.DataFrame,
    profiles: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Compare model-profile fit diagnostics on the current fitting rows."""
    _validate_model_profile_comparison_supported(config)
    profile_names = _normalise_comparison_profiles(profiles)
    validate_campaign_data(config, df)
    observed = get_observed_data(config, df)
    model_class = _model_class_name(config)
    encoded_dim = encoded_dimension(config)
    training: TrainingTensors | None = None
    observed_rows_used = 0
    train_yvar_used = False
    if not observed.empty:
        training = dataframe_to_training_tensors(config, observed)
        observed_rows_used = int(training.train_x.shape[0])
        train_yvar_used = training.train_yvar is not None

    if training is None or observed_rows_used < 2:
        return pd.DataFrame(
            [
                _comparison_row(
                    config,
                    profile,
                    model_class=model_class,
                    encoded_dim=encoded_dim,
                    observed_rows_used=observed_rows_used,
                    train_yvar_used=train_yvar_used,
                    fit_status="insufficient_observed",
                    fit_message="At least two fitting rows are required.",
                )
                for profile in profile_names
            ],
            columns=_MODEL_PROFILE_COMPARISON_COLUMNS,
        )

    rows: list[dict[str, object]] = []
    previous_metadata = dict(_LAST_FIT_METADATA)
    try:
        for profile in profile_names:
            profile_config = replace(config, model=ModelConfig(profile=profile))
            try:
                model = fit_gp_model(profile_config, observed)
                metadata = _comparison_fit_metadata(profile)
                posterior = model.posterior(training.train_x)
                predicted = posterior.mean.detach().reshape(-1)
                observed_model = training.train_y.detach().reshape(-1)
                predicted_std = (
                    posterior.variance.detach().clamp_min(0.0).sqrt().reshape(-1)
                )
                residual = observed_model - predicted
                rmse = float(torch.sqrt(torch.mean(residual.square())).item())
                mae = float(torch.mean(torch.abs(residual)).item())
                mean_std = float(torch.mean(predicted_std).item())
                rows.append(
                    _comparison_row(
                        profile_config,
                        profile,
                        model_class=model_class,
                        encoded_dim=encoded_dim,
                        observed_rows_used=observed_rows_used,
                        train_yvar_used=train_yvar_used,
                        fit_status=str(metadata.get("fit_status", "ok")),
                        fit_warning_count=int(metadata.get("fit_warning_count", 0)),
                        rmse_model_space=rmse,
                        mae_model_space=mae,
                        mean_predicted_std=mean_std,
                    )
                )
            except Exception as exc:
                metadata = _comparison_fit_metadata(profile)
                rows.append(
                    _comparison_row(
                        profile_config,
                        profile,
                        model_class=model_class,
                        encoded_dim=encoded_dim,
                        observed_rows_used=observed_rows_used,
                        train_yvar_used=train_yvar_used,
                        fit_status="failed",
                        fit_message=str(exc) or exc.__class__.__name__,
                        fit_warning_count=int(metadata.get("fit_warning_count", 0)),
                    )
                )
    finally:
        _LAST_FIT_METADATA.clear()
        _LAST_FIT_METADATA.update(previous_metadata)

    return pd.DataFrame(rows, columns=_MODEL_PROFILE_COMPARISON_COLUMNS)


def _validate_model_profile_comparison_supported(config: CampaignConfig) -> None:
    if config.is_multi_objective:
        raise ConfigError("model_profile_comparison() requires a single-objective config.")
    if config.fidelity is not None:
        raise ConfigError(
            "model_profile_comparison() does not support multi-fidelity configs."
        )
    if config.is_structured_campaign:
        raise ConfigError(
            "model_profile_comparison() does not support structured configs."
        )


def _normalise_comparison_profiles(
    profiles: list[str] | tuple[str, ...] | None,
) -> list[str]:
    requested = list(_DEFAULT_COMPARISON_PROFILES if profiles is None else profiles)
    if not requested:
        raise ConfigError("model_profile_comparison() requires at least one profile.")
    normalised: list[str] = []
    for profile in requested:
        if profile not in _DEFAULT_COMPARISON_PROFILES:
            choices = ", ".join(_DEFAULT_COMPARISON_PROFILES)
            raise ConfigError(f"Unknown model profile '{profile}'. Expected one of: {choices}.")
        if profile in normalised:
            raise ConfigError(f"Duplicate model profile requested: {profile}.")
        normalised.append(profile)
    return normalised


def _comparison_fit_metadata(profile: str) -> dict[str, object]:
    if _LAST_FIT_METADATA.get("profile") != profile:
        return {}
    return dict(_LAST_FIT_METADATA)


def _comparison_row(
    config: CampaignConfig,
    profile: str,
    *,
    model_class: str,
    encoded_dim: int,
    observed_rows_used: int,
    train_yvar_used: bool,
    fit_status: str,
    fit_message: str = "",
    fit_warning_count: int = 0,
    rmse_model_space: float = float("nan"),
    mae_model_space: float = float("nan"),
    mean_predicted_std: float = float("nan"),
) -> dict[str, object]:
    return {
        "model_profile": profile,
        "model_class": model_class,
        "covariance_profile": _covariance_profile_name(config),
        "fit_status": fit_status,
        "fit_message": fit_message,
        "fit_warning_count": fit_warning_count,
        "observed_rows_used_for_fitting": observed_rows_used,
        "encoded_dimension": encoded_dim,
        "train_yvar_used": train_yvar_used,
        "rmse_model_space": rmse_model_space,
        "mae_model_space": mae_model_space,
        "mean_predicted_std": mean_predicted_std,
    }


def _covar_module_for_profile(config: CampaignConfig, dimension: int):
    profile = config.model.profile
    if profile in {"default", "robust"}:
        return None
    if profile == "smooth":
        return ScaleKernel(RBFKernel(ard_num_dims=dimension))
    if profile == "rough":
        return ScaleKernel(MaternKernel(nu=1.5, ard_num_dims=dimension))
    raise AssertionError(f"Unexpected model profile: {profile}")


def _fit_mll_with_profile_metadata(
    config: CampaignConfig,
    mll: ExactMarginalLogLikelihood,
    training: TrainingTensors,
    *,
    model_class: str,
) -> None:
    covariance_profile = _covariance_profile_name(config)
    if config.model.profile != "robust":
        fit_gpytorch_mll(mll)
        _record_fit_metadata(
            config,
            training,
            model_class=model_class,
            covariance_profile=covariance_profile,
            fit_status="ok",
            fit_warnings=[],
            fallback_status="not_needed",
        )
        return

    caught_warnings: list[str] = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit_gpytorch_mll(mll)
            caught_warnings = [str(item.message) for item in caught]
    except Exception:
        _record_fit_metadata(
            config,
            training,
            model_class=model_class,
            covariance_profile=covariance_profile,
            fit_status="failed",
            fit_warnings=caught_warnings,
            fallback_status="raised",
        )
        raise

    _record_fit_metadata(
        config,
        training,
        model_class=model_class,
        covariance_profile=covariance_profile,
        fit_status="ok_with_warnings" if caught_warnings else "ok",
        fit_warnings=caught_warnings,
        fallback_status="not_needed",
    )


def _record_fit_metadata(
    config: CampaignConfig,
    training: TrainingTensors,
    *,
    model_class: str,
    covariance_profile: str,
    fit_status: str,
    fit_warnings: list[str],
    fallback_status: str,
) -> None:
    _LAST_FIT_METADATA.clear()
    _LAST_FIT_METADATA.update(
        {
            "campaign_name": config.campaign_name,
            "profile": config.model.profile,
            "model_class": model_class,
            "covariance_profile": covariance_profile,
            "encoded_dimension": int(training.train_x.shape[-1]),
            "observed_rows_used_for_fitting": int(training.train_x.shape[0]),
            "objective_count": int(training.train_y.shape[-1]),
            "train_yvar_used": training.train_yvar is not None,
            "training_fingerprint": _training_fingerprint(training),
            "fit_status": fit_status,
            "fit_warning_count": len(fit_warnings),
            "fit_warnings": "; ".join(fit_warnings),
            "fallback_status": fallback_status,
        }
    )


def _matching_last_fit_metadata(
    config: CampaignConfig,
    expected: dict[str, object],
) -> dict[str, object]:
    if (
        _LAST_FIT_METADATA.get("campaign_name") != config.campaign_name
        or _LAST_FIT_METADATA.get("profile") != config.model.profile
    ):
        return {}
    for key, value in expected.items():
        if _LAST_FIT_METADATA.get(key) != value:
            return {}
    return dict(_LAST_FIT_METADATA)


def _training_fingerprint(training: TrainingTensors) -> str:
    digest = hashlib.sha256()
    for name, tensor in (
        ("train_x", training.train_x),
        ("train_y", training.train_y),
        ("train_yvar", training.train_yvar),
    ):
        digest.update(name.encode("utf-8"))
        if tensor is None:
            digest.update(b"<none>")
            continue
        detached = tensor.detach().cpu().contiguous()
        digest.update(str(tuple(detached.shape)).encode("utf-8"))
        digest.update(str(detached.dtype).encode("utf-8"))
        digest.update(detached.numpy().tobytes())
    return digest.hexdigest()


def _model_class_name(config: CampaignConfig) -> str:
    if config.fidelity is not None:
        return "SingleTaskMultiFidelityGP"
    return "SingleTaskGP"


def _covariance_profile_name(config: CampaignConfig) -> str:
    if config.fidelity is not None:
        return "multi_fidelity"
    if config.model.profile == "smooth":
        return "RBF/ARD"
    if config.model.profile == "rough":
        return "Matern-1.5/ARD"
    if config.model.profile == "robust":
        return "default/robust"
    return "default"


def _yvar_tensor(
    yvar_df: pd.DataFrame | None,
    objective_names: list[str],
) -> torch.Tensor | None:
    if yvar_df is None:
        return None
    return torch.tensor(
        yvar_df[objective_names].astype(float).to_numpy(),
        dtype=torch.double,
    )
