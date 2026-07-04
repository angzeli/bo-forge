"""Model fitting helpers."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import pandas as pd
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

from bo_forge.config import CampaignConfig
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
    observed_rows_used = 0
    train_yvar_used = False
    if not observed.empty:
        if config.is_structured_campaign:
            observed_rows_used = int(len(observed))
        else:
            training = dataframe_to_training_tensors(config, observed)
            observed_rows_used = int(training.train_x.shape[0])
            train_yvar_used = training.train_yvar is not None

    metadata = _matching_last_fit_metadata(config)
    rows = [
        ("model_profile", config.model.profile),
        ("model_class", _model_class_name(config)),
        ("covariance_profile", _covariance_profile_name(config)),
        ("encoded_dimension", encoded_dimension(config)),
        ("observed_rows_used_for_fitting", observed_rows_used),
        ("objective_count", len(config.objective_names)),
        ("train_yvar_used", train_yvar_used),
        ("last_fit_status", metadata.get("fit_status")),
        ("last_fit_warning_count", metadata.get("fit_warning_count", 0)),
        ("last_fit_warnings", metadata.get("fit_warnings", "")),
        ("fallback_status", metadata.get("fallback_status")),
    ]
    return pd.DataFrame(rows, columns=["field", "value"])


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
            "fit_status": fit_status,
            "fit_warning_count": len(fit_warnings),
            "fit_warnings": "; ".join(fit_warnings),
            "fallback_status": fallback_status,
        }
    )


def _matching_last_fit_metadata(config: CampaignConfig) -> dict[str, object]:
    if (
        _LAST_FIT_METADATA.get("campaign_name") == config.campaign_name
        and _LAST_FIT_METADATA.get("profile") == config.model.profile
    ):
        return dict(_LAST_FIT_METADATA)
    return {}


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
