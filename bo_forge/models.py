"""Model fitting helpers."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

from bo_forge.config import CampaignConfig
from bo_forge.multi_objective import objectives_to_model_space
from bo_forge.replicates import modeling_observed_data_with_variance
from bo_forge.transforms import dataframe_to_unit_cube, objective_to_model_space


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
    model = SingleTaskGP(
        training.train_x,
        training.train_y,
        input_transform=Normalize(d=training.train_x.shape[-1]),
        outcome_transform=Standardize(m=training.train_y.shape[-1]),
        **kwargs,
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


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
