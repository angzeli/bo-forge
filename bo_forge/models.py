"""Model fitting helpers."""

from __future__ import annotations

import pandas as pd
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

from bo_forge.config import CampaignConfig
from bo_forge.transforms import objective_to_model_space, to_unit_cube


def dataframe_to_tensors(
    config: CampaignConfig, observed_df: pd.DataFrame
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert observed campaign rows to model-space tensors."""
    x_user = torch.tensor(
        observed_df[config.variable_names].astype(float).to_numpy(),
        dtype=torch.double,
    )
    y_user = torch.tensor(
        observed_df[[config.objective.name]].astype(float).to_numpy(),
        dtype=torch.double,
    )
    return to_unit_cube(config, x_user), objective_to_model_space(config, y_user)


def fit_gp_model(config: CampaignConfig, observed_df: pd.DataFrame) -> SingleTaskGP:
    """Fit a standard BoTorch SingleTaskGP to observed campaign data."""
    train_x, train_y = dataframe_to_tensors(config, observed_df)
    model = SingleTaskGP(
        train_x,
        train_y,
        input_transform=Normalize(d=train_x.shape[-1]),
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model

