"""Acquisition construction and optimization."""

from __future__ import annotations

import torch
from botorch.acquisition import LogExpectedImprovement
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement
from botorch.models.model import Model
from botorch.optim import optimize_acqf, optimize_acqf_mixed
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning

from bo_forge.config import CampaignConfig


def optimize_log_ei(
    config: CampaignConfig,
    model: Model,
    train_y_model: torch.Tensor,
    batch_size: int,
    *,
    model_dim: int | None = None,
    fixed_features_list: list[dict[int, float]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Optimize LogEI/qLogEI in unit-cube model space."""
    dimension = model_dim if model_dim is not None else len(config.variables)
    bounds = torch.tensor(
        [[0.0] * dimension, [1.0] * dimension],
        dtype=torch.double,
    )
    best_f = train_y_model.max()

    if batch_size == 1:
        acquisition = LogExpectedImprovement(model=model, best_f=best_f)
        source = "log_ei"
    else:
        sampler = SobolQMCNormalSampler(
            sample_shape=torch.Size([config.bo.mc_samples]),
            seed=config.bo.random_seed,
        )
        acquisition = qLogExpectedImprovement(
            model=model,
            best_f=best_f,
            sampler=sampler,
        )
        source = "qlog_ei"

    optimize_kwargs = {
        "acq_function": acquisition,
        "bounds": bounds,
        "q": batch_size,
        "num_restarts": config.bo.num_restarts,
        "raw_samples": config.bo.raw_samples,
        "options": {"batch_limit": 5, "maxiter": 200},
    }
    if fixed_features_list:
        candidates, acquisition_value = optimize_acqf_mixed(
            **optimize_kwargs,
            fixed_features_list=fixed_features_list,
        )
    else:
        candidates, acquisition_value = optimize_acqf(**optimize_kwargs)
    return candidates.detach(), acquisition_value.detach(), source


def optimize_qlog_ehvi(
    config: CampaignConfig,
    model: Model,
    train_y_model: torch.Tensor,
    ref_point: torch.Tensor,
    batch_size: int,
    *,
    model_dim: int,
    fixed_features_list: list[dict[int, float]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Optimize qLogEHVI in unit-cube model space."""
    bounds = torch.tensor(
        [[0.0] * model_dim, [1.0] * model_dim],
        dtype=torch.double,
    )
    acquisition = build_qlog_ehvi_acquisition(
        config=config,
        model=model,
        train_y_model=train_y_model,
        ref_point=ref_point,
    )

    optimize_kwargs = {
        "acq_function": acquisition,
        "bounds": bounds,
        "q": batch_size,
        "num_restarts": config.bo.num_restarts,
        "raw_samples": config.bo.raw_samples,
        "options": {"batch_limit": 5, "maxiter": 200},
    }
    if fixed_features_list:
        candidates, acquisition_value = optimize_acqf_mixed(
            **optimize_kwargs,
            fixed_features_list=fixed_features_list,
        )
    else:
        candidates, acquisition_value = optimize_acqf(**optimize_kwargs)
    return candidates.detach(), acquisition_value.detach(), "qlog_ehvi"


def build_qlog_ehvi_acquisition(
    config: CampaignConfig,
    model: Model,
    train_y_model: torch.Tensor,
    ref_point: torch.Tensor,
) -> qLogExpectedHypervolumeImprovement:
    """Construct the qLogEHVI acquisition used by optimizer and fallback paths."""
    sampler = SobolQMCNormalSampler(
        sample_shape=torch.Size([config.bo.mc_samples]),
        seed=config.bo.random_seed,
    )
    partitioning = NondominatedPartitioning(ref_point=ref_point, Y=train_y_model)
    return qLogExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point,
        partitioning=partitioning,
        sampler=sampler,
    )
