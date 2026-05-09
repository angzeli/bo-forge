"""Acquisition construction and optimization."""

from __future__ import annotations

import torch
from botorch.acquisition import LogExpectedImprovement
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.models.model import Model
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler

from bo_forge.config import CampaignConfig


def optimize_log_ei(
    config: CampaignConfig,
    model: Model,
    train_y_model: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Optimize LogEI/qLogEI in unit-cube model space."""
    bounds = torch.tensor(
        [[0.0] * len(config.variables), [1.0] * len(config.variables)],
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

    candidates, acquisition_value = optimize_acqf(
        acq_function=acquisition,
        bounds=bounds,
        q=batch_size,
        num_restarts=config.bo.num_restarts,
        raw_samples=config.bo.raw_samples,
        options={"batch_limit": 5, "maxiter": 200},
    )
    return candidates.detach(), acquisition_value.detach(), source

