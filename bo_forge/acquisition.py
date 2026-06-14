"""Acquisition construction and optimization."""

from __future__ import annotations

import torch
from botorch.acquisition import LogExpectedImprovement, PosteriorMean
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement
from botorch.models.model import Model
from botorch.optim import optimize_acqf, optimize_acqf_mixed
from botorch.optim.initializers import gen_one_shot_kg_initial_conditions
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning

from bo_forge.config import CampaignConfig
from bo_forge.multifidelity import (
    affine_fidelity_cost_model,
    target_fidelities,
    target_fidelity_projection,
)


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


def optimize_posterior_mean_at_target_fidelity(
    config: CampaignConfig,
    model: Model,
    *,
    model_dim: int,
    fixed_features_list: list[dict[int, float]] | None = None,
) -> torch.Tensor:
    """Optimize posterior mean at the configured target fidelity."""
    bounds = torch.tensor(
        [[0.0] * model_dim, [1.0] * model_dim],
        dtype=torch.double,
    )
    acquisition = PosteriorMean(model=model)
    target = target_fidelities(config)
    optimize_kwargs = {
        "acq_function": acquisition,
        "bounds": bounds,
        "q": 1,
        "num_restarts": config.bo.num_restarts,
        "raw_samples": config.bo.raw_samples,
        "options": {"batch_limit": 5, "maxiter": 200},
    }
    if fixed_features_list:
        candidates, current_value = optimize_acqf_mixed(
            **optimize_kwargs,
            fixed_features_list=[
                {**fixed_features, **target}
                for fixed_features in fixed_features_list
            ],
        )
    else:
        candidates, current_value = optimize_acqf(
            **optimize_kwargs,
            fixed_features=target,
        )
    del candidates
    return current_value.detach().reshape(1)


def optimize_qmf_kg(
    config: CampaignConfig,
    model: Model,
    current_value: torch.Tensor,
    *,
    model_dim: int,
    fixed_features_list: list[dict[int, float]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Optimize qMFKG in unit-cube model space."""
    bounds = torch.tensor(
        [[0.0] * model_dim, [1.0] * model_dim],
        dtype=torch.double,
    )
    acquisition = qMultiFidelityKnowledgeGradient(
        model=model,
        num_fantasies=config.fidelity.num_fantasies if config.fidelity else 64,
        current_value=current_value,
        cost_aware_utility=InverseCostWeightedUtility(
            cost_model=affine_fidelity_cost_model(config),
        ),
        project=target_fidelity_projection(config),
    )
    optimize_kwargs = {
        "acq_function": acquisition,
        "bounds": bounds,
        "q": 1,
        "num_restarts": config.bo.num_restarts,
        "raw_samples": config.bo.raw_samples,
        "options": {"batch_limit": 5, "maxiter": 200},
        "ic_generator": gen_one_shot_kg_initial_conditions,
    }
    if fixed_features_list:
        candidates, acquisition_value = optimize_acqf_mixed(
            **optimize_kwargs,
            fixed_features_list=fixed_features_list,
        )
    else:
        candidates, acquisition_value = optimize_acqf(**optimize_kwargs)
    candidates = _extract_qmf_kg_candidates(
        acquisition,
        candidates,
        q=1,
        num_fantasies=config.fidelity.num_fantasies if config.fidelity else 64,
    )
    return candidates.detach(), acquisition_value.detach(), "qmf_kg"


def _extract_qmf_kg_candidates(
    acquisition: qMultiFidelityKnowledgeGradient,
    candidates: torch.Tensor,
    *,
    q: int,
    num_fantasies: int,
) -> torch.Tensor:
    """Return user candidates from a qMFKG optimizer result."""
    candidate_count = int(candidates.shape[-2])
    if candidate_count == q:
        return candidates
    if candidate_count == q + num_fantasies:
        extracted = acquisition.extract_candidates(candidates)
        if int(extracted.shape[-2]) != q:
            raise RuntimeError(
                "qMFKG candidate extraction returned an unexpected candidate count: "
                f"expected={q}, actual={int(extracted.shape[-2])}."
            )
        return extracted
    raise RuntimeError(
        "qMFKG optimizer returned an unexpected candidate count: "
        f"expected {q} or {q + num_fantasies}, actual={candidate_count}."
    )


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
