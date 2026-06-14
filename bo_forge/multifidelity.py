"""Helpers for single-objective multi-fidelity campaigns."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import torch
from botorch.acquisition.utils import project_to_target_fidelity
from botorch.models.cost import AffineFidelityCostModel

from bo_forge.config import CampaignConfig, VariableConfig
from bo_forge.transforms import encoded_dimension, encoded_feature_indices


def fidelity_variable(config: CampaignConfig) -> VariableConfig:
    """Return the configured fidelity variable."""
    if config.fidelity is None:
        raise ValueError("Campaign config does not define a fidelity section.")
    for variable in config.variables:
        if variable.name == config.fidelity.variable:
            return variable
    raise ValueError(
        f"fidelity.variable references unknown variable '{config.fidelity.variable}'."
    )


def fidelity_variable_index(config: CampaignConfig) -> int:
    """Return the user-space variable index of the fidelity variable."""
    if config.fidelity is None:
        raise ValueError("Campaign config does not define a fidelity section.")
    return config.variable_names.index(config.fidelity.variable)


def fidelity_feature_index(config: CampaignConfig) -> int:
    """Return the model-space feature index of the fidelity variable."""
    if config.fidelity is None:
        raise ValueError("Campaign config does not define a fidelity section.")
    indices = encoded_feature_indices(config)[config.fidelity.variable]
    if len(indices) != 1:
        raise ValueError("Multi-fidelity campaigns require one continuous fidelity feature.")
    return indices[0]


def target_fidelity_unit_value(config: CampaignConfig) -> float:
    """Return the target fidelity in unit-cube model coordinates."""
    if config.fidelity is None:
        raise ValueError("Campaign config does not define a fidelity section.")
    variable = fidelity_variable(config)
    if variable.lower is None or variable.upper is None:
        raise ValueError("fidelity.variable must have finite lower and upper bounds.")
    return (config.fidelity.target - variable.lower) / (variable.upper - variable.lower)


def target_fidelities(config: CampaignConfig) -> dict[int, float]:
    """Return BoTorch target-fidelity mapping in model-space feature coordinates."""
    return {fidelity_feature_index(config): target_fidelity_unit_value(config)}


def target_fidelity_projection(config: CampaignConfig) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a BoTorch projection callable to the configured target fidelity."""
    return partial(
        project_to_target_fidelity,
        target_fidelities=target_fidelities(config),
        d=encoded_dimension(config),
    )


def affine_fidelity_cost_model(config: CampaignConfig) -> AffineFidelityCostModel:
    """Return BoTorch's affine fidelity cost model for qMFKG."""
    if config.fidelity is None:
        raise ValueError("Campaign config does not define a fidelity section.")
    return AffineFidelityCostModel(
        fidelity_weights={
            fidelity_feature_index(config): config.fidelity.fidelity_cost_weight,
        },
        fixed_cost=config.fidelity.fixed_cost,
    )
