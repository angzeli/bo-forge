"""Internal transforms between user units and model space."""

from __future__ import annotations

import torch

from bo_forge.config import CampaignConfig


def bounds_tensor(config: CampaignConfig) -> torch.Tensor:
    """Return user-space bounds as a 2 x d tensor."""
    lower = [variable.lower for variable in config.variables]
    upper = [variable.upper for variable in config.variables]
    return torch.tensor([lower, upper], dtype=torch.double)


def to_unit_cube(config: CampaignConfig, x_user: torch.Tensor) -> torch.Tensor:
    """Transform user-space inputs to the unit cube."""
    bounds = bounds_tensor(config).to(dtype=x_user.dtype, device=x_user.device)
    lower = bounds[0]
    width = bounds[1] - bounds[0]
    return (x_user - lower) / width


def from_unit_cube(config: CampaignConfig, x_unit: torch.Tensor) -> torch.Tensor:
    """Transform unit-cube inputs back to user units."""
    bounds = bounds_tensor(config).to(dtype=x_unit.dtype, device=x_unit.device)
    lower = bounds[0]
    width = bounds[1] - bounds[0]
    return lower + x_unit * width


def objective_to_model_space(config: CampaignConfig, y_user: torch.Tensor) -> torch.Tensor:
    """Convert objective values so the model/acquisition always maximizes."""
    return y_user * config.direction_sign


def objective_from_model_space(config: CampaignConfig, y_model: torch.Tensor) -> torch.Tensor:
    """Convert model-space objective values back to the configured direction."""
    return y_model * config.direction_sign

