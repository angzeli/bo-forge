"""Helpers for single-objective multi-fidelity campaigns."""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import partial

import pandas as pd
import torch
from botorch.acquisition.utils import project_to_target_fidelity
from botorch.models.cost import AffineFidelityCostModel

from bo_forge.config import CampaignConfig, VariableConfig
from bo_forge.transforms import encoded_dimension, encoded_feature_indices
from bo_forge.validation import get_observed_data, validate_campaign_data


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


def fidelity_summary(config: CampaignConfig, df: pd.DataFrame) -> pd.DataFrame:
    """Return read-only summary fields for a multi-fidelity campaign."""
    if config.fidelity is None:
        raise ValueError("fidelity_summary() requires a config with a fidelity section.")
    validate_campaign_data(config, df)

    fidelity_name = config.fidelity.variable
    target = float(config.fidelity.target)
    observed = get_observed_data(config, df)
    suggested = df["status"].astype(str) == "suggested"
    qmfkg = df["source"].astype(str) == "qmf_kg"
    if config.review.enabled:
        blocking_review = df["review_status"].isin({"pending", "accepted"})
    else:
        blocking_review = pd.Series(True, index=df.index)
    pending_qmfkg = int((suggested & qmfkg & blocking_review).sum())

    rows: list[tuple[str, object]] = [
        ("fidelity_variable", fidelity_name),
        ("target_fidelity", target),
        ("observed_rows", len(observed)),
        ("lower_fidelity_observed_rows", 0),
        ("target_fidelity_observed_rows", 0),
        ("min_observed_fidelity", None),
        ("max_observed_fidelity", None),
        ("pending_qmfkg_suggestions", pending_qmfkg),
        ("best_observed_row_id", None),
        ("best_observed_objective", None),
        ("best_target_fidelity_row_id", None),
        ("best_target_fidelity_objective", None),
    ]
    if observed.empty:
        return pd.DataFrame(rows, columns=["field", "value"])

    fidelity_values = pd.to_numeric(observed[fidelity_name])
    target_mask = fidelity_values.map(lambda value: _is_target_fidelity(value, target))
    lower_mask = (fidelity_values < target) & ~target_mask
    best = _best_fidelity_row(config, observed)
    target_best = _best_fidelity_row(config, observed.loc[target_mask])
    values = dict(rows)
    values.update(
        {
            "lower_fidelity_observed_rows": int(lower_mask.sum()),
            "target_fidelity_observed_rows": int(target_mask.sum()),
            "min_observed_fidelity": float(fidelity_values.min()),
            "max_observed_fidelity": float(fidelity_values.max()),
            "best_observed_row_id": None if best is None else str(best["row_id"]),
            "best_observed_objective": None
            if best is None
            else float(best[config.objective.name]),
            "best_target_fidelity_row_id": None
            if target_best is None
            else str(target_best["row_id"]),
            "best_target_fidelity_objective": None
            if target_best is None
            else float(target_best[config.objective.name]),
        }
    )
    return pd.DataFrame(list(values.items()), columns=["field", "value"])


def _is_target_fidelity(value: object, target: float) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isclose(numeric, target, rel_tol=1e-9, abs_tol=1e-9)


def _best_fidelity_row(
    config: CampaignConfig,
    observed: pd.DataFrame,
) -> pd.Series | None:
    if observed.empty:
        return None
    objective = config.objective.name
    values = pd.to_numeric(observed[objective])
    best_index = values.idxmax() if config.objective.direction == "maximize" else values.idxmin()
    return observed.loc[best_index]
