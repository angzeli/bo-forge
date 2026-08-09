"""Helpers for single-objective multi-fidelity campaigns."""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Any

import pandas as pd

from bo_forge.config import CampaignConfig, VariableConfig, fidelity_values_match
from bo_forge.validation import get_observed_data, validate_campaign_data

FIDELITY_COVERAGE_COLUMNS = [
    "fidelity",
    "is_target",
    "modeled_evaluation_cost",
    "observed_rows",
    "active_suggestions",
    "objective_mean",
    "objective_std",
    "objective_best",
    "best_row_id",
    "latest_observed_iteration",
]

if TYPE_CHECKING:
    from botorch.models.cost import AffineFidelityCostModel
    from torch import Tensor
else:
    Tensor = Any
    AffineFidelityCostModel = Any


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
    from bo_forge.transforms import encoded_feature_indices

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


def fidelity_level_unit_values(config: CampaignConfig) -> tuple[float, ...] | None:
    """Return configured discrete fidelity levels in unit-cube coordinates."""
    if config.fidelity is None:
        raise ValueError("Campaign config does not define a fidelity section.")
    if config.fidelity.levels is None:
        return None
    variable = fidelity_variable(config)
    assert variable.lower is not None and variable.upper is not None
    width = variable.upper - variable.lower
    return tuple((level - variable.lower) / width for level in config.fidelity.levels)


def fidelity_level_fixed_features(config: CampaignConfig) -> list[dict[int, float]]:
    """Return one fixed-feature map per configured discrete fidelity level."""
    unit_values = fidelity_level_unit_values(config)
    if unit_values is None:
        return []
    feature_index = fidelity_feature_index(config)
    return [{feature_index: value} for value in unit_values]


def map_initial_fidelity_to_levels(
    config: CampaignConfig,
    x_unit: Tensor,
) -> Tensor:
    """Map initial-design fidelity coordinates into equal-width level bins."""
    import torch

    if config.fidelity is None:
        return x_unit
    unit_values = fidelity_level_unit_values(config)
    if unit_values is None:
        return x_unit
    mapped = x_unit.clone()
    index = fidelity_feature_index(config)
    coordinates = mapped[..., index].clamp(min=0.0, max=1.0)
    bin_indices = torch.clamp(
        torch.floor(coordinates * len(unit_values)).to(dtype=torch.long),
        max=len(unit_values) - 1,
    )
    level_tensor = torch.tensor(unit_values, dtype=mapped.dtype, device=mapped.device)
    mapped[..., index] = level_tensor[bin_indices]
    return mapped


def target_fidelities(config: CampaignConfig) -> dict[int, float]:
    """Return BoTorch target-fidelity mapping in model-space feature coordinates."""
    return {fidelity_feature_index(config): target_fidelity_unit_value(config)}


def target_fidelity_projection(config: CampaignConfig) -> Callable[[Tensor], Tensor]:
    """Return a BoTorch projection callable to the configured target fidelity."""
    from botorch.acquisition.utils import project_to_target_fidelity

    from bo_forge.transforms import encoded_dimension

    return partial(
        project_to_target_fidelity,
        target_fidelities=target_fidelities(config),
        d=encoded_dimension(config),
    )


def affine_fidelity_cost_model(config: CampaignConfig) -> AffineFidelityCostModel:
    """Return BoTorch's affine fidelity cost model for qMFKG."""
    from botorch.models.cost import AffineFidelityCostModel

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
        ("fidelity_mode", "discrete" if config.fidelity.levels is not None else "continuous"),
        (
            "configured_fidelity_levels",
            None
            if config.fidelity.levels is None
            else ", ".join(f"{level:g}" for level in config.fidelity.levels),
        ),
        (
            "configured_fidelity_level_count",
            None if config.fidelity.levels is None else len(config.fidelity.levels),
        ),
        (
            "observed_fidelity_level_count",
            None if config.fidelity.levels is None else 0,
        ),
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
            "observed_fidelity_level_count": (
                None
                if config.fidelity.levels is None
                else sum(
                    any(
                        math.isclose(float(value), level, rel_tol=1e-9, abs_tol=1e-9)
                        for value in fidelity_values
                    )
                    for level in config.fidelity.levels
                )
            ),
        }
    )
    return pd.DataFrame(list(values.items()), columns=["field", "value"])


def fidelity_coverage(config: CampaignConfig, df: pd.DataFrame) -> pd.DataFrame:
    """Return observed and active-suggestion coverage by fidelity value."""
    if config.fidelity is None:
        raise ValueError("fidelity_coverage() requires a config with a fidelity section.")
    validate_campaign_data(config, df)

    observed = get_observed_data(config, df)
    active = _active_fidelity_suggestions(config, df)
    fidelity_name = config.fidelity.variable
    if config.fidelity.levels is not None:
        fidelity_values = [float(level) for level in config.fidelity.levels]
        observed_groups = _discrete_fidelity_groups(observed, fidelity_name, fidelity_values)
        active_groups = _discrete_fidelity_groups(active, fidelity_name, fidelity_values)
    else:
        values = [
            *pd.to_numeric(observed[fidelity_name]).tolist(),
            *pd.to_numeric(active[fidelity_name]).tolist(),
        ]
        fidelity_values = sorted({float(item) for item in values})
        observed_groups = pd.to_numeric(observed[fidelity_name])
        active_groups = pd.to_numeric(active[fidelity_name])
    if not fidelity_values:
        return pd.DataFrame(columns=FIDELITY_COVERAGE_COLUMNS)

    variable = fidelity_variable(config)
    assert variable.lower is not None and variable.upper is not None
    width = variable.upper - variable.lower
    objective = config.objective.name
    rows: list[dict[str, object]] = []
    for index, fidelity in enumerate(fidelity_values):
        observed_at_fidelity = observed.loc[observed_groups == fidelity]
        active_at_fidelity = active.loc[active_groups == fidelity]
        best = _best_fidelity_row(config, observed_at_fidelity)
        objective_values = pd.to_numeric(observed_at_fidelity[objective])
        rows.append(
            {
                "fidelity": fidelity,
                "is_target": (
                    index == len(fidelity_values) - 1
                    if config.fidelity.levels is not None
                    else _is_target_fidelity(fidelity, config.fidelity.target)
                ),
                "modeled_evaluation_cost": config.fidelity.fixed_cost
                + config.fidelity.fidelity_cost_weight
                * ((fidelity - variable.lower) / width),
                "observed_rows": len(observed_at_fidelity),
                "active_suggestions": len(active_at_fidelity),
                "objective_mean": (
                    None if observed_at_fidelity.empty else float(objective_values.mean())
                ),
                "objective_std": (
                    None
                    if len(observed_at_fidelity) < 2
                    else float(objective_values.std(ddof=1))
                ),
                "objective_best": None if best is None else float(best[objective]),
                "best_row_id": None if best is None else str(best["row_id"]),
                "latest_observed_iteration": (
                    None
                    if observed_at_fidelity.empty
                    else int(pd.to_numeric(observed_at_fidelity["iteration"]).max())
                ),
            }
        )
    result = pd.DataFrame(rows, columns=FIDELITY_COVERAGE_COLUMNS)
    for column in (
        "objective_mean",
        "objective_std",
        "objective_best",
        "best_row_id",
        "latest_observed_iteration",
    ):
        result[column] = result[column].astype(object)
    result.loc[result["observed_rows"] == 0, "objective_mean"] = None
    result.loc[result["observed_rows"] < 2, "objective_std"] = None
    result.loc[result["observed_rows"] == 0, "objective_best"] = None
    result.loc[result["observed_rows"] == 0, "best_row_id"] = None
    result.loc[result["observed_rows"] == 0, "latest_observed_iteration"] = None
    return result


def _active_fidelity_suggestions(
    config: CampaignConfig,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Return suggested rows that still represent active experiments."""
    active = df["status"].astype(str) == "suggested"
    if config.review.enabled:
        active &= df["review_status"].isin({"pending", "accepted"})
    return df.loc[active].copy()


def _discrete_fidelity_groups(
    df: pd.DataFrame,
    fidelity_name: str,
    levels: list[float],
) -> pd.Series:
    """Map validated discrete-fidelity rows to exactly one configured level."""
    if df.empty:
        return pd.Series(index=df.index, dtype=float)

    def matched_level(value: object) -> float:
        matches = [level for level in levels if fidelity_values_match(value, level)]
        if len(matches) != 1:
            raise ValueError(
                "Discrete fidelity value must match exactly one configured level: "
                f"value={value!r}, matching_levels={matches}."
            )
        return matches[0]

    return pd.to_numeric(df[fidelity_name]).map(matched_level)


def _is_target_fidelity(value: object, target: float) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return fidelity_values_match(numeric, target)


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
