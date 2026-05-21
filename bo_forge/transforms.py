"""Internal transforms between user units and model space."""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import product

import pandas as pd
import torch

from bo_forge.config import CampaignConfig, VariableConfig


def bounds_tensor(config: CampaignConfig) -> torch.Tensor:
    """Return continuous user-space bounds as a 2 x d tensor."""
    _raise_if_not_all_continuous(config)
    lower = [variable.lower for variable in config.variables]
    upper = [variable.upper for variable in config.variables]
    return torch.tensor([lower, upper], dtype=torch.double)


def to_unit_cube(config: CampaignConfig, x_user: torch.Tensor) -> torch.Tensor:
    """Transform user-space inputs to the unit cube."""
    _raise_if_not_all_continuous(config)
    bounds = bounds_tensor(config).to(dtype=x_user.dtype, device=x_user.device)
    lower = bounds[0]
    width = bounds[1] - bounds[0]
    return (x_user - lower) / width


def from_unit_cube(config: CampaignConfig, x_unit: torch.Tensor) -> torch.Tensor:
    """Transform unit-cube inputs back to user units."""
    _raise_if_not_all_continuous(config)
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


def has_mixed_variables(config: CampaignConfig) -> bool:
    """Return True when any configured variable is not continuous."""
    return any(variable.type != "continuous" for variable in config.variables)


def encoded_feature_names(config: CampaignConfig) -> list[str]:
    """Return model-space feature names after mixed-variable encoding."""
    names: list[str] = []
    for variable in config.variables:
        if variable.type == "categorical":
            names.extend(f"{variable.name}={value}" for value in variable.values)
        else:
            names.append(variable.name)
    return names


def encoded_dimension(config: CampaignConfig) -> int:
    """Return the model-space dimension after mixed-variable encoding."""
    return len(encoded_feature_names(config))


def encoded_feature_indices(config: CampaignConfig) -> dict[str, tuple[int, ...]]:
    """Map each user variable name to its model-space feature indices."""
    indices: dict[str, tuple[int, ...]] = {}
    cursor = 0
    for variable in config.variables:
        width = len(variable.values) if variable.type == "categorical" else 1
        indices[variable.name] = tuple(range(cursor, cursor + width))
        cursor += width
    return indices


def categorical_combination_count(config: CampaignConfig) -> int:
    """Return the number of possible categorical assignments."""
    count = 1
    for variable in config.variables:
        if variable.type == "categorical":
            count *= len(variable.values)
    return count


def categorical_feature_assignments(config: CampaignConfig) -> list[dict[int, float]]:
    """Return fixed one-hot feature maps for all categorical assignments."""
    categorical_variables = [
        variable for variable in config.variables if variable.type == "categorical"
    ]
    if not categorical_variables:
        return []

    feature_indices = encoded_feature_indices(config)
    assignments: list[dict[int, float]] = []
    for category_indices in product(
        *[range(len(variable.values)) for variable in categorical_variables]
    ):
        fixed_features: dict[int, float] = {}
        for variable, active_index in zip(
            categorical_variables, category_indices, strict=True
        ):
            for offset, model_index in enumerate(feature_indices[variable.name]):
                fixed_features[model_index] = 1.0 if offset == active_index else 0.0
        assignments.append(fixed_features)
    return assignments


def dataframe_to_unit_cube(config: CampaignConfig, df: pd.DataFrame) -> torch.Tensor:
    """Encode user-facing campaign variables into latent unit-cube coordinates."""
    rows = [
        [row[variable.name] for variable in config.variables]
        for _, row in df.iterrows()
    ]
    return values_to_unit_cube(config, rows)


def values_to_unit_cube(
    config: CampaignConfig,
    rows: Sequence[Sequence[object]],
) -> torch.Tensor:
    """Encode user-facing variable values into latent unit-cube coordinates."""
    encoded_rows = [
        _encode_variable_values(config, row)
        for row in rows
    ]
    return torch.tensor(encoded_rows, dtype=torch.double)


def unit_cube_to_user_values(
    config: CampaignConfig,
    x_unit: torch.Tensor,
) -> list[tuple[object, ...]]:
    """Decode model-space latent candidates into valid user-facing values."""
    if x_unit.ndim == 1:
        x_unit = x_unit.unsqueeze(0)

    rows: list[tuple[object, ...]] = []
    for row in x_unit.detach().cpu().tolist():
        values = _decode_model_row(config, row)
        rows.append(values)
    return rows


def unit_cube_to_design_values(
    config: CampaignConfig,
    x_unit: torch.Tensor,
) -> list[tuple[object, ...]]:
    """Decode one scalar per user variable into valid user-facing values."""
    if x_unit.ndim == 1:
        x_unit = x_unit.unsqueeze(0)

    rows: list[tuple[object, ...]] = []
    for row in x_unit.detach().cpu().tolist():
        values = tuple(
            _decode_scalar_variable_value(variable, float(value))
            for variable, value in zip(config.variables, row, strict=True)
        )
        rows.append(values)
    return rows


def dataframe_to_variable_coverage(
    config: CampaignConfig,
    df: pd.DataFrame,
) -> torch.Tensor:
    """Encode rows as one normalized coverage value per user variable."""
    rows = [
        [row[variable.name] for variable in config.variables]
        for _, row in df.iterrows()
    ]
    encoded_rows = [
        [
            _encode_scalar_variable_value(variable, value)
            for variable, value in zip(config.variables, row, strict=True)
        ]
        for row in rows
    ]
    return torch.tensor(encoded_rows, dtype=torch.double)


def _encode_variable_values(
    config: CampaignConfig,
    row: Sequence[object],
) -> list[float]:
    encoded: list[float] = []
    for variable, value in zip(config.variables, row, strict=True):
        if variable.type == "categorical":
            encoded.extend(_encode_categorical_one_hot(variable, value))
        else:
            encoded.append(_encode_scalar_variable_value(variable, value))
    return encoded


def _encode_categorical_one_hot(variable: VariableConfig, value: object) -> list[float]:
    parsed = str(value)
    try:
        index = list(variable.values).index(parsed)
    except ValueError as exc:
        raise ValueError(
            f"Variable '{variable.name}' has value outside configured categorical choices: "
            f"value={value!r}."
        ) from exc
    return [1.0 if offset == index else 0.0 for offset in range(len(variable.values))]


def _encode_scalar_variable_value(variable: VariableConfig, value: object) -> float:
    if variable.type == "continuous":
        parsed = _finite_float(value, variable.name)
        lower = _required_bound(variable, "lower")
        upper = _required_bound(variable, "upper")
        return (parsed - lower) / (upper - lower)

    if variable.type == "integer":
        parsed = _finite_float(value, variable.name)
        if parsed % 1 != 0:
            raise ValueError(
                f"Variable '{variable.name}' must be integer-valued: value={value!r}."
            )
        lower = int(_required_bound(variable, "lower"))
        upper = int(_required_bound(variable, "upper"))
        return _level_unit(int(parsed) - lower, upper - lower + 1)

    if variable.type == "discrete":
        parsed = _finite_float(value, variable.name)
        numeric_values = [float(item) for item in variable.values]
        for index, allowed in enumerate(numeric_values):
            if math.isclose(parsed, allowed, rel_tol=1e-12, abs_tol=1e-12):
                return _level_unit(index, len(numeric_values))
        raise ValueError(
            f"Variable '{variable.name}' has value outside configured discrete choices: "
            f"value={value!r}."
        )

    if variable.type == "categorical":
        parsed = str(value)
        try:
            index = list(variable.values).index(parsed)
        except ValueError as exc:
            raise ValueError(
                f"Variable '{variable.name}' has value outside configured categorical choices: "
                f"value={value!r}."
            ) from exc
        return _level_unit(index, len(variable.values))

    raise ValueError(f"Variable '{variable.name}' has unsupported type '{variable.type}'.")


def _decode_model_row(config: CampaignConfig, row: Sequence[float]) -> tuple[object, ...]:
    values: list[object] = []
    cursor = 0
    for variable in config.variables:
        if variable.type == "categorical":
            width = len(variable.values)
            block = row[cursor : cursor + width]
            if len(block) != width:
                raise ValueError(
                    f"Encoded row is missing categorical block for variable "
                    f"'{variable.name}'."
                )
            active_index = max(range(width), key=lambda index: block[index])
            values.append(str(variable.values[active_index]))
            cursor += width
        else:
            if cursor >= len(row):
                raise ValueError(
                    f"Encoded row is missing value for variable '{variable.name}'."
                )
            values.append(_decode_scalar_variable_value(variable, float(row[cursor])))
            cursor += 1
    if cursor != len(row):
        raise ValueError(
            f"Encoded row has unexpected extra model features: expected={cursor}, "
            f"actual={len(row)}."
        )
    return tuple(values)


def _decode_scalar_variable_value(variable: VariableConfig, value: float) -> object:
    clipped = min(max(value, 0.0), 1.0)
    if variable.type == "continuous":
        lower = _required_bound(variable, "lower")
        upper = _required_bound(variable, "upper")
        return lower + clipped * (upper - lower)

    if variable.type == "integer":
        lower = int(_required_bound(variable, "lower"))
        upper = int(_required_bound(variable, "upper"))
        index = _unit_to_level_index(clipped, upper - lower + 1)
        return lower + index

    if variable.type == "discrete":
        index = _unit_to_level_index(clipped, len(variable.values))
        return float(variable.values[index])

    if variable.type == "categorical":
        index = _unit_to_level_index(clipped, len(variable.values))
        return str(variable.values[index])

    raise ValueError(f"Variable '{variable.name}' has unsupported type '{variable.type}'.")


def _level_unit(index: int, count: int) -> float:
    if count < 1:
        raise ValueError("count must be >= 1.")
    return (index + 0.5) / count


def _unit_to_level_index(value: float, count: int) -> int:
    if count < 1:
        raise ValueError("count must be >= 1.")
    clipped = min(max(value, 0.0), math.nextafter(1.0, 0.0))
    return min(int(clipped * count), count - 1)


def _required_bound(variable: VariableConfig, key: str) -> float:
    value = variable.lower if key == "lower" else variable.upper
    if value is None:
        raise ValueError(f"Variable '{variable.name}' is missing bound '{key}'.")
    return float(value)


def _finite_float(value: object, variable_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Variable '{variable_name}' must be numeric: value={value!r}."
        ) from exc
    if not math.isfinite(parsed):
        raise ValueError(f"Variable '{variable_name}' must be finite: value={value!r}.")
    return parsed


def _raise_if_not_all_continuous(config: CampaignConfig) -> None:
    if has_mixed_variables(config):
        raise ValueError("This transform is only valid for continuous-only campaigns.")
