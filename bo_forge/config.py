"""Campaign configuration parsing and validation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from bo_forge.errors import ConfigError

RESERVED_COLUMNS = {
    "row_id",
    "iteration",
    "status",
    "source",
    "predicted_mean",
    "predicted_std",
    "acquisition",
}


@dataclass(frozen=True)
class VariableConfig:
    name: str
    type: str
    lower: float | None = None
    upper: float | None = None
    values: tuple[str | float, ...] = ()


@dataclass(frozen=True)
class ObjectiveConfig:
    name: str
    direction: str


@dataclass(frozen=True)
class BOConfig:
    batch_size: int = 1
    initial_design_size: int = 8
    acquisition: str = "log_ei"
    initial_design_method: str = "sobol"
    random_seed: int = 0
    raw_samples: int = 128
    num_restarts: int = 5
    mc_samples: int = 128


@dataclass(frozen=True)
class CampaignConfig:
    campaign_name: str
    objective: ObjectiveConfig
    variables: tuple[VariableConfig, ...]
    bo: BOConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> CampaignConfig:
        """Load a campaign config from YAML."""
        config_path = Path(path)
        try:
            with config_path.open("r", encoding="utf-8") as handle:
                raw = yaml.safe_load(handle)
        except OSError as exc:
            raise ConfigError(f"Could not read config file '{config_path}': {exc}") from exc
        return parse_campaign_config(raw)

    @property
    def variable_names(self) -> list[str]:
        """Return variable names in configured order."""
        return [variable.name for variable in self.variables]

    @property
    def direction_sign(self) -> float:
        """Return the multiplier that converts objective values to maximization."""
        return 1.0 if self.objective.direction == "maximize" else -1.0


def parse_campaign_config(raw: Any) -> CampaignConfig:
    """Parse raw YAML data into a validated campaign config."""
    if not isinstance(raw, dict):
        raise ConfigError("Config file must contain a YAML mapping at the top level.")

    campaign_name = _required_str(raw, "campaign_name", "campaign")
    objective = _parse_objective(raw.get("objective"))
    variables = _parse_variables(raw.get("variables"), objective.name)
    bo = _parse_bo(raw.get("bo", {}))

    return CampaignConfig(
        campaign_name=campaign_name,
        objective=objective,
        variables=tuple(variables),
        bo=bo,
    )


def _parse_objective(raw: Any) -> ObjectiveConfig:
    if not isinstance(raw, dict):
        raise ConfigError("Config key 'objective' must be a mapping.")

    name = _required_str(raw, "name", "objective")
    direction = _required_str(raw, "direction", "objective")
    if direction not in {"maximize", "minimize"}:
        raise ConfigError(
            f"Objective '{name}' has invalid direction '{direction}'. "
            "Expected 'maximize' or 'minimize'."
        )
    if name in RESERVED_COLUMNS:
        raise ConfigError(f"Objective name '{name}' conflicts with a reserved CSV column.")
    return ObjectiveConfig(name=name, direction=direction)


def _parse_variables(raw: Any, objective_name: str) -> list[VariableConfig]:
    if not isinstance(raw, list) or not raw:
        raise ConfigError("Config key 'variables' must be a non-empty list.")

    variables: list[VariableConfig] = []
    seen_names: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ConfigError(f"Variable at index {index} must be a mapping.")

        name = _required_str(item, "name", f"variables[{index}]")
        variable_type = _required_str(item, "type", f"variable '{name}'")
        if variable_type not in {"continuous", "integer", "discrete", "categorical"}:
            raise ConfigError(
                f"Variable '{name}' has unsupported type '{variable_type}'. "
                "Expected one of ['categorical', 'continuous', 'discrete', 'integer']."
            )
        _reject_unsupported_variable_keys(item, name, variable_type)
        if name in seen_names:
            raise ConfigError(f"Duplicate variable name '{name}'.")
        if name == objective_name:
            raise ConfigError(
                f"Variable '{name}' conflicts with objective name '{objective_name}'."
            )
        if name in RESERVED_COLUMNS:
            raise ConfigError(f"Variable name '{name}' conflicts with a reserved CSV column.")

        if variable_type == "continuous":
            lower = _required_float(item, "lower", f"variable '{name}'")
            upper = _required_float(item, "upper", f"variable '{name}'")
            if lower >= upper:
                raise ConfigError(
                    f"Variable '{name}' has lower >= upper: "
                    f"lower={lower:g}, upper={upper:g}."
                )
            variable = VariableConfig(
                name=name,
                type=variable_type,
                lower=lower,
                upper=upper,
            )
        elif variable_type == "integer":
            lower = _required_integer_bound(item, "lower", f"variable '{name}'")
            upper = _required_integer_bound(item, "upper", f"variable '{name}'")
            if lower > upper:
                raise ConfigError(
                    f"Variable '{name}' has lower > upper: "
                    f"lower={lower:g}, upper={upper:g}."
                )
            variable = VariableConfig(
                name=name,
                type=variable_type,
                lower=lower,
                upper=upper,
            )
        elif variable_type == "discrete":
            variable = VariableConfig(
                name=name,
                type=variable_type,
                values=_required_discrete_values(item, name),
            )
        else:
            variable = VariableConfig(
                name=name,
                type=variable_type,
                values=_required_categorical_values(item, name),
            )

        variables.append(variable)
        seen_names.add(name)

    return variables


def _parse_bo(raw: Any) -> BOConfig:
    if not isinstance(raw, dict):
        raise ConfigError("Config key 'bo' must be a mapping when provided.")

    acquisition = str(raw.get("acquisition", "log_ei"))
    if acquisition != "log_ei":
        raise ConfigError(
            f"Unsupported acquisition '{acquisition}'. "
            "BO Forge currently supports only 'log_ei'."
        )
    initial_design_method = str(raw.get("initial_design_method", "sobol"))
    if initial_design_method not in {"sobol", "random"}:
        raise ConfigError(
            f"Unsupported initial_design_method '{initial_design_method}'. "
            "Expected 'sobol' or 'random'."
        )

    return BOConfig(
        batch_size=_positive_int(raw.get("batch_size", 1), "bo.batch_size"),
        initial_design_size=_positive_int(
            raw.get("initial_design_size", 8), "bo.initial_design_size"
        ),
        acquisition=acquisition,
        initial_design_method=initial_design_method,
        random_seed=_non_negative_int(raw.get("random_seed", 0), "bo.random_seed"),
        raw_samples=_positive_int(raw.get("raw_samples", 128), "bo.raw_samples"),
        num_restarts=_positive_int(raw.get("num_restarts", 5), "bo.num_restarts"),
        mc_samples=_positive_int(raw.get("mc_samples", 128), "bo.mc_samples"),
    )


def _required_str(raw: dict[str, Any], key: str, context: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{context} must define non-empty string key '{key}'.")
    return value.strip()


def _required_float(raw: dict[str, Any], key: str, context: str) -> float:
    value = raw.get(key)
    if isinstance(value, bool):
        raise ConfigError(f"{context} must define numeric key '{key}', not a boolean.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{context} must define numeric key '{key}'.") from exc
    if not math.isfinite(parsed):
        raise ConfigError(f"{context} must define finite numeric key '{key}'.")
    return parsed


def _required_integer_bound(raw: dict[str, Any], key: str, context: str) -> float:
    parsed = _required_float(raw, key, context)
    if parsed % 1 != 0:
        raise ConfigError(f"{context} must define integer-valued key '{key}': value={parsed:g}.")
    return parsed


def _required_discrete_values(raw: dict[str, Any], name: str) -> tuple[float, ...]:
    values = raw.get("values")
    if not isinstance(values, list) or not values:
        raise ConfigError(f"Variable '{name}' must define non-empty list key 'values'.")

    parsed_values: list[float] = []
    seen: set[float] = set()
    for index, value in enumerate(values):
        if isinstance(value, bool):
            raise ConfigError(
                f"Variable '{name}' has non-numeric discrete value at index {index}: "
                f"value={value!r}."
            )
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                f"Variable '{name}' has non-numeric discrete value at index {index}: "
                f"value={value!r}."
            ) from exc
        if not math.isfinite(parsed):
            raise ConfigError(
                f"Variable '{name}' has non-finite discrete value at index {index}: "
                f"value={value!r}."
            )
        if parsed in seen:
            raise ConfigError(
                f"Variable '{name}' has duplicate discrete value after numeric parsing: "
                f"value={parsed:g}."
            )
        seen.add(parsed)
        parsed_values.append(parsed)
    return tuple(parsed_values)


def _required_categorical_values(raw: dict[str, Any], name: str) -> tuple[str, ...]:
    values = raw.get("values")
    if not isinstance(values, list) or not values:
        raise ConfigError(f"Variable '{name}' must define non-empty list key 'values'.")

    parsed_values: list[str] = []
    seen: set[str] = set()
    for index, value in enumerate(values):
        if not isinstance(value, str):
            raise ConfigError(
                f"Variable '{name}' has non-string categorical value at index {index}: "
                f"value={value!r}."
            )
        if value == "" or value.strip() != value:
            raise ConfigError(
                f"Variable '{name}' has blank or whitespace-padded categorical value "
                f"at index {index}: value={value!r}."
            )
        if value in seen:
            raise ConfigError(
                f"Variable '{name}' has duplicate categorical value: value={value!r}."
            )
        seen.add(value)
        parsed_values.append(value)
    return tuple(parsed_values)


def _reject_unsupported_variable_keys(
    raw: dict[str, Any],
    name: str,
    variable_type: str,
) -> None:
    if variable_type in {"continuous", "integer"}:
        allowed = {"name", "type", "lower", "upper"}
    else:
        allowed = {"name", "type", "values"}
    unsupported = sorted(set(raw) - allowed)
    if unsupported:
        raise ConfigError(
            f"Variable '{name}' has unsupported keys for type='{variable_type}': "
            f"{unsupported}."
        )


def _positive_int(value: Any, context: str) -> int:
    parsed = _int_value(value, context)
    if parsed < 1:
        raise ConfigError(f"{context} must be >= 1: value={parsed}.")
    return parsed


def _non_negative_int(value: Any, context: str) -> int:
    parsed = _int_value(value, context)
    if parsed < 0:
        raise ConfigError(f"{context} must be >= 0: value={parsed}.")
    return parsed


def _int_value(value: Any, context: str) -> int:
    if isinstance(value, bool):
        raise ConfigError(f"{context} must be an integer, not a boolean.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{context} must be an integer.") from exc
    if parsed != value and not (isinstance(value, str) and str(parsed) == value):
        raise ConfigError(f"{context} must be an integer: value={value!r}.")
    return parsed
