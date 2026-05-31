"""Campaign configuration parsing and validation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from bo_forge.errors import ConfigError

RESERVED_COLUMNS = {
    "row_id",
    "iteration",
    "status",
    "source",
    "review_status",
    "review_note",
    "replicate_group",
    "replicate_index",
    "predicted_mean",
    "predicted_std",
    "acquisition",
    "cost_estimate",
    "cost_actual",
    "utility",
}
RESERVED_COLUMN_PREFIXES = ("predicted_mean_", "predicted_std_")


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
    reference_point: float | None = None


@dataclass(frozen=True)
class ConstraintConfig:
    name: str
    expression: str


@dataclass(frozen=True)
class CostConfig:
    expression: str
    weight: float = 1.0
    budget: float | None = None
    candidate_pool_size: int = 256
    top_k: int = 24


@dataclass(frozen=True)
class ReviewConfig:
    enabled: bool = False


@dataclass(frozen=True)
class ReplicateConfig:
    enabled: bool = False


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
    min_normalized_distance: float = 0.0


@dataclass(frozen=True)
class CampaignConfig:
    campaign_name: str
    objective: ObjectiveConfig | None
    variables: tuple[VariableConfig, ...]
    bo: BOConfig
    objectives: tuple[ObjectiveConfig, ...] = ()
    constraints: tuple[ConstraintConfig, ...] = ()
    cost: CostConfig | None = None
    review: ReviewConfig = field(default_factory=ReviewConfig)
    replicates: ReplicateConfig = field(default_factory=ReplicateConfig)

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
    def objective_names(self) -> list[str]:
        """Return objective names in configured order."""
        if self.objectives:
            return [objective.name for objective in self.objectives]
        if self.objective is None:
            return []
        return [self.objective.name]

    @property
    def is_multi_objective(self) -> bool:
        """Return True when this campaign has multiple objectives."""
        return bool(self.objectives)

    @property
    def direction_sign(self) -> float:
        """Return the multiplier that converts objective values to maximization."""
        if self.objective is None:
            raise ConfigError("direction_sign is only available for single-objective configs.")
        return 1.0 if self.objective.direction == "maximize" else -1.0


def parse_campaign_config(raw: Any) -> CampaignConfig:
    """Parse raw YAML data into a validated campaign config."""
    if not isinstance(raw, dict):
        raise ConfigError("Config file must contain a YAML mapping at the top level.")

    campaign_name = _required_str(raw, "campaign_name", "campaign")
    objective, objectives = _parse_objective_section(raw)
    objective_names = [item.name for item in objectives] if objectives else [objective.name]
    variables = _parse_variables(raw.get("variables"), set(objective_names))
    constraints = _parse_constraints(raw.get("constraints", []), variables)
    cost = _parse_cost(raw.get("cost"), variables)
    review = _parse_review(raw.get("review"))
    replicates = _parse_replicates(raw.get("replicates"))
    if objectives and cost is not None:
        raise ConfigError("Multi-objective configs do not support 'cost' in v1.1.0.")
    if objectives and review.enabled:
        raise ConfigError("Multi-objective configs do not support review.enabled: true in v1.1.0.")
    if objectives and replicates.enabled:
        raise ConfigError(
            "Multi-objective configs do not support replicates.enabled: true in v1.1.0."
        )
    bo = _parse_bo(raw.get("bo", {}), multi_objective=bool(objectives))

    return CampaignConfig(
        campaign_name=campaign_name,
        objective=objective,
        variables=tuple(variables),
        bo=bo,
        objectives=tuple(objectives),
        constraints=tuple(constraints),
        cost=cost,
        review=review,
        replicates=replicates,
    )


def _parse_objective_section(raw: dict[str, Any]) -> tuple[ObjectiveConfig, list[ObjectiveConfig]]:
    has_single = "objective" in raw
    has_multi = "objectives" in raw
    if has_single and has_multi:
        raise ConfigError("Config must define either 'objective' or 'objectives', not both.")
    if has_multi:
        objectives = _parse_objectives(raw.get("objectives"))
        return objectives[0], objectives
    return _parse_objective(raw.get("objective"))


def _parse_objective(raw: Any) -> tuple[ObjectiveConfig, list[ObjectiveConfig]]:
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
    _reject_reserved_prefix_name(name, "Objective")
    return ObjectiveConfig(name=name, direction=direction), []


def _parse_objectives(raw: Any) -> list[ObjectiveConfig]:
    if not isinstance(raw, list):
        raise ConfigError("Config key 'objectives' must be a list.")
    if len(raw) != 2:
        raise ConfigError(
            "Config key 'objectives' must contain exactly two objectives in v1.1.0."
        )

    objectives: list[ObjectiveConfig] = []
    seen_names: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ConfigError(f"Objective at index {index} must be a mapping.")
        unsupported = sorted(set(item) - {"name", "direction", "reference_point"})
        if unsupported:
            raise ConfigError(
                f"Objective at index {index} has unsupported keys: {unsupported}."
            )
        name = _required_str(item, "name", f"objectives[{index}]")
        direction = _required_str(item, "direction", f"objective '{name}'")
        if direction not in {"maximize", "minimize"}:
            raise ConfigError(
                f"Objective '{name}' has invalid direction '{direction}'. "
                "Expected 'maximize' or 'minimize'."
            )
        if name in seen_names:
            raise ConfigError(f"Duplicate objective name '{name}'.")
        if name in RESERVED_COLUMNS:
            raise ConfigError(f"Objective name '{name}' conflicts with a reserved CSV column.")
        _reject_reserved_prefix_name(name, "Objective")
        reference_point = _required_float(item, "reference_point", f"objective '{name}'")
        objectives.append(
            ObjectiveConfig(
                name=name,
                direction=direction,
                reference_point=reference_point,
            )
        )
        seen_names.add(name)
    return objectives


def _parse_variables(raw: Any, objective_names: set[str]) -> list[VariableConfig]:
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
        if name in objective_names:
            raise ConfigError(
                f"Variable '{name}' conflicts with configured objective names."
            )
        if name in RESERVED_COLUMNS:
            raise ConfigError(f"Variable name '{name}' conflicts with a reserved CSV column.")
        _reject_reserved_prefix_name(name, "Variable")

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


def _parse_bo(raw: Any, *, multi_objective: bool = False) -> BOConfig:
    if not isinstance(raw, dict):
        raise ConfigError("Config key 'bo' must be a mapping when provided.")

    default_acquisition = "qlog_ehvi" if multi_objective else "log_ei"
    acquisition = str(raw.get("acquisition", default_acquisition))
    supported = {"qlog_ehvi"} if multi_objective else {"log_ei"}
    if acquisition not in supported:
        raise ConfigError(
            f"Unsupported acquisition '{acquisition}'. "
            f"Expected one of {sorted(supported)}."
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
        min_normalized_distance=_non_negative_float(
            raw.get("min_normalized_distance", 0.0),
            "bo.min_normalized_distance",
        ),
    )


def _parse_constraints(
    raw: Any,
    variables: list[VariableConfig],
) -> list[ConstraintConfig]:
    if raw in (None, []):
        return []
    if not isinstance(raw, list):
        raise ConfigError("Config key 'constraints' must be a list when provided.")

    from bo_forge.constraints import validate_constraint_expression

    variable_names = {variable.name for variable in variables}
    constraints: list[ConstraintConfig] = []
    seen_names: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ConfigError(f"Constraint at index {index} must be a mapping.")
        unsupported = sorted(set(item) - {"name", "expression"})
        if unsupported:
            raise ConfigError(
                f"Constraint at index {index} has unsupported keys: {unsupported}."
            )
        name = _required_str(item, "name", f"constraints[{index}]")
        expression = _required_str(item, "expression", f"constraint '{name}'")
        if name in seen_names:
            raise ConfigError(f"Duplicate constraint name '{name}'.")
        validate_constraint_expression(
            name=name,
            expression=expression,
            variable_names=variable_names,
        )
        constraints.append(ConstraintConfig(name=name, expression=expression))
        seen_names.add(name)
    return constraints


def _parse_cost(raw: Any, variables: list[VariableConfig]) -> CostConfig | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ConfigError("Config key 'cost' must be a mapping when provided.")
    unsupported = sorted(
        set(raw) - {"expression", "weight", "budget", "candidate_pool_size", "top_k"}
    )
    if unsupported:
        raise ConfigError(f"Config key 'cost' has unsupported keys: {unsupported}.")

    from bo_forge.costs import validate_cost_expression

    expression = _required_str(raw, "expression", "cost")
    validate_cost_expression(
        expression=expression,
        variable_names={variable.name for variable in variables},
    )
    weight = _non_negative_float(raw.get("weight", 1.0), "cost.weight")
    budget = None
    if raw.get("budget") is not None:
        budget = _non_negative_float(raw.get("budget"), "cost.budget")
    candidate_pool_size = _positive_int(
        raw.get("candidate_pool_size", 256),
        "cost.candidate_pool_size",
    )
    top_k = _positive_int(raw.get("top_k", 24), "cost.top_k")
    if top_k > candidate_pool_size:
        raise ConfigError(
            "cost.top_k must be <= cost.candidate_pool_size: "
            f"top_k={top_k}, candidate_pool_size={candidate_pool_size}."
        )
    return CostConfig(
        expression=expression,
        weight=weight,
        budget=budget,
        candidate_pool_size=candidate_pool_size,
        top_k=top_k,
    )


def _parse_review(raw: Any) -> ReviewConfig:
    if raw is None:
        return ReviewConfig()
    if not isinstance(raw, dict):
        raise ConfigError("Config key 'review' must be a mapping when provided.")
    unsupported = sorted(set(raw) - {"enabled"})
    if unsupported:
        raise ConfigError(f"Config key 'review' has unsupported keys: {unsupported}.")
    enabled = raw.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ConfigError("review.enabled must be a boolean.")
    return ReviewConfig(enabled=enabled)


def _parse_replicates(raw: Any) -> ReplicateConfig:
    if raw is None:
        return ReplicateConfig()
    if not isinstance(raw, dict):
        raise ConfigError("Config key 'replicates' must be a mapping when provided.")
    unsupported = sorted(set(raw) - {"enabled"})
    if unsupported:
        raise ConfigError(f"Config key 'replicates' has unsupported keys: {unsupported}.")
    enabled = raw.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ConfigError("replicates.enabled must be a boolean.")
    return ReplicateConfig(enabled=enabled)


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


def _reject_reserved_prefix_name(name: str, context: str) -> None:
    for prefix in RESERVED_COLUMN_PREFIXES:
        if name.startswith(prefix):
            raise ConfigError(
                f"{context} name '{name}' conflicts with reserved CSV column prefix "
                f"'{prefix}'."
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


def _non_negative_float(value: Any, context: str) -> float:
    if isinstance(value, bool):
        raise ConfigError(f"{context} must be numeric, not a boolean.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{context} must be numeric.") from exc
    if not math.isfinite(parsed):
        raise ConfigError(f"{context} must be finite.")
    if parsed < 0:
        raise ConfigError(f"{context} must be >= 0: value={parsed:g}.")
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
