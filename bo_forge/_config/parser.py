"""Parse YAML mappings and validate supported feature combinations."""

from __future__ import annotations

import math
from typing import Any

from bo_forge._config.sections import (
    _non_negative_float,
    _non_negative_int,
    _normalise_config_variable_value,
    _parse_replicates,
    _parse_review,
    _parse_stages,
    _positive_float,
    _positive_int,
    _reject_reserved_prefix_name,
    _reject_unsupported_variable_keys,
    _required_categorical_values,
    _required_discrete_values,
    _required_float,
    _required_integer_bound,
    _required_str,
)
from bo_forge.config import (
    FIDELITY_MATCH_ABS_TOL,
    FIDELITY_MATCH_REL_TOL,
    RESERVED_COLUMNS,
    BOConfig,
    CampaignConfig,
    ConstraintConfig,
    ContextConfig,
    CostConfig,
    FidelityConfig,
    ModelConfig,
    ObjectiveConfig,
    ReplicateConfig,
    StageConfig,
    VariableConfig,
    fidelity_values_match,
)
from bo_forge.errors import ConfigError


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
    fidelity = _parse_fidelity(raw.get("fidelity"), variables)
    stages = _parse_stages(raw.get("stages"), variables)
    context = _parse_context(raw.get("context"), variables)
    if stages and cost is not None:
        raise ConfigError(
            "Structured campaigns with cost are currently unsupported; "
            "remove either 'stages' or 'cost'."
        )
    review = _parse_review(raw.get("review"))
    replicates = _parse_replicates(
        raw.get("replicates"),
        multi_objective=bool(objectives),
    )
    model = _parse_model(raw.get("model"))
    bo = _parse_bo(
        raw.get("bo", {}),
        multi_objective=bool(objectives),
        has_fidelity=fidelity is not None,
    )
    _validate_qlog_nehvi_combinations(
        bo=bo,
        multi_objective=bool(objectives),
        objective_count=len(objectives) if objectives else 1,
        fidelity=fidelity,
        stages=stages,
        context=context,
        cost=cost,
        replicates=replicates,
    )
    _validate_context_combinations(
        context=context,
        multi_objective=bool(objectives),
        stages=stages,
        fidelity=fidelity,
        cost=cost,
        replicates=replicates,
    )
    _validate_fidelity_combinations(
        fidelity=fidelity,
        bo=bo,
        variables=variables,
        multi_objective=bool(objectives),
        stages=stages,
        cost=cost,
        replicates=replicates,
    )
    _validate_model_combinations(
        model=model,
        bo=bo,
        multi_objective=bool(objectives),
        fidelity=fidelity,
        stages=stages,
    )
    _validate_qlog_nei_combinations(
        bo=bo,
        multi_objective=bool(objectives),
        fidelity=fidelity,
        stages=stages,
        context=context,
        cost=cost,
        replicates=replicates,
    )

    return CampaignConfig(
        campaign_name=campaign_name,
        objective=objective,
        variables=tuple(variables),
        bo=bo,
        objectives=tuple(objectives),
        constraints=tuple(constraints),
        cost=cost,
        fidelity=fidelity,
        context=context,
        model=model,
        review=review,
        replicates=replicates,
        stages=tuple(stages),
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
    if len(raw) < 2:
        raise ConfigError(
            "Config key 'objectives' must contain at least two objectives."
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
        variable = _parse_variable(item, index, objective_names, seen_names)
        variables.append(variable)
        seen_names.add(variable.name)

    return variables


def _parse_variable(
    raw: Any,
    index: int,
    objective_names: set[str],
    seen_names: set[str],
) -> VariableConfig:
    if not isinstance(raw, dict):
        raise ConfigError(f"Variable at index {index} must be a mapping.")
    name = _required_str(raw, "name", f"variables[{index}]")
    variable_type = _required_str(raw, "type", f"variable '{name}'")
    if variable_type not in {"continuous", "integer", "discrete", "categorical"}:
        raise ConfigError(
            f"Variable '{name}' has unsupported type '{variable_type}'. "
            "Expected one of ['categorical', 'continuous', 'discrete', 'integer']."
        )
    _reject_unsupported_variable_keys(raw, name, variable_type)
    _validate_variable_name(name, objective_names, seen_names)
    if variable_type in {"continuous", "integer"}:
        return _parse_bounded_variable(raw, name, variable_type)
    values = (
        _required_discrete_values(raw, name)
        if variable_type == "discrete"
        else _required_categorical_values(raw, name)
    )
    return VariableConfig(name=name, type=variable_type, values=values)


def _validate_variable_name(
    name: str,
    objective_names: set[str],
    seen_names: set[str],
) -> None:
    if name in seen_names:
        raise ConfigError(f"Duplicate variable name '{name}'.")
    if name in objective_names:
        raise ConfigError(f"Variable '{name}' conflicts with configured objective names.")
    if name in RESERVED_COLUMNS:
        raise ConfigError(f"Variable name '{name}' conflicts with a reserved CSV column.")
    _reject_reserved_prefix_name(name, "Variable")


def _parse_bounded_variable(
    raw: dict[str, Any],
    name: str,
    variable_type: str,
) -> VariableConfig:
    bound_reader = _required_float if variable_type == "continuous" else _required_integer_bound
    lower = bound_reader(raw, "lower", f"variable '{name}'")
    upper = bound_reader(raw, "upper", f"variable '{name}'")
    invalid = lower >= upper if variable_type == "continuous" else lower > upper
    if invalid:
        operator = ">=" if variable_type == "continuous" else ">"
        raise ConfigError(
            f"Variable '{name}' has lower {operator} upper: lower={lower:g}, upper={upper:g}."
        )
    return VariableConfig(name=name, type=variable_type, lower=lower, upper=upper)


def _parse_bo(
    raw: Any,
    *,
    multi_objective: bool = False,
    has_fidelity: bool = False,
) -> BOConfig:
    if not isinstance(raw, dict):
        raise ConfigError("Config key 'bo' must be a mapping when provided.")

    if has_fidelity:
        default_acquisition = "qmf_kg"
    elif multi_objective:
        default_acquisition = "qlog_ehvi"
    else:
        default_acquisition = "log_ei"
    acquisition = str(raw.get("acquisition", default_acquisition))
    if acquisition == "qmf_kg" and not has_fidelity:
        raise ConfigError("bo.acquisition='qmf_kg' requires a 'fidelity' config section.")
    if has_fidelity:
        supported = {"qmf_kg"}
    elif multi_objective:
        supported = {"qlog_ehvi", "qlog_nehvi"}
    else:
        supported = {"log_ei", "qlog_nei"}
    if acquisition not in supported and acquisition not in {"qlog_nei", "qlog_nehvi"}:
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


def _parse_fidelity(
    raw: Any,
    variables: list[VariableConfig],
) -> FidelityConfig | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ConfigError("Config key 'fidelity' must be a mapping when provided.")
    unsupported = sorted(
        set(raw)
        - {
            "variable",
            "target",
            "fixed_cost",
            "fidelity_cost_weight",
            "num_fantasies",
            "levels",
            "optimizer_maxiter",
            "optimizer_timeout_seconds",
        }
    )
    if unsupported:
        raise ConfigError(f"Config key 'fidelity' has unsupported keys: {unsupported}.")
    variable_name = _required_str(raw, "variable", "fidelity")
    variable_by_name = {variable.name: variable for variable in variables}
    if variable_name not in variable_by_name:
        raise ConfigError(
            f"fidelity.variable references unknown variable '{variable_name}'."
        )
    variable = variable_by_name[variable_name]
    if variable.type != "continuous":
        raise ConfigError(
            f"fidelity.variable '{variable_name}' must be a continuous variable."
        )
    target = _required_float(raw, "target", "fidelity")
    assert variable.lower is not None and variable.upper is not None
    if target < variable.lower or target > variable.upper:
        raise ConfigError(
            f"fidelity.target must be within variable '{variable_name}' bounds: "
            f"target={target:g}, lower={variable.lower:g}, upper={variable.upper:g}."
        )
    levels = _parse_fidelity_levels(raw.get("levels"), variable, target)
    optimizer_timeout_seconds = None
    if raw.get("optimizer_timeout_seconds") is not None:
        optimizer_timeout_seconds = _positive_float(
            raw["optimizer_timeout_seconds"],
            "fidelity.optimizer_timeout_seconds",
        )
    return FidelityConfig(
        variable=variable_name,
        target=target,
        fixed_cost=_positive_float(raw.get("fixed_cost", 0.01), "fidelity.fixed_cost"),
        fidelity_cost_weight=_positive_float(
            raw.get("fidelity_cost_weight", 1.0),
            "fidelity.fidelity_cost_weight",
        ),
        num_fantasies=_positive_int(raw.get("num_fantasies", 64), "fidelity.num_fantasies"),
        levels=levels,
        optimizer_maxiter=_positive_int(
            raw.get("optimizer_maxiter", 200),
            "fidelity.optimizer_maxiter",
        ),
        optimizer_timeout_seconds=optimizer_timeout_seconds,
    )


def _parse_fidelity_levels(
    raw: Any,
    variable: VariableConfig,
    target: float,
) -> tuple[float, ...] | None:
    if raw is None:
        return None
    if not isinstance(raw, list) or len(raw) < 2:
        raise ConfigError("fidelity.levels must be a list containing at least two values.")
    levels: list[float] = []
    for index, value in enumerate(raw):
        if isinstance(value, bool):
            raise ConfigError(f"fidelity.levels[{index}] must be a finite number.")
        try:
            level = float(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                f"fidelity.levels[{index}] must be a finite number: value={value!r}."
            ) from exc
        if not math.isfinite(level):
            raise ConfigError(f"fidelity.levels[{index}] must be a finite number.")
        levels.append(level)

    if any(
        current <= previous
        for previous, current in zip(levels, levels[1:], strict=False)
    ):
        raise ConfigError("fidelity.levels must be strictly increasing.")
    for previous, current in zip(levels, levels[1:], strict=False):
        matching_radius = max(
            FIDELITY_MATCH_ABS_TOL,
            FIDELITY_MATCH_REL_TOL * max(abs(previous), abs(current)),
        )
        if current - previous <= 2 * matching_radius:
            raise ConfigError(
                "fidelity.levels must be separated enough to map CSV values "
                "unambiguously under the 1e-9 numeric tolerance: "
                f"levels={previous:g}, {current:g}."
            )
    assert variable.lower is not None and variable.upper is not None
    outside = [
        level
        for level in levels
        if level < variable.lower or level > variable.upper
    ]
    if outside:
        raise ConfigError(
            f"fidelity.levels must stay within variable '{variable.name}' bounds: "
            f"outside={outside}, lower={variable.lower:g}, upper={variable.upper:g}."
        )
    if not fidelity_values_match(levels[-1], target):
        raise ConfigError(
            "fidelity.target must equal the highest configured fidelity level: "
            f"target={target:g}, highest_level={levels[-1]:g}."
        )
    return tuple(levels)


def _parse_context(
    raw: Any,
    variables: list[VariableConfig],
) -> ContextConfig | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ConfigError("Config key 'context' must be a mapping when provided.")
    unsupported = sorted(set(raw) - {"variables", "default_values"})
    if unsupported:
        raise ConfigError(f"Config key 'context' has unsupported keys: {unsupported}.")

    configured_variables = {variable.name: variable for variable in variables}
    context_variables = _parse_context_variable_names(
        raw.get("variables"),
        configured_variables,
    )
    if len(context_variables) == len(variables):
        raise ConfigError(
            "context.variables cannot include every configured variable; at least one "
            "non-context decision variable is required."
        )
    default_values = _parse_context_default_values(
        raw.get("default_values", {}),
        context_variables,
        configured_variables,
    )
    return ContextConfig(variables=tuple(context_variables), default_values=default_values)


def _parse_context_variable_names(
    raw: Any,
    configured_variables: dict[str, VariableConfig],
) -> list[str]:
    if not isinstance(raw, list) or not raw:
        raise ConfigError("context.variables must be a non-empty list.")
    context_variables: list[str] = []
    seen: set[str] = set()
    for index, value in enumerate(raw):
        if not isinstance(value, str) or not value.strip() or value.strip() != value:
            raise ConfigError(
                f"context.variables[{index}] must be a non-empty unpadded string."
            )
        if value in seen:
            raise ConfigError(f"Duplicate context variable '{value}'.")
        if value not in configured_variables:
            raise ConfigError(f"context.variables references unknown variable '{value}'.")
        context_variables.append(value)
        seen.add(value)
    return context_variables


def _parse_context_default_values(
    raw_defaults: Any,
    context_variables: list[str],
    configured_variables: dict[str, VariableConfig],
) -> dict[str, object]:
    if raw_defaults is None:
        raw_defaults = {}
    if not isinstance(raw_defaults, dict):
        raise ConfigError("context.default_values must be a mapping when provided.")
    context_set = set(context_variables)
    default_values: dict[str, object] = {}
    for name, value in raw_defaults.items():
        if not isinstance(name, str) or not name.strip() or name.strip() != name:
            raise ConfigError("context.default_values keys must be non-empty strings.")
        if name not in context_set:
            raise ConfigError(
                f"context.default_values contains non-context variable '{name}'."
            )
        default_values[name] = _normalise_config_variable_value(
            configured_variables[name],
            value,
            f"context.default_values.{name}",
        )
    return default_values


def _parse_model(raw: Any) -> ModelConfig:
    if raw is None:
        return ModelConfig()
    if not isinstance(raw, dict):
        raise ConfigError("Config key 'model' must be a mapping when provided.")
    unsupported = sorted(set(raw) - {"profile"})
    if unsupported:
        raise ConfigError(f"Config key 'model' has unsupported keys: {unsupported}.")
    profile = str(raw.get("profile", "default"))
    if profile not in {"default", "smooth", "rough", "robust"}:
        raise ConfigError(
            "model.profile must be one of ['default', 'rough', 'robust', 'smooth']."
        )
    return ModelConfig(profile=profile)


def _validate_model_combinations(
    *,
    model: ModelConfig,
    bo: BOConfig,
    multi_objective: bool,
    fidelity: FidelityConfig | None,
    stages: list[StageConfig],
) -> None:
    if model.profile == "default":
        return
    if multi_objective:
        raise ConfigError(
            "Non-default model profiles are only supported for single-objective "
            "campaigns configured with bo.acquisition: log_ei or qlog_nei; "
            "use model.profile: default for multi-objective campaigns."
        )
    if fidelity is not None:
        raise ConfigError(
            "Non-default model profiles cannot be combined with fidelity campaigns "
            "because that model path requires model.profile: default."
        )
    if stages:
        raise ConfigError(
            "Non-default model profiles cannot be combined with structured campaign "
            "stages; use model.profile: default."
        )
    if bo.acquisition not in {"log_ei", "qlog_nei"}:
        raise ConfigError(
            "Non-default model profiles require bo.acquisition: log_ei or "
            "qlog_nei."
        )


def _validate_qlog_nei_combinations(
    *,
    bo: BOConfig,
    multi_objective: bool,
    fidelity: FidelityConfig | None,
    stages: list[StageConfig],
    context: ContextConfig | None,
    cost: CostConfig | None,
    replicates: ReplicateConfig,
) -> None:
    if bo.acquisition != "qlog_nei":
        return
    if multi_objective:
        raise ConfigError("bo.acquisition='qlog_nei' is single-objective only.")
    if fidelity is not None:
        raise ConfigError("bo.acquisition='qlog_nei' cannot be combined with fidelity.")
    if stages:
        raise ConfigError(
            "bo.acquisition='qlog_nei' cannot be combined with structured stages."
        )
    if context is not None:
        raise ConfigError("bo.acquisition='qlog_nei' cannot be combined with context.")
    if cost is not None:
        raise ConfigError(
            "bo.acquisition='qlog_nei' cannot be combined with cost-aware campaigns."
        )
    if replicates.enabled and replicates.suggestion_policy == "uncertain_best":
        raise ConfigError(
            "bo.acquisition='qlog_nei' supports replicate campaigns only with "
            "replicates.suggestion_policy: new_only."
        )


def _validate_qlog_nehvi_combinations(
    *,
    bo: BOConfig,
    multi_objective: bool,
    objective_count: int,
    fidelity: FidelityConfig | None,
    stages: list[StageConfig],
    context: ContextConfig | None,
    cost: CostConfig | None,
    replicates: ReplicateConfig,
) -> None:
    if bo.acquisition != "qlog_nehvi":
        return
    if not multi_objective:
        raise ConfigError(
            "bo.acquisition='qlog_nehvi' is only supported for coupled "
            "multi-objective campaigns."
        )
    if objective_count > 4:
        raise ConfigError(
            "bo.acquisition='qlog_nehvi' supports at most 4 objectives: "
            f"configured={objective_count}."
        )
    if fidelity is not None:
        raise ConfigError(
            "bo.acquisition='qlog_nehvi' cannot be combined with fidelity."
        )
    if stages:
        raise ConfigError(
            "bo.acquisition='qlog_nehvi' cannot be combined with structured stages."
        )
    if context is not None:
        raise ConfigError(
            "bo.acquisition='qlog_nehvi' cannot be combined with context."
        )
    if cost is not None:
        raise ConfigError(
            "bo.acquisition='qlog_nehvi' cannot be combined with cost-aware campaigns."
        )
    if replicates.enabled:
        raise ConfigError(
            "bo.acquisition='qlog_nehvi' cannot be combined with replicate campaigns."
        )


def _validate_context_combinations(
    *,
    context: ContextConfig | None,
    multi_objective: bool,
    stages: list[StageConfig],
    fidelity: FidelityConfig | None,
    cost: CostConfig | None,
    replicates: ReplicateConfig,
) -> None:
    if context is None:
        return
    if multi_objective:
        raise ConfigError("context is only supported for single-objective campaigns.")
    if stages:
        raise ConfigError("context cannot be combined with structured campaign stages.")
    if fidelity is not None:
        raise ConfigError("context cannot be combined with fidelity campaigns.")


def _validate_fidelity_combinations(
    *,
    fidelity: FidelityConfig | None,
    bo: BOConfig,
    variables: list[VariableConfig],
    multi_objective: bool,
    stages: list[StageConfig],
    cost: CostConfig | None,
    replicates: ReplicateConfig,
) -> None:
    if fidelity is None:
        return
    if bo.batch_size > 4:
        raise ConfigError(
            "qMFKG supports bo.batch_size from 1 through 4: "
            f"configured={bo.batch_size}."
        )
    if multi_objective:
        raise ConfigError("fidelity is only supported for single-objective campaigns.")
    if stages:
        raise ConfigError("fidelity cannot be combined with structured campaign stages.")
    if cost is not None:
        raise ConfigError("fidelity cannot be combined with cost-aware campaigns.")
    if replicates.enabled:
        raise ConfigError("fidelity cannot be combined with replicate campaigns.")
    unsupported = [variable.name for variable in variables if variable.type != "continuous"]
    if unsupported:
        raise ConfigError(
            "fidelity campaigns only support continuous variables: "
            f"non_continuous={unsupported}."
        )
