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
    "stage",
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
class FidelityConfig:
    variable: str
    target: float
    fixed_cost: float = 0.01
    fidelity_cost_weight: float = 1.0
    num_fantasies: int = 64


@dataclass(frozen=True)
class ContextConfig:
    variables: tuple[str, ...]
    default_values: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ReviewConfig:
    enabled: bool = False


@dataclass(frozen=True)
class ReplicateConfig:
    enabled: bool = False
    suggestion_policy: str = "uncertain_best"
    replicate_threshold: float = 0.10
    min_repeats_at_best: int = 3
    max_repeats_per_group: int = 5
    noise_floor: float = 1.0e-8


@dataclass(frozen=True)
class ModelConfig:
    profile: str = "default"


@dataclass(frozen=True)
class StageConfig:
    name: str
    variables: tuple[str, ...]


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
    fidelity: FidelityConfig | None = None
    context: ContextConfig | None = None
    model: ModelConfig = field(default_factory=ModelConfig)
    review: ReviewConfig = field(default_factory=ReviewConfig)
    replicates: ReplicateConfig = field(default_factory=ReplicateConfig)
    stages: tuple[StageConfig, ...] = ()

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
    def is_structured_campaign(self) -> bool:
        """Return True when this campaign defines staged variable activity."""
        return bool(self.stages)

    @property
    def is_contextual_campaign(self) -> bool:
        """Return True when this campaign fixes context variables at suggestion time."""
        return self.context is not None

    @property
    def context_variable_names(self) -> list[str]:
        """Return configured context variable names in YAML order."""
        if self.context is None:
            return []
        return list(self.context.variables)

    @property
    def decision_variable_names(self) -> list[str]:
        """Return variables optimized by suggestions after fixing context."""
        context_names = set(self.context_variable_names)
        return [name for name in self.variable_names if name not in context_names]

    @property
    def stage_names(self) -> list[str]:
        """Return configured stage names in YAML order."""
        return [stage.name for stage in self.stages]

    def active_variable_names_for_stage(self, stage_name: str) -> list[str]:
        """Return active variable names for a configured stage."""
        for stage in self.stages:
            if stage.name == stage_name:
                return list(stage.variables)
        raise ConfigError(f"Unknown campaign stage '{stage_name}'.")

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
    fidelity = _parse_fidelity(raw.get("fidelity"), variables)
    stages = _parse_stages(raw.get("stages"), variables)
    context = _parse_context(raw.get("context"), variables)
    if stages and cost is not None:
        raise ConfigError(
            "Structured campaigns with cost are not supported in v1.4.0; "
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


def is_structured_campaign(config: CampaignConfig) -> bool:
    """Return True when a campaign config defines structured stages."""
    return config.is_structured_campaign


def configured_stage_names(config: CampaignConfig) -> list[str]:
    """Return structured campaign stage names in configured order."""
    return config.stage_names


def active_variables_for_stage(config: CampaignConfig, stage_name: str) -> list[str]:
    """Return active variable names for one structured campaign stage."""
    return config.active_variable_names_for_stage(stage_name)


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
        - {"variable", "target", "fixed_cost", "fidelity_cost_weight", "num_fantasies"}
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
    return FidelityConfig(
        variable=variable_name,
        target=target,
        fixed_cost=_positive_float(raw.get("fixed_cost", 0.01), "fidelity.fixed_cost"),
        fidelity_cost_weight=_positive_float(
            raw.get("fidelity_cost_weight", 1.0),
            "fidelity.fidelity_cost_weight",
        ),
        num_fantasies=_positive_int(raw.get("num_fantasies", 64), "fidelity.num_fantasies"),
    )


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

    raw_variables = raw.get("variables")
    if not isinstance(raw_variables, list) or not raw_variables:
        raise ConfigError("context.variables must be a non-empty list.")
    configured_variables = {variable.name: variable for variable in variables}
    context_variables: list[str] = []
    seen: set[str] = set()
    for index, value in enumerate(raw_variables):
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
    if len(context_variables) == len(variables):
        raise ConfigError(
            "context.variables cannot include every configured variable; at least one "
            "non-context decision variable is required."
        )

    raw_defaults = raw.get("default_values", {})
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

    return ContextConfig(variables=tuple(context_variables), default_values=default_values)


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
            "campaigns configured with bo.acquisition: log_ei or qlog_nei "
            "in v2.3.x; "
            "use model.profile: default for multi-objective campaigns."
        )
    if fidelity is not None:
        raise ConfigError(
            "Non-default model profiles cannot be combined with fidelity campaigns "
            "in v2.3.x; use model.profile: default."
        )
    if stages:
        raise ConfigError(
            "Non-default model profiles cannot be combined with structured campaign "
            "stages in v2.3.x; use model.profile: default."
        )
    if bo.acquisition not in {"log_ei", "qlog_nei"}:
        raise ConfigError(
            "Non-default model profiles require bo.acquisition: log_ei or "
            "qlog_nei in v2.3.x."
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
        raise ConfigError("bo.acquisition='qlog_nei' is single-objective only in v2.3.x.")
    if fidelity is not None:
        raise ConfigError("bo.acquisition='qlog_nei' cannot be combined with fidelity in v2.3.x.")
    if stages:
        raise ConfigError(
            "bo.acquisition='qlog_nei' cannot be combined with structured stages in v2.3.x."
        )
    if context is not None:
        raise ConfigError("bo.acquisition='qlog_nei' cannot be combined with context in v2.3.x.")
    if cost is not None:
        raise ConfigError(
            "bo.acquisition='qlog_nei' cannot be combined with cost-aware campaigns in v2.3.x."
        )
    if replicates.enabled and replicates.suggestion_policy == "uncertain_best":
        raise ConfigError(
            "bo.acquisition='qlog_nei' supports replicate campaigns only with "
            "replicates.suggestion_policy: new_only in v2.3.x."
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
            "multi-objective campaigns in v2.3.0."
        )
    if objective_count > 4:
        raise ConfigError(
            "bo.acquisition='qlog_nehvi' supports at most 4 objectives in v2.3.0: "
            f"configured={objective_count}."
        )
    if fidelity is not None:
        raise ConfigError(
            "bo.acquisition='qlog_nehvi' cannot be combined with fidelity in v2.3.0."
        )
    if stages:
        raise ConfigError(
            "bo.acquisition='qlog_nehvi' cannot be combined with structured stages "
            "in v2.3.0."
        )
    if context is not None:
        raise ConfigError(
            "bo.acquisition='qlog_nehvi' cannot be combined with context in v2.3.0."
        )
    if cost is not None:
        raise ConfigError(
            "bo.acquisition='qlog_nehvi' cannot be combined with cost-aware campaigns "
            "in v2.3.0."
        )
    if replicates.enabled:
        raise ConfigError(
            "bo.acquisition='qlog_nehvi' cannot be combined with replicate campaigns "
            "in v2.3.0."
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
        raise ConfigError("context is only supported for single-objective campaigns in v2.3.0.")
    if stages:
        raise ConfigError("context cannot be combined with structured campaign stages in v2.3.0.")
    if fidelity is not None:
        raise ConfigError("context cannot be combined with fidelity campaigns in v2.3.0.")
    if replicates.enabled:
        raise ConfigError("context cannot be combined with replicate campaigns in v2.3.0.")


def _validate_fidelity_combinations(
    *,
    fidelity: FidelityConfig | None,
    variables: list[VariableConfig],
    multi_objective: bool,
    stages: list[StageConfig],
    cost: CostConfig | None,
    replicates: ReplicateConfig,
) -> None:
    if fidelity is None:
        return
    if multi_objective:
        raise ConfigError("fidelity is only supported for single-objective campaigns in v1.4.0.")
    if stages:
        raise ConfigError("fidelity cannot be combined with structured campaign stages in v1.4.0.")
    if cost is not None:
        raise ConfigError("fidelity cannot be combined with cost-aware campaigns in v1.4.0.")
    if replicates.enabled:
        raise ConfigError("fidelity cannot be combined with replicate campaigns in v1.4.0.")
    unsupported = [variable.name for variable in variables if variable.type != "continuous"]
    if unsupported:
        raise ConfigError(
            "fidelity campaigns only support continuous variables in v1.4.0: "
            f"non_continuous={unsupported}."
        )


def _parse_stages(raw: Any, variables: list[VariableConfig]) -> list[StageConfig]:
    if raw is None:
        return []
    if not isinstance(raw, list) or not raw:
        raise ConfigError("Config key 'stages' must be a non-empty list when provided.")

    configured_variables = {variable.name for variable in variables}
    stages: list[StageConfig] = []
    seen_names: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ConfigError(f"Stage at index {index} must be a mapping.")
        unsupported = sorted(set(item) - {"name", "variables"})
        if unsupported:
            raise ConfigError(f"Stage at index {index} has unsupported keys: {unsupported}.")
        name = _required_str(item, "name", f"stages[{index}]")
        if name in seen_names:
            raise ConfigError(f"Duplicate stage name '{name}'.")
        variables_raw = item.get("variables")
        if not isinstance(variables_raw, list) or not variables_raw:
            raise ConfigError(
                f"Stage '{name}' must define non-empty list key 'variables'."
            )
        active_variables: list[str] = []
        seen_variables: set[str] = set()
        for variable_index, variable_name in enumerate(variables_raw):
            if not isinstance(variable_name, str) or not variable_name.strip():
                raise ConfigError(
                    f"Stage '{name}' variable at index {variable_index} must be "
                    "a non-empty string."
                )
            cleaned = variable_name.strip()
            if cleaned != variable_name:
                raise ConfigError(
                    f"Stage '{name}' variable at index {variable_index} has "
                    f"surrounding whitespace: value={variable_name!r}."
                )
            if cleaned in seen_variables:
                raise ConfigError(
                    f"Stage '{name}' lists duplicate variable '{cleaned}'."
                )
            if cleaned not in configured_variables:
                raise ConfigError(
                    f"Stage '{name}' references unknown variable '{cleaned}'."
                )
            active_variables.append(cleaned)
            seen_variables.add(cleaned)
        stages.append(StageConfig(name=name, variables=tuple(active_variables)))
        seen_names.add(name)
    return stages


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


def _parse_replicates(raw: Any, *, multi_objective: bool = False) -> ReplicateConfig:
    if raw is None:
        return ReplicateConfig()
    if not isinstance(raw, dict):
        raise ConfigError("Config key 'replicates' must be a mapping when provided.")
    supported = {
        "enabled",
        "suggestion_policy",
        "replicate_threshold",
        "min_repeats_at_best",
        "max_repeats_per_group",
        "noise_floor",
    }
    unsupported = sorted(set(raw) - supported)
    if unsupported:
        raise ConfigError(f"Config key 'replicates' has unsupported keys: {unsupported}.")
    enabled = raw.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ConfigError("replicates.enabled must be a boolean.")
    default_policy = "new_only" if multi_objective and enabled else "uncertain_best"
    suggestion_policy = str(raw.get("suggestion_policy", default_policy))
    if suggestion_policy not in {"uncertain_best", "new_only"}:
        raise ConfigError(
            "replicates.suggestion_policy must be one of "
            "['new_only', 'uncertain_best']."
        )
    if multi_objective and enabled and suggestion_policy == "uncertain_best":
        raise ConfigError(
            "replicates.suggestion_policy='uncertain_best' is only supported for "
            "single-objective campaigns in v1.1.2; use 'new_only' for "
            "multi-objective replicate campaigns."
        )
    replicate_threshold = _positive_float(
        raw.get("replicate_threshold", 0.10),
        "replicates.replicate_threshold",
    )
    min_repeats_at_best = _positive_int(
        raw.get("min_repeats_at_best", 3),
        "replicates.min_repeats_at_best",
    )
    max_repeats_per_group = _positive_int(
        raw.get("max_repeats_per_group", 5),
        "replicates.max_repeats_per_group",
    )
    if min_repeats_at_best > max_repeats_per_group:
        raise ConfigError(
            "replicates.min_repeats_at_best must be <= "
            "replicates.max_repeats_per_group: "
            f"min_repeats_at_best={min_repeats_at_best}, "
            f"max_repeats_per_group={max_repeats_per_group}."
        )
    noise_floor = _positive_float(
        raw.get("noise_floor", 1.0e-8),
        "replicates.noise_floor",
    )
    return ReplicateConfig(
        enabled=enabled,
        suggestion_policy=suggestion_policy,
        replicate_threshold=replicate_threshold,
        min_repeats_at_best=min_repeats_at_best,
        max_repeats_per_group=max_repeats_per_group,
        noise_floor=noise_floor,
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


def _positive_float(value: Any, context: str) -> float:
    parsed = _non_negative_float(value, context)
    if parsed <= 0:
        raise ConfigError(f"{context} must be > 0: value={parsed:g}.")
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


def _normalise_config_variable_value(
    variable: VariableConfig,
    value: Any,
    context: str,
) -> object:
    if variable.type == "continuous":
        parsed = _finite_config_float(value, context)
        assert variable.lower is not None and variable.upper is not None
        if parsed < variable.lower or parsed > variable.upper:
            raise ConfigError(
                f"{context} is outside variable '{variable.name}' bounds: "
                f"value={parsed:g}, lower={variable.lower:g}, upper={variable.upper:g}."
            )
        return parsed
    if variable.type == "integer":
        parsed = _finite_config_float(value, context)
        if parsed % 1 != 0:
            raise ConfigError(f"{context} must be integer-valued: value={value!r}.")
        assert variable.lower is not None and variable.upper is not None
        if parsed < variable.lower or parsed > variable.upper:
            raise ConfigError(
                f"{context} is outside variable '{variable.name}' bounds: "
                f"value={parsed:g}, lower={variable.lower:g}, upper={variable.upper:g}."
            )
        return int(parsed)
    if variable.type == "discrete":
        parsed = _finite_config_float(value, context)
        allowed = [float(item) for item in variable.values]
        for allowed_value in allowed:
            if math.isclose(parsed, allowed_value, rel_tol=1e-12, abs_tol=1e-12):
                return float(allowed_value)
        raise ConfigError(
            f"{context} is not one of variable '{variable.name}' choices: "
            f"value={value!r}, allowed={allowed}."
        )
    if variable.type == "categorical":
        parsed = str(value)
        allowed = [str(item) for item in variable.values]
        if parsed not in allowed:
            raise ConfigError(
                f"{context} is not one of variable '{variable.name}' choices: "
                f"value={value!r}, allowed={allowed}."
            )
        return parsed
    raise ConfigError(f"Variable '{variable.name}' has unsupported type '{variable.type}'.")


def _finite_config_float(value: Any, context: str) -> float:
    if isinstance(value, bool):
        raise ConfigError(f"{context} must be numeric, not a boolean.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{context} must be numeric.") from exc
    if not math.isfinite(parsed):
        raise ConfigError(f"{context} must be finite.")
    return parsed
