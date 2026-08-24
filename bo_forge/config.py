"""Public campaign configuration dataclasses and compatibility helpers."""

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
FIDELITY_MATCH_REL_TOL = 1e-9
FIDELITY_MATCH_ABS_TOL = 1e-9

def fidelity_values_match(left: object, right: object) -> bool:
    """Return whether two fidelity values match under the public CSV tolerance."""
    try:
        left_value = float(left)
        right_value = float(right)
    except (TypeError, ValueError):
        return False
    return math.isclose(
        left_value,
        right_value,
        rel_tol=FIDELITY_MATCH_REL_TOL,
        abs_tol=FIDELITY_MATCH_ABS_TOL,
    )


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
    levels: tuple[float, ...] | None = None
    optimizer_maxiter: int = 200
    optimizer_timeout_seconds: float | None = None


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


def is_structured_campaign(config: CampaignConfig) -> bool:
    """Return True when a campaign config defines structured stages."""
    return config.is_structured_campaign


def configured_stage_names(config: CampaignConfig) -> list[str]:
    """Return structured campaign stage names in configured order."""
    return config.stage_names


def active_variables_for_stage(config: CampaignConfig, stage_name: str) -> list[str]:
    """Return active variable names for one structured campaign stage."""
    return config.active_variable_names_for_stage(stage_name)


def parse_campaign_config(raw: Any) -> CampaignConfig:
    """Parse raw YAML data into a validated campaign config."""
    from bo_forge._config.parser import parse_campaign_config as _parse_campaign_config

    return _parse_campaign_config(raw)
