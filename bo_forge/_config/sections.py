"""Parse structured, review, replicate, and scalar config fields."""

from __future__ import annotations

import math
from typing import Any

from bo_forge.config import (
    RESERVED_COLUMN_PREFIXES,
    ReplicateConfig,
    ReviewConfig,
    StageConfig,
    VariableConfig,
)
from bo_forge.errors import ConfigError


def _parse_stages(raw: Any, variables: list[VariableConfig]) -> list[StageConfig]:
    if raw is None:
        return []
    if not isinstance(raw, list) or not raw:
        raise ConfigError("Config key 'stages' must be a non-empty list when provided.")

    configured_variables = {variable.name for variable in variables}
    stages: list[StageConfig] = []
    seen_names: set[str] = set()
    for index, item in enumerate(raw):
        stage = _parse_stage(item, index, configured_variables, seen_names)
        stages.append(stage)
        seen_names.add(stage.name)
    return stages


def _parse_stage(
    raw: Any,
    index: int,
    configured_variables: set[str],
    seen_names: set[str],
) -> StageConfig:
    if not isinstance(raw, dict):
        raise ConfigError(f"Stage at index {index} must be a mapping.")
    unsupported = sorted(set(raw) - {"name", "variables"})
    if unsupported:
        raise ConfigError(f"Stage at index {index} has unsupported keys: {unsupported}.")
    name = _required_str(raw, "name", f"stages[{index}]")
    if name in seen_names:
        raise ConfigError(f"Duplicate stage name '{name}'.")
    return StageConfig(
        name=name,
        variables=tuple(
            _parse_stage_variables(name, raw.get("variables"), configured_variables)
        ),
    )


def _parse_stage_variables(
    stage_name: str,
    raw: Any,
    configured_variables: set[str],
) -> list[str]:
    if not isinstance(raw, list) or not raw:
        raise ConfigError(
            f"Stage '{stage_name}' must define non-empty list key 'variables'."
        )
    active_variables: list[str] = []
    for index, variable_name in enumerate(raw):
        if not isinstance(variable_name, str) or not variable_name.strip():
            raise ConfigError(
                f"Stage '{stage_name}' variable at index {index} must be "
                "a non-empty string."
            )
        cleaned = variable_name.strip()
        if cleaned != variable_name:
            raise ConfigError(
                f"Stage '{stage_name}' variable at index {index} has "
                f"surrounding whitespace: value={variable_name!r}."
            )
        if cleaned in active_variables:
            raise ConfigError(
                f"Stage '{stage_name}' lists duplicate variable '{cleaned}'."
            )
        if cleaned not in configured_variables:
            raise ConfigError(
                f"Stage '{stage_name}' references unknown variable '{cleaned}'."
            )
        active_variables.append(cleaned)
    return active_variables


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
            "single-objective campaigns; use 'new_only' for "
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
    if isinstance(value, bool):
        raise ConfigError(f"{context} must be numeric, not a boolean.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{context} must be numeric.") from exc
    if not math.isfinite(parsed):
        raise ConfigError(f"{context} must be finite.")
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
