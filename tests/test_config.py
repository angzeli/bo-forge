from pathlib import Path

import pytest

from bo_forge.config import (
    CampaignConfig,
    active_variables_for_stage,
    configured_stage_names,
    is_structured_campaign,
)
from bo_forge.costs import evaluate_cost
from bo_forge.errors import ConfigError, LogValidationError


def write_yaml(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_config_from_yaml_parses_valid_config(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path / "campaign.yaml",
        """
campaign_name: photocatalyst_loading
objective:
  name: activity
  direction: maximize
variables:
  - name: precursor_ratio
    type: continuous
    lower: 0.0
    upper: 1.0
bo:
  batch_size: 2
  initial_design_size: 4
  acquisition: log_ei
""",
    )

    config = CampaignConfig.from_yaml(path)

    assert config.campaign_name == "photocatalyst_loading"
    assert config.objective.name == "activity"
    assert config.objective.direction == "maximize"
    assert config.variable_names == ["precursor_ratio"]
    assert config.bo.batch_size == 2


def test_config_from_yaml_parses_mixed_variables(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path / "campaign.yaml",
        """
campaign_name: mixed
objective:
  name: yield
  direction: maximize
variables:
  - name: loading
    type: continuous
    lower: 0.02
    upper: 0.2
  - name: repeats
    type: integer
    lower: 1
    upper: 5
  - name: base_ratio
    type: discrete
    values: [0.1, 0.2, 0.5]
  - name: solvent
    type: categorical
    values: [MeCN, EtOH, Water]
bo:
  initial_design_method: random
""",
    )

    config = CampaignConfig.from_yaml(path)

    assert config.variable_names == ["loading", "repeats", "base_ratio", "solvent"]
    assert config.variables[1].type == "integer"
    assert config.variables[2].values == (0.1, 0.2, 0.5)
    assert config.variables[3].values == ("MeCN", "EtOH", "Water")
    assert config.bo.initial_design_method == "random"


def test_config_from_yaml_parses_constraints_and_distance_threshold(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path / "campaign.yaml",
        """
campaign_name: constrained
objective:
  name: yield
  direction: maximize
variables:
  - name: temperature
    type: continuous
    lower: -20
    upper: 100
  - name: solvent
    type: categorical
    values: [MeCN, Water]
constraints:
  - name: no_cold_water
    expression: "solvent != 'Water' or temperature >= -10"
bo:
  min_normalized_distance: 0.05
""",
    )

    config = CampaignConfig.from_yaml(path)

    assert config.constraints[0].name == "no_cold_water"
    assert config.constraints[0].expression == "solvent != 'Water' or temperature >= -10"
    assert config.bo.min_normalized_distance == 0.05


def test_config_from_yaml_parses_cost_and_review(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path / "campaign.yaml",
        """
campaign_name: cost_review
objective:
  name: yield
  direction: maximize
variables:
  - name: reaction_time
    type: integer
    lower: 10
    upper: 60
  - name: solvent
    type: categorical
    values: [MeCN, Water]
cost:
  expression: "1.0 + 0.04 * reaction_time + 2.0 * (solvent == 'Water')"
  weight: 0.5
  budget: 30
  candidate_pool_size: 64
  top_k: 8
review:
  enabled: true
""",
    )

    config = CampaignConfig.from_yaml(path)

    assert config.cost is not None
    assert config.cost.weight == 0.5
    assert config.cost.budget == 30.0
    assert config.cost.candidate_pool_size == 64
    assert config.cost.top_k == 8
    assert config.review.enabled
    assert evaluate_cost(config, (20, "Water")) == pytest.approx(3.8)


def test_config_from_yaml_parses_replicates(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path / "campaign.yaml",
        """
campaign_name: replicate_test
objective:
  name: activity
  direction: maximize
variables:
  - name: x
    type: continuous
    lower: 0
    upper: 1
replicates:
  enabled: true
  suggestion_policy: new_only
  replicate_threshold: 0.25
  min_repeats_at_best: 2
  max_repeats_per_group: 4
  noise_floor: 1.0e-6
""",
    )

    config = CampaignConfig.from_yaml(path)

    assert config.replicates.enabled
    assert config.replicates.suggestion_policy == "new_only"
    assert config.replicates.replicate_threshold == 0.25
    assert config.replicates.min_repeats_at_best == 2
    assert config.replicates.max_repeats_per_group == 4
    assert config.replicates.noise_floor == pytest.approx(1.0e-6)


def test_config_from_yaml_parses_structured_stages(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path / "campaign.yaml",
        """
campaign_name: structured_test
objective:
  name: activity
  direction: maximize
variables:
  - name: precursor_ratio
    type: continuous
    lower: 0
    upper: 1
  - name: annealing_temperature
    type: continuous
    lower: 300
    upper: 900
  - name: electrolyte
    type: categorical
    values: [KOH, NaOH]
stages:
  - name: screen
    variables: [precursor_ratio, electrolyte]
  - name: refine
    variables: [precursor_ratio, annealing_temperature]
""",
    )

    config = CampaignConfig.from_yaml(path)

    assert is_structured_campaign(config)
    assert configured_stage_names(config) == ["screen", "refine"]
    assert active_variables_for_stage(config, "screen") == [
        "precursor_ratio",
        "electrolyte",
    ]
    assert config.active_variable_names_for_stage("refine") == [
        "precursor_ratio",
        "annealing_temperature",
    ]


def test_structured_stages_reject_empty_stage_list(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path / "campaign.yaml",
        """
campaign_name: structured_test
objective:
  name: activity
  direction: maximize
variables:
  - name: x
    type: continuous
    lower: 0
    upper: 1
stages: []
""",
    )

    with pytest.raises(ConfigError, match="stages.*non-empty list"):
        CampaignConfig.from_yaml(path)


def test_structured_stages_reject_duplicate_stage_names(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path / "campaign.yaml",
        """
campaign_name: structured_test
objective:
  name: activity
  direction: maximize
variables:
  - name: x
    type: continuous
    lower: 0
    upper: 1
stages:
  - name: screen
    variables: [x]
  - name: screen
    variables: [x]
""",
    )

    with pytest.raises(ConfigError, match="Duplicate stage name 'screen'"):
        CampaignConfig.from_yaml(path)


def test_structured_stages_reject_unknown_variable_reference(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path / "campaign.yaml",
        """
campaign_name: structured_test
objective:
  name: activity
  direction: maximize
variables:
  - name: x
    type: continuous
    lower: 0
    upper: 1
stages:
  - name: screen
    variables: [x, missing_variable]
""",
    )

    with pytest.raises(ConfigError, match="references unknown variable 'missing_variable'"):
        CampaignConfig.from_yaml(path)


def test_structured_stages_reject_cost_until_stage_aware_cost_support(
    tmp_path: Path,
) -> None:
    path = write_yaml(
        tmp_path / "campaign.yaml",
        """
campaign_name: structured_cost
objective:
  name: activity
  direction: maximize
variables:
  - name: x
    type: continuous
    lower: 0
    upper: 1
  - name: temperature
    type: continuous
    lower: 300
    upper: 900
cost:
  expression: "1.0 + temperature / 1000"
stages:
  - name: screen
    variables: [x]
  - name: refine
    variables: [x, temperature]
""",
    )

    with pytest.raises(ConfigError, match="Structured campaigns with cost"):
        CampaignConfig.from_yaml(path)


def test_replicates_defaults_preserve_noisy_repeat_policy(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path / "campaign.yaml",
        """
campaign_name: replicate_defaults
objective:
  name: activity
  direction: maximize
variables:
  - name: x
    type: continuous
    lower: 0
    upper: 1
replicates:
  enabled: true
""",
    )

    config = CampaignConfig.from_yaml(path)

    assert config.replicates.suggestion_policy == "uncertain_best"
    assert config.replicates.replicate_threshold == pytest.approx(0.10)
    assert config.replicates.min_repeats_at_best == 3
    assert config.replicates.max_repeats_per_group == 5
    assert config.replicates.noise_floor == pytest.approx(1.0e-8)


def test_single_objective_replicates_default_to_uncertain_best(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path / "campaign.yaml",
        """
campaign_name: single_replicates
objective:
  name: activity
  direction: maximize
variables:
  - name: x
    type: continuous
    lower: 0
    upper: 1
replicates:
  enabled: true
""",
    )

    config = CampaignConfig.from_yaml(path)

    assert config.replicates.suggestion_policy == "uncertain_best"


def test_multi_objective_replicates_default_to_new_only_when_policy_omitted(
    tmp_path: Path,
) -> None:
    path = write_yaml(
        tmp_path / "campaign.yaml",
        """
campaign_name: multi_replicates
objectives:
  - name: yield_score
    direction: maximize
    reference_point: 0
  - name: waste_score
    direction: minimize
    reference_point: 1
variables:
  - name: x
    type: continuous
    lower: 0
    upper: 1
replicates:
  enabled: true
""",
    )

    config = CampaignConfig.from_yaml(path)

    assert config.replicates.enabled
    assert config.replicates.suggestion_policy == "new_only"


def test_multi_objective_replicates_accept_explicit_new_only(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path / "campaign.yaml",
        """
campaign_name: multi_replicates
objectives:
  - name: yield_score
    direction: maximize
    reference_point: 0
  - name: waste_score
    direction: minimize
    reference_point: 1
variables:
  - name: x
    type: continuous
    lower: 0
    upper: 1
replicates:
  enabled: true
  suggestion_policy: new_only
""",
    )

    config = CampaignConfig.from_yaml(path)

    assert config.replicates.suggestion_policy == "new_only"


def test_multi_objective_replicates_reject_explicit_uncertain_best(
    tmp_path: Path,
) -> None:
    path = write_yaml(
        tmp_path / "campaign.yaml",
        """
campaign_name: multi_replicates
objectives:
  - name: yield_score
    direction: maximize
    reference_point: 0
  - name: waste_score
    direction: minimize
    reference_point: 1
variables:
  - name: x
    type: continuous
    lower: 0
    upper: 1
replicates:
  enabled: true
  suggestion_policy: uncertain_best
""",
    )

    with pytest.raises(ConfigError, match="single-objective campaigns"):
        CampaignConfig.from_yaml(path)


def test_replicates_unknown_key_fails(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path / "campaign.yaml",
        """
campaign_name: bad_replicates
objective:
  name: activity
  direction: maximize
variables:
  - name: x
    type: continuous
    lower: 0
    upper: 1
replicates:
  enabled: true
  repeats: 2
""",
    )

    with pytest.raises(ConfigError, match="replicates.*unsupported keys"):
        CampaignConfig.from_yaml(path)


def test_replicates_enabled_must_be_boolean(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path / "campaign.yaml",
        """
campaign_name: bad_replicates
objective:
  name: activity
  direction: maximize
variables:
  - name: x
    type: continuous
    lower: 0
    upper: 1
replicates:
  enabled: "true"
""",
    )

    with pytest.raises(ConfigError, match="replicates.enabled must be a boolean"):
        CampaignConfig.from_yaml(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("suggestion_policy", "always_repeat", "suggestion_policy"),
        ("replicate_threshold", 0, "replicate_threshold"),
        ("noise_floor", -1, "noise_floor"),
        ("min_repeats_at_best", 0, "min_repeats_at_best"),
        ("max_repeats_per_group", 0, "max_repeats_per_group"),
    ],
)
def test_replicates_invalid_policy_controls_fail(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    rendered = f'"{value}"' if isinstance(value, str) else value
    path = write_yaml(
        tmp_path / "campaign.yaml",
        f"""
campaign_name: bad_replicates
objective:
  name: activity
  direction: maximize
variables:
  - name: x
    type: continuous
    lower: 0
    upper: 1
replicates:
  enabled: true
  {field}: {rendered}
""",
    )

    with pytest.raises(ConfigError, match=message):
        CampaignConfig.from_yaml(path)


def test_replicates_min_repeats_must_not_exceed_max(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path / "campaign.yaml",
        """
campaign_name: bad_replicates
objective:
  name: activity
  direction: maximize
variables:
  - name: x
    type: continuous
    lower: 0
    upper: 1
replicates:
  enabled: true
  min_repeats_at_best: 6
  max_repeats_per_group: 5
""",
    )

    with pytest.raises(ConfigError, match="min_repeats_at_best.*<="):
        CampaignConfig.from_yaml(path)


def test_cost_expression_bare_boolean_fails_at_evaluation(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path / "campaign.yaml",
        """
campaign_name: bad_cost_value
objective:
  name: yield
  direction: maximize
variables:
  - name: solvent
    type: categorical
    values: [MeCN, Water]
cost:
  expression: "solvent == 'Water'"
""",
    )

    config = CampaignConfig.from_yaml(path)

    with pytest.raises(LogValidationError, match="numeric value"):
        evaluate_cost(config, ("Water",))


def test_example_minimize_config_parses() -> None:
    config = CampaignConfig.from_yaml("configs/02_simple_2d_minimise_qlogei.yaml")

    assert config.campaign_name == "process_defect_minimisation"
    assert config.objective.name == "defect_rate"
    assert config.objective.direction == "minimize"
    assert config.direction_sign == -1.0
    assert config.variable_names == ["catalyst_loading", "cure_temperature"]
    assert config.bo.batch_size == 2


def test_example_3d_config_parses() -> None:
    config = CampaignConfig.from_yaml("configs/03_simple_3d_maximise_logei.yaml")

    assert config.campaign_name == "three_variable_photocatalyst"
    assert config.objective.name == "activity"
    assert config.objective.direction == "maximize"
    assert config.variable_names == [
        "precursor_ratio",
        "annealing_temperature",
        "electrolyte_concentration",
    ]
    assert config.bo.batch_size == 1


def test_example_4d_config_parses() -> None:
    config = CampaignConfig.from_yaml("configs/04_simple_4d_maximise_logei.yaml")

    assert config.campaign_name == "four_variable_photocatalyst"
    assert config.objective.name == "activity"
    assert config.objective.direction == "maximize"
    assert config.variable_names == [
        "precursor_ratio",
        "annealing_temperature",
        "electrolyte_concentration",
        "reaction_time",
    ]
    assert config.bo.batch_size == 1


def test_example_mixed_config_parses() -> None:
    config = CampaignConfig.from_yaml("configs/05_simple_mixed_logei.yaml")

    assert config.campaign_name == "mixed_catalyst_screen"
    assert config.objective.name == "yield_score"
    assert config.variable_names == [
        "catalyst_loading",
        "reaction_time",
        "base_equivalents",
        "solvent",
    ]
    assert [variable.type for variable in config.variables] == [
        "continuous",
        "integer",
        "discrete",
        "categorical",
    ]
    assert config.bo.batch_size == 2


def test_example_constrained_mixed_config_parses() -> None:
    config = CampaignConfig.from_yaml("configs/06_mixed_constrained_logei.yaml")

    assert config.campaign_name == "constrained_mixed_catalyst_screen"
    assert config.objective.name == "yield_score"
    assert config.variable_names == [
        "catalyst_loading",
        "reaction_time",
        "base_equivalents",
        "solvent",
    ]
    assert [constraint.name for constraint in config.constraints] == [
        "no_water_high_base",
        "water_needs_longer_time",
    ]
    assert config.bo.min_normalized_distance == 0.03


def test_example_cost_review_config_parses() -> None:
    config = CampaignConfig.from_yaml("configs/07_cost_aware_human_review_logei.yaml")

    assert config.campaign_name == "cost_aware_human_review_catalyst_screen"
    assert config.cost is not None
    assert config.cost.weight == 0.5
    assert config.cost.budget == 60.0
    assert config.cost.candidate_pool_size == 128
    assert config.cost.top_k == 24
    assert config.review.enabled


def test_example_replicate_config_parses() -> None:
    config = CampaignConfig.from_yaml("configs/08_replicate_aware_logei.yaml")

    assert config.campaign_name == "replicate_aware_photocatalyst"
    assert config.replicates.enabled
    assert config.variable_names == ["precursor_ratio", "annealing_temperature"]
    assert config.bo.initial_design_size == 4


def test_example_structured_config_parses() -> None:
    config = CampaignConfig.from_yaml("configs/13_structured_campaign_core.yaml")

    assert config.campaign_name == "structured_campaign_core"
    assert config.is_structured_campaign
    assert config.stage_names == ["screen", "refine"]
    assert config.active_variable_names_for_stage("screen") == [
        "precursor_ratio",
        "electrolyte",
    ]
    assert config.active_variable_names_for_stage("refine") == [
        "precursor_ratio",
        "annealing_temperature",
    ]


def test_structured_tutorial_config_parses() -> None:
    config = CampaignConfig.from_yaml("configs/14_structured_campaign_tutorial.yaml")

    assert config.campaign_name == "structured_photocatalyst_tutorial"
    assert config.is_structured_campaign
    assert config.stage_names == ["screening", "refinement"]
    assert config.active_variable_names_for_stage("screening") == [
        "catalyst_loading",
        "base",
    ]
    assert config.active_variable_names_for_stage("refinement") == [
        "catalyst_loading",
        "base",
        "temperature",
        "residence_time",
    ]
    assert [constraint.name for constraint in config.constraints] == [
        "refinement_temperature_limit",
        "refinement_loading_time_limit",
    ]


def test_config_rejects_invalid_bounds(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path / "campaign.yaml",
        """
campaign_name: bad_bounds
objective:
  name: activity
  direction: maximize
variables:
  - name: temperature
    type: continuous
    lower: 800
    upper: 300
""",
    )

    with pytest.raises(ConfigError, match="Variable 'temperature' has lower >= upper"):
        CampaignConfig.from_yaml(path)


def test_config_rejects_unknown_variable_type(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path / "campaign.yaml",
        """
campaign_name: mixed
objective:
  name: activity
  direction: maximize
variables:
  - name: catalyst
    type: molecular
    lower: 0
    upper: 1
""",
    )

    with pytest.raises(ConfigError, match="unsupported type 'molecular'"):
        CampaignConfig.from_yaml(path)


def test_config_rejects_unsupported_variable_keys(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path / "campaign.yaml",
        """
campaign_name: bad_keys
objective:
  name: yield
  direction: maximize
variables:
  - name: solvent
    type: categorical
    lower: 0
    upper: 1
""",
    )

    with pytest.raises(ConfigError, match="unsupported keys for type='categorical'"):
        CampaignConfig.from_yaml(path)


def test_config_rejects_integer_non_integer_bounds(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path / "campaign.yaml",
        """
campaign_name: bad_integer
objective:
  name: yield
  direction: maximize
variables:
  - name: repeats
    type: integer
    lower: 1.5
    upper: 5
""",
    )

    with pytest.raises(ConfigError, match="integer-valued key 'lower'"):
        CampaignConfig.from_yaml(path)


def test_config_rejects_duplicate_discrete_numeric_values(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path / "campaign.yaml",
        """
campaign_name: bad_discrete
objective:
  name: yield
  direction: maximize
variables:
  - name: dose
    type: discrete
    values: [1, 1.0]
""",
    )

    with pytest.raises(ConfigError, match="duplicate discrete value"):
        CampaignConfig.from_yaml(path)


def test_config_rejects_blank_categorical_values(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path / "campaign.yaml",
        """
campaign_name: bad_category
objective:
  name: yield
  direction: maximize
variables:
  - name: solvent
    type: categorical
    values: [MeCN, " EtOH"]
""",
    )

    with pytest.raises(ConfigError, match="whitespace-padded categorical value"):
        CampaignConfig.from_yaml(path)


def test_config_rejects_duplicate_variable_names(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path / "campaign.yaml",
        """
campaign_name: duplicate
objective:
  name: activity
  direction: maximize
variables:
  - name: x
    type: continuous
    lower: 0
    upper: 1
  - name: x
    type: continuous
    lower: 0
    upper: 1
""",
    )

    with pytest.raises(ConfigError, match="Duplicate variable name 'x'"):
        CampaignConfig.from_yaml(path)


def test_config_rejects_invalid_objective_direction(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path / "campaign.yaml",
        """
campaign_name: bad_direction
objective:
  name: activity
  direction: largest
variables:
  - name: x
    type: continuous
    lower: 0
    upper: 1
""",
    )

    with pytest.raises(ConfigError, match="invalid direction 'largest'"):
        CampaignConfig.from_yaml(path)


def test_config_rejects_invalid_initial_design_method(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path / "campaign.yaml",
        """
campaign_name: bad_initial_method
objective:
  name: activity
  direction: maximize
variables:
  - name: x
    type: continuous
    lower: 0
    upper: 1
bo:
  initial_design_method: latin_hypercube
""",
    )

    with pytest.raises(ConfigError, match="Unsupported initial_design_method"):
        CampaignConfig.from_yaml(path)


def test_config_rejects_duplicate_constraint_names(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path / "campaign.yaml",
        """
campaign_name: bad_constraints
objective:
  name: score
  direction: maximize
variables:
  - name: x
    type: continuous
    lower: 0
    upper: 1
constraints:
  - name: keep_positive
    expression: "x >= 0"
  - name: keep_positive
    expression: "x <= 1"
""",
    )

    with pytest.raises(ConfigError, match="Duplicate constraint name 'keep_positive'"):
        CampaignConfig.from_yaml(path)


def test_config_rejects_unknown_constraint_variable(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path / "campaign.yaml",
        """
campaign_name: bad_constraints
objective:
  name: score
  direction: maximize
variables:
  - name: x
    type: continuous
    lower: 0
    upper: 1
constraints:
  - name: unknown_name
    expression: "y >= 0"
""",
    )

    with pytest.raises(ConfigError, match="references unknown variable 'y'"):
        CampaignConfig.from_yaml(path)


def test_config_rejects_unsafe_constraint_expression(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path / "campaign.yaml",
        """
campaign_name: bad_constraints
objective:
  name: score
  direction: maximize
variables:
  - name: x
    type: continuous
    lower: 0
    upper: 1
constraints:
  - name: unsafe
    expression: "abs(x) <= 1"
""",
    )

    with pytest.raises(ConfigError, match="unsupported syntax: Call"):
        CampaignConfig.from_yaml(path)


def test_config_rejects_unknown_cost_variable(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path / "campaign.yaml",
        """
campaign_name: bad_cost
objective:
  name: score
  direction: maximize
variables:
  - name: x
    type: continuous
    lower: 0
    upper: 1
cost:
  expression: "missing + 1"
""",
    )

    with pytest.raises(ConfigError, match="references unknown variable 'missing'"):
        CampaignConfig.from_yaml(path)


def test_config_rejects_invalid_cost_settings(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path / "campaign.yaml",
        """
campaign_name: bad_cost
objective:
  name: score
  direction: maximize
variables:
  - name: x
    type: continuous
    lower: 0
    upper: 1
cost:
  expression: "1 + x"
  weight: -0.1
""",
    )

    with pytest.raises(ConfigError, match="cost.weight must be >= 0"):
        CampaignConfig.from_yaml(path)


def test_config_rejects_invalid_review_enabled(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path / "campaign.yaml",
        """
campaign_name: bad_review
objective:
  name: score
  direction: maximize
variables:
  - name: x
    type: continuous
    lower: 0
    upper: 1
review:
  enabled: yes please
""",
    )

    with pytest.raises(ConfigError, match="review.enabled must be a boolean"):
        CampaignConfig.from_yaml(path)


def test_config_rejects_invalid_min_normalized_distance(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path / "campaign.yaml",
        """
campaign_name: bad_distance
objective:
  name: score
  direction: maximize
variables:
  - name: x
    type: continuous
    lower: 0
    upper: 1
bo:
  min_normalized_distance: -0.1
""",
    )

    with pytest.raises(ConfigError, match="bo.min_normalized_distance must be >= 0"):
        CampaignConfig.from_yaml(path)
