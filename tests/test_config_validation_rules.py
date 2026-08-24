"""Configuration validation rules and example compatibility tests."""

from tests._config_support import (
    CampaignConfig,
    ConfigError,
    LogValidationError,
    Path,
    evaluate_cost,
    pytest,
    write_yaml,
)


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (
            """
objectives:
  - name: activity
    direction: maximize
    reference_point: 0
  - name: stability
    direction: maximize
    reference_point: 0
""",
            "single-objective",
        ),
        (
            """
cost:
  expression: "1.0 + x"
""",
            "cost-aware",
        ),
        (
            """
replicates:
  enabled: true
""",
            "replicate",
        ),
        (
            """
stages:
  - name: screen
    variables: [x, fidelity]
""",
            "structured campaign stages",
        ),
    ],
)
def test_fidelity_rejects_unsupported_combinations(
    tmp_path: Path,
    extra: str,
    message: str,
) -> None:
    objective = (
        ""
        if "objectives:" in extra
        else """
objective:
  name: activity
  direction: maximize
"""
    )
    path = write_yaml(
        tmp_path / "campaign.yaml",
        f"""
campaign_name: bad_fidelity
{objective}
variables:
  - name: x
    type: continuous
    lower: 0
    upper: 1
  - name: fidelity
    type: continuous
    lower: 0.2
    upper: 1.0
{extra}
fidelity:
  variable: fidelity
  target: 1.0
bo:
  acquisition: qmf_kg
""",
    )

    with pytest.raises(ConfigError, match=message):
        CampaignConfig.from_yaml(path)

def test_fidelity_requires_configured_continuous_variable(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path / "campaign.yaml",
        """
campaign_name: bad_fidelity_variable
objective:
  name: activity
  direction: maximize
variables:
  - name: x
    type: continuous
    lower: 0
    upper: 1
  - name: fidelity
    type: integer
    lower: 1
    upper: 3
fidelity:
  variable: fidelity
  target: 1
bo:
  acquisition: qmf_kg
""",
    )

    with pytest.raises(ConfigError, match="must be a continuous variable"):
        CampaignConfig.from_yaml(path)

def test_fidelity_rejects_non_continuous_design_variables(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path / "campaign.yaml",
        """
campaign_name: bad_fidelity_mixed
objective:
  name: activity
  direction: maximize
variables:
  - name: x
    type: continuous
    lower: 0
    upper: 1
  - name: solvent
    type: categorical
    values: [Water, MeCN]
  - name: fidelity
    type: continuous
    lower: 0.2
    upper: 1.0
fidelity:
  variable: fidelity
  target: 1.0
bo:
  acquisition: qmf_kg
""",
    )

    with pytest.raises(ConfigError, match="only support continuous variables"):
        CampaignConfig.from_yaml(path)

def test_fidelity_target_must_be_within_bounds(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path / "campaign.yaml",
        """
campaign_name: bad_fidelity_target
objective:
  name: activity
  direction: maximize
variables:
  - name: x
    type: continuous
    lower: 0
    upper: 1
  - name: fidelity
    type: continuous
    lower: 0.2
    upper: 1.0
fidelity:
  variable: fidelity
  target: 1.2
bo:
  acquisition: qmf_kg
""",
    )

    with pytest.raises(ConfigError, match="fidelity.target must be within"):
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

def test_example_multi_fidelity_config_parses() -> None:
    config = CampaignConfig.from_yaml("configs/15_multi_fidelity_qmfkg.yaml")

    assert config.campaign_name == "multi_fidelity_photocatalyst_qmfkg"
    assert config.fidelity is not None
    assert config.fidelity.variable == "fidelity"
    assert config.fidelity.target == 1.0
    assert config.bo.acquisition == "qmf_kg"
    assert config.bo.min_normalized_distance == 0.0
    assert config.variable_names == ["catalyst_loading", "fidelity"]

def test_example_model_profile_config_parses() -> None:
    config = CampaignConfig.from_yaml("configs/17_model_profile_logei.yaml")

    assert config.campaign_name == "model_profile_logei"
    assert config.model.profile == "smooth"
    assert config.bo.acquisition == "log_ei"
    assert config.variable_names == ["catalyst_loading", "reaction_temperature"]

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

def test_config_accepts_qlog_nei_with_non_default_model_profile(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path / "campaign.yaml",
        """
campaign_name: noisy_nei
objective:
  name: score
  direction: maximize
variables:
  - name: x
    type: continuous
    lower: 0
    upper: 1
model:
  profile: robust
bo:
  acquisition: qlog_nei
""",
    )

    config = CampaignConfig.from_yaml(path)

    assert config.bo.acquisition == "qlog_nei"
    assert config.model.profile == "robust"

def test_config_accepts_supported_qlog_nehvi_multi_objective_config(
    tmp_path: Path,
) -> None:
    path = write_yaml(
        tmp_path / "campaign.yaml",
        """
campaign_name: noisy_mobo_review
objectives:
  - name: yield
    direction: maximize
    reference_point: 0
  - name: waste
    direction: minimize
    reference_point: 10
variables:
  - name: x
    type: continuous
    lower: 0
    upper: 1
bo:
  acquisition: qlog_nehvi
""",
    )

    config = CampaignConfig.from_yaml(path)

    assert config.is_multi_objective
    assert config.bo.acquisition == "qlog_nehvi"

@pytest.mark.parametrize(
    ("yaml_text", "message"),
    [
        (
            """
campaign_name: bad_single
objective:
  name: score
  direction: maximize
variables:
  - name: x
    type: continuous
    lower: 0
    upper: 1
bo:
  acquisition: qlog_nehvi
""",
            "only supported for coupled multi-objective",
        ),
        (
            """
campaign_name: bad_cost
objectives:
  - name: yield
    direction: maximize
    reference_point: 0
  - name: waste
    direction: minimize
    reference_point: 10
variables:
  - name: x
    type: continuous
    lower: 0
    upper: 1
cost:
  expression: "1.0 + x"
bo:
  acquisition: qlog_nehvi
""",
            "cannot be combined with cost-aware",
        ),
        (
            """
campaign_name: bad_replicates
objectives:
  - name: yield
    direction: maximize
    reference_point: 0
  - name: waste
    direction: minimize
    reference_point: 10
variables:
  - name: x
    type: continuous
    lower: 0
    upper: 1
replicates:
  enabled: true
  suggestion_policy: new_only
bo:
  acquisition: qlog_nehvi
""",
            "cannot be combined with replicate",
        ),
        (
            """
campaign_name: bad_stages
objectives:
  - name: yield
    direction: maximize
    reference_point: 0
  - name: waste
    direction: minimize
    reference_point: 10
variables:
  - name: x
    type: continuous
    lower: 0
    upper: 1
stages:
  - name: screen
    variables: [x]
bo:
  acquisition: qlog_nehvi
""",
            "cannot be combined with structured stages",
        ),
        (
            """
campaign_name: bad_context
objectives:
  - name: yield
    direction: maximize
    reference_point: 0
  - name: waste
    direction: minimize
    reference_point: 10
variables:
  - name: x
    type: continuous
    lower: 0
    upper: 1
  - name: lot
    type: categorical
    values: [A, B]
context:
  variables: [lot]
bo:
  acquisition: qlog_nehvi
""",
            "cannot be combined with context",
        ),
        (
            """
campaign_name: bad_fidelity
objectives:
  - name: yield
    direction: maximize
    reference_point: 0
  - name: waste
    direction: minimize
    reference_point: 10
variables:
  - name: x
    type: continuous
    lower: 0
    upper: 1
  - name: fidelity
    type: continuous
    lower: 0.1
    upper: 1
fidelity:
  variable: fidelity
  target: 1
bo:
  acquisition: qlog_nehvi
""",
            "cannot be combined with fidelity",
        ),
        (
            """
campaign_name: bad_objective_count
objectives:
  - name: a
    direction: maximize
    reference_point: 0
  - name: b
    direction: maximize
    reference_point: 0
  - name: c
    direction: minimize
    reference_point: 10
  - name: d
    direction: minimize
    reference_point: 10
  - name: e
    direction: maximize
    reference_point: 0
variables:
  - name: x
    type: continuous
    lower: 0
    upper: 1
bo:
  acquisition: qlog_nehvi
""",
            "at most 4 objectives",
        ),
    ],
)
def test_config_rejects_unsupported_qlog_nehvi_combinations(
    tmp_path: Path,
    yaml_text: str,
    message: str,
) -> None:
    path = write_yaml(tmp_path / "campaign.yaml", yaml_text)

    with pytest.raises(
        ConfigError,
        match=message,
    ):
        CampaignConfig.from_yaml(path)

@pytest.mark.parametrize(
    ("extra_yaml", "message"),
    [
        (
            """
cost:
  expression: "1 + x"
""",
            "cannot be combined with cost-aware campaigns",
        ),
        (
            """
context:
  variables: [x]
""",
            "context cannot include every configured variable|cannot be combined with context",
        ),
        (
            """
stages:
  - name: screen
    variables: [x]
""",
            "cannot be combined with structured stages",
        ),
        (
            """
replicates:
  enabled: true
  suggestion_policy: uncertain_best
""",
            "replicates.suggestion_policy: new_only",
        ),
    ],
)
def test_config_rejects_qlog_nei_unsupported_combinations(
    tmp_path: Path,
    extra_yaml: str,
    message: str,
) -> None:
    variables_yaml = """
variables:
  - name: x
    type: continuous
    lower: 0
    upper: 1
  - name: y
    type: continuous
    lower: 0
    upper: 1
"""
    path = write_yaml(
        tmp_path / "campaign.yaml",
        f"""
campaign_name: bad_nei
objective:
  name: score
  direction: maximize
{variables_yaml}
bo:
  acquisition: qlog_nei
{extra_yaml}
""",
    )

    with pytest.raises(ConfigError, match=message):
        CampaignConfig.from_yaml(path)
