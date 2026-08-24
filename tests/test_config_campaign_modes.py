"""Configuration parsing tests for campaign modes and fidelity controls."""

from tests._config_support import (
    CampaignConfig,
    ConfigError,
    Path,
    active_variables_for_stage,
    configured_stage_names,
    evaluate_cost,
    is_structured_campaign,
    pytest,
    write_yaml,
)


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

@pytest.mark.parametrize(
    ("extra", "expect_cost", "expect_review"),
    [
        ("review:\n  enabled: true\n", False, True),
        (
            'cost:\n  expression: "1.0 + 0.04 * reaction_time + 0.5 * acidity"\n',
            True,
            False,
        ),
        (
            'cost:\n  expression: "1.0 + 0.04 * reaction_time + 0.5 * acidity"\n'
            "review:\n  enabled: true\n",
            True,
            True,
        ),
    ],
)
def test_config_accepts_contextual_review_cost_combinations(
    tmp_path: Path,
    extra: str,
    expect_cost: bool,
    expect_review: bool,
) -> None:
    path = write_yaml(
        tmp_path / "campaign.yaml",
        f"""
campaign_name: contextual_combo
objective:
  name: yield
  direction: maximize
variables:
  - name: reaction_time
    type: integer
    lower: 10
    upper: 60
  - name: acidity
    type: continuous
    lower: 0
    upper: 1
context:
  variables: [acidity]
  default_values:
    acidity: 0.5
{extra}bo:
  acquisition: log_ei
""",
    )

    config = CampaignConfig.from_yaml(path)

    assert config.context is not None
    assert (config.cost is not None) is expect_cost
    assert config.review.enabled is expect_review
    if config.cost is not None:
        assert evaluate_cost(config, (20, 0.5)) == pytest.approx(2.05)

@pytest.mark.parametrize(
    ("yaml_text", "message"),
    [
        (
            """
campaign_name: contextual_qlog_nei
objective:
  name: yield
  direction: maximize
variables:
  - name: reaction_time
    type: integer
    lower: 10
    upper: 60
  - name: acidity
    type: continuous
    lower: 0
    upper: 1
context:
  variables: [acidity]
bo:
  acquisition: qlog_nei
""",
            "cannot be combined with context",
        ),
        (
            """
campaign_name: contextual_multi
objectives:
  - name: yield
    direction: maximize
    reference_point: 0
  - name: waste
    direction: minimize
    reference_point: 10
variables:
  - name: reaction_time
    type: integer
    lower: 10
    upper: 60
  - name: acidity
    type: continuous
    lower: 0
    upper: 1
context:
  variables: [acidity]
""",
            "single-objective",
        ),
        (
            """
campaign_name: contextual_structured
objective:
  name: yield
  direction: maximize
variables:
  - name: reaction_time
    type: integer
    lower: 10
    upper: 60
  - name: acidity
    type: continuous
    lower: 0
    upper: 1
context:
  variables: [acidity]
stages:
  - name: screen
    variables: [reaction_time]
""",
            "structured campaign stages",
        ),
        (
            """
campaign_name: contextual_fidelity
objective:
  name: yield
  direction: maximize
variables:
  - name: reaction_time
    type: continuous
    lower: 10
    upper: 60
  - name: acidity
    type: continuous
    lower: 0
    upper: 1
  - name: fidelity
    type: continuous
    lower: 0.1
    upper: 1
context:
  variables: [acidity]
fidelity:
  variable: fidelity
  target: 1
bo:
  acquisition: qmf_kg
""",
            "fidelity campaigns",
        ),
    ],
)
def test_config_rejects_unsupported_contextual_combinations(
    tmp_path: Path,
    yaml_text: str,
    message: str,
) -> None:
    path = write_yaml(tmp_path / "campaign.yaml", yaml_text)

    with pytest.raises(ConfigError, match=message):
        CampaignConfig.from_yaml(path)

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

def test_config_from_yaml_parses_model_profile(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path / "campaign.yaml",
        """
campaign_name: model_profile
objective:
  name: activity
  direction: maximize
variables:
  - name: x
    type: continuous
    lower: 0
    upper: 1
model:
  profile: smooth
bo:
  acquisition: log_ei
""",
    )

    config = CampaignConfig.from_yaml(path)

    assert config.model.profile == "smooth"

def test_config_from_yaml_rejects_unknown_model_profile(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path / "campaign.yaml",
        """
campaign_name: bad_model_profile
objective:
  name: activity
  direction: maximize
variables:
  - name: x
    type: continuous
    lower: 0
    upper: 1
model:
  profile: too_fancy
""",
    )

    with pytest.raises(ConfigError, match="model.profile"):
        CampaignConfig.from_yaml(path)

def test_non_default_model_profiles_reject_unknown_acquisition(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path / "campaign.yaml",
        """
campaign_name: bad_model_profile_acquisition
objective:
  name: activity
  direction: maximize
variables:
  - name: x
    type: continuous
    lower: 0
    upper: 1
model:
  profile: smooth
bo:
  acquisition: qlog_ei
""",
    )

    with pytest.raises(ConfigError, match=r"Expected one of \['log_ei', 'qlog_nei'\]"):
        CampaignConfig.from_yaml(path)

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
            "multi-objective",
        ),
        (
            """
fidelity:
  variable: fidelity
  target: 1.0
bo:
  acquisition: qmf_kg
""",
            "fidelity campaigns",
        ),
        (
            """
stages:
  - name: screen
    variables: [x]
""",
            "structured campaign",
        ),
        (
            """
bo:
  acquisition: qmf_kg
""",
            "requires",
        ),
    ],
)
def test_non_default_model_profiles_reject_unsupported_combinations(
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
    variables = """
variables:
  - name: x
    type: continuous
    lower: 0
    upper: 1
"""
    if "fidelity:" in extra:
        variables += """
  - name: fidelity
    type: continuous
    lower: 0.2
    upper: 1.0
"""
    path = write_yaml(
        tmp_path / "campaign.yaml",
        f"""
campaign_name: bad_model_profile
{objective}
{variables}
model:
  profile: rough
{extra}
""",
    )

    with pytest.raises(ConfigError, match=message):
        CampaignConfig.from_yaml(path)

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

    with pytest.raises(
        ConfigError,
        match="Structured campaigns with cost are currently unsupported",
    ):
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

def test_config_from_yaml_parses_fidelity_config(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path / "campaign.yaml",
        """
campaign_name: fidelity_test
objective:
  name: activity
  direction: maximize
variables:
  - name: catalyst_loading
    type: continuous
    lower: 0.05
    upper: 0.5
  - name: fidelity
    type: continuous
    lower: 0.2
    upper: 1.0
fidelity:
  variable: fidelity
  target: 1.0
  fixed_cost: 0.02
  fidelity_cost_weight: 2.5
  num_fantasies: 8
bo:
  initial_design_size: 4
""",
    )

    config = CampaignConfig.from_yaml(path)

    assert config.fidelity is not None
    assert config.fidelity.variable == "fidelity"
    assert config.fidelity.target == 1.0
    assert config.fidelity.fixed_cost == 0.02
    assert config.fidelity.fidelity_cost_weight == 2.5
    assert config.fidelity.num_fantasies == 8
    assert config.fidelity.levels is None
    assert config.fidelity.optimizer_maxiter == 200
    assert config.fidelity.optimizer_timeout_seconds is None
    assert config.bo.acquisition == "qmf_kg"

def test_config_from_yaml_parses_fidelity_optimizer_controls(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path / "campaign.yaml",
        """
campaign_name: fidelity_runtime_controls
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
  target: 1.0
  optimizer_maxiter: 75
  optimizer_timeout_seconds: 12.5
bo:
  acquisition: qmf_kg
""",
    )

    config = CampaignConfig.from_yaml(path)

    assert config.fidelity is not None
    assert config.fidelity.optimizer_maxiter == 75
    assert config.fidelity.optimizer_timeout_seconds == pytest.approx(12.5)

@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("optimizer_maxiter", "0", "must be >= 1"),
        ("optimizer_maxiter", "1.5", "must be an integer"),
        ("optimizer_timeout_seconds", "0", "must be > 0"),
        ("optimizer_timeout_seconds", "-1", "must be > 0"),
        ("optimizer_timeout_seconds", ".nan", "must be finite"),
        ("optimizer_timeout_seconds", "true", "must be numeric, not a boolean"),
    ],
)
def test_fidelity_rejects_invalid_optimizer_controls(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    path = write_yaml(
        tmp_path / "campaign.yaml",
        f"""
campaign_name: bad_fidelity_runtime_control
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
  target: 1.0
  {field}: {value}
bo:
  acquisition: qmf_kg
""",
    )

    with pytest.raises(ConfigError, match=message):
        CampaignConfig.from_yaml(path)

def test_config_from_yaml_parses_discrete_fidelity_levels(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path / "campaign.yaml",
        """
campaign_name: discrete_fidelity
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
  target: 1.0
  levels: [0.25, 0.5, 0.75, 1.0]
bo:
  acquisition: qmf_kg
  batch_size: 4
""",
    )

    config = CampaignConfig.from_yaml(path)

    assert config.fidelity is not None
    assert config.fidelity.levels == (0.25, 0.5, 0.75, 1.0)
    assert config.bo.batch_size == 4

@pytest.mark.parametrize(
    ("levels", "target", "message"),
    [
        ("[1.0]", "1.0", "at least two"),
        ("[0.25, bad, 1.0]", "1.0", "finite number"),
        ("[0.25, .nan, 1.0]", "1.0", "finite number"),
        ("[0.25, 0.25, 1.0]", "1.0", "strictly increasing"),
        ("[0.5, 0.25, 1.0]", "1.0", "strictly increasing"),
        ("[0.25, 0.2500000005, 1.0]", "1.0", "separated enough"),
        ("[0.1, 0.5, 1.0]", "1.0", "within.*bounds"),
        ("[0.25, 0.5, 0.75]", "1.0", "highest configured fidelity level"),
    ],
)
def test_discrete_fidelity_rejects_invalid_levels(
    tmp_path: Path,
    levels: str,
    target: str,
    message: str,
) -> None:
    path = write_yaml(
        tmp_path / "campaign.yaml",
        f"""
campaign_name: bad_discrete_fidelity
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
  target: {target}
  levels: {levels}
bo:
  acquisition: qmf_kg
""",
    )

    with pytest.raises(ConfigError, match=message):
        CampaignConfig.from_yaml(path)

def test_discrete_fidelity_rejects_relative_tolerance_overlap(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path / "campaign.yaml",
        """
campaign_name: large_scale_discrete_fidelity
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
    lower: 1000000000
    upper: 1000000010
fidelity:
  variable: fidelity
  target: 1000000010
  levels: [1000000000, 1000000001, 1000000010]
bo:
  acquisition: qmf_kg
""",
    )

    with pytest.raises(ConfigError, match="separated enough"):
        CampaignConfig.from_yaml(path)

def test_qmfkg_rejects_configured_batch_size_above_four(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path / "campaign.yaml",
        """
campaign_name: oversized_qmfkg_batch
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
  target: 1.0
bo:
  acquisition: qmf_kg
  batch_size: 5
""",
    )

    with pytest.raises(ConfigError, match="batch_size from 1 through 4"):
        CampaignConfig.from_yaml(path)

def test_qmf_kg_requires_fidelity_config(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path / "campaign.yaml",
        """
campaign_name: bad_qmfkg
objective:
  name: activity
  direction: maximize
variables:
  - name: x
    type: continuous
    lower: 0
    upper: 1
bo:
  acquisition: qmf_kg
""",
    )

    with pytest.raises(ConfigError, match="qmf_kg.*requires"):
        CampaignConfig.from_yaml(path)
