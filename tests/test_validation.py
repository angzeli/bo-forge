import pandas as pd
import pytest

from bo_forge.config import (
    BOConfig,
    CampaignConfig,
    ConstraintConfig,
    CostConfig,
    ObjectiveConfig,
    ReviewConfig,
    VariableConfig,
)
from bo_forge.errors import LogValidationError
from bo_forge.logs import load_campaign_log
from bo_forge.validation import canonical_columns, design_tuples, validate_campaign_data


def config() -> CampaignConfig:
    return CampaignConfig(
        campaign_name="test",
        objective=ObjectiveConfig(name="activity", direction="maximize"),
        variables=(
            VariableConfig("x", "continuous", 0.0, 1.0),
            VariableConfig("temperature", "continuous", 300.0, 800.0),
        ),
        bo=BOConfig(batch_size=2, initial_design_size=3),
    )


def valid_df() -> pd.DataFrame:
    cfg = config()
    return pd.DataFrame(
        [
            {
                "row_id": "row_1",
                "iteration": 0,
                "status": "observed",
                "source": "manual",
                "x": 0.2,
                "temperature": 500.0,
                "activity": 1.3,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            },
            {
                "row_id": "row_2",
                "iteration": 1,
                "status": "suggested",
                "source": "sobol",
                "x": 0.5,
                "temperature": 650.0,
                "activity": "",
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            },
        ],
        columns=canonical_columns(cfg),
    )


def mixed_config() -> CampaignConfig:
    return CampaignConfig(
        campaign_name="mixed",
        objective=ObjectiveConfig(name="yield", direction="maximize"),
        variables=(
            VariableConfig("loading", "continuous", 0.0, 1.0),
            VariableConfig("repeats", "integer", 1.0, 5.0),
            VariableConfig("base_ratio", "discrete", values=(0.1, 0.2, 0.5)),
            VariableConfig("solvent", "categorical", values=("MeCN", "EtOH", "Water")),
        ),
        bo=BOConfig(batch_size=2, initial_design_size=3),
    )


def constrained_mixed_config() -> CampaignConfig:
    cfg = mixed_config()
    return CampaignConfig(
        campaign_name=cfg.campaign_name,
        objective=cfg.objective,
        variables=cfg.variables,
        bo=cfg.bo,
        constraints=(
            ConstraintConfig(
                name="no_water_high_base",
                expression="not (solvent == 'Water' and base_ratio >= 0.5)",
            ),
        ),
    )


def cost_review_config() -> CampaignConfig:
    return CampaignConfig(
        campaign_name="cost_review",
        objective=ObjectiveConfig(name="score", direction="maximize"),
        variables=(VariableConfig("x", "continuous", 0.0, 1.0),),
        bo=BOConfig(batch_size=1, initial_design_size=1),
        cost=CostConfig(expression="1.0 + x", weight=0.5, budget=10.0),
        review=ReviewConfig(enabled=True),
    )


def mixed_df() -> pd.DataFrame:
    cfg = mixed_config()
    return pd.DataFrame(
        [
            {
                "row_id": "mixed_1",
                "iteration": 0,
                "status": "observed",
                "source": "manual",
                "loading": 0.5,
                "repeats": 3.0,
                "base_ratio": "0.10",
                "solvent": "MeCN",
                "yield": 72.5,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            }
        ],
        columns=canonical_columns(cfg),
    )


def cost_review_df() -> pd.DataFrame:
    cfg = cost_review_config()
    return pd.DataFrame(
        [
            {
                "row_id": "obs_0",
                "iteration": 0,
                "status": "observed",
                "source": "manual",
                "review_status": "accepted",
                "review_note": "",
                "x": 0.2,
                "score": 1.0,
                "cost_estimate": 1.2,
                "cost_actual": 1.1,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
                "utility": "",
            },
            {
                "row_id": "suggested_0",
                "iteration": 1,
                "status": "suggested",
                "source": "cost_log_ei",
                "review_status": "pending",
                "review_note": "",
                "x": 0.6,
                "score": "",
                "cost_estimate": 1.6,
                "cost_actual": "",
                "predicted_mean": 1.2,
                "predicted_std": 0.1,
                "acquisition": 0.02,
                "utility": -0.78,
            },
        ],
        columns=canonical_columns(cfg),
    )


def test_validate_campaign_data_accepts_valid_log() -> None:
    validate_campaign_data(config(), valid_df())


def test_validate_campaign_data_accepts_valid_mixed_log() -> None:
    validate_campaign_data(mixed_config(), mixed_df())


def test_validate_campaign_data_accepts_feasible_constrained_log() -> None:
    validate_campaign_data(constrained_mixed_config(), mixed_df())


def test_validate_campaign_data_accepts_cost_review_log() -> None:
    validate_campaign_data(cost_review_config(), cost_review_df())


def test_validate_campaign_data_accepts_3d_example_log() -> None:
    config_3d = CampaignConfig.from_yaml("configs/03_simple_3d_maximise_logei.yaml")
    df = load_campaign_log("examples/03_simple_3d_maximise_logei_campaign_log.csv", config_3d)

    validate_campaign_data(config_3d, df)
    assert config_3d.variable_names == [
        "precursor_ratio",
        "annealing_temperature",
        "electrolyte_concentration",
    ]


def test_validate_campaign_data_accepts_4d_example_log() -> None:
    config_4d = CampaignConfig.from_yaml("configs/04_simple_4d_maximise_logei.yaml")
    df = load_campaign_log("examples/04_simple_4d_maximise_logei_campaign_log.csv", config_4d)

    validate_campaign_data(config_4d, df)
    assert config_4d.variable_names == [
        "precursor_ratio",
        "annealing_temperature",
        "electrolyte_concentration",
        "reaction_time",
    ]


def test_validate_campaign_data_accepts_mixed_example_log() -> None:
    config_mixed = CampaignConfig.from_yaml("configs/05_simple_mixed_logei.yaml")
    df = load_campaign_log("examples/05_simple_mixed_logei_campaign_log.csv", config_mixed)

    validate_campaign_data(config_mixed, df)
    assert config_mixed.variable_names == [
        "catalyst_loading",
        "reaction_time",
        "base_equivalents",
        "solvent",
    ]


def test_validate_campaign_data_accepts_constrained_mixed_example_log() -> None:
    config_mixed = CampaignConfig.from_yaml("configs/06_mixed_constrained_logei.yaml")
    df = load_campaign_log(
        "examples/06_mixed_constrained_logei_campaign_log.csv",
        config_mixed,
    )

    validate_campaign_data(config_mixed, df)
    assert [constraint.name for constraint in config_mixed.constraints] == [
        "no_water_high_base",
        "water_needs_longer_time",
    ]


def test_validate_campaign_data_accepts_cost_review_example_log() -> None:
    config_cost = CampaignConfig.from_yaml("configs/07_cost_aware_human_review_logei.yaml")
    df = load_campaign_log(
        "examples/07_cost_aware_human_review_campaign_log.csv",
        config_cost,
    )

    validate_campaign_data(config_cost, df)
    assert config_cost.cost is not None
    assert config_cost.review.enabled


def test_validate_campaign_data_rejects_missing_column() -> None:
    df = valid_df().drop(columns=["source"])

    with pytest.raises(LogValidationError, match="missing required columns"):
        validate_campaign_data(config(), df)


def test_validate_campaign_data_rejects_review_columns_without_review_config() -> None:
    cfg = cost_review_config()
    df = cost_review_df()

    plain_cfg = CampaignConfig(
        campaign_name=cfg.campaign_name,
        objective=cfg.objective,
        variables=cfg.variables,
        bo=cfg.bo,
    )

    with pytest.raises(LogValidationError, match="unsupported extra columns"):
        validate_campaign_data(plain_cfg, df)


def test_validate_campaign_data_rejects_duplicate_row_id() -> None:
    df = valid_df()
    df.loc[1, "row_id"] = "row_1"

    with pytest.raises(LogValidationError, match="Duplicate row_id 'row_1'"):
        validate_campaign_data(config(), df)


def test_validate_campaign_data_rejects_blank_observed_objective() -> None:
    df = valid_df()
    df.loc[0, "activity"] = ""

    with pytest.raises(LogValidationError, match="status='observed'.*activity.*blank"):
        validate_campaign_data(config(), df)


def test_validate_campaign_data_rejects_filled_suggested_objective() -> None:
    df = valid_df()
    df.loc[1, "activity"] = 1.8

    with pytest.raises(LogValidationError, match="status='suggested'.*activity.*filled"):
        validate_campaign_data(config(), df)


def test_validate_campaign_data_rejects_observed_review_not_accepted() -> None:
    df = cost_review_df()
    df.loc[0, "review_status"] = "pending"

    with pytest.raises(LogValidationError, match="review_status is 'pending', not 'accepted'"):
        validate_campaign_data(cost_review_config(), df)


def test_validate_campaign_data_rejects_review_note_newline() -> None:
    df = cost_review_df()
    df.loc[1, "review_note"] = "line one\nline two"

    with pytest.raises(LogValidationError, match="review_note containing a newline"):
        validate_campaign_data(cost_review_config(), df)


def test_validate_campaign_data_rejects_negative_cost_value() -> None:
    df = cost_review_df()
    df.loc[1, "cost_estimate"] = -1.0

    with pytest.raises(LogValidationError, match="negative value"):
        validate_campaign_data(cost_review_config(), df)


def test_validate_campaign_data_rejects_mismatched_cost_estimate() -> None:
    df = cost_review_df()
    df.loc[1, "cost_estimate"] = 9.9

    with pytest.raises(LogValidationError, match="cost_estimate inconsistent"):
        validate_campaign_data(cost_review_config(), df)


def test_validate_campaign_data_rejects_negative_cost_expression_result() -> None:
    cfg = CampaignConfig(
        campaign_name="bad_cost_result",
        objective=ObjectiveConfig(name="score", direction="maximize"),
        variables=(VariableConfig("x", "continuous", 0.0, 1.0),),
        bo=BOConfig(batch_size=1, initial_design_size=1),
        cost=CostConfig(expression="-1.0 + x"),
        review=ReviewConfig(enabled=True),
    )
    df = cost_review_df()

    with pytest.raises(LogValidationError, match="negative value"):
        validate_campaign_data(cfg, df)


def test_validate_campaign_data_rejects_non_finite_objective() -> None:
    df = valid_df()
    df.loc[0, "activity"] = float("inf")

    with pytest.raises(LogValidationError, match="non-finite objective 'activity'"):
        validate_campaign_data(config(), df)


def test_validate_campaign_data_rejects_out_of_bounds_variable() -> None:
    df = valid_df()
    df.loc[0, "temperature"] = 900.0

    with pytest.raises(LogValidationError, match="outside bounds"):
        validate_campaign_data(config(), df)


def test_validate_campaign_data_rejects_invalid_source() -> None:
    df = valid_df()
    df.loc[0, "source"] = "bad_source"

    with pytest.raises(LogValidationError, match="invalid source 'bad_source'"):
        validate_campaign_data(config(), df)


def test_validate_campaign_data_accepts_random_source() -> None:
    df = valid_df()
    df.loc[0, "source"] = "random"

    validate_campaign_data(config(), df)


def test_validate_campaign_data_rejects_invalid_integer_value() -> None:
    df = mixed_df()
    df.loc[0, "repeats"] = 3.5

    with pytest.raises(LogValidationError, match="non-integer value"):
        validate_campaign_data(mixed_config(), df)


def test_validate_campaign_data_rejects_invalid_discrete_value() -> None:
    df = mixed_df()
    df.loc[0, "base_ratio"] = "0.3"

    with pytest.raises(LogValidationError, match="outside configured choices"):
        validate_campaign_data(mixed_config(), df)


def test_validate_campaign_data_rejects_categorical_case_mismatch() -> None:
    df = mixed_df()
    df.loc[0, "solvent"] = "mecn"

    with pytest.raises(LogValidationError, match="outside configured choices"):
        validate_campaign_data(mixed_config(), df)


def test_validate_campaign_data_rejects_categorical_whitespace() -> None:
    df = mixed_df()
    df.loc[0, "solvent"] = " MeCN "

    with pytest.raises(LogValidationError, match="whitespace-padded categorical value"):
        validate_campaign_data(mixed_config(), df)


@pytest.mark.parametrize("source", ["manual", "sobol", "random", "log_ei", "qlog_ei"])
def test_constraints_apply_to_all_sources_and_statuses(source: str) -> None:
    cfg = constrained_mixed_config()
    df = mixed_df()
    df["yield"] = df["yield"].astype(object)
    df.loc[0, "source"] = source
    df.loc[0, "base_ratio"] = "0.5"
    df.loc[0, "solvent"] = "Water"
    if source != "manual":
        df.loc[0, "status"] = "suggested"
        df.loc[0, "yield"] = ""

    with pytest.raises(LogValidationError, match="violates constraint 'no_water_high_base'"):
        validate_campaign_data(cfg, df)


def test_constraint_or_short_circuits() -> None:
    cfg = CampaignConfig(
        campaign_name="short_circuit_or",
        objective=ObjectiveConfig(name="score", direction="maximize"),
        variables=(VariableConfig("x", "continuous", 0.0, 1.0),),
        bo=BOConfig(),
        constraints=(
            ConstraintConfig(
                name="allow_zero",
                expression="x == 0 or 1 / x > 0",
            ),
        ),
    )
    df = pd.DataFrame(
        [
            {
                "row_id": "row_0",
                "iteration": 0,
                "status": "observed",
                "source": "manual",
                "x": 0.0,
                "score": 1.0,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            }
        ],
        columns=canonical_columns(cfg),
    )

    validate_campaign_data(cfg, df)


def test_constraint_and_short_circuits_to_violation() -> None:
    cfg = CampaignConfig(
        campaign_name="short_circuit_and",
        objective=ObjectiveConfig(name="score", direction="maximize"),
        variables=(VariableConfig("x", "continuous", 0.0, 1.0),),
        bo=BOConfig(),
        constraints=(
            ConstraintConfig(
                name="positive_inverse",
                expression="x != 0 and 1 / x > 0",
            ),
        ),
    )
    df = pd.DataFrame(
        [
            {
                "row_id": "row_0",
                "iteration": 0,
                "status": "observed",
                "source": "manual",
                "x": 0.0,
                "score": 1.0,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            }
        ],
        columns=canonical_columns(cfg),
    )

    with pytest.raises(LogValidationError, match="violates constraint 'positive_inverse'"):
        validate_campaign_data(cfg, df)


def test_design_tuples_preserve_mixed_user_space_values() -> None:
    keys = design_tuples(mixed_config(), mixed_df())

    assert keys == {(0.5, 3, 0.1, "MeCN")}
