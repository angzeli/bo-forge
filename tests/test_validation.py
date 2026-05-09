import pandas as pd
import pytest

from bo_forge.config import BOConfig, CampaignConfig, ObjectiveConfig, VariableConfig
from bo_forge.errors import LogValidationError
from bo_forge.validation import canonical_columns, validate_campaign_data


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


def test_validate_campaign_data_accepts_valid_log() -> None:
    validate_campaign_data(config(), valid_df())


def test_validate_campaign_data_rejects_missing_column() -> None:
    df = valid_df().drop(columns=["source"])

    with pytest.raises(LogValidationError, match="missing required columns"):
        validate_campaign_data(config(), df)


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


def test_validate_campaign_data_rejects_out_of_bounds_variable() -> None:
    df = valid_df()
    df.loc[0, "temperature"] = 900.0

    with pytest.raises(LogValidationError, match="outside bounds"):
        validate_campaign_data(config(), df)


def test_validate_campaign_data_rejects_invalid_source() -> None:
    df = valid_df()
    df.loc[0, "source"] = "random"

    with pytest.raises(LogValidationError, match="invalid source 'random'"):
        validate_campaign_data(config(), df)

