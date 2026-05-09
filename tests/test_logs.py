from pathlib import Path

import pandas as pd
import pytest

from bo_forge.config import BOConfig, CampaignConfig, ObjectiveConfig, VariableConfig
from bo_forge.errors import LogWriteError
from bo_forge.logs import append_suggestions, load_campaign_log, mark_observed
from bo_forge.validation import canonical_columns


def config() -> CampaignConfig:
    return CampaignConfig(
        campaign_name="test",
        objective=ObjectiveConfig(name="activity", direction="maximize"),
        variables=(VariableConfig("x", "continuous", 0.0, 1.0),),
        bo=BOConfig(batch_size=1, initial_design_size=1),
    )


def suggestion(row_id: str = "suggested_1") -> pd.DataFrame:
    cfg = config()
    return pd.DataFrame(
        [
            {
                "row_id": row_id,
                "iteration": 0,
                "status": "suggested",
                "source": "sobol",
                "x": 0.4,
                "activity": "",
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            }
        ],
        columns=canonical_columns(cfg),
    )


def test_append_suggestions_and_mark_observed_round_trip(tmp_path: Path) -> None:
    cfg = config()
    log_path = tmp_path / "campaign.csv"

    append_suggestions(log_path, suggestion())
    mark_observed(log_path, "suggested_1", 1.7)

    df = load_campaign_log(log_path, cfg)
    assert len(df) == 1
    assert df.loc[0, "row_id"] == "suggested_1"
    assert df.loc[0, "status"] == "observed"
    assert float(df.loc[0, "activity"]) == pytest.approx(1.7)
    assert float(df.loc[0, "x"]) == pytest.approx(0.4)


def test_append_suggestions_rejects_observed_rows(tmp_path: Path) -> None:
    log_path = tmp_path / "campaign.csv"
    rows = suggestion()
    rows.loc[0, "status"] = "observed"
    rows.loc[0, "activity"] = "1.0"

    with pytest.raises(LogWriteError, match="expected status='suggested'"):
        append_suggestions(log_path, rows)


def test_append_suggestions_rejects_duplicate_row_id(tmp_path: Path) -> None:
    log_path = tmp_path / "campaign.csv"
    append_suggestions(log_path, suggestion("same"))

    with pytest.raises(LogWriteError, match="duplicate row_id 'same'"):
        append_suggestions(log_path, suggestion("same"))


def test_mark_observed_rejects_missing_row_id(tmp_path: Path) -> None:
    log_path = tmp_path / "campaign.csv"
    append_suggestions(log_path, suggestion("present"))

    with pytest.raises(LogWriteError, match="row_id was not found"):
        mark_observed(log_path, "missing", 1.0)


def test_mark_observed_rejects_already_observed_row(tmp_path: Path) -> None:
    log_path = tmp_path / "campaign.csv"
    append_suggestions(log_path, suggestion("row_1"))
    mark_observed(log_path, "row_1", 1.0)

    with pytest.raises(LogWriteError, match="status is 'observed', not 'suggested'"):
        mark_observed(log_path, "row_1", 1.2)
