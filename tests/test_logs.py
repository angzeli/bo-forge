from pathlib import Path

import pandas as pd
import pytest

from bo_forge.config import (
    BOConfig,
    CampaignConfig,
    CostConfig,
    FidelityConfig,
    ObjectiveConfig,
    ReplicateConfig,
    ReviewConfig,
    StageConfig,
    VariableConfig,
)
from bo_forge.errors import LogValidationError, LogWriteError
from bo_forge.logs import append_suggestions, load_campaign_log, mark_observed, review_suggestion
from bo_forge.validation import canonical_columns


def config() -> CampaignConfig:
    return CampaignConfig(
        campaign_name="test",
        objective=ObjectiveConfig(name="activity", direction="maximize"),
        variables=(VariableConfig("x", "continuous", 0.0, 1.0),),
        bo=BOConfig(batch_size=1, initial_design_size=1),
    )


def cost_review_config() -> CampaignConfig:
    cfg = config()
    return CampaignConfig(
        campaign_name=cfg.campaign_name,
        objective=cfg.objective,
        variables=cfg.variables,
        bo=cfg.bo,
        cost=CostConfig(expression="1.0 + x", budget=10.0),
        review=ReviewConfig(enabled=True),
    )


def replicate_config() -> CampaignConfig:
    cfg = config()
    return CampaignConfig(
        campaign_name=cfg.campaign_name,
        objective=cfg.objective,
        variables=cfg.variables,
        bo=cfg.bo,
        replicates=ReplicateConfig(enabled=True),
    )


def fidelity_config() -> CampaignConfig:
    return CampaignConfig(
        campaign_name="fidelity_test",
        objective=ObjectiveConfig(name="activity", direction="maximize"),
        variables=(
            VariableConfig("x", "continuous", 0.0, 1.0),
            VariableConfig("fidelity", "continuous", 0.2, 1.0),
        ),
        bo=BOConfig(batch_size=1, initial_design_size=1, acquisition="qmf_kg"),
        fidelity=FidelityConfig(variable="fidelity", target=1.0),
    )


def structured_config(*, review: bool = False) -> CampaignConfig:
    return CampaignConfig(
        campaign_name="structured_test",
        objective=ObjectiveConfig(name="activity", direction="maximize"),
        variables=(
            VariableConfig("x", "continuous", 0.0, 1.0),
            VariableConfig("temperature", "continuous", 300.0, 900.0),
        ),
        bo=BOConfig(batch_size=1, initial_design_size=1),
        review=ReviewConfig(enabled=review),
        stages=(
            StageConfig("screen", ("x",)),
            StageConfig("refine", ("x", "temperature")),
        ),
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


def cost_review_suggestion(row_id: str = "suggested_1") -> pd.DataFrame:
    cfg = cost_review_config()
    return pd.DataFrame(
        [
            {
                "row_id": row_id,
                "iteration": 0,
                "status": "suggested",
                "source": "sobol",
                "review_status": "pending",
                "review_note": "",
                "x": 0.4,
                "activity": "",
                "cost_estimate": 1.4,
                "cost_actual": "",
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
                "utility": "",
            }
        ],
        columns=canonical_columns(cfg),
    )


def structured_suggestion(*, review: bool = False) -> pd.DataFrame:
    cfg = structured_config(review=review)
    row = {
        "row_id": "structured_1",
        "iteration": 0,
        "status": "suggested",
        "source": "manual",
        "stage": "screen",
        "x": 0.4,
        "temperature": "",
        "activity": "",
        "predicted_mean": "",
        "predicted_std": "",
        "acquisition": "",
    }
    if review:
        row["review_status"] = "pending"
        row["review_note"] = ""
    return pd.DataFrame([row], columns=canonical_columns(cfg))


def qmfkg_suggestion(row_id: str = "qmfkg_1") -> pd.DataFrame:
    cfg = fidelity_config()
    return pd.DataFrame(
        [
            {
                "row_id": row_id,
                "iteration": 1,
                "status": "suggested",
                "source": "qmf_kg",
                "x": 0.4,
                "fidelity": 0.8,
                "activity": "",
                "predicted_mean": 1.2,
                "predicted_std": 0.1,
                "acquisition": 0.01,
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


def test_append_suggestions_without_config_still_supports_non_replicate_logs(
    tmp_path: Path,
) -> None:
    cfg = config()
    log_path = tmp_path / "campaign.csv"

    append_suggestions(log_path, suggestion("non_replicate"))

    df = load_campaign_log(log_path, cfg)
    assert df.loc[0, "row_id"] == "non_replicate"


def test_review_suggestion_and_mark_observed_with_actual_cost(tmp_path: Path) -> None:
    cfg = cost_review_config()
    log_path = tmp_path / "campaign.csv"

    append_suggestions(log_path, cost_review_suggestion())
    review_suggestion(log_path, "suggested_1", "accept", " approved ")
    mark_observed(log_path, "suggested_1", 1.7, actual_cost=1.25)

    df = load_campaign_log(log_path, cfg)
    assert df.loc[0, "status"] == "observed"
    assert df.loc[0, "review_status"] == "accepted"
    assert df.loc[0, "review_note"] == "approved"
    assert float(df.loc[0, "cost_actual"]) == pytest.approx(1.25)


def test_mark_observed_rejects_unaccepted_review_row(tmp_path: Path) -> None:
    log_path = tmp_path / "campaign.csv"
    append_suggestions(log_path, cost_review_suggestion())

    with pytest.raises(LogWriteError, match="review_status is 'pending', not 'accepted'"):
        mark_observed(log_path, "suggested_1", 1.7)


def test_review_suggestion_rejects_newline_note(tmp_path: Path) -> None:
    log_path = tmp_path / "campaign.csv"
    append_suggestions(log_path, cost_review_suggestion())

    with pytest.raises(LogWriteError, match="review_note cannot contain newline"):
        review_suggestion(log_path, "suggested_1", "accept", "first\nsecond")


def test_review_suggestion_rejects_non_review_log(tmp_path: Path) -> None:
    log_path = tmp_path / "campaign.csv"
    append_suggestions(log_path, suggestion())

    with pytest.raises(LogWriteError, match="review is not enabled"):
        review_suggestion(log_path, "suggested_1", "accept")


def test_mark_observed_rejects_actual_cost_without_cost_columns(tmp_path: Path) -> None:
    log_path = tmp_path / "campaign.csv"
    append_suggestions(log_path, suggestion())

    with pytest.raises(LogWriteError, match="no cost columns"):
        mark_observed(log_path, "suggested_1", 1.7, actual_cost=1.2)


def test_mark_observed_rejects_negative_actual_cost(tmp_path: Path) -> None:
    log_path = tmp_path / "campaign.csv"
    append_suggestions(log_path, cost_review_suggestion())
    review_suggestion(log_path, "suggested_1", "accept")

    with pytest.raises(LogWriteError, match="finite and >= 0"):
        mark_observed(log_path, "suggested_1", 1.7, actual_cost=-1.0)


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


def test_append_suggestions_rejects_duplicate_replicate_pair_structurally(
    tmp_path: Path,
) -> None:
    cfg = replicate_config()
    rows = pd.DataFrame(
        [
            {
                "row_id": "repeat_0",
                "iteration": 0,
                "status": "suggested",
                "source": "log_ei",
                "replicate_group": "group_0",
                "replicate_index": 1,
                "x": 0.4,
                "activity": "",
                "predicted_mean": 1.0,
                "predicted_std": 0.1,
                "acquisition": 0.0,
            },
            {
                "row_id": "repeat_1",
                "iteration": 0,
                "status": "suggested",
                "source": "log_ei",
                "replicate_group": "group_0",
                "replicate_index": 1,
                "x": 0.4,
                "activity": "",
                "predicted_mean": 1.0,
                "predicted_std": 0.1,
                "acquisition": 0.0,
            },
        ],
        columns=canonical_columns(cfg),
    )
    log_path = tmp_path / "campaign.csv"

    with pytest.raises(LogValidationError, match="Duplicate replicate row"):
        append_suggestions(log_path, rows)

    assert not log_path.exists()


def test_append_suggestions_with_config_rejects_typed_equivalent_replicate_group_without_mutation(
    tmp_path: Path,
) -> None:
    cfg = replicate_config()
    existing = pd.DataFrame(
        [
            {
                "row_id": "observed_0",
                "iteration": 0,
                "status": "observed",
                "source": "manual",
                "replicate_group": "group_0",
                "replicate_index": 0,
                "x": 0.4,
                "activity": 1.0,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            }
        ],
        columns=canonical_columns(cfg),
    )
    log_path = tmp_path / "campaign.csv"
    existing.to_csv(log_path, index=False)
    suggestions = pd.DataFrame(
        [
            {
                "row_id": "bad_repeat",
                "iteration": 1,
                "status": "suggested",
                "source": "log_ei",
                "replicate_group": "group_1",
                "replicate_index": 0,
                "x": "0.4000",
                "activity": "",
                "predicted_mean": 1.0,
                "predicted_std": 0.1,
                "acquisition": 0.0,
            }
        ],
        columns=canonical_columns(cfg),
    )
    before = log_path.read_bytes()

    with pytest.raises(LogValidationError, match="same design must share"):
        append_suggestions(log_path, suggestions, config=cfg)

    assert log_path.read_bytes() == before


def test_append_suggestions_requires_config_for_replicate_logs_without_mutation(
    tmp_path: Path,
) -> None:
    cfg = replicate_config()
    existing = pd.DataFrame(
        [
            {
                "row_id": "observed_0",
                "iteration": 0,
                "status": "observed",
                "source": "manual",
                "replicate_group": "group_0",
                "replicate_index": 0,
                "x": 0.4,
                "activity": 1.0,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            }
        ],
        columns=canonical_columns(cfg),
    )
    log_path = tmp_path / "campaign.csv"
    existing.to_csv(log_path, index=False)
    suggestions = pd.DataFrame(
        [
            {
                "row_id": "bad_repeat",
                "iteration": 1,
                "status": "suggested",
                "source": "log_ei",
                "replicate_group": "group_1",
                "replicate_index": 0,
                "x": 0.4,
                "activity": "",
                "predicted_mean": 1.0,
                "predicted_std": 0.1,
                "acquisition": 0.0,
            }
        ],
        columns=canonical_columns(cfg),
    )
    before = log_path.read_bytes()

    with pytest.raises(LogWriteError, match="Replicate append requires config-aware validation"):
        append_suggestions(log_path, suggestions)

    assert log_path.read_bytes() == before


def test_append_suggestions_requires_config_for_qmfkg_logs_without_mutation(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "campaign.csv"
    qmfkg_suggestion("existing").to_csv(log_path, index=False)
    before = log_path.read_bytes()

    with pytest.raises(LogWriteError, match="qMFKG append requires config-aware validation"):
        append_suggestions(log_path, qmfkg_suggestion())

    assert log_path.read_bytes() == before


def test_append_suggestions_with_config_accepts_qmfkg_logs(tmp_path: Path) -> None:
    cfg = fidelity_config()
    log_path = tmp_path / "campaign.csv"

    append_suggestions(log_path, qmfkg_suggestion(), config=cfg)

    df = load_campaign_log(log_path, cfg)
    assert df.loc[0, "source"] == "qmf_kg"


def test_mark_observed_requires_config_for_qmfkg_logs_without_mutation(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "campaign.csv"
    qmfkg_suggestion().to_csv(log_path, index=False)
    before = log_path.read_bytes()

    with pytest.raises(LogWriteError, match="qMFKG mark_observed requires"):
        mark_observed(log_path, "qmfkg_1", 1.7)

    assert log_path.read_bytes() == before


def test_mark_observed_requires_config_for_structured_logs_without_mutation(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "campaign.csv"
    structured_suggestion().to_csv(log_path, index=False)
    before = log_path.read_bytes()

    with pytest.raises(LogWriteError, match="Structured campaign mark_observed requires"):
        mark_observed(log_path, "structured_1", 1.7)

    assert log_path.read_bytes() == before


def test_mark_observed_with_config_supports_structured_logs(tmp_path: Path) -> None:
    cfg = structured_config()
    log_path = tmp_path / "campaign.csv"
    structured_suggestion().to_csv(log_path, index=False)

    mark_observed(log_path, "structured_1", 1.7, config=cfg)

    df = load_campaign_log(log_path, cfg)
    assert df.loc[0, "status"] == "observed"
    assert float(df.loc[0, "activity"]) == pytest.approx(1.7)


def test_review_suggestion_requires_config_for_structured_logs_without_mutation(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "campaign.csv"
    structured_suggestion(review=True).to_csv(log_path, index=False)
    before = log_path.read_bytes()

    with pytest.raises(LogWriteError, match="Structured campaign review_suggestion requires"):
        review_suggestion(log_path, "structured_1", "accept")

    assert log_path.read_bytes() == before


def test_review_suggestion_with_config_supports_structured_logs(tmp_path: Path) -> None:
    cfg = structured_config(review=True)
    log_path = tmp_path / "campaign.csv"
    structured_suggestion(review=True).to_csv(log_path, index=False)

    review_suggestion(log_path, "structured_1", "accept", config=cfg)

    df = load_campaign_log(log_path, cfg)
    assert df.loc[0, "review_status"] == "accepted"


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
