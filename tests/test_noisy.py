import pandas as pd
import pytest

from bo_forge.config import (
    BOConfig,
    CampaignConfig,
    ObjectiveConfig,
    ReplicateConfig,
    ReviewConfig,
    VariableConfig,
)
from bo_forge.noisy import qlog_nei_summary
from bo_forge.validation import canonical_columns


def qlog_nei_config(
    *,
    review: bool = True,
    initial_design_size: int = 4,
    replicates: bool = False,
) -> CampaignConfig:
    return CampaignConfig(
        campaign_name="qlog_nei_summary_test",
        objective=ObjectiveConfig(name="activity", direction="maximize"),
        variables=(
            VariableConfig("x", "continuous", 0.0, 1.0),
            VariableConfig("temperature", "continuous", 300.0, 800.0),
        ),
        bo=BOConfig(
            batch_size=1,
            initial_design_size=initial_design_size,
            acquisition="qlog_nei",
            random_seed=5,
            raw_samples=8,
            num_restarts=1,
            mc_samples=8,
        ),
        review=ReviewConfig(enabled=review),
        replicates=ReplicateConfig(enabled=replicates, suggestion_policy="new_only"),
    )


def qlog_nei_log(cfg: CampaignConfig) -> pd.DataFrame:
    rows = []
    for index, (x_value, temperature, activity) in enumerate(
        [(0.1, 350.0, 0.5), (0.3, 500.0, 1.1), (0.6, 650.0, 1.8), (0.9, 780.0, 1.2)]
    ):
        row = {
            "row_id": f"obs_{index}",
            "iteration": index,
            "status": "observed",
            "source": "manual",
            "x": x_value,
            "temperature": temperature,
            "activity": activity,
            "predicted_mean": "",
            "predicted_std": "",
            "acquisition": "",
        }
        if cfg.review.enabled:
            row["review_status"] = "accepted"
            row["review_note"] = ""
        if cfg.replicates.enabled:
            row["replicate_group"] = f"group_{index}"
            row["replicate_index"] = 0
        rows.append(row)
    return pd.DataFrame(rows, columns=canonical_columns(cfg))


def summary_values(summary: pd.DataFrame) -> dict[str, object]:
    return {str(row["field"]): row["value"] for _, row in summary.iterrows()}


def suggested_row(
    cfg: CampaignConfig,
    *,
    row_id: str,
    source: str,
    review_status: str,
    x: float,
    temperature: float,
) -> dict[str, object]:
    row: dict[str, object] = {
        "row_id": row_id,
        "iteration": 5,
        "status": "suggested",
        "source": source,
        "x": x,
        "temperature": temperature,
        "activity": "",
        "predicted_mean": "",
        "predicted_std": "",
        "acquisition": "",
    }
    if cfg.review.enabled:
        row["review_status"] = review_status
        row["review_note"] = ""
    if cfg.replicates.enabled:
        row["replicate_group"] = row_id
        row["replicate_index"] = 0
    return row


def test_qlog_nei_summary_counts_pending_review_and_initial_rows() -> None:
    cfg = qlog_nei_config(initial_design_size=5)
    df = qlog_nei_log(cfg)
    extra = [
        suggested_row(
            cfg,
            row_id="accepted_bo",
            source="qlog_nei",
            review_status="accepted",
            x=0.2,
            temperature=420.0,
        ),
        suggested_row(
            cfg,
            row_id="accepted_initial",
            source="sobol",
            review_status="accepted",
            x=0.4,
            temperature=520.0,
        ),
        suggested_row(
            cfg,
            row_id="blocking",
            source="qlog_nei",
            review_status="pending",
            x=0.5,
            temperature=620.0,
        ),
        suggested_row(
            cfg,
            row_id="rejected",
            source="qlog_nei",
            review_status="rejected",
            x=0.7,
            temperature=720.0,
        ),
        suggested_row(
            cfg,
            row_id="deferred",
            source="qlog_nei",
            review_status="deferred",
            x=0.8,
            temperature=760.0,
        ),
    ]
    df = pd.concat([df, pd.DataFrame(extra, columns=canonical_columns(cfg))], ignore_index=True)

    values = summary_values(qlog_nei_summary(cfg, df))

    assert values["observed_baseline_rows"] == 4
    assert values["active_pending_rows"] == 2
    assert values["active_pending_initial_rows"] == 1
    assert values["blocking_review_pending_rows"] == 1
    assert values["rejected_or_deferred_pending_rows"] == 2
    assert values["initial_design_remaining"] == 0
    assert values["ready_for_qlog_nei"] is False
    assert values["x_pending_used"] is False


def test_qlog_nei_summary_reports_ready_after_observed_initial_design() -> None:
    cfg = qlog_nei_config(initial_design_size=4)
    df = qlog_nei_log(cfg)
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                [
                    suggested_row(
                        cfg,
                        row_id="accepted_bo",
                        source="qlog_nei",
                        review_status="accepted",
                        x=0.2,
                        temperature=420.0,
                    )
                ],
                columns=canonical_columns(cfg),
            ),
        ],
        ignore_index=True,
    )

    values = summary_values(qlog_nei_summary(cfg, df))

    assert values["initial_design_remaining"] == 0
    assert values["ready_for_qlog_nei"] is True
    assert values["x_pending_used"] is True


def test_qlog_nei_summary_reports_not_ready_before_initial_design() -> None:
    cfg = qlog_nei_config(initial_design_size=4)
    df = qlog_nei_log(cfg).iloc[:2].copy()

    values = summary_values(qlog_nei_summary(cfg, df))

    assert values["observed_baseline_rows"] == 2
    assert values["initial_design_remaining"] == 2
    assert values["ready_for_qlog_nei"] is False
    assert values["x_pending_used"] is False


def test_qlog_nei_summary_handles_empty_observed_log() -> None:
    cfg = qlog_nei_config(initial_design_size=4)
    df = pd.DataFrame(columns=canonical_columns(cfg))

    values = summary_values(qlog_nei_summary(cfg, df))

    assert values["observed_baseline_rows"] == 0
    assert values["initial_design_remaining"] == 4
    assert values["ready_for_qlog_nei"] is False
    assert values["train_yvar_available"] is False


def test_qlog_nei_summary_reports_train_yvar_for_replicate_new_only() -> None:
    cfg = qlog_nei_config(review=False, replicates=True)
    df = qlog_nei_log(cfg).iloc[:2].copy()
    replicate = df.iloc[0].copy()
    replicate["row_id"] = "obs_0_repeat"
    replicate["replicate_index"] = 1
    replicate["activity"] = 0.9
    df = pd.concat([df, replicate.to_frame().T], ignore_index=True)

    values = summary_values(qlog_nei_summary(cfg, df))

    assert values["observed_baseline_rows"] == 2
    assert values["train_yvar_available"] is True


def test_qlog_nei_summary_rejects_unsupported_config() -> None:
    cfg = qlog_nei_config()
    unsupported = CampaignConfig(
        campaign_name=cfg.campaign_name,
        objective=cfg.objective,
        variables=cfg.variables,
        bo=BOConfig(),
    )

    with pytest.raises(ValueError, match="bo.acquisition: qlog_nei"):
        qlog_nei_summary(unsupported, qlog_nei_log(cfg))
