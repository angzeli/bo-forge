import math

import pandas as pd
import pytest

from bo_forge.config import (
    BOConfig,
    CampaignConfig,
    ObjectiveConfig,
    ReplicateConfig,
    VariableConfig,
)
from bo_forge.replicates import (
    aggregate_observed_replicates,
    best_replicate_group,
    modeling_observed_data,
    replicate_summary,
)
from bo_forge.validation import canonical_columns


def replicate_config(direction: str = "maximize") -> CampaignConfig:
    return CampaignConfig(
        campaign_name="replicate_test",
        objective=ObjectiveConfig(name="activity", direction=direction),
        variables=(
            VariableConfig("x", "continuous", 0.0, 1.0),
            VariableConfig("temperature", "integer", 300.0, 800.0),
        ),
        bo=BOConfig(batch_size=1, initial_design_size=2, random_seed=11),
        replicates=ReplicateConfig(enabled=True),
    )


def replicate_log(cfg: CampaignConfig) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "row_id": "rep_0a",
                "iteration": 0,
                "status": "observed",
                "source": "manual",
                "replicate_group": "group_0",
                "replicate_index": 0,
                "x": 0.2,
                "temperature": 500,
                "activity": 1.0,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            },
            {
                "row_id": "rep_0b",
                "iteration": 0,
                "status": "observed",
                "source": "manual",
                "replicate_group": "group_0",
                "replicate_index": 1,
                "x": 0.2,
                "temperature": 500,
                "activity": 1.4,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            },
            {
                "row_id": "rep_1a",
                "iteration": 1,
                "status": "observed",
                "source": "manual",
                "replicate_group": "group_1",
                "replicate_index": 0,
                "x": 0.7,
                "temperature": 650,
                "activity": 1.3,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            },
        ],
        columns=canonical_columns(cfg),
    )


def test_aggregate_observed_replicates_returns_exact_columns_and_stats() -> None:
    cfg = replicate_config()
    observed = replicate_log(cfg)

    aggregate = aggregate_observed_replicates(cfg, observed)

    assert list(aggregate.columns) == [
        "replicate_group",
        "x",
        "temperature",
        "n_replicates",
        "objective_mean",
        "objective_std",
        "objective_sem",
        "objective_min",
        "objective_max",
    ]
    group_0 = aggregate.loc[aggregate["replicate_group"] == "group_0"].iloc[0]
    assert group_0["n_replicates"] == 2
    assert group_0["objective_mean"] == pytest.approx(1.2)
    assert group_0["objective_std"] == pytest.approx(math.sqrt(0.08))
    assert group_0["objective_sem"] == pytest.approx(0.2)
    assert group_0["objective_min"] == pytest.approx(1.0)
    assert group_0["objective_max"] == pytest.approx(1.4)


def test_single_replicate_group_has_nan_std_and_sem() -> None:
    cfg = replicate_config()

    summary = replicate_summary(cfg, replicate_log(cfg))
    group_1 = summary.loc[summary["replicate_group"] == "group_1"].iloc[0]

    assert group_1["n_replicates"] == 1
    assert pd.isna(group_1["objective_std"])
    assert pd.isna(group_1["objective_sem"])


def test_modeling_observed_data_uses_group_means() -> None:
    cfg = replicate_config()

    model_df = modeling_observed_data(cfg, replicate_log(cfg))

    assert list(model_df.columns) == ["x", "temperature", "activity"]
    assert len(model_df) == 2
    assert model_df["activity"].tolist() == pytest.approx([1.2, 1.3])


def test_best_replicate_group_respects_maximize_and_minimize_direction() -> None:
    max_cfg = replicate_config(direction="maximize")
    min_cfg = replicate_config(direction="minimize")
    df = replicate_log(max_cfg)

    assert best_replicate_group(max_cfg, df)["replicate_group"].iloc[0] == "group_1"
    assert best_replicate_group(min_cfg, df)["replicate_group"].iloc[0] == "group_0"
