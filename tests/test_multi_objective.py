from pathlib import Path

import pandas as pd
import pytest

from bo_forge.config import BOConfig, CampaignConfig, ObjectiveConfig, VariableConfig
from bo_forge.errors import ConfigError, LogValidationError, LogWriteError
from bo_forge.logs import append_suggestions, mark_observed
from bo_forge.multi_objective import (
    hypervolume,
    hypervolume_progress,
    objectives_to_model_space,
    pareto_front,
    reference_point_to_model_space,
)
from bo_forge.session import CampaignSession
from bo_forge.suggestions import suggest_next
from bo_forge.validation import canonical_columns, validate_campaign_data


def multi_config(batch_size: int = 2, initial_design_size: int = 3) -> CampaignConfig:
    return CampaignConfig(
        campaign_name="multi",
        objective=ObjectiveConfig("yield_score", "maximize", 40.0),
        objectives=(
            ObjectiveConfig("yield_score", "maximize", 40.0),
            ObjectiveConfig("waste_score", "minimize", 25.0),
        ),
        variables=(
            VariableConfig("temperature", "continuous", 20.0, 100.0),
            VariableConfig("solvent", "categorical", values=("MeCN", "Water")),
        ),
        bo=BOConfig(
            batch_size=batch_size,
            initial_design_size=initial_design_size,
            acquisition="qlog_ehvi",
            random_seed=4,
            raw_samples=8,
            num_restarts=2,
            mc_samples=8,
        ),
    )


def observed_multi_log(cfg: CampaignConfig) -> pd.DataFrame:
    rows = []
    for index, (temperature, solvent, yield_score, waste_score) in enumerate(
        [
            (30.0, "MeCN", 50.0, 20.0),
            (45.0, "Water", 65.0, 18.0),
            (65.0, "MeCN", 58.0, 12.0),
            (85.0, "Water", 72.0, 16.0),
        ]
    ):
        rows.append(
            {
                "row_id": f"obs_{index}",
                "iteration": index,
                "status": "observed",
                "source": "manual",
                "temperature": temperature,
                "solvent": solvent,
                "yield_score": yield_score,
                "waste_score": waste_score,
                "predicted_mean_yield_score": "",
                "predicted_std_yield_score": "",
                "predicted_mean_waste_score": "",
                "predicted_std_waste_score": "",
                "acquisition": "",
            }
        )
    return pd.DataFrame(rows, columns=canonical_columns(cfg))


def test_two_objective_config_parses_reference_points(tmp_path: Path) -> None:
    path = tmp_path / "multi.yaml"
    path.write_text(
        """
campaign_name: multi
objectives:
  - name: yield_score
    direction: maximize
    reference_point: 40
  - name: waste_score
    direction: minimize
    reference_point: 25
variables:
  - name: temperature
    type: continuous
    lower: 20
    upper: 100
bo:
  acquisition: qlog_ehvi
""",
        encoding="utf-8",
    )

    config = CampaignConfig.from_yaml(path)

    assert config.is_multi_objective
    assert config.objective_names == ["yield_score", "waste_score"]
    assert [objective.reference_point for objective in config.objectives] == [40.0, 25.0]


@pytest.mark.parametrize(
    "yaml_text, message",
    [
        (
            """
campaign_name: bad
objectives:
  - name: yield_score
    direction: maximize
variables: []
""",
            "exactly two",
        ),
        (
            """
campaign_name: bad
objective:
  name: yield_score
  direction: maximize
objectives:
  - name: yield_score
    direction: maximize
    reference_point: 0
  - name: waste_score
    direction: minimize
    reference_point: 1
variables: []
""",
            "either 'objective' or 'objectives'",
        ),
        (
            """
campaign_name: bad
objectives:
  - name: yield_score
    direction: maximize
    reference_point: 0
  - name: yield_score
    direction: minimize
    reference_point: 1
variables: []
""",
            "Duplicate objective",
        ),
        (
            """
campaign_name: bad
objectives:
  - name: yield_score
    direction: maximize
    reference_point: nan
  - name: waste_score
    direction: minimize
    reference_point: 1
variables: []
""",
            "finite numeric",
        ),
    ],
)
def test_invalid_multi_objective_configs_fail(
    tmp_path: Path,
    yaml_text: str,
    message: str,
) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(yaml_text, encoding="utf-8")

    with pytest.raises(ConfigError, match=message):
        CampaignConfig.from_yaml(path)


def test_multi_objective_rejects_cost_review_and_replicates(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        """
campaign_name: bad
objectives:
  - name: yield_score
    direction: maximize
    reference_point: 40
  - name: waste_score
    direction: minimize
    reference_point: 25
variables:
  - name: temperature
    type: continuous
    lower: 20
    upper: 100
review:
  enabled: true
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="review.enabled"):
        CampaignConfig.from_yaml(path)


def test_multi_objective_canonical_schema_and_validation() -> None:
    cfg = multi_config()
    df = observed_multi_log(cfg)

    assert canonical_columns(cfg) == [
        "row_id",
        "iteration",
        "status",
        "source",
        "temperature",
        "solvent",
        "yield_score",
        "waste_score",
        "predicted_mean_yield_score",
        "predicted_std_yield_score",
        "predicted_mean_waste_score",
        "predicted_std_waste_score",
        "acquisition",
    ]
    validate_campaign_data(cfg, df)


def test_multi_objective_example_config_and_log_validate() -> None:
    cfg = CampaignConfig.from_yaml(
        "configs/10_multi_objective_mixed_constrained_qlogehvi.yaml"
    )
    df = pd.read_csv(
        "examples/10_multi_objective_mixed_constrained_campaign_log.csv",
        keep_default_na=False,
    )

    validate_campaign_data(cfg, df)
    assert cfg.is_multi_objective
    assert cfg.objective_names == ["yield_score", "waste_score"]


def test_observed_rows_require_both_objectives() -> None:
    cfg = multi_config()
    df = observed_multi_log(cfg).astype(object)
    df.loc[0, "waste_score"] = ""

    with pytest.raises(LogValidationError, match="waste_score.*blank"):
        validate_campaign_data(cfg, df)


def test_suggested_rows_require_blank_objectives() -> None:
    cfg = multi_config()
    row = observed_multi_log(cfg).iloc[[0]].copy()
    row.loc[row.index[0], "status"] = "suggested"

    with pytest.raises(LogValidationError, match="yield_score.*filled"):
        validate_campaign_data(cfg, row)


def test_multi_objective_direction_and_reference_transforms() -> None:
    cfg = multi_config()
    values = pd.DataFrame(
        [{"yield_score": 55.0, "waste_score": 20.0}],
        columns=cfg.objective_names,
    )
    tensor = objectives_to_model_space(
        cfg,
        pd_to_tensor(values),
    )

    assert tensor.tolist() == [[55.0, -20.0]]
    assert reference_point_to_model_space(cfg).tolist() == [40.0, -25.0]


def test_pareto_front_and_hypervolume() -> None:
    cfg = multi_config()
    df = observed_multi_log(cfg)

    front = pareto_front(cfg, df)
    progress = hypervolume_progress(cfg, df)

    assert set(front["row_id"]) == {"obs_2", "obs_3"}
    assert hypervolume(cfg, df) > 0.0
    assert list(progress.columns) == ["observation", "row_id", "hypervolume"]


def test_hypervolume_returns_zero_when_no_observation_dominates_reference() -> None:
    cfg = multi_config()
    df = observed_multi_log(cfg)
    df["yield_score"] = 30.0
    df["waste_score"] = 30.0

    assert hypervolume(cfg, df) == 0.0


def test_qlog_ehvi_suggestions_are_valid_and_non_mutating() -> None:
    cfg = multi_config()
    df = observed_multi_log(cfg)
    before = df.copy(deep=True)

    suggestions = suggest_next(cfg, df, batch_size=2)

    pd.testing.assert_frame_equal(df, before)
    validate_campaign_data(cfg, suggestions)
    assert set(suggestions["source"]) == {"qlog_ehvi"}
    assert suggestions[["yield_score", "waste_score"]].map(lambda value: value == "").all().all()


def test_mark_observed_writes_both_objectives(tmp_path: Path) -> None:
    cfg = multi_config(initial_design_size=10)
    log_path = tmp_path / "campaign.csv"
    observed_multi_log(cfg).to_csv(log_path, index=False)
    suggestions = suggest_next(cfg, observed_multi_log(cfg), batch_size=1)
    append_suggestions(log_path, suggestions)
    row_id = str(suggestions["row_id"].iloc[0])

    mark_observed(
        log_path,
        row_id,
        objective_values={"yield_score": 61.0, "waste_score": 14.0},
    )

    written = pd.read_csv(log_path, keep_default_na=False)
    row = written.loc[written["row_id"] == row_id].iloc[0]
    assert row["status"] == "observed"
    assert float(row["yield_score"]) == 61.0
    assert float(row["waste_score"]) == 14.0


def test_mark_observed_rejects_single_objective_value_for_multi(tmp_path: Path) -> None:
    cfg = multi_config(initial_design_size=10)
    log_path = tmp_path / "campaign.csv"
    df = observed_multi_log(cfg)
    suggestions = suggest_next(cfg, df, batch_size=1)
    pd.concat([df, suggestions], ignore_index=True).to_csv(log_path, index=False)

    with pytest.raises(LogWriteError, match="objective_value is not valid"):
        mark_observed(log_path, str(suggestions["row_id"].iloc[0]), objective_value=1.0)


def test_session_multi_objective_helpers(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    log_path = tmp_path / "campaign.csv"
    config_path.write_text(
        """
campaign_name: multi
objectives:
  - name: yield_score
    direction: maximize
    reference_point: 40
  - name: waste_score
    direction: minimize
    reference_point: 25
variables:
  - name: temperature
    type: continuous
    lower: 20
    upper: 100
  - name: solvent
    type: categorical
    values: [MeCN, Water]
bo:
  acquisition: qlog_ehvi
  initial_design_size: 3
""",
        encoding="utf-8",
    )
    cfg = CampaignConfig.from_yaml(config_path)
    observed_multi_log(cfg).to_csv(log_path, index=False)

    campaign = CampaignSession.from_files(config_path, log_path)

    assert not campaign.pareto_front().empty
    assert "hypervolume" in set(campaign.summary()["field"])
    with pytest.raises(ValueError, match="pareto_front"):
        campaign.best_observation()


def test_cli_multi_objective_mark_observed_errors(tmp_path: Path) -> None:
    from bo_forge.cli import run

    cfg = multi_config(initial_design_size=10)
    config_path = tmp_path / "config.yaml"
    log_path = tmp_path / "campaign.csv"
    config_path.write_text(
        """
campaign_name: multi
objectives:
  - name: yield_score
    direction: maximize
    reference_point: 40
  - name: waste_score
    direction: minimize
    reference_point: 25
variables:
  - name: temperature
    type: continuous
    lower: 20
    upper: 100
  - name: solvent
    type: categorical
    values: [MeCN, Water]
bo:
  acquisition: qlog_ehvi
  initial_design_size: 10
""",
        encoding="utf-8",
    )
    df = observed_multi_log(cfg)
    suggestions = suggest_next(cfg, df, batch_size=1)
    pd.concat([df, suggestions], ignore_index=True).to_csv(log_path, index=False)
    row_id = str(suggestions["row_id"].iloc[0])

    assert run(
        [
            "mark-observed",
            "--config",
            str(config_path),
            "--log",
            str(log_path),
            "--row-id",
            row_id,
            "--objective-value",
            "1.0",
        ]
    ) == 1
    assert run(
        [
            "mark-observed",
            "--config",
            str(config_path),
            "--log",
            str(log_path),
            "--row-id",
            row_id,
            "--objective",
            "yield_score=60",
        ]
    ) == 1
    assert run(
        [
            "mark-observed",
            "--config",
            str(config_path),
            "--log",
            str(log_path),
            "--row-id",
            row_id,
            "--objective",
            "yield_score=60",
            "--objective",
            "waste_score=12",
        ]
    ) == 0


def pd_to_tensor(df: pd.DataFrame):
    import torch

    return torch.tensor(df.astype(float).to_numpy(), dtype=torch.double)
