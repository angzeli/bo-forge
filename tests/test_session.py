from pathlib import Path

import matplotlib
import pandas as pd
import pytest

import bo_forge.session as session_module
from bo_forge import CampaignSession
from bo_forge.config import (
    BOConfig,
    CampaignConfig,
    CostConfig,
    ObjectiveConfig,
    ReplicateConfig,
    ReviewConfig,
    StageConfig,
    VariableConfig,
)
from bo_forge.errors import LogValidationError
from bo_forge.io import empty_campaign_log
from bo_forge.logs import append_suggestions, mark_observed
from bo_forge.validation import canonical_columns

matplotlib.use("Agg")


def write_config(path: Path, *, direction: str = "maximize", initial_design_size: int = 2) -> Path:
    path.write_text(
        f"""
campaign_name: session_test
objective:
  name: score
  direction: {direction}
variables:
  - name: x
    type: continuous
    lower: 0
    upper: 1
bo:
  batch_size: 1
  initial_design_size: {initial_design_size}
  acquisition: log_ei
  random_seed: 5
  raw_samples: 16
  num_restarts: 2
  mc_samples: 16
""",
        encoding="utf-8",
    )
    return path


def write_mixed_config(path: Path, *, initial_design_size: int = 3) -> Path:
    path.write_text(
        f"""
campaign_name: mixed_session_test
objective:
  name: score
  direction: maximize
variables:
  - name: x
    type: continuous
    lower: 0
    upper: 1
  - name: repeats
    type: integer
    lower: 1
    upper: 3
  - name: dose
    type: discrete
    values: [0.1, 0.2, 0.5]
  - name: solvent
    type: categorical
    values: [MeCN, EtOH]
bo:
  batch_size: 1
  initial_design_size: {initial_design_size}
  acquisition: log_ei
  random_seed: 5
  raw_samples: 16
  num_restarts: 2
  mc_samples: 16
""",
        encoding="utf-8",
    )
    return path


def write_cost_review_config(path: Path, *, initial_design_size: int = 2) -> Path:
    path.write_text(
        f"""
campaign_name: cost_review_session_test
objective:
  name: score
  direction: maximize
variables:
  - name: x
    type: continuous
    lower: 0
    upper: 1
cost:
  expression: "1.0 + x"
  weight: 0.5
  budget: 10
  candidate_pool_size: 16
  top_k: 8
review:
  enabled: true
bo:
  batch_size: 1
  initial_design_size: {initial_design_size}
  acquisition: log_ei
  random_seed: 5
  raw_samples: 16
  num_restarts: 2
  mc_samples: 16
""",
        encoding="utf-8",
    )
    return path


def config(direction: str = "maximize", initial_design_size: int = 2) -> CampaignConfig:
    return CampaignConfig(
        campaign_name="session_test",
        objective=ObjectiveConfig(name="score", direction=direction),
        variables=(VariableConfig("x", "continuous", 0.0, 1.0),),
        bo=BOConfig(
            batch_size=1,
            initial_design_size=initial_design_size,
            random_seed=5,
            raw_samples=16,
            num_restarts=2,
            mc_samples=16,
        ),
    )


def structured_config() -> CampaignConfig:
    cfg = config(initial_design_size=1)
    return CampaignConfig(
        campaign_name="structured_session_test",
        objective=cfg.objective,
        variables=(
            VariableConfig("x", "continuous", 0.0, 1.0),
            VariableConfig("temperature", "continuous", 300.0, 900.0),
        ),
        bo=cfg.bo,
        stages=(
            StageConfig("screen", ("x",)),
            StageConfig("refine", ("x", "temperature")),
        ),
    )


def structured_review_config() -> CampaignConfig:
    cfg = structured_config()
    return CampaignConfig(
        campaign_name=cfg.campaign_name,
        objective=cfg.objective,
        variables=cfg.variables,
        bo=cfg.bo,
        review=ReviewConfig(enabled=True),
        stages=cfg.stages,
    )


def structured_replicate_config() -> CampaignConfig:
    cfg = structured_config()
    return CampaignConfig(
        campaign_name="structured_replicate_session_test",
        objective=cfg.objective,
        variables=cfg.variables,
        bo=cfg.bo,
        replicates=ReplicateConfig(enabled=True),
        stages=cfg.stages,
    )


def structured_multi_objective_config() -> CampaignConfig:
    return CampaignConfig(
        campaign_name="structured_multi_session_test",
        objective=ObjectiveConfig("yield_score", "maximize", 0.0),
        objectives=(
            ObjectiveConfig("yield_score", "maximize", 0.0),
            ObjectiveConfig("waste_score", "minimize", 10.0),
        ),
        variables=(
            VariableConfig("x", "continuous", 0.0, 1.0),
            VariableConfig("temperature", "continuous", 300.0, 900.0),
        ),
        bo=BOConfig(
            batch_size=1,
            initial_design_size=1,
            acquisition="qlog_ehvi",
            random_seed=5,
            raw_samples=8,
            num_restarts=2,
            mc_samples=8,
        ),
        stages=(
            StageConfig("screen", ("x",)),
            StageConfig("refine", ("x", "temperature")),
        ),
    )


def mixed_config(initial_design_size: int = 3) -> CampaignConfig:
    return CampaignConfig(
        campaign_name="mixed_session_test",
        objective=ObjectiveConfig(name="score", direction="maximize"),
        variables=(
            VariableConfig("x", "continuous", 0.0, 1.0),
            VariableConfig("repeats", "integer", 1.0, 3.0),
            VariableConfig("dose", "discrete", values=(0.1, 0.2, 0.5)),
            VariableConfig("solvent", "categorical", values=("MeCN", "EtOH")),
        ),
        bo=BOConfig(
            batch_size=1,
            initial_design_size=initial_design_size,
            random_seed=5,
            raw_samples=16,
            num_restarts=2,
            mc_samples=16,
        ),
    )


def cost_review_config(initial_design_size: int = 2) -> CampaignConfig:
    cfg = config(initial_design_size=initial_design_size)
    return CampaignConfig(
        campaign_name="cost_review_session_test",
        objective=cfg.objective,
        variables=cfg.variables,
        bo=cfg.bo,
        cost=CostConfig(
            expression="1.0 + x",
            weight=0.5,
            budget=10.0,
            candidate_pool_size=16,
            top_k=8,
        ),
        review=ReviewConfig(enabled=True),
    )


def replicate_config(initial_design_size: int = 2) -> CampaignConfig:
    cfg = config(initial_design_size=initial_design_size)
    return CampaignConfig(
        campaign_name="replicate_session_test",
        objective=cfg.objective,
        variables=cfg.variables,
        bo=cfg.bo,
        replicates=ReplicateConfig(enabled=True),
    )


def observed_log(cfg: CampaignConfig, values: list[float]) -> pd.DataFrame:
    rows = []
    for index, value in enumerate(values):
        rows.append(
            {
                "row_id": f"obs_{index}",
                "iteration": index,
                "status": "observed",
                "source": "manual",
                "x": 0.2 + index * 0.2,
                "score": value,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            }
        )
    return pd.DataFrame(rows, columns=canonical_columns(cfg))


def structured_observed_log(cfg: CampaignConfig) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "row_id": "screen_0",
                "iteration": 0,
                "status": "observed",
                "source": "manual",
                "stage": "screen",
                "x": 0.2,
                "temperature": "",
                "score": 1.0,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            }
        ],
        columns=canonical_columns(cfg),
    )


def structured_pending_log(cfg: CampaignConfig) -> pd.DataFrame:
    row = {
        "row_id": "screen_1",
        "iteration": 1,
        "status": "suggested",
        "source": "manual",
        "stage": "screen",
        "x": 0.4,
        "temperature": "",
        "score": "",
        "predicted_mean": "",
        "predicted_std": "",
        "acquisition": "",
    }
    if cfg.review.enabled:
        row["review_status"] = "pending"
        row["review_note"] = ""
    return pd.DataFrame([row], columns=canonical_columns(cfg))


def structured_stage_log(cfg: CampaignConfig) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "row_id": "screen_0",
                "iteration": 0,
                "status": "observed",
                "source": "manual",
                "stage": "screen",
                "x": 0.2,
                "temperature": "",
                "score": 1.0,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            },
            {
                "row_id": "screen_1",
                "iteration": 1,
                "status": "observed",
                "source": "manual",
                "stage": "screen",
                "x": 0.7,
                "temperature": "",
                "score": 1.5,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            },
            {
                "row_id": "refine_pending",
                "iteration": 2,
                "status": "suggested",
                "source": "manual",
                "stage": "refine",
                "x": 0.6,
                "temperature": 650.0,
                "score": "",
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            },
        ],
        columns=canonical_columns(cfg),
    )


def structured_replicate_log(cfg: CampaignConfig) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "row_id": "group_0_rep_0",
                "iteration": 0,
                "status": "observed",
                "source": "manual",
                "stage": "screen",
                "replicate_group": "group_0",
                "replicate_index": 0,
                "x": 0.2,
                "temperature": "",
                "score": 1.0,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            },
            {
                "row_id": "group_0_rep_1",
                "iteration": 0,
                "status": "observed",
                "source": "manual",
                "stage": "screen",
                "replicate_group": "group_0",
                "replicate_index": 1,
                "x": 0.2,
                "temperature": "",
                "score": 3.0,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            },
            {
                "row_id": "group_1_rep_0",
                "iteration": 1,
                "status": "observed",
                "source": "manual",
                "stage": "screen",
                "replicate_group": "group_1",
                "replicate_index": 0,
                "x": 0.8,
                "temperature": "",
                "score": 2.5,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            },
        ],
        columns=canonical_columns(cfg),
    )


def structured_multi_objective_log(cfg: CampaignConfig) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "row_id": "screen_0",
                "iteration": 0,
                "status": "observed",
                "source": "manual",
                "stage": "screen",
                "x": 0.2,
                "temperature": "",
                "yield_score": 1.0,
                "waste_score": 5.0,
                "predicted_mean_yield_score": "",
                "predicted_std_yield_score": "",
                "predicted_mean_waste_score": "",
                "predicted_std_waste_score": "",
                "acquisition": "",
            },
            {
                "row_id": "screen_1",
                "iteration": 1,
                "status": "observed",
                "source": "manual",
                "stage": "screen",
                "x": 0.8,
                "temperature": "",
                "yield_score": 0.5,
                "waste_score": 1.0,
                "predicted_mean_yield_score": "",
                "predicted_std_yield_score": "",
                "predicted_mean_waste_score": "",
                "predicted_std_waste_score": "",
                "acquisition": "",
            },
        ],
        columns=canonical_columns(cfg),
    )


def mixed_observed_log(cfg: CampaignConfig) -> pd.DataFrame:
    rows = []
    for index, (x_value, repeats, dose, solvent, score) in enumerate(
        [
            (0.1, 1, 0.1, "MeCN", 1.0),
            (0.3, 2, 0.2, "EtOH", 1.4),
            (0.8, 3, 0.5, "MeCN", 1.2),
        ]
    ):
        rows.append(
            {
                "row_id": f"mixed_obs_{index}",
                "iteration": index,
                "status": "observed",
                "source": "manual",
                "x": x_value,
                "repeats": repeats,
                "dose": dose,
                "solvent": solvent,
                "score": score,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            }
        )
    return pd.DataFrame(rows, columns=canonical_columns(cfg))


def cost_review_log(cfg: CampaignConfig) -> pd.DataFrame:
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
                "source": "sobol",
                "review_status": "pending",
                "review_note": "",
                "x": 0.5,
                "score": "",
                "cost_estimate": 1.5,
                "cost_actual": "",
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
                "utility": "",
            },
        ],
        columns=canonical_columns(cfg),
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
                "score": 1.0,
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
                "score": 1.6,
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
                "x": 0.8,
                "score": 1.4,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            },
        ],
        columns=canonical_columns(cfg),
    )


def pending_log(cfg: CampaignConfig) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "row_id": "pending_0",
                "iteration": 0,
                "status": "suggested",
                "source": "sobol",
                "x": 0.5,
                "score": "",
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            }
        ],
        columns=canonical_columns(cfg),
    )


def write_log(path: Path, cfg: CampaignConfig, df: pd.DataFrame | None = None) -> Path:
    if df is None:
        df = empty_campaign_log(cfg)
    df.to_csv(path, index=False)
    return path


def summary_value(summary: pd.DataFrame, field: str):
    matches = summary.loc[summary["field"] == field, "value"]
    assert len(matches) == 1
    return matches.iloc[0]


def test_from_files_loads_config_and_log(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    cfg = config()
    log_path = write_log(tmp_path / "campaign.csv", cfg, observed_log(cfg, [1.0]))

    campaign = CampaignSession.from_files(config_path, log_path)

    assert campaign.config_path == config_path
    assert campaign.log_path == log_path
    assert campaign.config.campaign_name == "session_test"
    assert len(campaign.df) == 1


def test_structured_campaign_summary_includes_stage_metadata(tmp_path: Path) -> None:
    cfg = structured_config()
    campaign = CampaignSession(
        config_path=tmp_path / "structured.yaml",
        log_path=tmp_path / "structured.csv",
        config=cfg,
        df=structured_observed_log(cfg),
    )

    summary = campaign.summary()

    assert summary_value(summary, "structured_campaign") is True
    assert summary_value(summary, "stage_count") == 2
    assert summary_value(summary, "stages") == "screen, refine"
    assert summary_value(summary, "stage_active_variables") == (
        "screen: x; refine: x, temperature"
    )


def test_stage_summary_returns_deterministic_stage_rows(tmp_path: Path) -> None:
    cfg = structured_config()
    campaign = CampaignSession(
        config_path=tmp_path / "structured.yaml",
        log_path=tmp_path / "structured.csv",
        config=cfg,
        df=structured_stage_log(cfg),
    )

    summary = campaign.stage_summary()

    assert list(summary.columns) == [
        "stage",
        "active_variables",
        "inactive_variables",
        "total_rows",
        "observed_rows",
        "suggested_rows",
        "pending_rows",
        "best_row_id",
        "best_objective_value",
        "pareto_count",
        "warning",
        "transition_readiness",
    ]
    assert summary["stage"].tolist() == ["screen", "refine"]
    screen = summary.loc[summary["stage"] == "screen"].iloc[0]
    assert screen["active_variables"] == "x"
    assert screen["inactive_variables"] == "temperature"
    assert int(screen["observed_rows"]) == 2
    assert int(screen["pending_rows"]) == 0
    assert screen["best_row_id"] == "screen_1"
    assert float(screen["best_objective_value"]) == pytest.approx(1.5)
    assert screen["warning"] == ""
    assert screen["transition_readiness"] == "ready_for_suggestions"
    refine = summary.loc[summary["stage"] == "refine"].iloc[0]
    assert refine["active_variables"] == "x, temperature"
    assert refine["inactive_variables"] == ""
    assert int(refine["observed_rows"]) == 0
    assert int(refine["suggested_rows"]) == 1
    assert int(refine["pending_rows"]) == 1
    assert refine["warning"] == "No observed rows for stage."
    assert refine["transition_readiness"] == "resolve_pending"


def test_stage_summary_preserves_config_order_for_inactive_variables(
    tmp_path: Path,
) -> None:
    cfg = CampaignConfig(
        campaign_name="structured_order_session_test",
        objective=ObjectiveConfig("score", "maximize"),
        variables=(
            VariableConfig("x", "continuous", 0.0, 1.0),
            VariableConfig("zeta", "continuous", 0.0, 1.0),
            VariableConfig("alpha", "continuous", 0.0, 1.0),
        ),
        bo=BOConfig(batch_size=1, initial_design_size=1),
        stages=(StageConfig("screen", ("zeta",)),),
    )
    campaign = CampaignSession(
        config_path=tmp_path / "structured_order.yaml",
        log_path=tmp_path / "structured_order.csv",
        config=cfg,
        df=empty_campaign_log(cfg),
    )

    summary = campaign.stage_summary()

    assert summary.loc[0, "inactive_variables"] == "x, alpha"


def test_stage_summary_uses_replicate_group_mean_for_best_stage_row(
    tmp_path: Path,
) -> None:
    cfg = structured_replicate_config()
    campaign = CampaignSession(
        config_path=tmp_path / "structured_replicates.yaml",
        log_path=tmp_path / "structured_replicates.csv",
        config=cfg,
        df=structured_replicate_log(cfg),
    )

    summary = campaign.stage_summary()

    screen = summary.loc[summary["stage"] == "screen"].iloc[0]
    assert screen["best_row_id"] == "group_1"
    assert float(screen["best_objective_value"]) == pytest.approx(2.5)


def test_stage_summary_reports_multi_objective_pareto_count(tmp_path: Path) -> None:
    cfg = structured_multi_objective_config()
    campaign = CampaignSession(
        config_path=tmp_path / "structured_multi.yaml",
        log_path=tmp_path / "structured_multi.csv",
        config=cfg,
        df=structured_multi_objective_log(cfg),
    )

    summary = campaign.stage_summary()

    screen = summary.loc[summary["stage"] == "screen"].iloc[0]
    assert pd.isna(screen["best_row_id"])
    assert pd.isna(screen["best_objective_value"])
    assert int(screen["pareto_count"]) == 2
    refine = summary.loc[summary["stage"] == "refine"].iloc[0]
    assert int(refine["pareto_count"]) == 0
    assert refine["warning"] == "No observed rows for stage."


def test_structured_report_includes_stage_summary(tmp_path: Path) -> None:
    cfg = structured_config()
    log_path = write_log(tmp_path / "structured.csv", cfg, structured_stage_log(cfg))
    campaign = CampaignSession(
        config_path=tmp_path / "structured.yaml",
        log_path=log_path,
        config=cfg,
        df=pd.read_csv(log_path, keep_default_na=False),
    )

    report = campaign.report()
    report_path = campaign.export_report(tmp_path / "reports" / "structured.txt")
    text = report_path.read_text(encoding="utf-8")

    assert "stage_summary" in report
    assert "Stage Summary\n-------------" in text
    assert "active_variables" in text
    assert "No observed rows for stage." in text


def test_non_structured_report_has_no_stage_summary(tmp_path: Path) -> None:
    cfg = config()
    log_path = write_log(tmp_path / "campaign.csv", cfg, observed_log(cfg, [1.0]))
    campaign = CampaignSession(
        config_path=tmp_path / "campaign.yaml",
        log_path=log_path,
        config=cfg,
        df=pd.read_csv(log_path, keep_default_na=False),
    )

    report = campaign.report()
    text = session_module._format_campaign_report(report)

    assert "stage_summary" not in report
    assert "Stage Summary" not in text


def test_fidelity_summary_and_report_include_fidelity_section() -> None:
    campaign = CampaignSession.from_files(
        "configs/15_multi_fidelity_qmfkg.yaml",
        "examples/15_multi_fidelity_qmfkg_campaign_log.csv",
    )

    summary = campaign.fidelity_summary()
    report = campaign.report()
    text = session_module._format_campaign_report(report)

    assert summary_value(summary, "fidelity_variable") == "fidelity"
    assert summary_value(summary, "target_fidelity") == pytest.approx(1.0)
    assert summary_value(summary, "observed_rows") == 4
    assert summary_value(summary, "target_fidelity_observed_rows") == 1
    assert summary_value(summary, "best_observed_row_id") == "mf_seed_3"
    assert "fidelity_summary" in report
    assert "Fidelity Summary\n----------------" in text


def test_fidelity_summary_rejects_non_fidelity_session(tmp_path: Path) -> None:
    cfg = config()
    log_path = write_log(tmp_path / "campaign.csv", cfg, observed_log(cfg, [1.0]))
    campaign = CampaignSession(
        config_path=tmp_path / "campaign.yaml",
        log_path=log_path,
        config=cfg,
        df=pd.read_csv(log_path, keep_default_na=False),
    )

    with pytest.raises(ValueError, match="requires a config with a fidelity section"):
        campaign.fidelity_summary()


def test_context_summary_and_report_include_context_section() -> None:
    campaign = CampaignSession.from_files(
        "configs/16_contextual_logei.yaml",
        "examples/16_contextual_logei_campaign_log.csv",
    )

    summary = campaign.context_summary()
    report = campaign.report()
    text = session_module._format_campaign_report(report)

    assert summary["context_key"].tolist() == [
        "feedstock_acidity=0.3",
        "feedstock_acidity=0.7",
    ]
    assert "context_summary" in report
    assert "Context Summary\n---------------" in text
    assert "feedstock_acidity=0.3" in text


def test_contextual_report_handles_pending_only_log(tmp_path: Path) -> None:
    cfg = CampaignConfig.from_yaml("configs/16_contextual_logei.yaml")
    pending = {
        "row_id": "pending_0",
        "iteration": 0,
        "status": "suggested",
        "source": "sobol",
        "catalyst_loading": 0.5,
        "reaction_temperature": 80,
        "solvent": "MeCN",
        "feedstock_acidity": 0.25,
        "yield_score": "",
        "predicted_mean": "",
        "predicted_std": "",
        "acquisition": "",
    }
    log = pd.DataFrame(
        [[pending[column] for column in canonical_columns(cfg)]],
        columns=canonical_columns(cfg),
    )
    log_path = tmp_path / "contextual_pending_only.csv"
    log.to_csv(log_path, index=False)
    campaign = CampaignSession.from_files("configs/16_contextual_logei.yaml", log_path)

    summary = campaign.context_summary()
    report = campaign.report()
    text = session_module._format_campaign_report(report)

    assert summary["context_key"].tolist() == ["feedstock_acidity=0.25"]
    assert int(summary["pending_suggestions"].iloc[0]) == 1
    assert "context_summary" in report
    assert "Context Summary\n---------------" in text
    assert "feedstock_acidity=0.25" in text


def test_context_summary_rejects_non_context_session(tmp_path: Path) -> None:
    cfg = config()
    log_path = write_log(tmp_path / "campaign.csv", cfg, observed_log(cfg, [1.0]))
    campaign = CampaignSession(
        config_path=tmp_path / "campaign.yaml",
        log_path=log_path,
        config=cfg,
        df=pd.read_csv(log_path, keep_default_na=False),
    )

    with pytest.raises(ValueError, match="requires a config with a context section"):
        campaign.context_summary()


def test_model_summary_and_report_include_model_section() -> None:
    campaign = CampaignSession.from_files(
        "configs/17_model_profile_logei.yaml",
        "examples/17_model_profile_campaign_log.csv",
    )

    summary = campaign.model_summary()
    report = campaign.report()
    text = session_module._format_campaign_report(report)

    values = dict(zip(summary["field"], summary["value"], strict=True))
    assert values["model_profile"] == "smooth"
    assert values["covariance_profile"] == "RBF/ARD"
    assert values["observed_rows_used_for_fitting"] == 4
    assert "model_summary" in report
    assert "Model Summary\n-------------" in text


def test_structured_session_mutations_use_config_aware_validation(tmp_path: Path) -> None:
    cfg = structured_config()
    log_path = write_log(tmp_path / "structured.csv", cfg, structured_pending_log(cfg))
    campaign = CampaignSession(
        config_path=tmp_path / "structured.yaml",
        log_path=log_path,
        config=cfg,
        df=pd.read_csv(log_path, keep_default_na=False),
    )

    observed = campaign.mark_observed("screen_1", objective_value=1.7)

    assert observed.loc[0, "status"] == "observed"
    assert float(observed.loc[0, "score"]) == pytest.approx(1.7)

    review_cfg = structured_review_config()
    review_log_path = write_log(
        tmp_path / "structured_review.csv",
        review_cfg,
        structured_pending_log(review_cfg),
    )
    review_campaign = CampaignSession(
        config_path=tmp_path / "structured_review.yaml",
        log_path=review_log_path,
        config=review_cfg,
        df=pd.read_csv(review_log_path, keep_default_na=False),
    )

    reviewed = review_campaign.review_suggestion("screen_1", "accept")

    assert reviewed.loc[0, "review_status"] == "accepted"


def test_structured_session_suggest_next_accepts_stage_without_mutating(
    tmp_path: Path,
) -> None:
    cfg = structured_config()
    log_path = write_log(tmp_path / "structured.csv", cfg, empty_campaign_log(cfg))
    campaign = CampaignSession(
        config_path=tmp_path / "structured.yaml",
        log_path=log_path,
        config=cfg,
        df=pd.read_csv(log_path, keep_default_na=False),
    )
    before = log_path.read_bytes()

    suggestions = campaign.suggest_next(stage="screen")

    assert log_path.read_bytes() == before
    assert len(suggestions) == 1
    assert suggestions.loc[0, "stage"] == "screen"
    assert suggestions.loc[0, "x"] != ""
    assert suggestions.loc[0, "temperature"] == ""
    assert list(suggestions.columns) == canonical_columns(cfg)


def test_structured_next_action_mentions_explicit_stage(tmp_path: Path) -> None:
    cfg = structured_config()
    log_path = write_log(tmp_path / "structured.csv", cfg, empty_campaign_log(cfg))
    campaign = CampaignSession(
        config_path=tmp_path / "structured.yaml",
        log_path=log_path,
        config=cfg,
        df=pd.read_csv(log_path, keep_default_na=False),
    )

    action = campaign.next_action()

    assert "campaign.suggest_next(stage='STAGE_NAME')" in action.loc[0, "suggested_call"]


def test_mixed_session_loads_validates_reports_and_suggests(tmp_path: Path) -> None:
    config_path = write_mixed_config(tmp_path / "mixed.yaml")
    cfg = mixed_config()
    log_path = write_log(tmp_path / "mixed.csv", cfg, mixed_observed_log(cfg))
    campaign = CampaignSession.from_files(config_path, log_path)

    campaign.validate()
    summary = campaign.summary()
    report = campaign.report()
    suggestions = campaign.suggest_next(batch_size=1)

    assert summary_value(summary, "campaign_status") == "ready_for_bo"
    assert list(report) == [
        "summary",
        "next_action",
        "model_summary",
        "best_observation",
        "best_replicate_group",
        "replicate_summary",
        "pending_suggestions",
        "review_queue",
        "cost_summary",
    ]
    assert len(suggestions) == 1
    assert suggestions.loc[0, "source"] == "log_ei"
    assert suggestions.loc[0, "solvent"] in {"MeCN", "EtOH"}


def test_session_suggestion_quality_is_read_only(tmp_path: Path) -> None:
    config_path = write_mixed_config(tmp_path / "mixed.yaml")
    cfg = mixed_config()
    log_path = write_log(tmp_path / "mixed.csv", cfg, mixed_observed_log(cfg))
    campaign = CampaignSession.from_files(config_path, log_path)
    before = campaign.df.copy(deep=True)
    suggestions = campaign.suggest_next(batch_size=1)

    quality = campaign.suggestion_quality(suggestions)

    assert list(quality.columns) == [
        "row_id",
        "is_feasible",
        "violated_constraints",
        "is_exact_duplicate",
        "duplicate_allowed_by_replicates",
        "nearest_existing_distance",
        "nearest_batch_distance",
        "passes_distance_threshold",
    ]
    pd.testing.assert_frame_equal(campaign.df, before)
    pd.testing.assert_frame_equal(pd.read_csv(log_path, keep_default_na=False), before)


def test_from_files_loads_3d_example_campaign() -> None:
    campaign = CampaignSession.from_files(
        "configs/03_simple_3d_maximise_logei.yaml",
        "examples/03_simple_3d_maximise_logei_campaign_log.csv",
    )

    assert campaign.config.campaign_name == "three_variable_photocatalyst"
    assert campaign.config.variable_names == [
        "precursor_ratio",
        "annealing_temperature",
        "electrolyte_concentration",
    ]
    assert len(campaign.df) == 4


def test_from_files_loads_mixed_example_campaign() -> None:
    campaign = CampaignSession.from_files(
        "configs/05_simple_mixed_logei.yaml",
        "examples/05_simple_mixed_logei_campaign_log.csv",
    )

    assert campaign.config.campaign_name == "mixed_catalyst_screen"
    assert campaign.config.variable_names == [
        "catalyst_loading",
        "reaction_time",
        "base_equivalents",
        "solvent",
    ]
    assert len(campaign.df) == 4


def test_from_files_loads_cost_review_example_campaign() -> None:
    campaign = CampaignSession.from_files(
        "configs/07_cost_aware_human_review_logei.yaml",
        "examples/07_cost_aware_human_review_campaign_log.csv",
    )

    assert campaign.config.campaign_name == "cost_aware_human_review_catalyst_screen"
    assert campaign.config.cost is not None
    assert campaign.config.review.enabled
    assert len(campaign.df) == 4


def test_from_files_loads_replicate_example_campaign() -> None:
    campaign = CampaignSession.from_files(
        "configs/08_replicate_aware_logei.yaml",
        "examples/08_replicate_aware_campaign_log.csv",
    )

    assert campaign.config.campaign_name == "replicate_aware_photocatalyst"
    assert campaign.config.replicates.enabled
    assert len(campaign.df) == 5


def test_summary_shape_counts_status_and_no_observed_rows(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "campaign.yaml", initial_design_size=3)
    cfg = config(initial_design_size=3)
    log_path = write_log(tmp_path / "campaign.csv", cfg)
    campaign = CampaignSession.from_files(config_path, log_path)

    summary = campaign.summary()

    assert list(summary.columns) == ["field", "value"]
    assert campaign.campaign_status() == "ready_for_initial_design"
    assert summary_value(summary, "campaign_status") == "ready_for_initial_design"
    assert summary_value(summary, "total_rows") == 0
    assert summary_value(summary, "observed_rows") == 0
    assert summary_value(summary, "pending_suggestions") == 0
    assert summary_value(summary, "initial_design_remaining") == 3
    assert summary_value(summary, "best_row_id") is None
    assert summary_value(summary, "best_objective_value") is None
    pd.testing.assert_frame_equal(
        campaign.best_observation(),
        pd.DataFrame(columns=canonical_columns(campaign.config)),
    )


def test_next_action_pending_suggestions(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    cfg = config()
    log_path = write_log(tmp_path / "campaign.csv", cfg, pending_log(cfg))
    campaign = CampaignSession.from_files(config_path, log_path)

    action = campaign.next_action()

    assert list(action.columns) == ["campaign_status", "action", "reason", "suggested_call"]
    assert len(action) == 1
    assert action.loc[0, "campaign_status"] == "has_pending_suggestions"
    assert action.loc[0, "action"] == "resolve_pending_suggestions"
    assert "campaign.pending_suggestions()" in action.loc[0, "suggested_call"]
    assert "campaign.mark_observed(row_id, objective_value)" in action.loc[0, "suggested_call"]


def test_next_action_initial_design(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "campaign.yaml", initial_design_size=2)
    cfg = config(initial_design_size=2)
    log_path = write_log(tmp_path / "campaign.csv", cfg)
    campaign = CampaignSession.from_files(config_path, log_path)

    action = campaign.next_action()

    assert list(action.columns) == ["campaign_status", "action", "reason", "suggested_call"]
    assert len(action) == 1
    assert action.loc[0, "campaign_status"] == "ready_for_initial_design"
    assert action.loc[0, "action"] == "suggest_initial_design"
    assert "campaign.suggest_next()" in action.loc[0, "suggested_call"]
    assert "campaign.append_suggestions(suggestions)" in action.loc[0, "suggested_call"]


def test_next_action_ready_for_bo(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "campaign.yaml", initial_design_size=2)
    cfg = config(initial_design_size=2)
    log_path = write_log(tmp_path / "campaign.csv", cfg, observed_log(cfg, [1.0, 2.5]))
    campaign = CampaignSession.from_files(config_path, log_path)

    action = campaign.next_action()

    assert list(action.columns) == ["campaign_status", "action", "reason", "suggested_call"]
    assert len(action) == 1
    assert action.loc[0, "campaign_status"] == "ready_for_bo"
    assert action.loc[0, "action"] == "suggest_bo"
    assert "campaign.suggest_next(batch_size=...)" in action.loc[0, "suggested_call"]
    assert "campaign.append_suggestions(suggestions)" in action.loc[0, "suggested_call"]


def test_summary_status_priority_pending_wins(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "campaign.yaml", initial_design_size=3)
    cfg = config(initial_design_size=3)
    log_path = write_log(tmp_path / "campaign.csv", cfg)
    campaign = CampaignSession.from_files(config_path, log_path)

    suggestions = campaign.suggest_next(batch_size=1)
    campaign.append_suggestions(suggestions)

    assert campaign.campaign_status() == "has_pending_suggestions"
    assert summary_value(campaign.summary(), "campaign_status") == "has_pending_suggestions"
    assert len(campaign.pending_suggestions()) == 1


def test_summary_ready_for_bo_and_best_maximize(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "campaign.yaml", direction="maximize")
    cfg = config(direction="maximize")
    log_path = write_log(tmp_path / "campaign.csv", cfg, observed_log(cfg, [1.0, 2.5]))
    campaign = CampaignSession.from_files(config_path, log_path)

    summary = campaign.summary()
    best = campaign.best_observation()

    assert campaign.campaign_status() == "ready_for_bo"
    assert summary_value(summary, "campaign_status") == "ready_for_bo"
    assert summary_value(summary, "best_row_id") == "obs_1"
    assert summary_value(summary, "best_objective_value") == pytest.approx(2.5)
    assert list(best.columns) == canonical_columns(campaign.config)
    assert best["row_id"].iloc[0] == "obs_1"
    assert float(best["score"].iloc[0]) == pytest.approx(2.5)


def test_summary_best_minimize(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "campaign.yaml", direction="minimize")
    cfg = config(direction="minimize")
    log_path = write_log(tmp_path / "campaign.csv", cfg, observed_log(cfg, [1.0, 0.4]))
    campaign = CampaignSession.from_files(config_path, log_path)

    summary = campaign.summary()
    best = campaign.best_observation()

    assert summary_value(summary, "best_row_id") == "obs_1"
    assert summary_value(summary, "best_objective_value") == pytest.approx(0.4)
    assert best["row_id"].iloc[0] == "obs_1"
    assert float(best["score"].iloc[0]) == pytest.approx(0.4)


def test_best_observation_returns_copy(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "campaign.yaml", direction="maximize")
    cfg = config(direction="maximize")
    log_path = write_log(tmp_path / "campaign.csv", cfg, observed_log(cfg, [1.0, 2.5]))
    campaign = CampaignSession.from_files(config_path, log_path)

    best = campaign.best_observation()
    best.loc[best.index[0], "score"] = 99.0

    assert float(campaign.df.loc[campaign.df["row_id"] == "obs_1", "score"].iloc[0]) == 2.5


def test_read_only_helpers_do_not_mutate_df_or_disk(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "campaign.yaml", direction="maximize")
    cfg = config(direction="maximize")
    log_path = write_log(tmp_path / "campaign.csv", cfg, observed_log(cfg, [1.0, 2.5]))
    campaign = CampaignSession.from_files(config_path, log_path)
    before_df = campaign.df.copy(deep=True)
    before_csv = log_path.read_text(encoding="utf-8")

    assert campaign.campaign_status() == "ready_for_bo"
    assert not campaign.best_observation().empty
    action = campaign.next_action()
    assert action.loc[0, "action"] == "suggest_bo"

    pd.testing.assert_frame_equal(campaign.df, before_df)
    assert log_path.read_text(encoding="utf-8") == before_csv


def test_report_returns_read_only_dataframes(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "campaign.yaml", direction="maximize")
    cfg = config(direction="maximize")
    log_path = write_log(tmp_path / "campaign.csv", cfg, observed_log(cfg, [1.0, 2.5]))
    campaign = CampaignSession.from_files(config_path, log_path)
    before_df = campaign.df.copy(deep=True)
    before_csv = log_path.read_text(encoding="utf-8")
    before_paths = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))

    report = campaign.report()

    assert list(report) == [
        "summary",
        "next_action",
        "model_summary",
        "best_observation",
        "best_replicate_group",
        "replicate_summary",
        "pending_suggestions",
        "review_queue",
        "cost_summary",
    ]
    assert all(isinstance(value, pd.DataFrame) for value in report.values())
    pd.testing.assert_frame_equal(campaign.df, before_df)
    assert log_path.read_text(encoding="utf-8") == before_csv
    assert sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")) == before_paths


def test_export_report_writes_text_to_nested_path_without_mutating_campaign(
    tmp_path: Path,
) -> None:
    config_path = write_config(tmp_path / "campaign.yaml", direction="maximize")
    cfg = config(direction="maximize")
    df = pd.concat([observed_log(cfg, [1.0, 2.5]), pending_log(cfg)], ignore_index=True)
    log_path = write_log(tmp_path / "campaign.csv", cfg, df)
    campaign = CampaignSession.from_files(config_path, log_path)
    before_df = campaign.df.copy(deep=True)
    before_csv = log_path.read_text(encoding="utf-8")

    report_path = campaign.export_report(tmp_path / "reports" / "latest_campaign_report.txt")

    assert report_path == tmp_path / "reports" / "latest_campaign_report.txt"
    assert report_path.exists()
    text = report_path.read_text(encoding="utf-8")
    assert "BO Forge Campaign Report\n========================" in text
    assert "Summary\n-------" in text
    assert "Next Action\n-----------" in text
    assert "Best Raw Observation\n--------------------" in text
    assert "Best Replicate Group By Mean Objective" in text
    assert "Replicate Summary\n-----------------" in text
    assert "Pending Suggestions\n-------------------" in text
    assert "Campaign status: has_pending_suggestions" in text
    assert "Action: resolve_pending_suggestions" in text
    assert "Reason:\n  There are unresolved suggested rows" in text
    assert "Suggested call:\n  campaign.pending_suggestions()" in text
    assert "objective" in text
    assert "score" in text
    assert "observed_rows" in text
    assert "pending_suggestions" in text
    assert "row_id: obs_1" in text
    assert "status: observed" in text
    assert "score: 2.5" in text
    assert "pending_0" in text
    pd.testing.assert_frame_equal(campaign.df, before_df)
    assert log_path.read_text(encoding="utf-8") == before_csv


def test_export_report_renders_empty_sections(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "campaign.yaml", initial_design_size=3)
    cfg = config(initial_design_size=3)
    log_path = write_log(tmp_path / "campaign.csv", cfg)
    campaign = CampaignSession.from_files(config_path, log_path)

    report_path = campaign.export_report(tmp_path / "report.txt")
    text = report_path.read_text(encoding="utf-8")

    assert "No best observation yet." in text
    assert "No replicate groups observed." in text
    assert "No pending suggestions." in text
    assert "No suggestions awaiting review." in text
    assert "No cost model configured." in text


def test_cost_review_session_helpers_and_plot(tmp_path: Path) -> None:
    config_path = write_cost_review_config(tmp_path / "campaign.yaml")
    cfg = cost_review_config()
    log_path = write_log(tmp_path / "campaign.csv", cfg, cost_review_log(cfg))
    campaign = CampaignSession.from_files(config_path, log_path)
    before_csv = log_path.read_text(encoding="utf-8")

    summary = campaign.summary()
    review_queue = campaign.review_queue()
    cost_summary = campaign.cost_summary()
    report_path = campaign.export_report(tmp_path / "reports" / "cost_review.txt")
    result = campaign.plot_cost_progress(save_path=tmp_path / "reports" / "cost.png")
    report_text = report_path.read_text(encoding="utf-8")

    assert summary_value(summary, "budget") == pytest.approx(10.0)
    assert summary_value(summary, "pending_review") == 1
    assert summary_value(summary, "accepted_pending") == 0
    assert summary_value(summary, "rejected") == 0
    assert summary_value(summary, "deferred") == 0
    assert summary_value(summary, "observed_effective_cost") == pytest.approx(1.1)
    assert summary_value(summary, "accepted_pending_estimated_cost") == pytest.approx(0.0)
    assert summary_value(summary, "budget_remaining") == pytest.approx(8.9)
    assert list(cost_summary["field"]) == [
        "total_observed_cost",
        "accepted_pending_cost",
        "budget",
        "budget_remaining",
        "best_observed_objective",
    ]
    assert review_queue.iloc[0]["row_id"] == "suggested_0"
    assert "Review Queue" in report_text
    assert "Cost Summary" in report_text
    assert "pending_review" in report_text
    assert "accepted_pending" in report_text
    assert "total_observed_cost" in report_text
    assert "suggested_0" in report_text
    assert hasattr(result[0], "savefig")
    assert (tmp_path / "reports" / "cost.png").exists()
    assert log_path.read_text(encoding="utf-8") == before_csv


def test_replicate_session_helpers_summary_report_and_plot(tmp_path: Path) -> None:
    cfg = replicate_config(initial_design_size=2)
    config_path = tmp_path / "campaign.yaml"
    config_path.write_text(
        """
campaign_name: replicate_session_test
objective:
  name: score
  direction: maximize
variables:
  - name: x
    type: continuous
    lower: 0
    upper: 1
replicates:
  enabled: true
bo:
  batch_size: 1
  initial_design_size: 2
  acquisition: log_ei
  random_seed: 5
  raw_samples: 16
  num_restarts: 2
  mc_samples: 16
""",
        encoding="utf-8",
    )
    log_path = write_log(tmp_path / "campaign.csv", cfg, replicate_log(cfg))
    campaign = CampaignSession.from_files(config_path, log_path)
    before_csv = log_path.read_text(encoding="utf-8")

    summary = campaign.summary()
    replicate_summary = campaign.replicate_summary()
    best_group = campaign.best_replicate_group()
    report_path = campaign.export_report(tmp_path / "reports" / "replicate.txt")
    result = campaign.plot_replicates(save_path=tmp_path / "reports" / "replicates.png")
    report_text = report_path.read_text(encoding="utf-8")

    assert summary_value(summary, "campaign_status") == "ready_for_bo"
    assert summary_value(summary, "replicate_groups") == 2
    assert summary_value(summary, "replicated_groups") == 1
    assert summary_value(summary, "max_replicates_per_group") == 2
    assert summary_value(summary, "best_replicate_group") == "group_1"
    assert summary_value(summary, "best_replicate_mean") == pytest.approx(1.4)
    assert list(replicate_summary["replicate_group"]) == ["group_0", "group_1"]
    assert best_group["replicate_group"].iloc[0] == "group_1"
    assert "Best Raw Observation" in report_text
    assert "Best Replicate Group By Mean Objective" in report_text
    assert "Replicate Summary" in report_text
    assert hasattr(result[0], "savefig")
    assert (tmp_path / "reports" / "replicates.png").exists()
    assert log_path.read_text(encoding="utf-8") == before_csv


def test_accepted_pending_suggestions_reserve_budget(tmp_path: Path) -> None:
    config_path = write_cost_review_config(tmp_path / "campaign.yaml")
    cfg = cost_review_config()
    df = cost_review_log(cfg)
    df.loc[df["row_id"] == "suggested_0", "review_status"] = "accepted"
    log_path = write_log(tmp_path / "campaign.csv", cfg, df)
    campaign = CampaignSession.from_files(config_path, log_path)

    summary = campaign.summary()
    cost_summary = campaign.cost_summary()

    assert summary_value(summary, "pending_review") == 0
    assert summary_value(summary, "accepted_pending") == 1
    assert summary_value(summary, "accepted_pending_estimated_cost") == pytest.approx(1.5)
    assert summary_value(summary, "budget_remaining") == pytest.approx(7.4)
    assert summary_value(cost_summary, "accepted_pending_cost") == pytest.approx(1.5)


def test_summary_reports_rejected_and_deferred_review_counts(tmp_path: Path) -> None:
    config_path = write_cost_review_config(tmp_path / "campaign.yaml")
    cfg = cost_review_config()
    df = cost_review_log(cfg)
    rejected = df.loc[df["row_id"] == "suggested_0"].iloc[0].copy()
    rejected["row_id"] = "suggested_1"
    rejected["review_status"] = "rejected"
    rejected["x"] = 0.6
    rejected["cost_estimate"] = 1.6
    deferred = rejected.copy()
    deferred["row_id"] = "suggested_2"
    deferred["review_status"] = "deferred"
    deferred["x"] = 0.7
    deferred["cost_estimate"] = 1.7
    df = pd.concat([df, pd.DataFrame([rejected, deferred])], ignore_index=True)
    log_path = write_log(tmp_path / "campaign.csv", cfg, df)
    campaign = CampaignSession.from_files(config_path, log_path)

    summary = campaign.summary()

    assert summary_value(summary, "pending_review") == 1
    assert summary_value(summary, "accepted_pending") == 0
    assert summary_value(summary, "rejected") == 1
    assert summary_value(summary, "deferred") == 1


def test_next_action_review_pending_suggestions(tmp_path: Path) -> None:
    config_path = write_cost_review_config(tmp_path / "campaign.yaml")
    cfg = cost_review_config()
    log_path = write_log(tmp_path / "campaign.csv", cfg, cost_review_log(cfg))
    campaign = CampaignSession.from_files(config_path, log_path)

    action = campaign.next_action()

    assert action.loc[0, "campaign_status"] == "has_pending_suggestions"
    assert action.loc[0, "action"] == "review_pending_suggestions"
    assert "campaign.review_queue()" in action.loc[0, "suggested_call"]
    assert "campaign.review_suggestion(row_id, decision, note='')" in (
        action.loc[0, "suggested_call"]
    )


def test_next_action_review_accepted_suggestions(tmp_path: Path) -> None:
    config_path = write_cost_review_config(tmp_path / "campaign.yaml")
    cfg = cost_review_config()
    df = cost_review_log(cfg)
    df.loc[df["row_id"] == "suggested_0", "review_status"] = "accepted"
    log_path = write_log(tmp_path / "campaign.csv", cfg, df)
    campaign = CampaignSession.from_files(config_path, log_path)

    action = campaign.next_action()

    assert action.loc[0, "campaign_status"] == "has_pending_suggestions"
    assert action.loc[0, "action"] == "run_accepted_suggestions"
    assert "campaign.mark_observed(row_id, objective_value, actual_cost=...)" in (
        action.loc[0, "suggested_call"]
    )


def test_qlog_nei_accepted_pending_suggestions_are_ready_for_bo(tmp_path: Path) -> None:
    log_path = tmp_path / "qlog_nei.csv"
    log_path.write_text(
        Path("examples/18_noisy_pending_qlognei_campaign_log.csv").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
    campaign = CampaignSession.from_files("configs/18_noisy_pending_qlognei.yaml", log_path)

    action = campaign.next_action()

    assert campaign.campaign_status() == "ready_for_bo"
    assert action.loc[0, "campaign_status"] == "ready_for_bo"
    assert action.loc[0, "action"] == "suggest_bo"
    assert "X_pending" in action.loc[0, "reason"]


def test_qlog_nei_summary_counts_accepted_pending_initial_rows(
    tmp_path: Path,
) -> None:
    cfg = CampaignConfig(
        campaign_name="qlog_nei_initial_pending",
        objective=ObjectiveConfig(name="score", direction="maximize"),
        variables=(
            VariableConfig("x", "continuous", 0.0, 1.0),
            VariableConfig("temperature", "continuous", 300.0, 800.0),
        ),
        bo=BOConfig(batch_size=1, initial_design_size=4, acquisition="qlog_nei"),
        review=ReviewConfig(enabled=True),
    )
    rows = [
        {
            "row_id": f"obs_{index}",
            "iteration": index,
            "status": "observed",
            "source": "manual",
            "review_status": "accepted",
            "review_note": "",
            "x": x_value,
            "temperature": temperature,
            "score": score,
            "predicted_mean": "",
            "predicted_std": "",
            "acquisition": "",
        }
        for index, (x_value, temperature, score) in enumerate(
            [(0.1, 350.0, 0.5), (0.3, 500.0, 1.1), (0.6, 650.0, 1.8)]
        )
    ]
    rows.append(
        {
            "row_id": "initial_pending",
            "iteration": 3,
            "status": "suggested",
            "source": "sobol",
            "review_status": "accepted",
            "review_note": "",
            "x": 0.75,
            "temperature": 700.0,
            "score": "",
            "predicted_mean": "",
            "predicted_std": "",
            "acquisition": "",
        }
    )
    df = pd.DataFrame(rows, columns=canonical_columns(cfg))
    campaign = CampaignSession(
        config_path=tmp_path / "campaign.yaml",
        log_path=tmp_path / "campaign.csv",
        config=cfg,
        df=df,
    )

    summary = campaign.summary()

    assert summary_value(summary, "observed_rows") == 3
    assert summary_value(summary, "pending_suggestions") == 1
    assert summary_value(summary, "initial_design_remaining") == 0
    assert campaign.campaign_status() == "has_pending_suggestions"


def test_review_suggestion_and_mark_observed_with_actual_cost_reload(tmp_path: Path) -> None:
    config_path = write_cost_review_config(tmp_path / "campaign.yaml")
    cfg = cost_review_config()
    log_path = write_log(tmp_path / "campaign.csv", cfg, cost_review_log(cfg))
    campaign = CampaignSession.from_files(config_path, log_path)

    reviewed = campaign.review_suggestion("suggested_0", "accept", " approved ")
    assert reviewed is campaign.df
    assert campaign.df.loc[campaign.df["row_id"] == "suggested_0", "review_status"].iloc[0] == (
        "accepted"
    )
    assert campaign.df.loc[campaign.df["row_id"] == "suggested_0", "review_note"].iloc[0] == (
        "approved"
    )

    observed = campaign.mark_observed("suggested_0", 1.8, actual_cost=1.7)

    assert observed is campaign.df
    row = campaign.df.loc[campaign.df["row_id"] == "suggested_0"].iloc[0]
    assert row["status"] == "observed"
    assert float(row["score"]) == pytest.approx(1.8)
    assert float(row["cost_actual"]) == pytest.approx(1.7)


def test_suggest_next_does_not_mutate_df_or_disk(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    cfg = config()
    log_path = write_log(tmp_path / "campaign.csv", cfg)
    campaign = CampaignSession.from_files(config_path, log_path)
    before_df = campaign.df.copy(deep=True)
    before_csv = log_path.read_text(encoding="utf-8")

    suggestions = campaign.suggest_next(batch_size=1)

    assert len(suggestions) == 1
    pd.testing.assert_frame_equal(campaign.df, before_df)
    assert log_path.read_text(encoding="utf-8") == before_csv


def test_append_suggestions_and_mark_observed_auto_reload(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    cfg = config()
    log_path = write_log(tmp_path / "campaign.csv", cfg)
    campaign = CampaignSession.from_files(config_path, log_path)

    suggestions = campaign.suggest_next(batch_size=1)
    appended = campaign.append_suggestions(suggestions)

    assert appended is campaign.df
    assert len(campaign.pending_suggestions()) == 1

    row_id = str(suggestions.loc[0, "row_id"])
    observed = campaign.mark_observed(row_id, 1.2)

    assert observed is campaign.df
    assert campaign.pending_suggestions().empty
    assert campaign.df.loc[campaign.df["row_id"] == row_id, "status"].iloc[0] == "observed"
    observed_value = float(campaign.df.loc[campaign.df["row_id"] == row_id, "score"].iloc[0])
    assert observed_value == pytest.approx(1.2)


def test_session_append_invalid_replicate_suggestion_leaves_csv_bytes_unchanged(
    tmp_path: Path,
) -> None:
    base = replicate_config(initial_design_size=2)
    cfg = CampaignConfig(
        campaign_name=base.campaign_name,
        objective=base.objective,
        variables=base.variables,
        bo=base.bo,
        review=ReviewConfig(enabled=True),
        replicates=base.replicates,
    )
    df = replicate_log(base)
    df.insert(4, "review_status", "accepted")
    df.insert(5, "review_note", "")
    df = df.loc[:, canonical_columns(cfg)]
    log_path = write_log(tmp_path / "campaign.csv", cfg, df)
    campaign = CampaignSession(
        config_path=tmp_path / "campaign.yaml",
        log_path=log_path,
        config=cfg,
        df=pd.read_csv(log_path, keep_default_na=False),
    )
    bad_suggestion = campaign.df.loc[campaign.df["replicate_group"] == "group_1"].iloc[
        [0]
    ].copy().astype(object)
    bad_suggestion.loc[:, "row_id"] = "bad_repeat"
    bad_suggestion.loc[:, "status"] = "suggested"
    bad_suggestion.loc[:, "review_status"] = "pending"
    bad_suggestion.loc[:, "review_note"] = ""
    bad_suggestion.loc[:, "score"] = ""
    before = log_path.read_bytes()

    with pytest.raises(LogValidationError, match="Duplicate replicate row"):
        campaign.append_suggestions(bad_suggestion)

    assert log_path.read_bytes() == before


def test_session_append_suggestions_uses_config_aware_low_level_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = config(initial_design_size=1)
    campaign = CampaignSession(
        config_path=tmp_path / "campaign.yaml",
        log_path=tmp_path / "campaign.csv",
        config=cfg,
        df=empty_campaign_log(cfg),
    )
    suggestions = pd.DataFrame(
        [
            {
                "row_id": "suggested_0",
                "iteration": 0,
                "status": "suggested",
                "source": "sobol",
                "x": 0.4,
                "score": "",
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            }
        ],
        columns=canonical_columns(cfg),
    )
    captured: dict[str, object] = {}

    def fake_append(log_path, appended, config=None):
        captured["log_path"] = log_path
        captured["appended"] = appended
        captured["config"] = config

    monkeypatch.setattr(session_module, "_append_suggestions", fake_append)
    monkeypatch.setattr(campaign, "reload", lambda: campaign.df)

    campaign.append_suggestions(suggestions)

    assert captured["log_path"] == campaign.log_path
    assert captured["appended"] is suggestions
    assert captured["config"] is cfg


def test_reload_reflects_disk_changes(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    cfg = config()
    log_path = write_log(tmp_path / "campaign.csv", cfg)
    campaign = CampaignSession.from_files(config_path, log_path)

    suggestions = campaign.suggest_next(batch_size=1)
    append_suggestions(log_path, suggestions)
    mark_observed(log_path, str(suggestions.loc[0, "row_id"]), 0.8)

    reloaded = campaign.reload()

    assert reloaded is campaign.df
    assert len(campaign.observed_data()) == 1
    assert float(campaign.df.loc[0, "score"]) == pytest.approx(0.8)


def test_plot_methods_return_figure_and_axes_like_objects(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    cfg = config()
    log_path = write_log(tmp_path / "campaign.csv", cfg, observed_log(cfg, [1.0, 1.4]))
    campaign = CampaignSession.from_files(config_path, log_path)

    for result in [campaign.plot_progress(), campaign.plot_diagnostics()]:
        assert isinstance(result, tuple)
        assert len(result) >= 2
        figure, axes_like = result[0], result[1]
        assert hasattr(figure, "savefig")
        assert axes_like is not None


def test_plot_methods_save_paths_do_not_mutate_df_or_disk(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "campaign.yaml")
    cfg = config()
    log_path = write_log(tmp_path / "campaign.csv", cfg, observed_log(cfg, [1.0, 1.4]))
    campaign = CampaignSession.from_files(config_path, log_path)
    before_df = campaign.df.copy(deep=True)
    before_csv = log_path.read_text(encoding="utf-8")

    progress = campaign.plot_progress(save_path=tmp_path / "reports" / "progress.png")
    diagnostics = campaign.plot_diagnostics(save_path=tmp_path / "reports" / "diagnostics.png")

    assert (tmp_path / "reports" / "progress.png").exists()
    assert (tmp_path / "reports" / "diagnostics.png").exists()
    assert hasattr(progress[0], "savefig")
    assert hasattr(diagnostics[0], "savefig")
    pd.testing.assert_frame_equal(campaign.df, before_df)
    assert log_path.read_text(encoding="utf-8") == before_csv
