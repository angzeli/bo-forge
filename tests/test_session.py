from pathlib import Path

import matplotlib
import pandas as pd
import pytest

from bo_forge import CampaignSession
from bo_forge.config import BOConfig, CampaignConfig, ObjectiveConfig, VariableConfig
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
    assert list(report) == ["summary", "next_action", "best_observation", "pending_suggestions"]
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

    assert list(report) == ["summary", "next_action", "best_observation", "pending_suggestions"]
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
    assert "Best Observation\n----------------" in text
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
    assert "No pending suggestions." in text


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
