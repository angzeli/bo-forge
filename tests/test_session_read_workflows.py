"""Campaign session loading, summaries, reports, and read-only workflow tests."""

from tests._session_support import (
    BOConfig,
    CampaignConfig,
    CampaignSession,
    ObjectiveConfig,
    Path,
    ReviewConfig,
    StageConfig,
    SuggestionError,
    VariableConfig,
    canonical_columns,
    config,
    cost_review_config,
    cost_review_log,
    empty_campaign_log,
    mixed_config,
    mixed_observed_log,
    observed_log,
    pd,
    pending_log,
    pytest,
    replace,
    replicate_config,
    replicate_log,
    session_module,
    structured_config,
    structured_multi_objective_config,
    structured_multi_objective_log,
    structured_observed_log,
    structured_pending_log,
    structured_replicate_config,
    structured_replicate_log,
    structured_review_config,
    structured_stage_log,
    suggestions_module,
    summary_value,
    write_config,
    write_cost_review_config,
    write_log,
    write_mixed_config,
)


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
    coverage = campaign.fidelity_coverage()
    report = campaign.report()
    text = session_module._format_campaign_report(report)

    assert summary_value(summary, "fidelity_variable") == "fidelity"
    assert summary_value(summary, "target_fidelity") == pytest.approx(1.0)
    assert summary_value(summary, "observed_rows") == 4
    assert summary_value(summary, "target_fidelity_observed_rows") == 1
    assert summary_value(summary, "best_observed_row_id") == "mf_seed_3"
    assert "fidelity_summary" in report
    assert coverage.columns.tolist()[0:3] == [
        "fidelity",
        "is_target",
        "modeled_evaluation_cost",
    ]
    assert "fidelity_coverage" in report
    assert "Fidelity Summary\n----------------" in text
    assert "Fidelity Coverage\n-----------------" in text

def test_qmfkg_timeout_before_optimization_leaves_csv_bytes_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = Path("examples/22_discrete_multi_fidelity_qmfkg_campaign_log.csv")
    log_path = tmp_path / "campaign.csv"
    log_path.write_bytes(source.read_bytes())
    campaign = CampaignSession.from_files(
        "configs/22_discrete_multi_fidelity_qmfkg.yaml",
        log_path,
    )
    assert campaign.config.fidelity is not None
    campaign.config = replace(
        campaign.config,
        fidelity=replace(
            campaign.config.fidelity,
            optimizer_timeout_seconds=1.0,
        ),
    )
    before = log_path.read_bytes()
    times = iter([50.0, 51.0])
    monkeypatch.setattr(
        suggestions_module,
        "fit_multi_fidelity_gp_model",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(suggestions_module.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(
        suggestions_module,
        "optimize_posterior_mean_at_target_fidelity",
        lambda **_kwargs: pytest.fail("target optimization must not start"),
    )

    with pytest.raises(SuggestionError, match="acquisition optimization timed out"):
        campaign.suggest_next(batch_size=1)

    assert log_path.read_bytes() == before

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
    with pytest.raises(ValueError, match="requires a config with a fidelity section"):
        campaign.fidelity_coverage()
    with pytest.raises(ValueError, match="requires a config with fidelity"):
        campaign.plot_fidelity_progress()

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

def test_contextual_cost_review_report_includes_context_and_cost_sections() -> None:
    campaign = CampaignSession.from_files(
        "configs/20_contextual_cost_review_logei.yaml",
        "examples/20_contextual_cost_review_campaign_log.csv",
    )

    context = campaign.context_summary()
    cost = campaign.cost_summary()
    report = campaign.report()
    text = session_module._format_campaign_report(report)

    assert "context_summary" in report
    assert "cost_summary" in report
    assert context["context_key"].tolist() == [
        "feedstock_acidity=0.25",
        "feedstock_acidity=0.65",
    ]
    assert summary_value(cost, "budget") == pytest.approx(90.0)
    assert "Context Summary\n---------------" in text
    assert "Cost Summary\n------------" in text

def test_contextual_cost_review_cost_summary_reserves_accepted_pending_cost(
    tmp_path: Path,
) -> None:
    config_path = Path("configs/20_contextual_cost_review_logei.yaml")
    log_path = tmp_path / "contextual_cost_review.csv"
    df = pd.read_csv(
        "examples/20_contextual_cost_review_campaign_log.csv",
        keep_default_na=False,
    )
    accepted_pending = {
        "row_id": "ctx_cost_pending_0",
        "iteration": 1,
        "status": "suggested",
        "source": "cost_log_ei",
        "review_status": "accepted",
        "review_note": "approved",
        "catalyst_loading": 0.4,
        "reaction_temperature": 80,
        "solvent": "EtOH",
        "feedstock_acidity": 0.5,
        "yield_score": "",
        "cost_estimate": 3.8,
        "cost_actual": "",
        "predicted_mean": 0.7,
        "predicted_std": 0.05,
        "acquisition": 0.1,
        "utility": -1.23,
    }
    df = pd.concat([df, pd.DataFrame([accepted_pending])], ignore_index=True)
    df.to_csv(log_path, index=False)
    campaign = CampaignSession.from_files(config_path, log_path)

    values = dict(
        zip(
            campaign.cost_summary()["field"],
            campaign.cost_summary()["value"],
            strict=True,
        )
    )

    assert values["total_observed_cost"] == pytest.approx(18.2)
    assert values["accepted_pending_cost"] == pytest.approx(3.8)
    assert values["budget_remaining"] == pytest.approx(68.0)

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

def test_qlog_nei_summary_and_report_include_pending_state_section() -> None:
    campaign = CampaignSession.from_files(
        "configs/18_noisy_pending_qlognei.yaml",
        "examples/18_noisy_pending_qlognei_campaign_log.csv",
    )

    summary = campaign.qlog_nei_summary()
    report = campaign.report()
    text = session_module._format_campaign_report(report)

    assert summary_value(summary, "observed_baseline_rows") == 4
    assert summary_value(summary, "active_pending_rows") == 1
    assert summary_value(summary, "ready_for_qlog_nei") is True
    assert "qlog_nei_summary" in report
    assert "qLogNEI Summary\n---------------" in text

def test_qlog_nei_summary_rejects_non_qlog_nei_session(tmp_path: Path) -> None:
    cfg = config()
    log_path = write_log(tmp_path / "campaign.csv", cfg, observed_log(cfg, [1.0]))
    campaign = CampaignSession(
        config_path=tmp_path / "campaign.yaml",
        log_path=log_path,
        config=cfg,
        df=pd.read_csv(log_path, keep_default_na=False),
    )

    with pytest.raises(ValueError, match="bo.acquisition: qlog_nei"):
        campaign.qlog_nei_summary()

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

def test_qlog_nehvi_accepted_pending_suggestions_are_ready_for_bo(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "qlog_nehvi.csv"
    log_path.write_text(
        Path("examples/19_multi_objective_qlognehvi_campaign_log.csv").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
    campaign = CampaignSession.from_files(
        "configs/19_multi_objective_qlognehvi.yaml",
        log_path,
    )

    action = campaign.next_action()

    assert campaign.campaign_status() == "ready_for_bo"
    assert action.loc[0, "campaign_status"] == "ready_for_bo"
    assert action.loc[0, "action"] == "suggest_bo"
    assert "qLogNEHVI" in action.loc[0, "reason"]
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
