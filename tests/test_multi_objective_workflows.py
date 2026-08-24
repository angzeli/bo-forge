"""Multi-objective config, validation, suggestion, report, and plot tests."""

from tests._multi_objective_support import (
    CampaignConfig,
    CampaignSession,
    ConfigError,
    LogValidationError,
    LogWriteError,
    Path,
    append_suggestions,
    canonical_columns,
    design_key_for_values,
    design_tuples,
    four_objective_config,
    has_pending_suggestions,
    hypervolume,
    hypervolume_progress,
    mark_observed,
    multi_config,
    objectives_to_model_space,
    observed_four_objective_log,
    observed_multi_log,
    pareto_front,
    pd,
    pd_to_tensor,
    pytest,
    qlog_nehvi_config,
    reference_point_to_model_space,
    review_suggestion,
    suggest_next,
    validate_campaign_data,
)


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
    if config.replicates.enabled:
        assert config.replicates.suggestion_policy == "new_only"
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
            "at least two",
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

@pytest.mark.parametrize(
    "extra_yaml",
    [
        """
review:
  enabled: true
""",
        """
replicates:
  enabled: true
""",
        """
review:
  enabled: true
replicates:
  enabled: true
""",
    ],
)
def test_multi_objective_accepts_review_and_replicates(
    tmp_path: Path,
    extra_yaml: str,
) -> None:
    path = tmp_path / "campaign.yaml"
    path.write_text(
        f"""
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
{extra_yaml}
""",
        encoding="utf-8",
    )

    config = CampaignConfig.from_yaml(path)

    assert config.is_multi_objective

@pytest.mark.parametrize(
    "extra_yaml",
    [
        """
cost:
  expression: "1.0 + temperature"
""",
        """
cost:
  expression: "1.0 + temperature"
review:
  enabled: true
""",
        """
cost:
  expression: "1.0 + temperature"
replicates:
  enabled: true
""",
        """
cost:
  expression: "1.0 + temperature"
review:
  enabled: true
replicates:
  enabled: true
""",
    ],
)
def test_multi_objective_accepts_cost_combinations(
    tmp_path: Path,
    extra_yaml: str,
) -> None:
    path = tmp_path / "cost.yaml"
    path.write_text(
        f"""
campaign_name: cost_multi
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
{extra_yaml}
""",
        encoding="utf-8",
    )

    cfg = CampaignConfig.from_yaml(path)

    assert cfg.is_multi_objective
    assert cfg.cost is not None

@pytest.mark.parametrize(
    "objective_name, variable_name, message",
    [
        ("yield_score", "yield_score", "conflicts with configured objective names"),
        ("row_id", "temperature", "reserved CSV column"),
        ("predicted_mean_yield", "temperature", "reserved CSV column prefix"),
        ("predicted_std_yield", "temperature", "reserved CSV column prefix"),
    ],
)
def test_multi_objective_rejects_ambiguous_objective_names(
    tmp_path: Path,
    objective_name: str,
    variable_name: str,
    message: str,
) -> None:
    path = tmp_path / "bad_names.yaml"
    path.write_text(
        f"""
campaign_name: bad_names
objectives:
  - name: {objective_name}
    direction: maximize
    reference_point: 0
  - name: waste_score
    direction: minimize
    reference_point: 1
variables:
  - name: {variable_name}
    type: continuous
    lower: 0
    upper: 1
bo:
  acquisition: qlog_ehvi
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match=message):
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

def test_multi_objective_log_rejects_qlog_nehvi_source() -> None:
    cfg = multi_config()
    df = observed_multi_log(cfg)
    df.loc[0, "source"] = "qlog_nehvi"

    with pytest.raises(LogValidationError, match="invalid source 'qlog_nehvi'"):
        validate_campaign_data(cfg, df)

def test_qlog_nehvi_multi_objective_log_accepts_qlog_nehvi_source() -> None:
    cfg = qlog_nehvi_config()
    df = observed_multi_log(cfg)
    df.loc[0, "source"] = "qlog_nehvi"

    validate_campaign_data(cfg, df)

def test_qlog_nehvi_multi_objective_log_accepts_historical_qlog_ehvi_source() -> None:
    cfg = qlog_nehvi_config()
    df = observed_multi_log(cfg)
    df.loc[0, "source"] = "qlog_ehvi"

    validate_campaign_data(cfg, df)

def test_multi_objective_review_and_replicate_canonical_schema() -> None:
    cfg = multi_config(review=True, replicates=True)
    df = observed_multi_log(cfg)

    assert canonical_columns(cfg) == [
        "row_id",
        "iteration",
        "status",
        "source",
        "review_status",
        "review_note",
        "replicate_group",
        "replicate_index",
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

def test_multi_objective_cost_review_replicate_canonical_schema() -> None:
    cfg = multi_config(cost=True, review=True, replicates=True)
    df = observed_multi_log(cfg)

    assert canonical_columns(cfg) == [
        "row_id",
        "iteration",
        "status",
        "source",
        "review_status",
        "review_note",
        "replicate_group",
        "replicate_index",
        "temperature",
        "solvent",
        "yield_score",
        "waste_score",
        "cost_estimate",
        "cost_actual",
        "predicted_mean_yield_score",
        "predicted_std_yield_score",
        "predicted_mean_waste_score",
        "predicted_std_waste_score",
        "acquisition",
        "utility",
    ]
    validate_campaign_data(cfg, df)

def test_four_objective_canonical_schema_and_validation() -> None:
    cfg = four_objective_config()
    df = observed_four_objective_log(cfg)

    assert canonical_columns(cfg) == [
        "row_id",
        "iteration",
        "status",
        "source",
        "catalyst_loading",
        "reaction_time",
        "base_equivalents",
        "solvent",
        "yield",
        "selectivity",
        "waste",
        "energy_use",
        "predicted_mean_yield",
        "predicted_std_yield",
        "predicted_mean_selectivity",
        "predicted_std_selectivity",
        "predicted_mean_waste",
        "predicted_std_waste",
        "predicted_mean_energy_use",
        "predicted_std_energy_use",
        "acquisition",
    ]
    validate_campaign_data(cfg, df)

def test_cost_aware_multi_objective_example_config_and_log_validate() -> None:
    cfg = CampaignConfig.from_yaml(
        "configs/12_cost_aware_multi_objective_qlogehvi.yaml"
    )
    df = pd.read_csv(
        "examples/12_cost_aware_multi_objective_campaign_log.csv",
        keep_default_na=False,
    )

    assert cfg.is_multi_objective
    assert cfg.cost is not None
    assert cfg.objective_names == ["yield", "selectivity", "waste"]
    assert list(df.columns) == canonical_columns(cfg)
    validate_campaign_data(cfg, df)

def test_qlog_nehvi_example_config_and_log_validate() -> None:
    cfg = CampaignConfig.from_yaml("configs/19_multi_objective_qlognehvi.yaml")
    df = pd.read_csv(
        "examples/19_multi_objective_qlognehvi_campaign_log.csv",
        keep_default_na=False,
    )

    assert cfg.is_multi_objective
    assert cfg.bo.acquisition == "qlog_nehvi"
    assert cfg.review.enabled
    assert list(df.columns) == canonical_columns(cfg)
    validate_campaign_data(cfg, df)

def test_multi_objective_prediction_metadata_must_be_finite_when_filled() -> None:
    cfg = four_objective_config()
    df = observed_four_objective_log(cfg).astype(object)
    df.loc[0, "predicted_mean_yield"] = "inf"

    with pytest.raises(LogValidationError, match="predicted_mean_yield"):
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

def test_four_objective_example_config_and_log_validate() -> None:
    cfg = CampaignConfig.from_yaml(
        "configs/11_four_objective_mixed_constrained_qlogehvi.yaml"
    )
    df = pd.read_csv(
        "examples/11_four_objective_mixed_constrained_campaign_log.csv",
        keep_default_na=False,
    )

    validate_campaign_data(cfg, df)
    assert cfg.is_multi_objective
    assert cfg.objective_names == ["yield", "selectivity", "waste", "energy_use"]
    assert list(df.columns) == canonical_columns(cfg)

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
    assert list(progress.columns) == ["observation", "row_id", "iteration", "hypervolume"]
    assert progress["observation"].tolist() == [1, 2, 3, 4]

def test_multi_objective_replicate_pareto_uses_group_mean_vectors() -> None:
    cfg = multi_config(replicates=True)
    df = observed_multi_log(cfg).iloc[:3].copy()
    df.loc[1, ["replicate_group", "replicate_index"]] = ["group_0", 1]
    df.loc[1, ["temperature", "solvent"]] = df.loc[0, ["temperature", "solvent"]].to_numpy()
    df.loc[0, ["yield_score", "waste_score"]] = [50.0, 20.0]
    df.loc[1, ["yield_score", "waste_score"]] = [70.0, 10.0]

    front = pareto_front(cfg, df)
    group_0 = front.loc[front["replicate_group"] == "group_0"].iloc[0]

    assert "row_id" not in front.columns
    assert {"replicate_group", "n_replicates", "first_row_id", "last_iteration"}.issubset(
        front.columns
    )
    assert group_0["yield_score"] == pytest.approx(60.0)
    assert group_0["waste_score"] == pytest.approx(15.0)
    assert hypervolume(cfg, df) > 0.0
    progress = hypervolume_progress(cfg, df)
    assert progress["hypervolume"].is_monotonic_increasing

def test_multi_objective_replicate_hypervolume_progress_is_best_so_far() -> None:
    cfg = multi_config(replicates=True)
    df = observed_multi_log(cfg).iloc[:3].copy()
    df.loc[1, ["replicate_group", "replicate_index"]] = ["group_0", 1]
    df.loc[1, ["temperature", "solvent"]] = df.loc[0, ["temperature", "solvent"]].to_numpy()
    df.loc[0, ["yield_score", "waste_score"]] = [50.0, 20.0]
    df.loc[1, ["yield_score", "waste_score"]] = [10.0, 45.0]

    raw_prefix_values = [
        hypervolume(cfg, df.iloc[: index + 1].copy()) for index in range(len(df))
    ]

    assert raw_prefix_values[1] < raw_prefix_values[0]
    progress = hypervolume_progress(cfg, df)
    expected = pd.Series(raw_prefix_values).cummax().tolist()
    assert progress["hypervolume"].tolist() == pytest.approx(expected)
    assert progress["hypervolume"].is_monotonic_increasing

def test_four_objective_pareto_front_uses_full_space_and_stable_order() -> None:
    cfg = four_objective_config()
    df = observed_four_objective_log(cfg)
    df.loc[df["row_id"] == "obs_a", ["yield", "selectivity"]] = [0.90, 0.90]
    df.loc[df["row_id"] == "obs_a", ["waste", "energy_use"]] = [0.80, 0.80]
    df.loc[df["row_id"] == "obs_d", ["yield", "selectivity"]] = [0.70, 0.70]
    df.loc[df["row_id"] == "obs_d", ["waste", "energy_use"]] = [0.20, 0.20]

    front = pareto_front(cfg, df)

    assert "obs_d" in set(front["row_id"])
    assert list(front["row_id"]) == sorted(
        front["row_id"],
        key=lambda row_id: (
            -float(front.loc[front["row_id"] == row_id, "yield"].iloc[0]),
            -float(front.loc[front["row_id"] == row_id, "selectivity"].iloc[0]),
            float(front.loc[front["row_id"] == row_id, "waste"].iloc[0]),
            float(front.loc[front["row_id"] == row_id, "energy_use"].iloc[0]),
            row_id,
        ),
    )

def test_hypervolume_returns_zero_when_no_observation_dominates_reference() -> None:
    cfg = multi_config()
    df = observed_multi_log(cfg)
    df["yield_score"] = 30.0
    df["waste_score"] = 30.0

    assert hypervolume(cfg, df) == 0.0

def test_empty_multi_objective_utilities_have_stable_shapes(tmp_path: Path) -> None:
    cfg = four_objective_config()
    empty = pd.DataFrame(columns=canonical_columns(cfg))

    assert pareto_front(cfg, empty).empty
    summary = CampaignSession(
        config_path=tmp_path / "config.yaml",
        log_path=tmp_path / "log.csv",
        config=cfg,
        df=empty,
    ).pareto_summary()
    assert dict(summary.values)["objective_count"] == 4
    assert dict(summary.values)["pareto_count"] == 0
    assert hypervolume(cfg, empty) == 0.0
    assert list(hypervolume_progress(cfg, empty).columns) == [
        "observation",
        "row_id",
        "iteration",
        "hypervolume",
    ]

def test_hypervolume_progress_repeats_for_dominated_rows() -> None:
    cfg = four_objective_config()
    df = observed_four_objective_log(cfg).iloc[:2].copy()
    dominated = df.iloc[[0]].copy()
    dominated.loc[:, "row_id"] = "dominated"
    dominated.loc[:, "iteration"] = 99
    dominated.loc[:, "catalyst_loading"] = 0.03
    dominated.loc[:, "reaction_time"] = 90
    dominated.loc[:, "yield"] = 0.25
    dominated.loc[:, "selectivity"] = 0.25
    dominated.loc[:, "waste"] = 0.85
    dominated.loc[:, "energy_use"] = 0.85
    df = pd.concat([df, dominated], ignore_index=True)

    progress = hypervolume_progress(cfg, df)

    assert list(progress.columns) == ["observation", "row_id", "iteration", "hypervolume"]
    assert progress["observation"].tolist() == [1, 2, 3]
    assert progress["hypervolume"].is_monotonic_increasing
    assert progress["hypervolume"].iloc[-1] == progress["hypervolume"].iloc[-2]

def test_qlog_ehvi_suggestions_are_valid_and_non_mutating() -> None:
    cfg = multi_config()
    df = observed_multi_log(cfg)
    before = df.copy(deep=True)

    suggestions = suggest_next(cfg, df, batch_size=2)

    pd.testing.assert_frame_equal(df, before)
    validate_campaign_data(cfg, suggestions)
    assert set(suggestions["source"]) == {"qlog_ehvi"}
    assert suggestions[["yield_score", "waste_score"]].map(lambda value: value == "").all().all()

def test_initial_multi_objective_cost_suggestions_fill_cost_and_leave_utility_blank() -> None:
    cfg = multi_config(cost=True, initial_design_size=6)
    df = observed_multi_log(cfg)

    suggestions = suggest_next(cfg, df, batch_size=2)

    validate_campaign_data(cfg, suggestions)
    assert set(suggestions["source"]) == {"sobol"}
    assert suggestions["cost_estimate"].map(lambda value: float(value) >= 0).all()
    assert suggestions["cost_actual"].map(lambda value: value == "").all()
    assert suggestions["utility"].map(lambda value: value == "").all()

def test_cost_aware_multi_objective_qlog_ehvi_suggestions_fill_batch_utility() -> None:
    cfg = multi_config(cost=True)
    df = observed_multi_log(cfg)
    before = df.copy(deep=True)

    suggestions = suggest_next(cfg, df, batch_size=2)

    pd.testing.assert_frame_equal(df, before)
    validate_campaign_data(cfg, suggestions)
    assert set(suggestions["source"]) == {"cost_qlog_ehvi"}
    assert suggestions[["yield_score", "waste_score"]].map(lambda value: value == "").all().all()
    costs = pd.to_numeric(suggestions["cost_estimate"])
    acquisition = float(suggestions["acquisition"].iloc[0])
    utility = float(suggestions["utility"].iloc[0])
    assert suggestions["utility"].astype(float).nunique() == 1
    assert utility == pytest.approx(acquisition - cfg.cost.weight * float(costs.sum()))
    prediction_columns = [
        column
        for objective in cfg.objectives
        for column in [
            f"predicted_mean_{objective.name}",
            f"predicted_std_{objective.name}",
        ]
    ]
    assert suggestions[prediction_columns].apply(pd.to_numeric).map(pd.notna).all().all()

def test_four_objective_qlog_ehvi_suggestions_are_valid_and_non_mutating() -> None:
    cfg = four_objective_config()
    df = observed_four_objective_log(cfg)
    before = df.copy(deep=True)

    suggestions = suggest_next(cfg, df, batch_size=1)

    pd.testing.assert_frame_equal(df, before)
    validate_campaign_data(cfg, suggestions)
    assert set(suggestions["source"]) == {"qlog_ehvi"}
    assert suggestions[cfg.objective_names].map(lambda value: value == "").all().all()
    prediction_columns = [
        column
        for objective in cfg.objectives
        for column in [f"predicted_mean_{objective.name}", f"predicted_std_{objective.name}"]
    ]
    assert suggestions[prediction_columns].apply(pd.to_numeric).map(pd.notna).all().all()

def test_multi_objective_qlog_ehvi_suggestions_fill_review_and_replicate_fields() -> None:
    cfg = multi_config(review=True, replicates=True)
    df = observed_multi_log(cfg)

    suggestions = suggest_next(cfg, df, batch_size=1)

    validate_campaign_data(cfg, suggestions)
    row = suggestions.iloc[0]
    assert row["source"] == "qlog_ehvi"
    assert row["review_status"] == "pending"
    assert row["review_note"] == ""
    assert row["replicate_group"] == row["row_id"]
    assert row["replicate_group"] not in set(df["replicate_group"].astype(str))
    assert int(row["replicate_index"]) == 0

def test_multi_objective_cost_summary_report_and_cost_progress_plot(tmp_path: Path) -> None:
    cfg = multi_config(cost=True)
    df = observed_multi_log(cfg)
    campaign = CampaignSession(
        config_path=tmp_path / "campaign.yaml",
        log_path=tmp_path / "campaign.csv",
        config=cfg,
        df=df,
    )

    summary = campaign.cost_summary()
    assert set(summary["field"]) == {
        "total_observed_cost",
        "accepted_pending_cost",
        "budget",
        "budget_remaining",
        "current_hypervolume",
        "pareto_count",
    }
    report = campaign.report()
    assert "cost_summary" in report
    output_path = tmp_path / "cost-progress.png"
    campaign.plot_cost_progress(save_path=output_path)
    assert output_path.is_file()

@pytest.mark.parametrize(
    ("review_status", "blocks"),
    [
        ("pending", True),
        ("accepted", True),
        ("rejected", False),
        ("deferred", False),
    ],
)
def test_multi_objective_review_status_controls_suggestion_blocking(
    review_status: str,
    blocks: bool,
) -> None:
    cfg = multi_config(review=True)
    df = observed_multi_log(cfg)
    suggestions = suggest_next(cfg, df, batch_size=1)
    suggestions.loc[:, "review_status"] = review_status
    combined = pd.concat([df, suggestions], ignore_index=True)

    assert has_pending_suggestions(combined, cfg) is blocks

def test_multi_objective_next_action_uses_objective_values_for_pending_rows(
    tmp_path: Path,
) -> None:
    cfg = multi_config()
    df = observed_multi_log(cfg)
    suggestions = suggest_next(cfg, df, batch_size=1)
    campaign = CampaignSession(
        config_path=tmp_path / "config.yaml",
        log_path=tmp_path / "campaign.csv",
        config=cfg,
        df=pd.concat([df, suggestions], ignore_index=True),
    )

    assert "objective_values" in campaign.next_action()["suggested_call"].iloc[0]
    assert "objective_value)" not in campaign.next_action()["suggested_call"].iloc[0]

def test_rejected_multi_objective_suggestions_remain_duplicate_protected() -> None:
    cfg = multi_config(review=True)
    df = observed_multi_log(cfg)
    suggestions = suggest_next(cfg, df, batch_size=1)
    suggestions.loc[:, "review_status"] = "rejected"
    combined = pd.concat([df, suggestions], ignore_index=True)
    candidate = tuple(suggestions.iloc[0][variable.name] for variable in cfg.variables)

    assert design_key_for_values(cfg, candidate) in design_tuples(cfg, combined)

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

def test_multi_objective_review_append_and_mark_observed_round_trip(tmp_path: Path) -> None:
    cfg = multi_config(review=True, initial_design_size=10)
    log_path = tmp_path / "campaign.csv"
    df = observed_multi_log(cfg)
    df.to_csv(log_path, index=False)
    suggestions = suggest_next(cfg, df, batch_size=1)
    append_suggestions(log_path, suggestions)
    row_id = str(suggestions["row_id"].iloc[0])

    with pytest.raises(LogWriteError, match="review_status"):
        mark_observed(
            log_path,
            row_id,
            objective_values={"yield_score": 61.0, "waste_score": 14.0},
        )

    review_suggestion(log_path, row_id, "accept", "ready")
    mark_observed(
        log_path,
        row_id,
        objective_values={"yield_score": 61.0, "waste_score": 14.0},
    )

    written = pd.read_csv(log_path, keep_default_na=False)
    row = written.loc[written["row_id"] == row_id].iloc[0]
    assert row["status"] == "observed"
    assert row["review_status"] == "accepted"
    assert row["review_note"] == "ready"

def test_multi_objective_replicate_append_and_mark_observed_round_trip(
    tmp_path: Path,
) -> None:
    cfg = multi_config(replicates=True, initial_design_size=10)
    log_path = tmp_path / "campaign.csv"
    df = observed_multi_log(cfg)
    df.to_csv(log_path, index=False)
    suggestions = suggest_next(cfg, df, batch_size=1)
    append_suggestions(log_path, suggestions, config=cfg)
    row_id = str(suggestions["row_id"].iloc[0])

    mark_observed(
        log_path,
        row_id,
        objective_values={"yield_score": 61.0, "waste_score": 14.0},
    )

    written = pd.read_csv(log_path, keep_default_na=False)
    row = written.loc[written["row_id"] == row_id].iloc[0]
    assert row["status"] == "observed"
    assert row["replicate_group"] == row_id
    assert int(row["replicate_index"]) == 0
    assert float(row["yield_score"]) == 61.0
    assert float(row["waste_score"]) == 14.0

def test_multi_objective_review_replicate_append_and_mark_observed_round_trip(
    tmp_path: Path,
) -> None:
    cfg = multi_config(review=True, replicates=True, initial_design_size=10)
    log_path = tmp_path / "campaign.csv"
    df = observed_multi_log(cfg)
    df.to_csv(log_path, index=False)
    suggestions = suggest_next(cfg, df, batch_size=1)
    append_suggestions(log_path, suggestions, config=cfg)
    row_id = str(suggestions["row_id"].iloc[0])

    with pytest.raises(LogWriteError, match="review_status"):
        mark_observed(
            log_path,
            row_id,
            objective_values={"yield_score": 61.0, "waste_score": 14.0},
        )

    review_suggestion(log_path, row_id, "accept", "ready")
    mark_observed(
        log_path,
        row_id,
        objective_values={"yield_score": 61.0, "waste_score": 14.0},
    )

    written = pd.read_csv(log_path, keep_default_na=False)
    row = written.loc[written["row_id"] == row_id].iloc[0]
    assert row["status"] == "observed"
    assert row["review_status"] == "accepted"
    assert row["replicate_group"] == row_id
    assert int(row["replicate_index"]) == 0

def test_mark_observed_rejects_single_objective_value_for_multi(tmp_path: Path) -> None:
    cfg = multi_config(initial_design_size=10)
    log_path = tmp_path / "campaign.csv"
    df = observed_multi_log(cfg)
    suggestions = suggest_next(cfg, df, batch_size=1)
    pd.concat([df, suggestions], ignore_index=True).to_csv(log_path, index=False)

    with pytest.raises(LogWriteError, match="objective_value is not valid"):
        mark_observed(log_path, str(suggestions["row_id"].iloc[0]), objective_value=1.0)

@pytest.mark.parametrize(
    "objective_values",
    [
        {"yield": 0.7, "selectivity": 0.8, "waste": 0.4},
        {"yield": 0.7, "selectivity": 0.8, "waste": 0.4, "unknown": 0.5},
        {"yield": 0.7, "selectivity": 0.8, "waste": 0.4, "energy_use": float("inf")},
    ],
)
def test_failed_multi_objective_mark_observed_leaves_csv_unchanged(
    tmp_path: Path,
    objective_values: dict[str, float],
) -> None:
    cfg = four_objective_config(initial_design_size=10)
    log_path = tmp_path / "campaign.csv"
    df = observed_four_objective_log(cfg)
    suggestions = suggest_next(cfg, df, batch_size=1)
    pd.concat([df, suggestions], ignore_index=True).to_csv(log_path, index=False)
    before = log_path.read_bytes()

    with pytest.raises(LogWriteError):
        mark_observed(
            log_path,
            str(suggestions["row_id"].iloc[0]),
            objective_values=objective_values,
        )

    assert log_path.read_bytes() == before

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

def test_multi_objective_review_replicate_report_includes_sections(tmp_path: Path) -> None:
    cfg = multi_config(review=True, replicates=True)
    df = observed_multi_log(cfg)
    campaign = CampaignSession(
        config_path=tmp_path / "config.yaml",
        log_path=tmp_path / "campaign.csv",
        config=cfg,
        df=df,
    )

    report = campaign.report()
    text = campaign.export_report(tmp_path / "report.txt").read_text()

    assert "review_queue" in report
    assert "replicate_summary" in report
    assert "Review Queue" in text
    assert "Replicate Summary" in text
    assert "objective_values" in campaign.next_action()["suggested_call"].iloc[0] or (
        campaign.campaign_status() != "has_pending_suggestions"
    )

def test_four_objective_session_report_and_plots(tmp_path: Path) -> None:
    cfg = four_objective_config()
    config_path = tmp_path / "config.yaml"
    log_path = tmp_path / "campaign.csv"
    config_path.write_text(
        """
campaign_name: four_objective
objectives:
  - name: yield
    direction: maximize
    reference_point: 0.2
  - name: selectivity
    direction: maximize
    reference_point: 0.2
  - name: waste
    direction: minimize
    reference_point: 0.9
  - name: energy_use
    direction: minimize
    reference_point: 0.9
variables:
  - name: catalyst_loading
    type: continuous
    lower: 0.02
    upper: 0.20
  - name: reaction_time
    type: integer
    lower: 20
    upper: 90
  - name: base_equivalents
    type: discrete
    values: [0.5, 1.0, 1.5]
  - name: solvent
    type: categorical
    values: [MeCN, DMF, Water]
constraints:
  - name: water_needs_time
    expression: "solvent != 'Water' or reaction_time >= 45"
bo:
  acquisition: qlog_ehvi
  initial_design_size: 4
  raw_samples: 8
  num_restarts: 2
  mc_samples: 8
""",
        encoding="utf-8",
    )
    observed_four_objective_log(cfg).to_csv(log_path, index=False)
    campaign = CampaignSession.from_files(config_path, log_path)

    report_path = tmp_path / "report.txt"
    pareto_path = tmp_path / "figures" / "pareto.png"
    parallel_path = tmp_path / "figures" / "parallel.png"
    hypervolume_path = tmp_path / "figures" / "hypervolume.png"

    assert "objective_count" in set(campaign.pareto_summary()["field"])
    assert "Pareto Front" in campaign.export_report(report_path).read_text()
    campaign.plot_pareto(save_path=pareto_path)
    campaign.plot_pareto_parallel(save_path=parallel_path)
    campaign.plot_hypervolume(save_path=hypervolume_path)

    assert pareto_path.exists()
    assert parallel_path.exists()
    assert hypervolume_path.exists()

def test_pairwise_pareto_plot_projects_full_space_membership(tmp_path: Path) -> None:
    cfg = four_objective_config()
    df = observed_four_objective_log(cfg)
    df.loc[df["row_id"] == "obs_a", ["yield", "selectivity"]] = [0.90, 0.90]
    df.loc[df["row_id"] == "obs_a", ["waste", "energy_use"]] = [0.80, 0.80]
    df.loc[df["row_id"] == "obs_d", ["yield", "selectivity"]] = [0.70, 0.70]
    df.loc[df["row_id"] == "obs_d", ["waste", "energy_use"]] = [0.20, 0.20]
    campaign = CampaignSession(
        config_path=tmp_path / "config.yaml",
        log_path=tmp_path / "log.csv",
        config=cfg,
        df=df,
    )
    front = campaign.pareto_front()

    _, axes = campaign.plot_pareto()

    first_panel = list(axes.flat)[0]
    pareto_offsets = first_panel.collections[1].get_offsets()
    assert len(pareto_offsets) == len(front)
    obs_d = front.loc[front["row_id"] == "obs_d"].iloc[0]
    assert any(
        float(x) == pytest.approx(float(obs_d["yield"]))
        and float(y) == pytest.approx(float(obs_d["selectivity"]))
        for x, y in pareto_offsets
    )

def test_multi_objective_plot_replicates_labels_every_objective(tmp_path: Path) -> None:
    cfg = multi_config(replicates=True)
    df = observed_multi_log(cfg)
    campaign = CampaignSession(
        config_path=tmp_path / "config.yaml",
        log_path=tmp_path / "log.csv",
        config=cfg,
        df=df,
    )

    _, axes = campaign.plot_replicates(save_path=tmp_path / "replicates.png")

    assert (tmp_path / "replicates.png").exists()
    titles = [axis.get_title() for axis in axes.flat if axis.get_visible()]
    assert any("yield_score" in title for title in titles)
    assert any("waste_score" in title for title in titles)

def test_pareto_parallel_plot_handles_mixed_directions_and_constant_objective(
    tmp_path: Path,
) -> None:
    cfg = four_objective_config()
    df = observed_four_objective_log(cfg)
    df["waste"] = 0.4
    campaign = CampaignSession(
        config_path=tmp_path / "config.yaml",
        log_path=tmp_path / "log.csv",
        config=cfg,
        df=df,
    )

    _, ax = campaign.plot_pareto_parallel(save_path=tmp_path / "parallel.png")

    assert (tmp_path / "parallel.png").exists()
    labels = [label.get_text() for label in ax.get_xticklabels()]
    assert labels == ["yield (max)", "selectivity (max)", "waste (min)", "energy_use (min)"]
    assert ax.lines
    for line in ax.lines:
        assert float(line.get_ydata()[2]) == pytest.approx(0.5)

def test_pareto_parallel_plot_empty_state(tmp_path: Path) -> None:
    cfg = four_objective_config()
    campaign = CampaignSession(
        config_path=tmp_path / "config.yaml",
        log_path=tmp_path / "log.csv",
        config=cfg,
        df=pd.DataFrame(columns=canonical_columns(cfg)),
    )

    _, ax = campaign.plot_pareto_parallel(save_path=tmp_path / "empty_parallel.png")

    assert (tmp_path / "empty_parallel.png").exists()
    assert any("No Pareto-front rows" in text.get_text() for text in ax.texts)
