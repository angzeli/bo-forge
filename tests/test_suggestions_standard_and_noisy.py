"""Standard, structured, replicate, qLogNEI, and qLogNEHVI suggestion tests."""

from tests._suggestions_support import (
    BOConfig,
    CampaignConfig,
    ConstraintConfig,
    CostConfig,
    ModelConfig,
    ObjectiveConfig,
    ReviewConfig,
    StageConfig,
    SuggestionError,
    VariableConfig,
    append_suggestions,
    canonical_columns,
    config,
    empty_campaign_log,
    load_campaign_log,
    mark_observed,
    math,
    multi_fidelity_config,
    observed_log,
    pd,
    pytest,
    qlog_nehvi_config,
    qlog_nehvi_log,
    qlog_nehvi_pending_row,
    qlog_nei_config,
    qlog_nei_log,
    reference_point_to_model_space,
    replicate_config,
    replicate_observed_log,
    structured_config,
    suggest_next,
    suggestion_quality_summary,
    suggestions_module,
    torch,
    values_to_unit_cube,
)


def test_suggest_next_returns_sobol_initial_suggestions() -> None:
    cfg = config(batch_size=2, initial_design_size=3)
    df = empty_campaign_log(cfg)

    suggestions = suggest_next(cfg, df)

    assert len(suggestions) == 2
    assert set(suggestions["status"]) == {"suggested"}
    assert set(suggestions["source"]) == {"sobol"}
    assert suggestions["x"].astype(float).between(0.0, 1.0).all()
    assert suggestions["temperature"].astype(float).between(300.0, 800.0).all()
    assert suggestions["activity"].astype(str).eq("").all()

def test_multi_fidelity_initial_design_includes_fidelity_values() -> None:
    cfg = multi_fidelity_config(initial_design_size=3)
    df = empty_campaign_log(cfg)

    suggestions = suggest_next(cfg, df, batch_size=2)

    assert len(suggestions) == 2
    assert set(suggestions["source"]) == {"sobol"}
    assert suggestions["x"].astype(float).between(0.0, 1.0).all()
    assert suggestions["fidelity"].astype(float).between(0.2, 1.0).all()
    assert list(suggestions.columns) == canonical_columns(cfg)

def test_structured_suggest_requires_stage_when_ambiguous() -> None:
    cfg = structured_config()
    df = empty_campaign_log(cfg)

    with pytest.raises(SuggestionError, match="require an explicit stage"):
        suggest_next(cfg, df)

def test_structured_suggest_rejects_unknown_stage() -> None:
    cfg = structured_config()
    df = empty_campaign_log(cfg)

    with pytest.raises(SuggestionError, match="Unknown structured campaign stage 'unknown'"):
        suggest_next(cfg, df, stage="unknown")

def test_structured_suggest_rejects_stage_with_no_active_variables() -> None:
    cfg = CampaignConfig(
        campaign_name="bad_structured",
        objective=ObjectiveConfig(name="activity", direction="maximize"),
        variables=(VariableConfig("x", "continuous", 0.0, 1.0),),
        bo=BOConfig(batch_size=1, initial_design_size=1),
        stages=(StageConfig("empty", ()),),
    )
    df = empty_campaign_log(cfg)

    with pytest.raises(SuggestionError, match="has no active variables"):
        suggest_next(cfg, df, stage="empty")

def test_structured_suggest_with_cost_fails_with_current_version_message() -> None:
    base = structured_config()
    cfg = CampaignConfig(
        campaign_name=base.campaign_name,
        objective=base.objective,
        variables=base.variables,
        bo=base.bo,
        cost=CostConfig(expression="1.0"),
        stages=base.stages,
    )
    df = empty_campaign_log(cfg)

    with pytest.raises(
        SuggestionError,
        match="Structured campaign suggestions with cost are currently unsupported",
    ):
        suggest_next(cfg, df, stage="screen")

def test_structured_suggest_populates_stage_and_only_active_variables() -> None:
    cfg = structured_config()
    df = empty_campaign_log(cfg)

    suggestions = suggest_next(cfg, df, stage="screen")

    assert len(suggestions) == 1
    assert suggestions.loc[0, "stage"] == "screen"
    assert suggestions.loc[0, "x"] != ""
    assert suggestions.loc[0, "temperature"] == ""
    assert suggestions.loc[0, "activity"] == ""

def test_structured_model_based_suggest_uses_selected_stage() -> None:
    cfg = structured_config()
    df = pd.DataFrame(
        [
            {
                "row_id": "screen_obs_0",
                "iteration": 0,
                "status": "observed",
                "source": "manual",
                "stage": "screen",
                "x": 0.2,
                "temperature": "",
                "activity": 1.0,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            },
            {
                "row_id": "screen_obs_1",
                "iteration": 1,
                "status": "observed",
                "source": "manual",
                "stage": "screen",
                "x": 0.8,
                "temperature": "",
                "activity": 1.3,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            },
        ],
        columns=canonical_columns(cfg),
    )

    suggestions = suggest_next(cfg, df, stage="screen")

    assert len(suggestions) == 1
    assert suggestions.loc[0, "source"] == "log_ei"
    assert suggestions.loc[0, "stage"] == "screen"
    assert suggestions.loc[0, "x"] != ""
    assert suggestions.loc[0, "temperature"] == ""
    assert math.isfinite(float(suggestions.loc[0, "predicted_mean"]))
    assert float(suggestions.loc[0, "predicted_std"]) >= 0.0
    assert math.isfinite(float(suggestions.loc[0, "acquisition"]))

def test_structured_single_stage_can_infer_stage() -> None:
    base = structured_config()
    cfg = CampaignConfig(
        campaign_name=base.campaign_name,
        objective=base.objective,
        variables=base.variables,
        bo=base.bo,
        stages=(StageConfig("screen", ("x",)),),
    )
    df = empty_campaign_log(cfg)

    suggestions = suggest_next(cfg, df)

    assert suggestions.loc[0, "stage"] == "screen"
    assert suggestions.loc[0, "temperature"] == ""

def test_structured_suggest_ignores_constraints_with_inactive_variables() -> None:
    base = structured_config()
    cfg = CampaignConfig(
        campaign_name=base.campaign_name,
        objective=base.objective,
        variables=base.variables,
        bo=base.bo,
        constraints=(ConstraintConfig("temperature_low", "temperature <= 300"),),
        stages=base.stages,
    )
    df = empty_campaign_log(cfg)

    suggestions = suggest_next(cfg, df, stage="screen")

    assert suggestions.loc[0, "stage"] == "screen"
    assert suggestions.loc[0, "temperature"] == ""

def test_structured_suggest_applies_constraints_with_active_variables() -> None:
    cfg = CampaignConfig(
        campaign_name="structured_active_constraint",
        objective=ObjectiveConfig(name="activity", direction="maximize"),
        variables=(VariableConfig("solvent", "categorical", values=("A",)),),
        bo=BOConfig(batch_size=1, initial_design_size=1, random_seed=3),
        constraints=(ConstraintConfig("only_a", "solvent == 'A'"),),
        stages=(StageConfig("screen", ("solvent",)),),
    )
    df = empty_campaign_log(cfg)

    suggestions = suggest_next(cfg, df, stage="screen")

    assert suggestions.loc[0, "solvent"] == "A"

def test_structured_duplicate_checks_are_stage_aware() -> None:
    cfg = CampaignConfig(
        campaign_name="structured_duplicate",
        objective=ObjectiveConfig(name="activity", direction="maximize"),
        variables=(VariableConfig("solvent", "categorical", values=("A",)),),
        bo=BOConfig(batch_size=1, initial_design_size=1, random_seed=3),
        stages=(
            StageConfig("screen", ("solvent",)),
            StageConfig("refine", ("solvent",)),
        ),
    )
    df = pd.DataFrame(
        [
            {
                "row_id": "refine_obs",
                "iteration": 0,
                "status": "observed",
                "source": "manual",
                "stage": "refine",
                "solvent": "A",
                "activity": 1.0,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            }
        ],
        columns=canonical_columns(cfg),
    )

    suggestions = suggest_next(cfg, df, stage="screen")

    assert suggestions.loc[0, "stage"] == "screen"
    assert suggestions.loc[0, "solvent"] == "A"

def test_replicate_suggestions_set_group_to_row_id_and_start_at_zero() -> None:
    cfg = replicate_config(initial_design_size=4)
    df = empty_campaign_log(cfg)

    suggestions = suggest_next(cfg, df)

    assert len(suggestions) == 1
    assert suggestions.loc[0, "replicate_group"] == suggestions.loc[0, "row_id"]
    assert int(suggestions.loc[0, "replicate_index"]) == 0

def test_replicate_initial_design_counts_groups_not_raw_rows() -> None:
    cfg = replicate_config(initial_design_size=4)
    df = replicate_observed_log(cfg).astype(object)

    suggestions = suggest_next(cfg, df)

    assert suggestions.loc[0, "source"] == "sobol"

def test_replicate_model_based_suggestions_use_aggregated_observations() -> None:
    cfg = replicate_config(initial_design_size=3, suggestion_policy="new_only")
    df = replicate_observed_log(cfg).astype(object)

    suggestions = suggest_next(cfg, df)

    assert len(suggestions) == 1
    assert suggestions.loc[0, "source"] == "log_ei"
    assert suggestions.loc[0, "replicate_group"] == suggestions.loc[0, "row_id"]
    existing_designs = {
        (float(row["x"]), float(row["temperature"]))
        for _, row in df.iterrows()
    }
    suggested_design = (
        float(suggestions.loc[0, "x"]),
        float(suggestions.loc[0, "temperature"]),
    )
    assert suggested_design not in existing_designs

def test_replicate_uncertain_best_policy_suggests_same_group_repeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = replicate_config(initial_design_size=3, replicate_threshold=0.10)
    df = replicate_observed_log(cfg)

    class FakePosterior:
        mean = torch.tensor([[0.5], [2.0], [1.0]], dtype=torch.double)
        variance = torch.tensor([[0.01], [0.04], [0.01]], dtype=torch.double)

    class FakeModel:
        def posterior(self, _x):
            return FakePosterior()

    monkeypatch.setattr(suggestions_module, "fit_gp_model", lambda *_args: FakeModel())

    suggestions = suggest_next(cfg, df)

    assert len(suggestions) == 1
    assert suggestions.loc[0, "source"] == "log_ei"
    assert suggestions.loc[0, "replicate_group"] == "group_1"
    assert int(suggestions.loc[0, "replicate_index"]) == 1
    assert float(suggestions.loc[0, "x"]) == pytest.approx(0.4)
    assert float(suggestions.loc[0, "temperature"]) == pytest.approx(550.0)
    assert float(suggestions.loc[0, "predicted_mean"]) == pytest.approx(2.0)
    assert float(suggestions.loc[0, "predicted_std"]) == pytest.approx(0.2)

def test_uncertain_best_uses_next_replicate_index_from_all_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = replicate_config(initial_design_size=3, replicate_threshold=0.10)
    cfg = CampaignConfig(
        campaign_name=base.campaign_name,
        objective=base.objective,
        variables=base.variables,
        bo=base.bo,
        review=ReviewConfig(enabled=True),
        replicates=base.replicates,
    )
    df = replicate_observed_log(cfg).astype(object)
    df.loc[:, "review_status"] = "accepted"
    df.loc[:, "review_note"] = ""
    rejected = df.loc[df["replicate_group"] == "group_1"].iloc[[0]].copy()
    rejected.loc[:, "row_id"] = "rejected_repeat"
    rejected.loc[:, "status"] = "suggested"
    rejected.loc[:, "review_status"] = "rejected"
    rejected.loc[:, "review_note"] = "do not run"
    rejected.loc[:, "replicate_index"] = 1
    rejected.loc[:, "activity"] = ""
    df = pd.concat([df, rejected], ignore_index=True)

    class FakePosterior:
        mean = torch.tensor([[0.5], [2.0], [1.0]], dtype=torch.double)
        variance = torch.tensor([[0.01], [0.04], [0.01]], dtype=torch.double)

    class FakeModel:
        def posterior(self, _x):
            return FakePosterior()

    monkeypatch.setattr(suggestions_module, "fit_gp_model", lambda *_args: FakeModel())

    suggestions = suggest_next(cfg, df)

    assert suggestions.loc[0, "replicate_group"] == "group_1"
    assert int(suggestions.loc[0, "replicate_index"]) == 2

def test_replicate_repeat_suggestion_round_trips_as_same_group_duplicate(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = replicate_config(initial_design_size=3, replicate_threshold=0.10)
    df = replicate_observed_log(cfg)

    class FakePosterior:
        mean = torch.tensor([[0.5], [2.0], [1.0]], dtype=torch.double)
        variance = torch.tensor([[0.01], [0.04], [0.01]], dtype=torch.double)

    class FakeModel:
        def posterior(self, _x):
            return FakePosterior()

    monkeypatch.setattr(suggestions_module, "fit_gp_model", lambda *_args: FakeModel())
    suggestions = suggest_next(cfg, df)
    log_path = tmp_path / "campaign.csv"
    df.to_csv(log_path, index=False)

    append_suggestions(log_path, suggestions, config=cfg)
    mark_observed(log_path, str(suggestions.loc[0, "row_id"]), objective_value=1.45)

    written = load_campaign_log(log_path, cfg)
    repeated = written.loc[written["replicate_group"] == "group_1"]
    assert len(repeated) == 2
    assert sorted(repeated["replicate_index"].astype(int).tolist()) == [0, 1]

def test_suggestion_quality_marks_intentional_replicate_without_duplicate_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = replicate_config(initial_design_size=3, replicate_threshold=0.10)
    df = replicate_observed_log(cfg)

    class FakePosterior:
        mean = torch.tensor([[0.5], [2.0], [1.0]], dtype=torch.double)
        variance = torch.tensor([[0.01], [0.04], [0.01]], dtype=torch.double)

    class FakeModel:
        def posterior(self, _x):
            return FakePosterior()

    monkeypatch.setattr(suggestions_module, "fit_gp_model", lambda *_args: FakeModel())
    suggestions = suggest_next(cfg, df)

    summary = suggestion_quality_summary(cfg, df, suggestions)

    assert bool(summary.loc[0, "duplicate_allowed_by_replicates"])
    assert not bool(summary.loc[0, "is_exact_duplicate"])
    assert summary.loc[0, "nearest_existing_distance"] == pytest.approx(0.0)
    assert bool(summary.loc[0, "passes_distance_threshold"])

def test_suggestion_quality_allows_intentional_replicate_batch_duplicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = replicate_config(
        initial_design_size=3,
        replicate_threshold=0.10,
        min_repeats_at_best=3,
    )
    df = replicate_observed_log(cfg)

    class FakePosterior:
        mean = torch.tensor([[0.5], [2.0], [1.0]], dtype=torch.double)
        variance = torch.tensor([[0.01], [0.04], [0.01]], dtype=torch.double)

    class FakeModel:
        def posterior(self, _x):
            return FakePosterior()

    monkeypatch.setattr(suggestions_module, "fit_gp_model", lambda *_args: FakeModel())
    suggestions = suggest_next(cfg, df, batch_size=2)

    summary = suggestion_quality_summary(cfg, df, suggestions)

    assert suggestions["replicate_index"].astype(int).tolist() == [1, 2]
    assert summary["duplicate_allowed_by_replicates"].astype(bool).all()
    assert not summary["is_exact_duplicate"].astype(bool).any()
    assert summary["passes_distance_threshold"].astype(bool).all()

def test_cost_replicate_uncertain_best_fills_cost_without_utility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = replicate_config(initial_design_size=3, replicate_threshold=0.10)
    cfg = CampaignConfig(
        campaign_name=base.campaign_name,
        objective=base.objective,
        variables=base.variables,
        bo=base.bo,
        cost=CostConfig(
            expression="1.0 + x",
            weight=0.5,
            budget=100.0,
            candidate_pool_size=16,
            top_k=8,
        ),
        replicates=base.replicates,
    )
    df = replicate_observed_log(cfg)

    class FakePosterior:
        mean = torch.tensor([[0.5], [2.0], [1.0]], dtype=torch.double)
        variance = torch.tensor([[0.01], [0.04], [0.01]], dtype=torch.double)

    class FakeModel:
        def posterior(self, _x):
            return FakePosterior()

    monkeypatch.setattr(suggestions_module, "fit_gp_model", lambda *_args: FakeModel())

    suggestions = suggest_next(cfg, df)

    assert suggestions.loc[0, "source"] == "log_ei"
    assert float(suggestions.loc[0, "cost_estimate"]) == pytest.approx(1.4)
    assert str(suggestions.loc[0, "utility"]) == ""

def test_uncertain_best_fills_remaining_batch_with_exploration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = replicate_config(
        initial_design_size=3,
        replicate_threshold=0.10,
        min_repeats_at_best=2,
    )
    df = replicate_observed_log(cfg)
    captured: dict[str, pd.DataFrame] = {}

    class FakePosterior:
        mean = torch.tensor([[0.5], [2.0], [1.0]], dtype=torch.double)
        variance = torch.tensor([[0.01], [0.04], [0.01]], dtype=torch.double)

    class FakeModel:
        def posterior(self, _x):
            return FakePosterior()

    def fake_model_based(**kwargs):
        captured["df"] = kwargs["df"]
        row = {
            "row_id": "new_suggestion",
            "iteration": 99,
            "status": "suggested",
            "source": "log_ei",
            "replicate_group": "new_suggestion",
            "replicate_index": 0,
            "x": 0.9,
            "temperature": 700.0,
            "activity": "",
            "predicted_mean": 1.5,
            "predicted_std": 0.1,
            "acquisition": 0.2,
        }
        return pd.DataFrame([row], columns=canonical_columns(cfg))

    monkeypatch.setattr(suggestions_module, "fit_gp_model", lambda *_args: FakeModel())
    monkeypatch.setattr(suggestions_module, "_suggest_model_based", fake_model_based)

    suggestions = suggest_next(cfg, df, batch_size=2)

    assert len(suggestions) == 2
    assert suggestions.loc[0, "replicate_group"] == "group_1"
    assert int(suggestions.loc[0, "replicate_index"]) == 1
    assert suggestions.loc[1, "replicate_group"] == "new_suggestion"
    assert suggestions["iteration"].astype(int).nunique() == 1
    staged = captured["df"].loc[captured["df"]["replicate_group"] == "group_1"]
    assert sorted(staged["replicate_index"].astype(int).tolist()) == [0, 1]

def test_uncertain_best_batch_fill_avoids_duplicate_repeat_design(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = replicate_config(
        initial_design_size=3,
        replicate_threshold=0.10,
        min_repeats_at_best=2,
    )
    df = replicate_observed_log(cfg)

    class FakePosterior:
        mean = torch.tensor([[0.5], [2.0], [1.0]], dtype=torch.double)
        variance = torch.tensor([[0.01], [0.04], [0.01]], dtype=torch.double)

    class FakeModel:
        def posterior(self, _x):
            return FakePosterior()

    def fake_model_based(**kwargs):
        staged_df = kwargs["df"]
        repeat_design = staged_df.loc[
            staged_df["replicate_group"] == "group_1",
            ["x", "temperature"],
        ].iloc[-1]
        assert float(repeat_design["x"]) == pytest.approx(0.4)
        assert float(repeat_design["temperature"]) == pytest.approx(550.0)
        row = {
            "row_id": "new_suggestion",
            "iteration": 99,
            "status": "suggested",
            "source": "log_ei",
            "replicate_group": "new_suggestion",
            "replicate_index": 0,
            "x": 0.95,
            "temperature": 730.0,
            "activity": "",
            "predicted_mean": 1.5,
            "predicted_std": 0.1,
            "acquisition": 0.2,
        }
        return pd.DataFrame([row], columns=canonical_columns(cfg))

    monkeypatch.setattr(suggestions_module, "fit_gp_model", lambda *_args: FakeModel())
    monkeypatch.setattr(suggestions_module, "_suggest_model_based", fake_model_based)

    suggestions = suggest_next(cfg, df, batch_size=2)

    designs = {
        (float(row["x"]), float(row["temperature"]))
        for _, row in suggestions.iterrows()
    }
    assert designs == {(0.4, 550.0), (0.95, 730.0)}

def test_cost_aware_uncertain_best_repeat_then_cost_exploration_fill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = replicate_config(
        initial_design_size=3,
        replicate_threshold=0.10,
        min_repeats_at_best=2,
    )
    cfg = CampaignConfig(
        campaign_name=base.campaign_name,
        objective=base.objective,
        variables=base.variables,
        bo=base.bo,
        cost=CostConfig(
            expression="1.0 + x",
            weight=0.5,
            budget=10.0,
            candidate_pool_size=16,
            top_k=8,
        ),
        replicates=base.replicates,
    )
    df = replicate_observed_log(cfg)
    captured: dict[str, object] = {}

    class FakePosterior:
        mean = torch.tensor([[0.5], [2.0], [1.0]], dtype=torch.double)
        variance = torch.tensor([[0.01], [0.04], [0.01]], dtype=torch.double)

    class FakeModel:
        def posterior(self, _x):
            return FakePosterior()

    def fake_cost_aware(**kwargs):
        captured["config"] = kwargs["config"]
        captured["df"] = kwargs["df"]
        row = {
            "row_id": "cost_suggestion",
            "iteration": 99,
            "status": "suggested",
            "source": "cost_log_ei",
            "replicate_group": "cost_suggestion",
            "replicate_index": 0,
            "x": 0.9,
            "temperature": 700.0,
            "activity": "",
            "cost_estimate": 1.9,
            "cost_actual": "",
            "predicted_mean": 1.5,
            "predicted_std": 0.1,
            "acquisition": 0.8,
            "utility": -0.15,
        }
        return pd.DataFrame([row], columns=canonical_columns(cfg))

    monkeypatch.setattr(suggestions_module, "fit_gp_model", lambda *_args: FakeModel())
    monkeypatch.setattr(suggestions_module, "_suggest_cost_aware_model_based", fake_cost_aware)

    suggestions = suggest_next(cfg, df, batch_size=2)

    assert len(suggestions) == 2
    assert suggestions.loc[0, "source"] == "log_ei"
    assert str(suggestions.loc[0, "utility"]) == ""
    assert suggestions.loc[1, "source"] == "cost_log_ei"
    assert captured["config"].cost.budget == pytest.approx(8.6)
    staged = captured["df"].loc[captured["df"]["replicate_group"] == "group_1"]
    assert sorted(staged["replicate_index"].astype(int).tolist()) == [0, 1]

def test_repeat_batch_fill_underfills_on_candidate_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = replicate_config(
        initial_design_size=3,
        replicate_threshold=0.10,
        min_repeats_at_best=2,
    )
    cfg = CampaignConfig(
        campaign_name=base.campaign_name,
        objective=base.objective,
        variables=base.variables,
        bo=base.bo,
        cost=CostConfig(
            expression="1.0 + x",
            weight=0.5,
            budget=6.81,
            candidate_pool_size=16,
            top_k=8,
        ),
        replicates=base.replicates,
    )
    df = replicate_observed_log(cfg)

    class FakePosterior:
        mean = torch.tensor([[0.5], [2.0], [1.0]], dtype=torch.double)
        variance = torch.tensor([[0.01], [0.04], [0.01]], dtype=torch.double)

    class FakeModel:
        def posterior(self, _x):
            return FakePosterior()

    def exhausted_cost_aware(**_kwargs):
        raise suggestions_module._CandidateGenerationExhausted(
            "remaining budget exhausted"
        )

    monkeypatch.setattr(suggestions_module, "fit_gp_model", lambda *_args: FakeModel())
    monkeypatch.setattr(
        suggestions_module,
        "_suggest_cost_aware_model_based",
        exhausted_cost_aware,
    )

    suggestions = suggest_next(cfg, df, batch_size=2)

    assert len(suggestions) == 1
    assert suggestions.loc[0, "replicate_group"] == "group_1"
    assert int(suggestions.loc[0, "replicate_index"]) == 1
    assert float(suggestions.loc[0, "cost_estimate"]) == pytest.approx(1.4)

def test_repeat_batch_fill_reraises_unexpected_suggestion_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = replicate_config(
        initial_design_size=3,
        replicate_threshold=0.10,
        min_repeats_at_best=2,
    )
    df = replicate_observed_log(cfg)

    class FakePosterior:
        mean = torch.tensor([[0.5], [2.0], [1.0]], dtype=torch.double)
        variance = torch.tensor([[0.01], [0.04], [0.01]], dtype=torch.double)

    class FakeModel:
        def posterior(self, _x):
            return FakePosterior()

    def unexpected_model_based(**_kwargs):
        raise SuggestionError("unexpected internal issue")

    monkeypatch.setattr(suggestions_module, "fit_gp_model", lambda *_args: FakeModel())
    monkeypatch.setattr(
        suggestions_module,
        "_suggest_model_based",
        unexpected_model_based,
    )

    with pytest.raises(SuggestionError) as exc_info:
        suggest_next(cfg, df, batch_size=2)
    message = str(exc_info.value)
    assert "Repeat suggestions were generated, but exploration fill failed" in message
    assert "unexpected internal issue" in message

def test_replicate_uncertain_best_policy_respects_max_repeat_cutoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = replicate_config(
        initial_design_size=3,
        replicate_threshold=0.10,
        max_repeats_per_group=5,
    )
    df = replicate_observed_log(cfg)
    extra_rows = []
    for replicate_index in range(1, 5):
        row = df.loc[df["replicate_group"] == "group_1"].iloc[0].copy()
        row["row_id"] = f"rep_1_extra_{replicate_index}"
        row["replicate_index"] = replicate_index
        row["activity"] = 1.4 + 0.01 * replicate_index
        extra_rows.append(row)
    df = pd.concat([df, pd.DataFrame(extra_rows)], ignore_index=True)

    class FakePosterior:
        mean = torch.tensor([[0.5], [2.0], [1.0]], dtype=torch.double)
        variance = torch.tensor([[0.01], [0.04], [0.01]], dtype=torch.double)

    class FakeModel:
        def posterior(self, _x):
            return FakePosterior()

    fallback = pd.DataFrame(
        [
            {
                "row_id": "new_suggestion",
                "iteration": 99,
                "status": "suggested",
                "source": "log_ei",
                "replicate_group": "new_suggestion",
                "replicate_index": 0,
                "x": 0.9,
                "temperature": 700.0,
                "activity": "",
                "predicted_mean": 1.5,
                "predicted_std": 0.1,
                "acquisition": 0.2,
            }
        ],
        columns=canonical_columns(cfg),
    )
    monkeypatch.setattr(suggestions_module, "fit_gp_model", lambda *_args: FakeModel())
    monkeypatch.setattr(suggestions_module, "_suggest_model_based", lambda **_kwargs: fallback)

    suggestions = suggest_next(cfg, df)

    assert suggestions.loc[0, "replicate_group"] == "new_suggestion"

def test_replicate_new_only_policy_skips_active_repeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = replicate_config(initial_design_size=3, suggestion_policy="new_only")
    df = replicate_observed_log(cfg)
    fallback = pd.DataFrame(
        [
            {
                "row_id": "new_suggestion",
                "iteration": 99,
                "status": "suggested",
                "source": "log_ei",
                "replicate_group": "new_suggestion",
                "replicate_index": 0,
                "x": 0.9,
                "temperature": 700.0,
                "activity": "",
                "predicted_mean": 1.5,
                "predicted_std": 0.1,
                "acquisition": 0.2,
            }
        ],
        columns=canonical_columns(cfg),
    )
    monkeypatch.setattr(suggestions_module, "_suggest_model_based", lambda **_kwargs: fallback)

    suggestions = suggest_next(cfg, df)

    assert suggestions.loc[0, "replicate_group"] == "new_suggestion"

def test_suggest_next_refuses_pending_suggestions() -> None:
    cfg = config(batch_size=1, initial_design_size=3)
    df = empty_campaign_log(cfg)
    pending = suggest_next(cfg, df)

    with pytest.raises(SuggestionError, match="unresolved status='suggested'"):
        suggest_next(cfg, pending)

def test_suggest_next_returns_model_based_single_suggestion() -> None:
    cfg = config(batch_size=1, initial_design_size=3)
    df = observed_log(cfg)

    suggestions = suggest_next(cfg, df)

    assert len(suggestions) == 1
    assert suggestions.loc[0, "status"] == "suggested"
    assert suggestions.loc[0, "source"] == "log_ei"
    assert float(suggestions.loc[0, "predicted_std"]) >= 0.0
    assert float(suggestions.loc[0, "x"]) >= 0.0
    assert float(suggestions.loc[0, "x"]) <= 1.0

def test_suggest_next_supports_non_default_model_profile_without_mutating() -> None:
    base = config(batch_size=1, initial_design_size=3)
    cfg = CampaignConfig(
        campaign_name=base.campaign_name,
        objective=base.objective,
        variables=base.variables,
        bo=base.bo,
        model=ModelConfig(profile="rough"),
    )
    df = observed_log(cfg)
    before = df.copy(deep=True)

    suggestions = suggest_next(cfg, df, batch_size=1)

    pd.testing.assert_frame_equal(df, before)
    assert len(suggestions) == 1
    assert suggestions.loc[0, "source"] == "log_ei"
    assert math.isfinite(float(suggestions.loc[0, "predicted_mean"]))
    assert float(suggestions.loc[0, "predicted_std"]) >= 0.0

def test_qlog_nei_suggestions_are_non_mutating_and_use_qlog_nei_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = qlog_nei_config()
    df = qlog_nei_log(cfg)
    before = df.copy(deep=True)
    candidate = values_to_unit_cube(cfg, [(0.45, 610.0)])

    def fake_optimizer(**kwargs: object) -> tuple[torch.Tensor, torch.Tensor, str]:
        assert kwargs["x_baseline"].shape[0] == 4
        assert kwargs["x_pending"] is None
        return candidate, torch.tensor(0.25, dtype=torch.double), "qlog_nei"

    monkeypatch.setattr(suggestions_module, "optimize_qlog_nei", fake_optimizer)

    suggestions = suggest_next(cfg, df, batch_size=1)

    pd.testing.assert_frame_equal(df, before)
    assert len(suggestions) == 1
    assert suggestions.loc[0, "source"] == "qlog_nei"
    assert math.isfinite(float(suggestions.loc[0, "predicted_mean"]))
    assert float(suggestions.loc[0, "predicted_std"]) >= 0.0
    assert float(suggestions.loc[0, "acquisition"]) == pytest.approx(0.25)

def test_qlog_nei_passes_accepted_review_suggestions_as_x_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = qlog_nei_config(review=True)
    df = qlog_nei_log(cfg)
    pending = {
        "row_id": "pending_0",
        "iteration": 4,
        "status": "suggested",
        "source": "qlog_nei",
        "review_status": "accepted",
        "review_note": "",
        "x": 0.75,
        "temperature": 700.0,
        "activity": "",
        "predicted_mean": 1.4,
        "predicted_std": 0.2,
        "acquisition": 0.1,
    }
    df = pd.concat([df, pd.DataFrame([pending], columns=canonical_columns(cfg))], ignore_index=True)
    candidate = values_to_unit_cube(cfg, [(0.45, 610.0)])
    captured: dict[str, object] = {}

    def fake_optimizer(**kwargs: object) -> tuple[torch.Tensor, torch.Tensor, str]:
        captured["x_pending"] = kwargs["x_pending"]
        return candidate, torch.tensor(0.25, dtype=torch.double), "qlog_nei"

    monkeypatch.setattr(suggestions_module, "optimize_qlog_nei", fake_optimizer)

    suggestions = suggest_next(cfg, df, batch_size=1)

    x_pending = captured["x_pending"]
    assert isinstance(x_pending, torch.Tensor)
    assert x_pending.shape == (1, 2)
    assert suggestions.loc[0, "source"] == "qlog_nei"

def test_qlog_nei_review_pending_rows_block_suggestions() -> None:
    cfg = qlog_nei_config(review=True)
    df = qlog_nei_log(cfg)
    pending = {
        "row_id": "review_pending",
        "iteration": 4,
        "status": "suggested",
        "source": "qlog_nei",
        "review_status": "pending",
        "review_note": "",
        "x": 0.75,
        "temperature": 700.0,
        "activity": "",
        "predicted_mean": 1.4,
        "predicted_std": 0.2,
        "acquisition": 0.1,
    }
    df = pd.concat([df, pd.DataFrame([pending], columns=canonical_columns(cfg))], ignore_index=True)

    with pytest.raises(SuggestionError, match="review_status='pending'"):
        suggest_next(cfg, df, batch_size=1)

@pytest.mark.parametrize("review_status", ["rejected", "deferred"])
def test_qlog_nei_rejected_and_deferred_review_rows_do_not_enter_x_pending(
    review_status: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = qlog_nei_config(review=True)
    df = qlog_nei_log(cfg)
    ignored = {
        "row_id": f"review_{review_status}",
        "iteration": 4,
        "status": "suggested",
        "source": "qlog_nei",
        "review_status": review_status,
        "review_note": "",
        "x": 0.75,
        "temperature": 700.0,
        "activity": "",
        "predicted_mean": 1.4,
        "predicted_std": 0.2,
        "acquisition": 0.1,
    }
    df = pd.concat([df, pd.DataFrame([ignored], columns=canonical_columns(cfg))], ignore_index=True)
    candidate = values_to_unit_cube(cfg, [(0.45, 610.0)])
    captured: dict[str, object] = {}

    def fake_optimizer(**kwargs: object) -> tuple[torch.Tensor, torch.Tensor, str]:
        captured["x_pending"] = kwargs["x_pending"]
        return candidate, torch.tensor(0.25, dtype=torch.double), "qlog_nei"

    monkeypatch.setattr(suggestions_module, "optimize_qlog_nei", fake_optimizer)

    suggestions = suggest_next(cfg, df, batch_size=1)

    assert captured["x_pending"] is None
    assert suggestions.loc[0, "source"] == "qlog_nei"

def test_qlog_nei_waits_for_pending_initial_design_rows() -> None:
    cfg = qlog_nei_config(review=True, initial_design_size=4)
    df = qlog_nei_log(cfg).iloc[:3].copy()
    pending_initial = {
        "row_id": "initial_pending",
        "iteration": 3,
        "status": "suggested",
        "source": "sobol",
        "review_status": "accepted",
        "review_note": "",
        "x": 0.75,
        "temperature": 700.0,
        "activity": "",
        "predicted_mean": "",
        "predicted_std": "",
        "acquisition": "",
    }
    df = pd.concat(
        [df, pd.DataFrame([pending_initial], columns=canonical_columns(cfg))],
        ignore_index=True,
    )

    with pytest.raises(SuggestionError, match="observe accepted pending initial suggestions"):
        suggest_next(cfg, df, batch_size=1)

def test_qlog_nei_can_fill_remaining_initial_design_with_pending_initial_rows() -> None:
    cfg = qlog_nei_config(review=True, initial_design_size=4)
    df = qlog_nei_log(cfg).iloc[:2].copy()
    pending_initial = {
        "row_id": "initial_pending",
        "iteration": 2,
        "status": "suggested",
        "source": "sobol",
        "review_status": "accepted",
        "review_note": "",
        "x": 0.75,
        "temperature": 700.0,
        "activity": "",
        "predicted_mean": "",
        "predicted_std": "",
        "acquisition": "",
    }
    df = pd.concat(
        [df, pd.DataFrame([pending_initial], columns=canonical_columns(cfg))],
        ignore_index=True,
    )

    suggestions = suggest_next(cfg, df, batch_size=2)

    assert len(suggestions) == 1
    assert suggestions.loc[0, "source"] == "sobol"

def test_qlog_nehvi_suggestions_are_non_mutating_and_use_design_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = qlog_nehvi_config()
    df = qlog_nehvi_log(cfg)
    before = df.copy(deep=True)
    candidate = values_to_unit_cube(cfg, [(72.0, "MeCN")])

    def fake_optimizer(**kwargs: object) -> tuple[torch.Tensor, torch.Tensor, str]:
        assert kwargs["x_baseline"].shape == (4, candidate.shape[1])
        assert kwargs["x_pending"] is None
        assert torch.equal(kwargs["ref_point"], reference_point_to_model_space(cfg))
        return candidate, torch.tensor(0.35, dtype=torch.double), "qlog_nehvi"

    monkeypatch.setattr(suggestions_module, "optimize_qlog_nehvi", fake_optimizer)

    suggestions = suggest_next(cfg, df, batch_size=1)

    pd.testing.assert_frame_equal(df, before)
    assert len(suggestions) == 1
    assert suggestions.loc[0, "source"] == "qlog_nehvi"
    assert suggestions[cfg.objective_names].map(lambda value: value == "").all().all()
    assert math.isfinite(float(suggestions.loc[0, "predicted_mean_yield_score"]))
    assert math.isfinite(float(suggestions.loc[0, "predicted_mean_waste_score"]))
    assert float(suggestions.loc[0, "acquisition"]) == pytest.approx(0.35)

def test_qlog_nehvi_non_review_pending_rows_enter_x_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = qlog_nehvi_config(review=False)
    df = qlog_nehvi_log(cfg)
    pending = qlog_nehvi_pending_row(
        cfg,
        row_id="pending_no_review",
        review_status=None,
        temperature=58.0,
        solvent="Water",
    )
    df = pd.concat([df, pd.DataFrame([pending], columns=canonical_columns(cfg))], ignore_index=True)
    candidate = values_to_unit_cube(cfg, [(72.0, "MeCN")])
    captured: dict[str, object] = {}

    def fake_optimizer(**kwargs: object) -> tuple[torch.Tensor, torch.Tensor, str]:
        captured["x_pending"] = kwargs["x_pending"]
        return candidate, torch.tensor(0.35, dtype=torch.double), "qlog_nehvi"

    monkeypatch.setattr(suggestions_module, "optimize_qlog_nehvi", fake_optimizer)

    suggestions = suggest_next(cfg, df, batch_size=1)

    x_pending = captured["x_pending"]
    assert isinstance(x_pending, torch.Tensor)
    assert x_pending.shape == (1, candidate.shape[1])
    assert suggestions.loc[0, "source"] == "qlog_nehvi"

def test_qlog_nehvi_review_accepted_rows_enter_x_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = qlog_nehvi_config(review=True)
    df = qlog_nehvi_log(cfg)
    pending = qlog_nehvi_pending_row(
        cfg,
        row_id="accepted_pending",
        review_status="accepted",
        temperature=58.0,
        solvent="Water",
    )
    df = pd.concat([df, pd.DataFrame([pending], columns=canonical_columns(cfg))], ignore_index=True)
    candidate = values_to_unit_cube(cfg, [(72.0, "MeCN")])
    captured: dict[str, object] = {}

    def fake_optimizer(**kwargs: object) -> tuple[torch.Tensor, torch.Tensor, str]:
        captured["x_pending"] = kwargs["x_pending"]
        return candidate, torch.tensor(0.35, dtype=torch.double), "qlog_nehvi"

    monkeypatch.setattr(suggestions_module, "optimize_qlog_nehvi", fake_optimizer)

    suggestions = suggest_next(cfg, df, batch_size=1)

    x_pending = captured["x_pending"]
    assert isinstance(x_pending, torch.Tensor)
    assert x_pending.shape == (1, candidate.shape[1])
    assert suggestions.loc[0, "source"] == "qlog_nehvi"

def test_qlog_nehvi_review_pending_rows_block_suggestions() -> None:
    cfg = qlog_nehvi_config(review=True)
    df = qlog_nehvi_log(cfg)
    pending = qlog_nehvi_pending_row(
        cfg,
        row_id="review_pending",
        review_status="pending",
        temperature=58.0,
        solvent="Water",
    )
    df = pd.concat([df, pd.DataFrame([pending], columns=canonical_columns(cfg))], ignore_index=True)

    with pytest.raises(SuggestionError, match="review_status='pending'"):
        suggest_next(cfg, df, batch_size=1)
