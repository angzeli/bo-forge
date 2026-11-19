"""Candidate filtering, quality, distance, and finite-space tests."""

from tests._suggestions_support import (
    MAX_DECODE_RETRIES,
    BOConfig,
    CampaignConfig,
    ConstraintConfig,
    ObjectiveConfig,
    SuggestionError,
    VariableConfig,
    append_suggestions,
    canonical_columns,
    constrained_mixed_config,
    cost_review_mixed_config,
    cost_review_mixed_observed_log,
    empty_campaign_log,
    load_campaign_log,
    mark_observed,
    mixed_config,
    mixed_observed_log,
    pd,
    pytest,
    qlog_nei_config,
    qlog_nei_log,
    replace,
    suggest_next,
    suggestion_quality_summary,
    suggestions_module,
    torch,
    values_to_unit_cube,
)


def test_cost_penalty_can_choose_cheaper_lower_acquisition_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = cost_review_mixed_config(
        batch_size=1,
        initial_design_size=3,
        budget=50.0,
        weight=1.0,
    )
    df = cost_review_mixed_observed_log(cfg)
    cheap = (0.2, 1, 0.1, "MeCN")
    costly = (0.9, 3, 0.5, "EtOH")

    def candidate_pool(*args, **kwargs):
        return [costly, cheap]

    def score_candidate(*, config, model, acquisition, candidate, cost_estimate):
        acquisition_value = 3.0 if candidate == costly else 2.0
        return {
            "candidate": candidate,
            "cost_estimate": cost_estimate,
            "acquisition": acquisition_value,
            "utility": acquisition_value - config.cost.weight * cost_estimate,
            "predicted_mean": 1.0,
            "predicted_std": 0.1,
        }

    monkeypatch.setattr(suggestions_module, "_cost_aware_candidate_pool", candidate_pool)
    monkeypatch.setattr(suggestions_module, "_score_cost_aware_candidate", score_candidate)

    suggestions = suggest_next(cfg, df)

    assert tuple(suggestions.loc[0, cfg.variable_names]) == cheap
    assert float(suggestions.loc[0, "acquisition"]) < 3.0

def test_suggestion_quality_summary_reports_constraints_duplicates_and_distances() -> None:
    cfg = constrained_mixed_config(
        batch_size=2,
        initial_design_size=3,
        min_normalized_distance=0.05,
    )
    df = mixed_observed_log(cfg)
    suggestions = pd.DataFrame(
        [
            {
                "row_id": "suggested_duplicate",
                "iteration": 4,
                "status": "suggested",
                "source": "log_ei",
                "x": 0.1,
                "repeats": 1,
                "dose": 0.1,
                "solvent": "MeCN",
                "score": "",
                "predicted_mean": 1.1,
                "predicted_std": 0.1,
                "acquisition": 0.01,
            },
            {
                "row_id": "suggested_infeasible",
                "iteration": 4,
                "status": "suggested",
                "source": "log_ei",
                "x": 0.2,
                "repeats": 2,
                "dose": 0.5,
                "solvent": "EtOH",
                "score": "",
                "predicted_mean": 1.2,
                "predicted_std": 0.1,
                "acquisition": 0.02,
            },
        ],
        columns=canonical_columns(cfg),
    )

    summary = suggestion_quality_summary(cfg, df, suggestions)

    assert list(summary.columns) == [
        "row_id",
        "is_feasible",
        "violated_constraints",
        "is_exact_duplicate",
        "duplicate_allowed_by_replicates",
        "nearest_existing_distance",
        "nearest_batch_distance",
        "passes_distance_threshold",
    ]
    assert bool(summary.loc[0, "is_exact_duplicate"])
    assert summary.loc[1, "violated_constraints"] == "no_etoh_high_dose"
    assert not bool(summary.loc[1, "is_feasible"])
    assert summary["nearest_existing_distance"].notna().all()

def test_categorical_combination_threshold_is_enforced() -> None:
    cfg = CampaignConfig(
        campaign_name="many_categories",
        objective=ObjectiveConfig(name="score", direction="maximize"),
        variables=tuple(
            VariableConfig(f"cat_{index}", "categorical", values=("a", "b"))
            for index in range(7)
        ),
        bo=BOConfig(batch_size=1, initial_design_size=1),
    )
    row = {
        "row_id": "obs_0",
        "iteration": 0,
        "status": "observed",
        "source": "manual",
        "score": 1.0,
        "predicted_mean": "",
        "predicted_std": "",
        "acquisition": "",
    }
    for variable in cfg.variables:
        row[variable.name] = "a"
    df = pd.DataFrame([row], columns=canonical_columns(cfg))

    with pytest.raises(SuggestionError, match="at most 64 categorical combinations"):
        suggest_next(cfg, df)

def test_duplicate_decoded_candidates_retry_then_fall_back_to_valid_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = mixed_config(batch_size=1, initial_design_size=3)
    df = mixed_observed_log(cfg)
    duplicate = (0.1, 1, 0.1, "MeCN")
    duplicate_x = values_to_unit_cube(cfg, [duplicate])
    calls = 0

    def duplicate_optimizer(*args, **kwargs):
        nonlocal calls
        calls += 1
        return duplicate_x, torch.tensor(0.0, dtype=torch.double), "log_ei"

    monkeypatch.setattr(suggestions_module, "optimize_log_ei", duplicate_optimizer)

    suggestions = suggest_next(cfg, df)

    assert calls == MAX_DECODE_RETRIES
    assert suggestions["source"].tolist() == ["log_ei"]
    candidate = tuple(suggestions.loc[0, name] for name in cfg.variable_names)
    assert candidate != duplicate
    assert not suggestion_quality_summary(cfg, df, suggestions).loc[0, "is_exact_duplicate"]


def _budget_constrained_config(batch_size: int) -> CampaignConfig:
    return CampaignConfig(
        campaign_name="budget_constrained",
        objective=ObjectiveConfig(name="score", direction="maximize"),
        variables=(
            VariableConfig("a", "continuous", 0.0, 1.0),
            VariableConfig("b", "continuous", 0.0, 1.0),
        ),
        constraints=(ConstraintConfig(name="budget", expression="a + b <= 1.0"),),
        bo=BOConfig(
            batch_size=batch_size,
            initial_design_size=4,
            random_seed=0,
            raw_samples=32,
            num_restarts=2,
            mc_samples=32,
        ),
    )


def _budget_constrained_log(cfg: CampaignConfig) -> pd.DataFrame:
    points = [(0.1, 0.1), (0.2, 0.5), (0.5, 0.2), (0.4, 0.4),
              (0.1, 0.6), (0.6, 0.1), (0.3, 0.3), (0.45, 0.5)]
    rows = [
        {
            "row_id": f"obs_{index}", "iteration": 0, "status": "observed",
            "source": "manual", "a": a, "b": b, "score": a + b,
            "predicted_mean": "", "predicted_std": "", "acquisition": "",
        }
        for index, (a, b) in enumerate(points)
    ]
    return pd.DataFrame(rows, columns=canonical_columns(cfg))


@pytest.mark.parametrize(("batch_size", "source"), [(1, "log_ei"), (3, "qlog_ei")])
def test_constraint_excluding_unconstrained_optimum_still_suggests(
    batch_size: int, source: str,
) -> None:
    # The objective rises toward (1, 1), which the constraint excludes, so the
    # unconstrained optimizer is rejected on every retry and the pool must answer.
    cfg = _budget_constrained_config(batch_size)
    df = _budget_constrained_log(cfg)

    suggestions = suggest_next(cfg, df)

    assert len(suggestions) == batch_size
    assert set(suggestions["source"]) == {source}
    quality = suggestion_quality_summary(cfg, df, suggestions)
    assert quality["is_feasible"].all()
    assert not quality["is_exact_duplicate"].any()
    totals = suggestions["a"].astype(float) + suggestions["b"].astype(float)
    assert totals.max() > 0.8


def test_qlog_nei_falls_back_to_feasible_pool_after_infeasible_optimizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = replace(
        qlog_nei_config(),
        constraints=(ConstraintConfig(name="x_cap", expression="x <= 0.95"),),
    )
    df = qlog_nei_log(cfg)
    infeasible_x = values_to_unit_cube(cfg, [(1.0, 500.0)])
    calls = 0

    def infeasible_optimizer(*args, **kwargs):
        nonlocal calls
        calls += 1
        return infeasible_x, torch.tensor(0.0, dtype=torch.double), "qlog_nei"

    monkeypatch.setattr(suggestions_module, "optimize_qlog_nei", infeasible_optimizer)

    suggestions = suggest_next(cfg, df)

    assert calls == MAX_DECODE_RETRIES
    assert suggestions["source"].tolist() == ["qlog_nei"]
    assert float(suggestions.loc[0, "x"]) <= 0.95

def test_near_duplicate_threshold_failure_has_clear_message() -> None:
    cfg = CampaignConfig(
        campaign_name="too_restrictive",
        objective=ObjectiveConfig(name="activity", direction="maximize"),
        variables=(VariableConfig("x", "continuous", 0.0, 1.0),),
        bo=BOConfig(
            batch_size=1,
            initial_design_size=1,
            random_seed=3,
            raw_samples=16,
            num_restarts=2,
            mc_samples=16,
            min_normalized_distance=1.1,
        ),
    )
    df = pd.DataFrame(
        [
            {
                "row_id": "obs_0",
                "iteration": 0,
                "status": "observed",
                "source": "manual",
                "x": 0.5,
                "activity": 1.0,
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            }
        ],
        columns=canonical_columns(cfg),
    )

    with pytest.raises(SuggestionError, match="constraints may be too restrictive"):
        suggest_next(cfg, df)

def test_mixed_append_round_trip_validates(tmp_path) -> None:
    cfg = mixed_config(batch_size=1, initial_design_size=3)
    log_path = tmp_path / "mixed.csv"
    df = empty_campaign_log(cfg)

    suggestions = suggest_next(cfg, df, batch_size=1)
    append_suggestions(log_path, suggestions)
    mark_observed(log_path, str(suggestions.loc[0, "row_id"]), 1.2)
    reloaded = load_campaign_log(log_path, cfg)

    assert len(reloaded) == 1
    assert reloaded.loc[0, "status"] == "observed"
    assert reloaded.loc[0, "solvent"] in {"MeCN", "EtOH"}

def test_finite_mixed_initial_space_exhaustion_raises() -> None:
    cfg = CampaignConfig(
        campaign_name="tiny",
        objective=ObjectiveConfig(name="score", direction="maximize"),
        variables=(VariableConfig("solvent", "categorical", values=("MeCN",)),),
        bo=BOConfig(batch_size=2, initial_design_size=2, random_seed=3),
    )
    df = empty_campaign_log(cfg)

    with pytest.raises(SuggestionError, match="finite design space is exhausted"):
        suggest_next(cfg, df)
