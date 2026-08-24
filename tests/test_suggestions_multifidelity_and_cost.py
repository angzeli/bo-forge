"""Multi-fidelity, mixed-space, contextual, and cost-aware suggestion tests."""

from tests._suggestions_support import (
    MAX_DECODE_RETRIES,
    BOConfig,
    CampaignConfig,
    ConstraintConfig,
    FidelityConfig,
    ObjectiveConfig,
    ReviewConfig,
    SuggestionError,
    VariableConfig,
    append_suggestions,
    canonical_columns,
    config,
    constrained_mixed_config,
    contextual_cost_review_config,
    contextual_cost_review_observed_log,
    cost_review_mixed_config,
    cost_review_mixed_observed_log,
    empty_campaign_log,
    evaluate_cost,
    load_campaign_log,
    mark_observed,
    math,
    mixed_config,
    mixed_observed_log,
    multi_fidelity_config,
    multi_fidelity_observed_log,
    observed_log,
    patch_multi_fidelity_test_model,
    pd,
    pytest,
    qlog_nehvi_config,
    qlog_nehvi_log,
    qlog_nehvi_pending_row,
    replace,
    replicate_config,
    replicate_observed_log,
    review_mixed_config,
    suggest_next,
    suggestions_module,
    torch,
    values_to_unit_cube,
    warnings,
)


@pytest.mark.parametrize("review_status", ["rejected", "deferred"])
def test_qlog_nehvi_rejected_and_deferred_rows_do_not_enter_x_pending(
    review_status: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = qlog_nehvi_config(review=True)
    df = qlog_nehvi_log(cfg)
    ignored = qlog_nehvi_pending_row(
        cfg,
        row_id=f"review_{review_status}",
        review_status=review_status,
        temperature=58.0,
        solvent="Water",
    )
    df = pd.concat([df, pd.DataFrame([ignored], columns=canonical_columns(cfg))], ignore_index=True)
    candidate = values_to_unit_cube(cfg, [(72.0, "MeCN")])
    captured: dict[str, object] = {}

    def fake_optimizer(**kwargs: object) -> tuple[torch.Tensor, torch.Tensor, str]:
        captured["x_pending"] = kwargs["x_pending"]
        return candidate, torch.tensor(0.35, dtype=torch.double), "qlog_nehvi"

    monkeypatch.setattr(suggestions_module, "optimize_qlog_nehvi", fake_optimizer)

    suggestions = suggest_next(cfg, df, batch_size=1)

    assert captured["x_pending"] is None
    assert suggestions.loc[0, "source"] == "qlog_nehvi"

def test_qlog_nehvi_waits_for_pending_initial_design_rows() -> None:
    cfg = qlog_nehvi_config(review=True, initial_design_size=4)
    df = qlog_nehvi_log(cfg).iloc[:3].copy()
    pending_initial = qlog_nehvi_pending_row(
        cfg,
        row_id="initial_pending",
        review_status="accepted",
        source="sobol",
        temperature=58.0,
        solvent="Water",
    )
    df = pd.concat(
        [df, pd.DataFrame([pending_initial], columns=canonical_columns(cfg))],
        ignore_index=True,
    )

    with pytest.raises(SuggestionError, match="observe accepted pending initial suggestions"):
        suggest_next(cfg, df, batch_size=1)

def test_qlog_nei_mixed_variables_use_fixed_feature_assignments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = mixed_config(batch_size=1, initial_design_size=3)
    cfg = replace(
        base,
        bo=replace(
            base.bo,
            acquisition="qlog_nei",
            raw_samples=8,
            num_restarts=1,
            mc_samples=8,
        ),
    )
    df = mixed_observed_log(cfg)
    candidate = values_to_unit_cube(cfg, [(0.45, 2, 0.2, "EtOH")])
    captured: dict[str, object] = {}

    def fake_optimizer(**kwargs: object) -> tuple[torch.Tensor, torch.Tensor, str]:
        captured["model_dim"] = kwargs["model_dim"]
        captured["fixed_features_list"] = kwargs["fixed_features_list"]
        return candidate, torch.tensor(0.25, dtype=torch.double), "qlog_nei"

    monkeypatch.setattr(suggestions_module, "optimize_qlog_nei", fake_optimizer)

    suggestions = suggest_next(cfg, df, batch_size=1)

    assert captured["model_dim"] == 5
    fixed_features = captured["fixed_features_list"]
    assert isinstance(fixed_features, list)
    assert len(fixed_features) == 2
    assert suggestions.loc[0, "source"] == "qlog_nei"
    assert suggestions.loc[0, "solvent"] in {"MeCN", "EtOH"}
    assert int(suggestions.loc[0, "repeats"]) in {1, 2, 3}
    assert float(suggestions.loc[0, "dose"]) in {0.1, 0.2, 0.5}

def test_qlog_nei_replicate_new_only_uses_replicate_train_yvar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = replicate_config(initial_design_size=3, suggestion_policy="new_only")
    cfg = replace(
        base,
        bo=replace(
            base.bo,
            acquisition="qlog_nei",
            raw_samples=8,
            num_restarts=1,
            mc_samples=8,
        ),
    )
    df = replicate_observed_log(cfg)
    candidate = values_to_unit_cube(cfg, [(0.55, 600.0)])
    captured: dict[str, object] = {}

    class FakePosterior:
        def __init__(self, x_unit: torch.Tensor) -> None:
            self.mean = torch.full((x_unit.shape[0], 1), 1.25, dtype=torch.double)
            self.variance = torch.full((x_unit.shape[0], 1), 0.04, dtype=torch.double)

    class FakeModel:
        def posterior(self, x_unit: torch.Tensor) -> FakePosterior:
            return FakePosterior(x_unit)

    def fake_fit_gp_model(config: CampaignConfig, observed_df: pd.DataFrame) -> FakeModel:
        training = suggestions_module.dataframe_to_training_tensors(config, observed_df)
        captured["train_yvar_used"] = training.train_yvar is not None
        return FakeModel()

    def fake_optimizer(**kwargs: object) -> tuple[torch.Tensor, torch.Tensor, str]:
        assert kwargs["x_pending"] is None
        return candidate, torch.tensor(0.25, dtype=torch.double), "qlog_nei"

    monkeypatch.setattr(suggestions_module, "fit_gp_model", fake_fit_gp_model)
    monkeypatch.setattr(suggestions_module, "optimize_qlog_nei", fake_optimizer)

    suggestions = suggest_next(cfg, df, batch_size=1)

    assert captured["train_yvar_used"] is True
    assert suggestions.loc[0, "source"] == "qlog_nei"
    assert suggestions.loc[0, "replicate_group"] == suggestions.loc[0, "row_id"]
    assert int(suggestions.loc[0, "replicate_index"]) == 0

def test_multi_fidelity_qmfkg_returns_one_valid_non_mutating_suggestion() -> None:
    cfg = multi_fidelity_config(initial_design_size=3)
    df = multi_fidelity_observed_log(cfg)
    before = df.copy(deep=True)

    suggestions = suggest_next(cfg, df, batch_size=1)

    pd.testing.assert_frame_equal(df, before)
    assert len(suggestions) == 1
    assert suggestions.loc[0, "source"] == "qmf_kg"
    assert suggestions["x"].astype(float).between(0.0, 1.0).all()
    assert suggestions["fidelity"].astype(float).between(0.2, 1.0).all()
    assert suggestions.loc[0, "activity"] == ""
    assert math.isfinite(float(suggestions.loc[0, "predicted_mean"]))
    assert float(suggestions.loc[0, "predicted_std"]) >= 0.0
    assert math.isfinite(float(suggestions.loc[0, "acquisition"]))
    assert list(suggestions.columns) == canonical_columns(cfg)

def test_multi_fidelity_qmfkg_rejects_batch_size_above_four() -> None:
    cfg = multi_fidelity_config(initial_design_size=3)
    df = multi_fidelity_observed_log(cfg)

    with pytest.raises(SuggestionError, match="batch_size from 1 through 4"):
        suggest_next(cfg, df, batch_size=5)

def test_discrete_qmfkg_batch_uses_levels_shared_iteration_and_acquisition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = multi_fidelity_config(initial_design_size=3)
    cfg = replace(
        base,
        bo=replace(base.bo, batch_size=2),
        fidelity=replace(base.fidelity, levels=(0.25, 0.5, 0.75, 1.0)),
    )
    df = multi_fidelity_observed_log(cfg)
    candidates = values_to_unit_cube(cfg, [(0.4, 0.5), (0.8, 1.0)])
    captured: dict[str, object] = {}

    class FakePosterior:
        mean = torch.tensor([[1.1], [1.4]], dtype=torch.double)
        variance = torch.tensor([[0.01], [0.04]], dtype=torch.double)

    class FakeModel:
        def posterior(self, x_unit: torch.Tensor) -> FakePosterior:
            assert x_unit.shape == torch.Size([2, 2])
            return FakePosterior()

    def fake_optimizer(**kwargs: object) -> tuple[torch.Tensor, torch.Tensor, str]:
        captured.update(kwargs)
        return candidates, torch.tensor(0.42, dtype=torch.double), "qmf_kg"

    monkeypatch.setattr(
        suggestions_module,
        "fit_multi_fidelity_gp_model",
        lambda *_args, **_kwargs: FakeModel(),
    )
    monkeypatch.setattr(
        suggestions_module,
        "optimize_posterior_mean_at_target_fidelity",
        lambda **_kwargs: torch.tensor([1.0], dtype=torch.double),
    )
    monkeypatch.setattr(suggestions_module, "optimize_qmf_kg", fake_optimizer)

    suggestions = suggest_next(cfg, df, batch_size=2)

    assert captured["batch_size"] == 2
    assert captured["fixed_features_list"] == [
        {1: pytest.approx(0.0625)},
        {1: pytest.approx(0.375)},
        {1: pytest.approx(0.6875)},
        {1: pytest.approx(1.0)},
    ]
    assert len(suggestions) == 2
    assert suggestions["fidelity"].astype(float).tolist() == pytest.approx([0.5, 1.0])
    assert suggestions["predicted_mean"].astype(float).tolist() == pytest.approx([1.1, 1.4])
    assert suggestions["predicted_std"].astype(float).tolist() == pytest.approx([0.1, 0.2])
    assert suggestions["iteration"].nunique() == 1
    assert suggestions["acquisition"].astype(float).tolist() == pytest.approx([0.42, 0.42])

@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_discrete_qmfkg_real_optimizer_returns_warning_free_batch(batch_size: int) -> None:
    base = multi_fidelity_config(initial_design_size=3)
    cfg = replace(
        base,
        bo=replace(base.bo, batch_size=batch_size, min_normalized_distance=0.0),
        fidelity=replace(
            base.fidelity,
            levels=(0.25, 0.5, 0.75, 1.0),
            optimizer_maxiter=7,
            optimizer_timeout_seconds=30.0,
        ),
    )
    df = multi_fidelity_observed_log(cfg)
    before = df.copy(deep=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        suggestions = suggest_next(cfg, df, batch_size=batch_size)

    pd.testing.assert_frame_equal(df, before)
    assert len(suggestions) == batch_size
    assert set(suggestions["fidelity"].astype(float)).issubset({0.25, 0.5, 0.75, 1.0})
    assert suggestions["iteration"].nunique() == 1
    assert suggestions["acquisition"].nunique() == 1
    assert not any("degrees of freedom is <= 0" in str(item.message) for item in caught)

@pytest.mark.parametrize("batch_size", [2, 4])
def test_continuous_qmfkg_real_optimizer_returns_batch(batch_size: int) -> None:
    base = multi_fidelity_config(initial_design_size=3)
    cfg = replace(
        base,
        bo=replace(base.bo, batch_size=batch_size, min_normalized_distance=0.0),
        fidelity=replace(
            base.fidelity,
            optimizer_maxiter=7,
            optimizer_timeout_seconds=30.0,
        ),
    )
    df = multi_fidelity_observed_log(cfg)
    before = df.copy(deep=True)

    suggestions = suggest_next(cfg, df, batch_size=batch_size)

    pd.testing.assert_frame_equal(df, before)
    assert len(suggestions) == batch_size
    assert suggestions["fidelity"].astype(float).between(0.2, 1.0).all()
    assert suggestions["iteration"].nunique() == 1
    assert suggestions["acquisition"].nunique() == 1

@pytest.mark.parametrize("method", ["sobol", "random"])
def test_discrete_fidelity_initial_design_uses_only_configured_levels(method: str) -> None:
    base = multi_fidelity_config(initial_design_size=4)
    cfg = replace(
        base,
        bo=replace(base.bo, batch_size=4, initial_design_method=method),
        fidelity=replace(base.fidelity, levels=(0.25, 0.5, 0.75, 1.0)),
    )
    empty = empty_campaign_log(cfg)

    first = suggest_next(cfg, empty, batch_size=4)
    second = suggest_next(cfg, empty, batch_size=4)

    pd.testing.assert_frame_equal(
        first.drop(columns="row_id").reset_index(drop=True),
        second.drop(columns="row_id").reset_index(drop=True),
    )
    assert set(first["fidelity"].astype(float)).issubset({0.25, 0.5, 0.75, 1.0})

def test_discrete_fidelity_only_initial_design_reports_exhausted_space() -> None:
    cfg = CampaignConfig(
        campaign_name="fidelity_only",
        objective=ObjectiveConfig(name="activity", direction="maximize"),
        variables=(VariableConfig("fidelity", "continuous", 0.25, 1.0),),
        bo=BOConfig(
            batch_size=1,
            initial_design_size=5,
            acquisition="qmf_kg",
            min_normalized_distance=0.0,
        ),
        fidelity=FidelityConfig(
            variable="fidelity",
            target=1.0,
            levels=(0.25, 0.5, 0.75, 1.0),
        ),
    )
    rows = [
        {
            "row_id": f"level_{index}",
            "iteration": index,
            "status": "observed",
            "source": "manual",
            "fidelity": level,
            "activity": float(index),
            "predicted_mean": "",
            "predicted_std": "",
            "acquisition": "",
        }
        for index, level in enumerate(cfg.fidelity.levels)
    ]
    df = pd.DataFrame(rows, columns=canonical_columns(cfg))

    with pytest.raises(SuggestionError, match="finite design space is exhausted"):
        suggest_next(cfg, df, batch_size=1)

def test_multi_fidelity_qmfkg_wraps_optimizer_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = multi_fidelity_config(initial_design_size=3)
    df = multi_fidelity_observed_log(cfg)

    monkeypatch.setattr(
        suggestions_module,
        "fit_multi_fidelity_gp_model",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        suggestions_module,
        "optimize_posterior_mean_at_target_fidelity",
        lambda **_kwargs: torch.tensor([0.0], dtype=torch.double),
    )

    def fail_qmfkg(*_args: object, **_kwargs: object) -> tuple[torch.Tensor, torch.Tensor, str]:
        raise RuntimeError("optimizer exploded")

    monkeypatch.setattr(suggestions_module, "optimize_qmf_kg", fail_qmfkg)

    with pytest.raises(
        SuggestionError,
        match="Could not generate qMFKG suggestion: optimizer exploded",
    ):
        suggest_next(cfg, df, batch_size=1)

def test_qmfkg_timeout_accepts_candidate_returned_at_shared_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = multi_fidelity_config(initial_design_size=3)
    assert base.fidelity is not None
    cfg = replace(
        base,
        fidelity=replace(base.fidelity, optimizer_timeout_seconds=10.0),
    )
    df = multi_fidelity_observed_log(cfg)
    candidate = values_to_unit_cube(cfg, [(0.4, 0.6)])
    times = iter([100.0, 102.0, 106.0, 110.0])
    captured: dict[str, float | None] = {}
    patch_multi_fidelity_test_model(monkeypatch)
    monkeypatch.setattr(suggestions_module.time, "monotonic", lambda: next(times))

    def target_optimizer(**kwargs: object) -> torch.Tensor:
        captured["target"] = kwargs["timeout_seconds"]  # type: ignore[assignment]
        return torch.tensor([0.0], dtype=torch.double)

    def candidate_optimizer(
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        captured["candidate"] = kwargs["timeout_seconds"]  # type: ignore[assignment]
        return candidate, torch.tensor(0.2), "qmf_kg"

    monkeypatch.setattr(
        suggestions_module,
        "optimize_posterior_mean_at_target_fidelity",
        target_optimizer,
    )
    monkeypatch.setattr(suggestions_module, "optimize_qmf_kg", candidate_optimizer)

    result = suggest_next(cfg, df, batch_size=1)

    assert len(result) == 1
    assert captured["target"] == pytest.approx(8.0)
    assert captured["candidate"] == pytest.approx(4.0)

def test_qmfkg_timeout_rejects_candidate_returned_after_shared_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = multi_fidelity_config(initial_design_size=3)
    assert base.fidelity is not None
    cfg = replace(
        base,
        fidelity=replace(base.fidelity, optimizer_timeout_seconds=10.0),
    )
    df = multi_fidelity_observed_log(cfg)
    before = df.copy(deep=True)
    candidate = values_to_unit_cube(cfg, [(0.4, 0.6)])
    times = iter([100.0, 102.0, 106.0, 111.0])
    patch_multi_fidelity_test_model(monkeypatch)
    monkeypatch.setattr(suggestions_module.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(
        suggestions_module,
        "optimize_posterior_mean_at_target_fidelity",
        lambda **_kwargs: torch.tensor([0.0], dtype=torch.double),
    )
    monkeypatch.setattr(
        suggestions_module,
        "optimize_qmf_kg",
        lambda **_kwargs: (candidate, torch.tensor(0.2), "qmf_kg"),
    )

    with pytest.raises(SuggestionError, match="acquisition optimization timed out"):
        suggest_next(cfg, df, batch_size=1)

    pd.testing.assert_frame_equal(df, before)

def test_qmfkg_model_fit_timeout_is_not_reported_as_acquisition_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = multi_fidelity_config(initial_design_size=3)
    assert base.fidelity is not None
    cfg = replace(
        base,
        fidelity=replace(base.fidelity, optimizer_timeout_seconds=10.0),
    )
    df = multi_fidelity_observed_log(cfg)
    before = df.copy(deep=True)
    monkeypatch.setattr(
        suggestions_module,
        "fit_multi_fidelity_gp_model",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(TimeoutError("fit stalled")),
    )

    with pytest.raises(
        SuggestionError,
        match="Could not generate qMFKG suggestion: fit stalled",
    ) as exc_info:
        suggest_next(cfg, df, batch_size=1)

    assert "acquisition optimization timed out" not in str(exc_info.value)
    pd.testing.assert_frame_equal(df, before)

def test_qmfkg_candidate_optimizer_timeout_is_reported_as_acquisition_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = multi_fidelity_config(initial_design_size=3)
    assert base.fidelity is not None
    cfg = replace(
        base,
        fidelity=replace(base.fidelity, optimizer_timeout_seconds=10.0),
    )
    df = multi_fidelity_observed_log(cfg)
    before = df.copy(deep=True)
    times = iter([100.0, 101.0, 102.0])
    patch_multi_fidelity_test_model(monkeypatch)
    monkeypatch.setattr(suggestions_module.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(
        suggestions_module,
        "optimize_posterior_mean_at_target_fidelity",
        lambda **_kwargs: torch.tensor([0.0], dtype=torch.double),
    )
    monkeypatch.setattr(
        suggestions_module,
        "optimize_qmf_kg",
        lambda **_kwargs: (_ for _ in ()).throw(TimeoutError("optimizer stopped")),
    )

    with pytest.raises(SuggestionError, match="acquisition optimization timed out"):
        suggest_next(cfg, df, batch_size=1)

    pd.testing.assert_frame_equal(df, before)

def test_qmfkg_timeout_between_target_and_candidate_optimization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = multi_fidelity_config(initial_design_size=3)
    assert base.fidelity is not None
    cfg = replace(
        base,
        fidelity=replace(base.fidelity, optimizer_timeout_seconds=5.0),
    )
    df = multi_fidelity_observed_log(cfg)
    before = df.copy(deep=True)
    times = iter([10.0, 11.0, 15.0])
    patch_multi_fidelity_test_model(monkeypatch)
    monkeypatch.setattr(suggestions_module.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(
        suggestions_module,
        "optimize_posterior_mean_at_target_fidelity",
        lambda **_kwargs: torch.tensor([0.0], dtype=torch.double),
    )
    monkeypatch.setattr(
        suggestions_module,
        "optimize_qmf_kg",
        lambda **_kwargs: pytest.fail("candidate optimization must not start"),
    )

    with pytest.raises(SuggestionError, match="acquisition optimization timed out"):
        suggest_next(cfg, df, batch_size=1)

    pd.testing.assert_frame_equal(df, before)

def test_qmfkg_timeout_after_rejected_batch_stops_before_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = multi_fidelity_config(initial_design_size=3)
    assert base.fidelity is not None
    cfg = replace(
        base,
        fidelity=replace(base.fidelity, optimizer_timeout_seconds=5.0),
    )
    df = multi_fidelity_observed_log(cfg)
    duplicate = values_to_unit_cube(cfg, [(0.1, 0.25)])
    times = iter([20.0, 20.5, 21.0, 24.9, 25.1])
    calls = 0
    patch_multi_fidelity_test_model(monkeypatch)
    monkeypatch.setattr(suggestions_module.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(
        suggestions_module,
        "optimize_posterior_mean_at_target_fidelity",
        lambda **_kwargs: torch.tensor([0.0], dtype=torch.double),
    )

    def duplicate_optimizer(
        **_kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        nonlocal calls
        calls += 1
        return duplicate, torch.tensor(0.0), "qmf_kg"

    monkeypatch.setattr(suggestions_module, "optimize_qmf_kg", duplicate_optimizer)

    with pytest.raises(
        SuggestionError,
        match="Last rejection: candidate duplicates an existing design exactly",
    ):
        suggest_next(cfg, df, batch_size=1)

    assert calls == 1

def test_qmfkg_retries_receive_remaining_shared_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = multi_fidelity_config(initial_design_size=3)
    assert base.fidelity is not None
    cfg = replace(
        base,
        fidelity=replace(base.fidelity, optimizer_timeout_seconds=10.0),
    )
    df = multi_fidelity_observed_log(cfg)
    batches = [
        values_to_unit_cube(cfg, [(0.1, 0.25)]),
        values_to_unit_cube(cfg, [(0.4, 0.6)]),
    ]
    times = iter([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    remaining: list[float] = []
    patch_multi_fidelity_test_model(monkeypatch)
    monkeypatch.setattr(suggestions_module.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(
        suggestions_module,
        "optimize_posterior_mean_at_target_fidelity",
        lambda **_kwargs: torch.tensor([0.0], dtype=torch.double),
    )

    def optimizer(**kwargs: object) -> tuple[torch.Tensor, torch.Tensor, str]:
        value = kwargs["timeout_seconds"]
        assert isinstance(value, float)
        remaining.append(value)
        return batches[len(remaining) - 1], torch.tensor(0.2), "qmf_kg"

    monkeypatch.setattr(suggestions_module, "optimize_qmf_kg", optimizer)

    result = suggest_next(cfg, df, batch_size=1)

    assert len(result) == 1
    assert remaining == pytest.approx([8.0, 6.0])

def test_qmfkg_without_timeout_preserves_optimizer_route_without_reading_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = multi_fidelity_config(initial_design_size=3)
    df = multi_fidelity_observed_log(cfg)
    candidate = values_to_unit_cube(cfg, [(0.4, 0.6)])
    captured: list[float | None] = []
    patch_multi_fidelity_test_model(monkeypatch)
    monkeypatch.setattr(
        suggestions_module.time,
        "monotonic",
        lambda: pytest.fail("unset timeout must not read the monotonic clock"),
    )

    def target_optimizer(**kwargs: object) -> torch.Tensor:
        captured.append(kwargs["timeout_seconds"])  # type: ignore[arg-type]
        return torch.tensor([0.0], dtype=torch.double)

    def candidate_optimizer(
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        captured.append(kwargs["timeout_seconds"])  # type: ignore[arg-type]
        return candidate, torch.tensor(0.2), "qmf_kg"

    monkeypatch.setattr(
        suggestions_module,
        "optimize_posterior_mean_at_target_fidelity",
        target_optimizer,
    )
    monkeypatch.setattr(suggestions_module, "optimize_qmf_kg", candidate_optimizer)

    result = suggest_next(cfg, df, batch_size=1)

    assert len(result) == 1
    assert captured == [None, None]

def test_discrete_qmfkg_retries_duplicate_batches_then_reports_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = multi_fidelity_config(initial_design_size=3)
    cfg = replace(
        base,
        bo=replace(base.bo, batch_size=2, min_normalized_distance=0.0),
        fidelity=replace(base.fidelity, levels=(0.25, 0.5, 0.75, 1.0)),
    )
    df = multi_fidelity_observed_log(cfg)
    duplicate_batch = values_to_unit_cube(cfg, [(0.1, 0.25), (0.3, 0.5)])
    calls = 0

    monkeypatch.setattr(
        suggestions_module,
        "fit_multi_fidelity_gp_model",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        suggestions_module,
        "optimize_posterior_mean_at_target_fidelity",
        lambda **_kwargs: torch.tensor([0.0], dtype=torch.double),
    )

    def duplicate_optimizer(**_kwargs: object) -> tuple[torch.Tensor, torch.Tensor, str]:
        nonlocal calls
        calls += 1
        return duplicate_batch, torch.tensor(0.0, dtype=torch.double), "qmf_kg"

    monkeypatch.setattr(suggestions_module, "optimize_qmf_kg", duplicate_optimizer)

    with pytest.raises(SuggestionError, match=f"{MAX_DECODE_RETRIES} retries"):
        suggest_next(cfg, df, batch_size=2)

    assert calls == MAX_DECODE_RETRIES

def test_discrete_qmfkg_review_pending_batch_blocks_new_suggestions() -> None:
    base = multi_fidelity_config(initial_design_size=3)
    cfg = replace(
        base,
        bo=replace(base.bo, batch_size=2),
        fidelity=replace(base.fidelity, levels=(0.25, 0.5, 0.75, 1.0)),
        review=ReviewConfig(enabled=True),
    )
    observed = multi_fidelity_observed_log(cfg)
    observed["review_status"] = "accepted"
    observed["review_note"] = ""
    observed = observed.loc[:, canonical_columns(cfg)]
    pending = {
        "row_id": "pending_qmfkg_batch",
        "iteration": 5,
        "status": "suggested",
        "source": "qmf_kg",
        "review_status": "pending",
        "review_note": "",
        "x": 0.95,
        "fidelity": 0.5,
        "activity": "",
        "predicted_mean": 1.0,
        "predicted_std": 0.2,
        "acquisition": 0.1,
    }
    df = pd.concat(
        [observed, pd.DataFrame([pending], columns=canonical_columns(cfg))],
        ignore_index=True,
    )

    with pytest.raises(SuggestionError, match="unresolved status='suggested'"):
        suggest_next(cfg, df, batch_size=2)

def test_discrete_qmfkg_retries_constraint_violating_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = multi_fidelity_config(initial_design_size=3)
    cfg = replace(
        base,
        bo=replace(base.bo, batch_size=2, min_normalized_distance=0.0),
        fidelity=replace(base.fidelity, levels=(0.25, 0.5, 0.75, 1.0)),
        constraints=(ConstraintConfig("x_limit", "x <= 0.9"),),
    )
    df = multi_fidelity_observed_log(cfg)
    candidate_batches = [
        values_to_unit_cube(cfg, [(0.95, 0.25), (0.7, 0.5)]),
        values_to_unit_cube(cfg, [(0.4, 0.25), (0.7, 0.5)]),
    ]
    calls = 0

    class FakePosterior:
        mean = torch.tensor([[1.0], [1.1]], dtype=torch.double)
        variance = torch.tensor([[0.04], [0.04]], dtype=torch.double)

    class FakeModel:
        def posterior(self, _x_unit: torch.Tensor) -> FakePosterior:
            return FakePosterior()

    monkeypatch.setattr(
        suggestions_module,
        "fit_multi_fidelity_gp_model",
        lambda *_args, **_kwargs: FakeModel(),
    )
    monkeypatch.setattr(
        suggestions_module,
        "optimize_posterior_mean_at_target_fidelity",
        lambda **_kwargs: torch.tensor([0.0], dtype=torch.double),
    )

    def optimizer(**_kwargs: object) -> tuple[torch.Tensor, torch.Tensor, str]:
        nonlocal calls
        candidate = candidate_batches[min(calls, len(candidate_batches) - 1)]
        calls += 1
        return candidate, torch.tensor(0.2, dtype=torch.double), "qmf_kg"

    monkeypatch.setattr(suggestions_module, "optimize_qmf_kg", optimizer)

    result = suggest_next(cfg, df, batch_size=2)

    assert calls == 2
    assert result["x"].astype(float).tolist() == pytest.approx([0.4, 0.7])

def test_suggest_next_returns_model_based_batch_suggestions() -> None:
    cfg = config(batch_size=2, initial_design_size=3)
    df = observed_log(cfg)

    suggestions = suggest_next(cfg, df)

    assert len(suggestions) == 2
    assert set(suggestions["source"]) == {"qlog_ei"}
    assert suggestions["temperature"].astype(float).between(300.0, 800.0).all()

def test_one_by_one_sobol_suggestions_do_not_repeat_after_csv_round_trip(tmp_path) -> None:
    cfg = config(batch_size=1, initial_design_size=4)
    log_path = tmp_path / "campaign.csv"
    df = empty_campaign_log(cfg)
    seen: set[tuple[float, float]] = set()

    for index in range(4):
        suggestions = suggest_next(cfg, df, batch_size=1)
        candidate = (
            float(suggestions.loc[0, "x"]),
            float(suggestions.loc[0, "temperature"]),
        )
        assert candidate not in seen
        seen.add(candidate)

        append_suggestions(log_path, suggestions, config=cfg)
        mark_observed(log_path, str(suggestions.loc[0, "row_id"]), float(index))
        df = load_campaign_log(log_path, cfg)

def test_mixed_sobol_initial_suggestions_are_valid_and_duplicate_free() -> None:
    cfg = mixed_config(batch_size=3, initial_design_size=4)
    df = empty_campaign_log(cfg)

    suggestions = suggest_next(cfg, df)

    assert len(suggestions) == 3
    assert set(suggestions["source"]) == {"sobol"}
    assert suggestions["x"].astype(float).between(0.0, 1.0).all()
    assert set(suggestions["repeats"].astype(int)).issubset({1, 2, 3})
    assert set(suggestions["dose"].astype(float)).issubset({0.1, 0.2, 0.5})
    assert set(suggestions["solvent"]).issubset({"MeCN", "EtOH"})
    assert len(suggestions[["x", "repeats", "dose", "solvent"]].drop_duplicates()) == 3

def test_mixed_random_initial_suggestions_are_seeded() -> None:
    cfg = mixed_config(batch_size=2, initial_design_size=4, initial_design_method="random")
    df = empty_campaign_log(cfg)

    first = suggest_next(cfg, df)
    second = suggest_next(cfg, df)

    pd.testing.assert_series_equal(first["x"], second["x"])
    pd.testing.assert_series_equal(first["repeats"], second["repeats"])
    pd.testing.assert_series_equal(first["dose"], second["dose"])
    pd.testing.assert_series_equal(first["solvent"], second["solvent"])
    assert set(first["source"]) == {"random"}

def test_constrained_mixed_initial_suggestions_are_feasible() -> None:
    cfg = constrained_mixed_config(batch_size=4, initial_design_size=4)
    df = empty_campaign_log(cfg)

    suggestions = suggest_next(cfg, df)

    assert not (
        (suggestions["solvent"] == "EtOH")
        & (suggestions["dose"].astype(float) >= 0.5)
    ).any()

def test_mixed_model_based_single_suggestion() -> None:
    cfg = mixed_config(batch_size=1, initial_design_size=3)
    df = mixed_observed_log(cfg)

    suggestions = suggest_next(cfg, df)

    assert len(suggestions) == 1
    assert suggestions.loc[0, "source"] == "log_ei"
    assert suggestions.loc[0, "solvent"] in {"MeCN", "EtOH"}
    assert int(suggestions.loc[0, "repeats"]) in {1, 2, 3}
    assert float(suggestions.loc[0, "dose"]) in {0.1, 0.2, 0.5}
    assert float(suggestions.loc[0, "predicted_std"]) >= 0.0

def test_mixed_model_based_batch_suggestions() -> None:
    cfg = mixed_config(batch_size=2, initial_design_size=3)
    df = mixed_observed_log(cfg)

    suggestions = suggest_next(cfg, df)

    assert len(suggestions) == 2
    assert set(suggestions["source"]) == {"qlog_ei"}
    assert set(suggestions["dose"].astype(float)).issubset({0.1, 0.2, 0.5})
    assert set(suggestions["repeats"].astype(int)).issubset({1, 2, 3})
    assert set(suggestions["solvent"]).issubset({"MeCN", "EtOH"})
    assert suggestions["solvent"].nunique() == 1

def test_constrained_mixed_model_based_suggestions_are_feasible() -> None:
    cfg = constrained_mixed_config(batch_size=2, initial_design_size=3)
    df = mixed_observed_log(cfg)

    suggestions = suggest_next(cfg, df)

    assert not (
        (suggestions["solvent"] == "EtOH")
        & (suggestions["dose"].astype(float) >= 0.5)
    ).any()

def test_cost_aware_initial_suggestions_fill_cost_but_not_utility() -> None:
    cfg = cost_review_mixed_config(batch_size=2, initial_design_size=4)
    df = empty_campaign_log(cfg)

    suggestions = suggest_next(cfg, df)

    assert len(suggestions) == 2
    assert set(suggestions["source"]) == {"sobol"}
    assert set(suggestions["review_status"]) == {"pending"}
    assert suggestions["cost_estimate"].astype(float).gt(0).all()
    assert suggestions["cost_actual"].astype(str).eq("").all()
    assert suggestions["utility"].astype(str).eq("").all()

def test_contextual_cost_review_initial_suggestion_fills_context_cost_and_review() -> None:
    cfg = contextual_cost_review_config(initial_design_size=4)
    df = empty_campaign_log(cfg)

    suggestions = suggest_next(
        cfg,
        df,
        context_values={"feedstock_acidity": 0.75},
    )

    assert len(suggestions) == 1
    assert suggestions.loc[0, "source"] == "sobol"
    assert suggestions.loc[0, "review_status"] == "pending"
    assert float(suggestions.loc[0, "feedstock_acidity"]) == pytest.approx(0.75)
    candidate = tuple(suggestions.loc[0, cfg.variable_names])
    assert float(suggestions.loc[0, "cost_estimate"]) == pytest.approx(
        evaluate_cost(cfg, candidate)
    )
    assert suggestions.loc[0, "cost_actual"] == ""
    assert suggestions.loc[0, "utility"] == ""

def test_cost_aware_model_suggestions_fill_cost_and_utility() -> None:
    cfg = cost_review_mixed_config(batch_size=1, initial_design_size=3)
    df = cost_review_mixed_observed_log(cfg)

    suggestions = suggest_next(cfg, df)

    assert len(suggestions) == 1
    assert suggestions.loc[0, "source"] == "cost_log_ei"
    assert suggestions.loc[0, "review_status"] == "pending"
    acquisition = float(suggestions.loc[0, "acquisition"])
    cost_estimate = float(suggestions.loc[0, "cost_estimate"])
    utility = float(suggestions.loc[0, "utility"])
    assert utility == pytest.approx(acquisition - cfg.cost.weight * cost_estimate)

def test_contextual_cost_review_model_suggestion_uses_fixed_context_and_cost() -> None:
    cfg = contextual_cost_review_config(initial_design_size=4)
    df = contextual_cost_review_observed_log(cfg)
    before = df.copy(deep=True)

    suggestions = suggest_next(
        cfg,
        df,
        context_values={"feedstock_acidity": 0.5},
    )

    pd.testing.assert_frame_equal(df, before)
    assert len(suggestions) == 1
    assert suggestions.loc[0, "source"] == "cost_log_ei"
    assert suggestions.loc[0, "review_status"] == "pending"
    assert float(suggestions.loc[0, "feedstock_acidity"]) == pytest.approx(0.5)
    candidate = tuple(suggestions.loc[0, cfg.variable_names])
    cost_estimate = float(suggestions.loc[0, "cost_estimate"])
    acquisition = float(suggestions.loc[0, "acquisition"])
    utility = float(suggestions.loc[0, "utility"])
    assert cost_estimate == pytest.approx(evaluate_cost(cfg, candidate))
    assert utility == pytest.approx(acquisition - cfg.cost.weight * cost_estimate)

def test_contextual_cost_aware_model_path_requires_resolved_context_values() -> None:
    cfg = contextual_cost_review_config(initial_design_size=4)
    df = contextual_cost_review_observed_log(cfg)

    with pytest.raises(
        SuggestionError,
        match="Contextual cost-aware suggestions require resolved context values",
    ):
        suggestions_module._suggest_cost_aware_model_based(
            config=cfg,
            df=df,
            observed_df=df,
            batch_size=1,
        )

def test_cost_aware_candidate_pool_falls_back_to_sobol_when_optimizer_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = cost_review_mixed_config(batch_size=1, initial_design_size=3)
    df = cost_review_mixed_observed_log(cfg)

    def failing_optimizer(*args, **kwargs):
        raise RuntimeError("optimizer failed")

    monkeypatch.setattr(suggestions_module, "optimize_log_ei", failing_optimizer)

    suggestions = suggest_next(cfg, df)

    assert len(suggestions) == 1
    assert suggestions.loc[0, "source"] == "cost_log_ei"
    assert suggestions.loc[0, "review_status"] == "pending"
    assert float(suggestions.loc[0, "cost_estimate"]) > 0
    assert suggestions.loc[0, "solvent"] in {"MeCN", "EtOH"}

def test_contextual_cost_duplicate_detection_uses_full_context_tuple(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = contextual_cost_review_config(initial_design_size=4)
    df = contextual_cost_review_observed_log(cfg)
    same_decision_different_context = (0.20, 70, "MeCN", 0.75)

    def candidate_pool(*args, **kwargs):
        return [
            (0.20, 70, "MeCN", 0.25),
            same_decision_different_context,
        ]

    def score_candidate(*, config, model, acquisition, candidate, cost_estimate):
        return {
            "candidate": candidate,
            "cost_estimate": cost_estimate,
            "acquisition": 2.0,
            "utility": 2.0 - config.cost.weight * cost_estimate,
            "predicted_mean": 0.8,
            "predicted_std": 0.1,
        }

    monkeypatch.setattr(suggestions_module, "_cost_aware_candidate_pool", candidate_pool)
    monkeypatch.setattr(suggestions_module, "_score_cost_aware_candidate", score_candidate)

    suggestions = suggest_next(
        cfg,
        df,
        context_values={"feedstock_acidity": 0.75},
    )

    assert tuple(suggestions.loc[0, cfg.variable_names]) == same_decision_different_context

@pytest.mark.parametrize(
    ("review_status", "blocks"),
    [
        ("pending", True),
        ("accepted", True),
        ("rejected", False),
        ("deferred", False),
    ],
)
def test_review_status_controls_suggestion_blocking(
    review_status: str,
    blocks: bool,
) -> None:
    cfg = review_mixed_config(batch_size=1, initial_design_size=2)
    df = pd.DataFrame(
        [
            {
                "row_id": "reviewed_0",
                "iteration": 0,
                "status": "suggested",
                "source": "sobol",
                "review_status": review_status,
                "review_note": "",
                "x": 0.1,
                "repeats": 1,
                "dose": 0.1,
                "solvent": "MeCN",
                "score": "",
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            }
        ],
        columns=canonical_columns(cfg),
    )

    if blocks:
        with pytest.raises(SuggestionError, match="unresolved status='suggested'"):
            suggest_next(cfg, df)
    else:
        suggestions = suggest_next(cfg, df)
        assert len(suggestions) == 1

def test_rejected_suggestions_do_not_block_but_remain_duplicate_protected() -> None:
    cfg = CampaignConfig(
        campaign_name="review_categories",
        objective=ObjectiveConfig(name="score", direction="maximize"),
        variables=(VariableConfig("solvent", "categorical", values=("MeCN", "EtOH")),),
        bo=BOConfig(batch_size=1, initial_design_size=2, random_seed=3),
        review=ReviewConfig(enabled=True),
    )
    df = pd.DataFrame(
        [
            {
                "row_id": "rejected_0",
                "iteration": 0,
                "status": "suggested",
                "source": "sobol",
                "review_status": "rejected",
                "review_note": "not practical",
                "solvent": "MeCN",
                "score": "",
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            }
        ],
        columns=canonical_columns(cfg),
    )

    suggestions = suggest_next(cfg, df)

    assert suggestions.loc[0, "solvent"] == "EtOH"

def test_cost_budget_exhaustion_raises_clear_error() -> None:
    cfg = cost_review_mixed_config(batch_size=1, initial_design_size=3, budget=0.1)
    df = cost_review_mixed_observed_log(cfg)

    with pytest.raises(SuggestionError, match="remaining budget may be too small"):
        suggest_next(cfg, df)

def test_initial_cost_budget_exhaustion_raises_clear_error() -> None:
    cfg = cost_review_mixed_config(batch_size=1, initial_design_size=2, budget=0.0)
    df = empty_campaign_log(cfg)

    with pytest.raises(
        SuggestionError,
        match=r"budget-feasible initial suggestions.*remaining_budget=0",
    ):
        suggest_next(cfg, df)

def test_initial_cost_budget_failure_reports_rejected_candidate_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = cost_review_mixed_config(batch_size=1, initial_design_size=2, budget=0.5)
    df = empty_campaign_log(cfg)
    monkeypatch.setattr(suggestions_module, "MAX_INITIAL_DESIGN_BATCHES", 0)

    with pytest.raises(
        SuggestionError,
        match=(
            r"remaining_budget=0\.5, minimum_rejected_candidate_cost="
            r"[0-9.]+, available_for_next_candidate=0\.5"
        ),
    ):
        suggest_next(cfg, df)
