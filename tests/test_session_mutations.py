"""Campaign session mutation, reload, conflict, and plotting tests."""

from tests._session_support import (
    CampaignConfig,
    CampaignSession,
    LogConflictError,
    LogValidationError,
    Path,
    ReviewConfig,
    append_suggestions,
    canonical_columns,
    config,
    cost_review_config,
    cost_review_log,
    empty_campaign_log,
    mark_observed,
    observed_log,
    pd,
    pytest,
    replicate_config,
    replicate_log,
    session_module,
    write_config,
    write_cost_review_config,
    write_log,
)


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

    def fake_append(
        log_path,
        appended,
        config=None,
        *,
        expected_log_fingerprint=None,
    ):
        captured["log_path"] = log_path
        captured["appended"] = appended
        captured["config"] = config
        captured["expected_log_fingerprint"] = expected_log_fingerprint

    monkeypatch.setattr(session_module, "_append_suggestions", fake_append)
    monkeypatch.setattr(campaign, "reload", lambda: campaign.df)

    campaign.append_suggestions(suggestions)

    assert captured["log_path"] == campaign.log_path
    assert captured["appended"] is suggestions
    assert captured["config"] is cfg
    assert captured["expected_log_fingerprint"] is None

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

def test_long_lived_session_rejects_stale_append_and_recovers_after_reload(
    tmp_path: Path,
) -> None:
    config_path = write_config(tmp_path / "campaign.yaml", initial_design_size=2)
    cfg = config(initial_design_size=2)
    log_path = write_log(tmp_path / "campaign.csv", cfg)
    first = CampaignSession.from_files(config_path, log_path)
    stale = CampaignSession.from_files(config_path, log_path)
    first_suggestion = first.suggest_next(batch_size=1)
    stale_suggestion = stale.suggest_next(batch_size=1).copy()
    stale_suggestion.loc[:, "row_id"] = "stale_session_row"

    first.append_suggestions(first_suggestion)
    before = log_path.read_bytes()
    with pytest.raises(LogConflictError, match="changed after it was loaded"):
        stale.append_suggestions(stale_suggestion)
    assert log_path.read_bytes() == before

    stale.reload()
    stale_suggestion.loc[:, "x"] = 0.123456
    stale.append_suggestions(stale_suggestion)
    assert "stale_session_row" in stale.df["row_id"].tolist()

def test_session_loaded_before_log_creation_detects_external_creation(
    tmp_path: Path,
) -> None:
    config_path = write_config(tmp_path / "campaign.yaml", initial_design_size=2)
    log_path = tmp_path / "campaign.csv"
    campaign = CampaignSession.from_files(config_path, log_path)
    suggestions = campaign.suggest_next(batch_size=1)
    external = suggestions.copy()
    external.loc[:, "row_id"] = "external_row"
    append_suggestions(log_path, external, config=campaign.config)
    before = log_path.read_bytes()

    with pytest.raises(LogConflictError, match="changed after it was loaded"):
        campaign.append_suggestions(suggestions)

    assert log_path.read_bytes() == before

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
