"""Streamlit campaign creation, staging, display, and export helper tests."""

from tests._streamlit_support import (
    CampaignConfig,
    Path,
    SimpleNamespace,
    append_disabled_reason,
    available_plot_kinds,
    build_campaign_yaml_text,
    campaign_report_text,
    canonical_columns,
    compact_dataframe,
    copy_example_log,
    create_campaign_files,
    dataframe_fingerprint,
    default_new_campaign_paths,
    drop_all_blank_columns,
    empty_campaign_log,
    empty_state_message,
    export_staged_suggestions_csv,
    extract_matplotlib_figure,
    feature_flags,
    file_fingerprint,
    format_dataframe_for_display,
    format_number_for_display,
    humanize_campaign_status,
    humanize_next_action,
    load_campaign_session,
    make_staged_suggestion_bundle,
    observable_row_options,
    observable_rows,
    parse_campaign_config_text,
    parse_categorical_values_text,
    parse_discrete_values_text,
    pd,
    plt,
    pytest,
    resolve_path_input,
    select_display_columns,
    simple_suggestions,
    staged_bundle_invalidation_reason,
    staged_bundle_is_appendable,
    staged_suggestions_from_bundle,
    status_tone,
    streamlit_app,
    streamlit_helpers,
    suggestions_module,
)


def test_resolve_path_input_accepts_nonblank_path() -> None:
    assert resolve_path_input(" configs/a.yaml ", "Config") == Path("configs/a.yaml")

def test_resolve_path_input_rejects_blank_path() -> None:
    with pytest.raises(ValueError, match="Config path is required"):
        resolve_path_input("   ", "Config")

def test_load_campaign_session_from_existing_files(tmp_path: Path) -> None:
    log_path = copy_example_log(tmp_path, "01_simple_2d_maximise_logei_campaign_log.csv")

    campaign = load_campaign_session(
        "configs/01_simple_2d_maximise_logei.yaml",
        log_path,
    )

    assert campaign.config.campaign_name == "photocatalyst_loading"
    assert len(campaign.df) == 2

def test_file_fingerprint_changes_when_content_changes(tmp_path: Path) -> None:
    path = tmp_path / "data.txt"
    path.write_text("first", encoding="utf-8")
    first = file_fingerprint(path)

    path.write_text("second", encoding="utf-8")

    assert file_fingerprint(path) != first

def test_dataframe_fingerprint_is_stable_for_identical_values() -> None:
    df = simple_suggestions()

    assert dataframe_fingerprint(df) == dataframe_fingerprint(df.copy(deep=True))

def test_staged_bundle_rejects_tampered_context_metadata(tmp_path: Path) -> None:
    config_path = tmp_path / "campaign.yaml"
    log_path = tmp_path / "campaign.csv"
    config_path.write_text("config", encoding="utf-8")
    log_path.write_text("log", encoding="utf-8")
    bundle = make_staged_suggestion_bundle(
        simple_suggestions(),
        config_path,
        log_path,
        context_values={"ctx": 0.25},
    )
    bundle["context_values"] = {"ctx": 0.75}

    assert (
        staged_bundle_invalidation_reason(bundle, config_path, log_path)
        == "Context values changed after suggestions were staged."
    )

def test_default_new_campaign_paths_are_derived_from_campaign_name() -> None:
    config_path, log_path = default_new_campaign_paths("My Catalyst Campaign!")

    assert config_path == Path("configs/my_catalyst_campaign.yaml")
    assert log_path == Path("examples/my_catalyst_campaign_campaign_log.csv")

def test_parse_discrete_values_text_is_strict() -> None:
    assert parse_discrete_values_text("0.1, 0.2, 1", "loading") == [0.1, 0.2, 1.0]

    with pytest.raises(ValueError, match="blank value"):
        parse_discrete_values_text("0.1, , 0.3", "loading")
    with pytest.raises(ValueError, match="non-numeric"):
        parse_discrete_values_text("0.1, high", "loading")

def test_parse_categorical_values_text_is_strict() -> None:
    assert parse_categorical_values_text("MeCN, DMF, THF", "solvent") == [
        "MeCN",
        "DMF",
        "THF",
    ]

    with pytest.raises(ValueError, match="blank label"):
        parse_categorical_values_text("MeCN, , THF", "solvent")
    with pytest.raises(ValueError, match="duplicate label"):
        parse_categorical_values_text("MeCN, DMF, MeCN", "solvent")

def test_build_campaign_yaml_text_parses_through_config_validation() -> None:
    text = build_campaign_yaml_text(
        campaign_name="app_created_campaign",
        objective_name="yield",
        objective_direction="maximize",
        variables=[
            {"name": "temperature", "type": "continuous", "lower": 20.0, "upper": 80.0},
            {"name": "solvent", "type": "categorical", "values": ["MeCN", "DMF"]},
        ],
        batch_size=2,
        initial_design_size=6,
        initial_design_method="sobol",
        random_seed=7,
        model={"profile": "rough"},
    )

    config = parse_campaign_config_text(text)

    assert config.campaign_name == "app_created_campaign"
    assert config.objective.name == "yield"
    assert config.variable_names == ["temperature", "solvent"]
    assert config.bo.batch_size == 2
    assert config.model.profile == "rough"

def test_build_campaign_yaml_text_supports_single_objective_qlog_nei() -> None:
    text = build_campaign_yaml_text(
        campaign_name="app_created_qlog_nei_campaign",
        objective_name="activity",
        objective_direction="maximize",
        variables=[
            {"name": "loading", "type": "continuous", "lower": 0.0, "upper": 1.0},
            {"name": "temperature", "type": "continuous", "lower": 40.0, "upper": 140.0},
        ],
        batch_size=1,
        initial_design_size=4,
        initial_design_method="sobol",
        random_seed=7,
        review_enabled=True,
        model={"profile": "smooth"},
        bo_overrides={"acquisition": "qlog_nei"},
    )

    config = parse_campaign_config_text(text)
    empty_log = empty_campaign_log(config)

    assert config.bo.acquisition == "qlog_nei"
    assert config.review.enabled
    assert config.model.profile == "smooth"
    assert list(empty_log.columns) == canonical_columns(config)

def test_build_campaign_yaml_text_supports_advanced_multi_objective_sections() -> None:
    text = build_campaign_yaml_text(
        campaign_name="advanced_app_campaign",
        objective_name="activity",
        objective_direction="maximize",
        objectives=[
            {"name": "yield", "direction": "maximize", "reference_point": 0.2},
            {"name": "waste", "direction": "minimize", "reference_point": 0.9},
        ],
        variables=[{"name": "x", "type": "continuous", "lower": 0.0, "upper": 1.0}],
        batch_size=2,
        initial_design_size=4,
        initial_design_method="sobol",
        random_seed=7,
        review_enabled=True,
        replicates_enabled=True,
        cost={"expression": "1.0 + x", "weight": 0.5, "budget": 10.0},
    )

    config = parse_campaign_config_text(text)

    assert config.is_multi_objective
    assert config.objective_names == ["yield", "waste"]
    assert config.review.enabled
    assert config.replicates.enabled
    assert config.cost is not None

def test_build_campaign_yaml_text_supports_contextual_replicate_settings() -> None:
    text = build_campaign_yaml_text(
        campaign_name="contextual_replicate_app_campaign",
        objective_name="activity",
        objective_direction="maximize",
        variables=[
            {"name": "loading", "type": "continuous", "lower": 0.0, "upper": 1.0},
            {"name": "feedstock", "type": "categorical", "values": ["A", "B"]},
        ],
        batch_size=2,
        initial_design_size=4,
        initial_design_method="sobol",
        random_seed=7,
        context={"variables": ["feedstock"], "default_values": {"feedstock": "A"}},
        replicates={
            "enabled": True,
            "suggestion_policy": "uncertain_best",
            "replicate_threshold": 0.2,
            "min_repeats_at_best": 2,
            "max_repeats_per_group": 4,
            "noise_floor": 1.0e-8,
        },
    )

    config = parse_campaign_config_text(text)
    empty_log = empty_campaign_log(config)

    assert config.context is not None
    assert config.replicates.enabled
    assert config.replicates.suggestion_policy == "uncertain_best"
    assert config.replicates.replicate_threshold == pytest.approx(0.2)
    assert list(empty_log.columns) == canonical_columns(config)

def test_build_campaign_yaml_text_supports_multi_fidelity_qmfkg() -> None:
    text = build_campaign_yaml_text(
        campaign_name="app_created_fidelity_campaign",
        objective_name="activity",
        objective_direction="maximize",
        variables=[
            {"name": "loading", "type": "continuous", "lower": 0.0, "upper": 1.0},
            {"name": "fidelity", "type": "continuous", "lower": 0.2, "upper": 1.0},
        ],
        batch_size=1,
        initial_design_size=4,
        initial_design_method="sobol",
        random_seed=7,
        review_enabled=True,
        fidelity={
            "variable": "fidelity",
            "target": 1.0,
            "fixed_cost": 0.01,
            "fidelity_cost_weight": 1.0,
            "num_fantasies": 8,
            "optimizer_maxiter": 80,
            "optimizer_timeout_seconds": 15.0,
        },
        bo_overrides={
            "acquisition": "qmf_kg",
            "batch_size": 1,
            "raw_samples": 8,
            "num_restarts": 1,
            "mc_samples": 16,
            "min_normalized_distance": 0.0,
        },
    )

    config = parse_campaign_config_text(text)
    empty_log = empty_campaign_log(config)

    assert config.fidelity is not None
    assert config.fidelity.variable == "fidelity"
    assert config.fidelity.target == pytest.approx(1.0)
    assert config.fidelity.optimizer_maxiter == 80
    assert config.fidelity.optimizer_timeout_seconds == pytest.approx(15.0)
    assert config.bo.acquisition == "qmf_kg"
    assert config.bo.batch_size == 1
    assert config.bo.raw_samples == 8
    assert config.bo.num_restarts == 1
    assert config.bo.mc_samples == 16
    assert config.bo.min_normalized_distance == pytest.approx(0.0)
    assert config.review.enabled
    assert list(empty_log.columns) == canonical_columns(config)

def test_build_campaign_yaml_text_supports_discrete_batch_qmfkg() -> None:
    text = build_campaign_yaml_text(
        campaign_name="app_created_discrete_fidelity",
        objective_name="activity",
        objective_direction="maximize",
        variables=[
            {"name": "loading", "type": "continuous", "lower": 0.0, "upper": 1.0},
            {"name": "fidelity", "type": "continuous", "lower": 0.2, "upper": 1.0},
        ],
        batch_size=3,
        initial_design_size=6,
        initial_design_method="sobol",
        random_seed=22,
        fidelity={
            "variable": "fidelity",
            "target": 1.0,
            "levels": [0.25, 0.5, 0.75, 1.0],
            "fixed_cost": 0.01,
            "fidelity_cost_weight": 1.0,
            "num_fantasies": 8,
        },
        bo_overrides={
            "acquisition": "qmf_kg",
            "raw_samples": 8,
            "num_restarts": 1,
            "mc_samples": 16,
            "min_normalized_distance": 0.0,
        },
    )

    config = parse_campaign_config_text(text)
    empty_log = empty_campaign_log(config)

    assert config.fidelity is not None
    assert config.fidelity.levels == (0.25, 0.5, 0.75, 1.0)
    assert config.fidelity.target == pytest.approx(1.0)
    assert config.fidelity.optimizer_maxiter == 200
    assert config.fidelity.optimizer_timeout_seconds is None
    assert config.bo.batch_size == 3
    assert list(empty_log.columns) == canonical_columns(config)

def test_suggest_form_preserves_non_fidelity_configured_batch_size() -> None:
    captured: dict[str, object] = {}

    class Form:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_args: object) -> None:
            return None

    class FakeStreamlit:
        @staticmethod
        def form(_key: str) -> Form:
            return Form()

        @staticmethod
        def number_input(_label: str, **kwargs: object) -> object:
            captured.update(kwargs)
            return kwargs["value"]

        @staticmethod
        def form_submit_button(*_args: object, **_kwargs: object) -> bool:
            return False

    campaign = SimpleNamespace(
        config=SimpleNamespace(
            fidelity=None,
            bo=SimpleNamespace(batch_size=8),
        )
    )

    batch_size, clicked = streamlit_app._render_suggestion_dry_run_form(
        FakeStreamlit,
        campaign,
    )

    assert batch_size == 8
    assert captured["max_value"] == 32
    assert captured["value"] == 8
    assert not clicked

def test_failed_qmfkg_dry_run_keeps_existing_staged_state() -> None:
    existing_bundle = {"suggestions_fingerprint": "existing"}

    class FakeStreamlit:
        session_state = {
            streamlit_app.STAGED_SUGGESTION_BUNDLE_KEY: existing_bundle,
        }
        errors: list[str] = []

        @classmethod
        def error(cls, message: str) -> None:
            cls.errors.append(message)

    class FailingCampaign:
        @staticmethod
        def suggest_dry_run(*_args: object, **_kwargs: object) -> None:
            raise suggestions_module.SuggestionError(
                "qMFKG acquisition optimization timed out"
            )

    request_state = streamlit_app._SuggestionRequestState(
        config_path=Path("config.yaml"),
        log_path=Path("campaign.csv"),
        selected_stage=None,
        context_values=None,
    )

    streamlit_app._generate_staged_suggestions(
        FakeStreamlit,
        FailingCampaign(),
        request_state,
        1,
    )

    assert FakeStreamlit.errors == ["qMFKG acquisition optimization timed out"]
    assert (
        FakeStreamlit.session_state[streamlit_app.STAGED_SUGGESTION_BUNDLE_KEY]
        is existing_bundle
    )

def test_build_campaign_yaml_text_supports_contextual_logei() -> None:
    text = build_campaign_yaml_text(
        campaign_name="app_created_contextual_campaign",
        objective_name="activity",
        objective_direction="maximize",
        variables=[
            {"name": "loading", "type": "continuous", "lower": 0.0, "upper": 1.0},
            {"name": "feedstock_acidity", "type": "continuous", "lower": 0.0, "upper": 1.0},
            {"name": "solvent", "type": "categorical", "values": ["MeCN", "EtOH"]},
        ],
        batch_size=1,
        initial_design_size=4,
        initial_design_method="sobol",
        random_seed=7,
        context={
            "variables": ["feedstock_acidity", "solvent"],
            "default_values": {"feedstock_acidity": 0.25, "solvent": "MeCN"},
        },
    )

    config = parse_campaign_config_text(text)
    empty_log = empty_campaign_log(config)

    assert config.context is not None
    assert config.context_variable_names == ["feedstock_acidity", "solvent"]
    assert config.context.default_values == {
        "feedstock_acidity": 0.25,
        "solvent": "MeCN",
    }
    assert config.bo.acquisition == "log_ei"
    assert list(empty_log.columns) == canonical_columns(config)

def test_build_campaign_yaml_text_supports_contextual_review_cost_logei() -> None:
    text = build_campaign_yaml_text(
        campaign_name="app_created_contextual_cost_campaign",
        objective_name="activity",
        objective_direction="maximize",
        variables=[
            {"name": "loading", "type": "continuous", "lower": 0.0, "upper": 1.0},
            {"name": "feedstock_acidity", "type": "continuous", "lower": 0.0, "upper": 1.0},
        ],
        batch_size=1,
        initial_design_size=4,
        initial_design_method="sobol",
        random_seed=7,
        review_enabled=True,
        cost={"expression": "1.0 + feedstock_acidity", "weight": 0.5, "budget": 25.0},
        context={
            "variables": ["feedstock_acidity"],
            "default_values": {"feedstock_acidity": 0.25},
        },
    )

    config = parse_campaign_config_text(text)
    empty_log = empty_campaign_log(config)

    assert config.context is not None
    assert config.context.default_values == {"feedstock_acidity": 0.25}
    assert config.review.enabled
    assert config.cost is not None
    assert config.cost.expression == "1.0 + feedstock_acidity"
    assert config.bo.acquisition == "log_ei"
    assert list(empty_log.columns) == canonical_columns(config)

def test_format_dataframe_for_display_stringifies_mixed_type_columns() -> None:
    df = pd.DataFrame({"field": ["a", "b"], "value": ["text", 3]})

    display_df = format_dataframe_for_display(df)

    assert display_df["value"].tolist() == ["text", "3"]

def test_display_status_helpers_are_stable() -> None:
    assert humanize_campaign_status("has_pending_suggestions") == "Pending suggestions"
    assert humanize_campaign_status("ready_for_initial_design") == "Ready for initial design"
    assert humanize_campaign_status("ready_for_bo") == "Ready for BO"
    assert humanize_next_action("resolve_pending_suggestions") == "Resolve pending suggestions"
    assert humanize_next_action("suggest_bo") == "Suggest BO candidates"
    assert status_tone("has_pending_suggestions") == "warning"
    assert status_tone("ready_for_bo") == "success"

def test_number_and_identifier_display_helpers_are_stable() -> None:
    assert format_number_for_display(10.0) == 10
    assert format_number_for_display(0.167123456) == 0.1671
    assert format_number_for_display(float("nan")) == ""
    assert (
        streamlit_helpers.shorten_identifier("8a69540f7cb847e9b2a4acb56b3a67ed")
        == "8a69540f...3a67ed"
    )

def test_dataframe_display_helpers_compact_without_mutating_source() -> None:
    df = pd.DataFrame(
        {
            "row_id": ["8a69540f7cb847e9b2a4acb56b3a67ed"],
            "blank": [""],
            "activity": [10.0],
            "precursor_ratio": [0.167123456],
        }
    )
    before = df.copy(deep=True)

    without_blank = drop_all_blank_columns(df)
    selected = select_display_columns(without_blank, ["activity", "row_id"])
    compact = compact_dataframe(df)

    pd.testing.assert_frame_equal(df, before)
    assert "blank" not in without_blank.columns
    assert selected.columns[:2].tolist() == ["activity", "row_id"]
    assert compact["row_id"].iloc[0] == "8a69540f...b3a67ed"
    assert compact["activity"].iloc[0] == 10
    assert compact["precursor_ratio"].iloc[0] == 0.1671

def test_compact_replicate_summary_keeps_multi_objective_sem_columns() -> None:
    summary = pd.DataFrame(
        [
            {
                "replicate_group": "group_0",
                "n_replicates": 2,
                "yield_score_mean": 0.7,
                "yield_score_std": 0.1,
                "yield_score_sem": 0.07,
                "yield_score_min": 0.6,
                "yield_score_max": 0.8,
                "waste_score_mean": 0.3,
                "waste_score_std": 0.1,
                "waste_score_sem": 0.07,
                "waste_score_min": 0.2,
                "waste_score_max": 0.4,
            }
        ]
    )

    compact = streamlit_app._compact_replicate_summary(summary)

    assert "yield_score_sem" in compact.columns
    assert "waste_score_sem" in compact.columns

def test_compact_context_summary_keeps_documented_columns() -> None:
    summary = pd.DataFrame(
        [
            {
                "context_key": "acid=0.2|solvent=MeCN",
                "acid": "0.2",
                "solvent": "MeCN",
                "observed_rows": 2,
                "pending_suggestions": 1,
                "best_row_id": "obs_1",
                "best_objective": 0.8,
            }
        ]
    )

    compact = streamlit_app._compact_context_summary(summary)

    assert compact.columns.tolist() == [
        "context_key",
        "acid",
        "solvent",
        "observed_rows",
        "pending_suggestions",
        "best_row_id",
        "best_objective",
    ]

def test_empty_state_messages_are_defined() -> None:
    title, detail = empty_state_message("staged_suggestions")

    assert title == "No staged suggestions yet."
    assert "dry-run" in detail

def test_create_campaign_files_writes_config_and_empty_log(tmp_path: Path) -> None:
    config_text = build_campaign_yaml_text(
        campaign_name="new_app_campaign",
        objective_name="activity",
        objective_direction="maximize",
        variables=[
            {"name": "x", "type": "continuous", "lower": 0.0, "upper": 1.0},
        ],
        batch_size=1,
        initial_design_size=4,
        initial_design_method="sobol",
        random_seed=0,
    )
    config_path = tmp_path / "configs" / "campaign.yaml"
    log_path = tmp_path / "logs" / "campaign.csv"

    campaign = create_campaign_files(
        config_text=config_text,
        config_path=config_path,
        log_path=log_path,
    )

    assert campaign.config.campaign_name == "new_app_campaign"
    assert config_path.read_text(encoding="utf-8") == config_text
    df = pd.read_csv(log_path, keep_default_na=False)
    assert df.empty
    assert list(df.columns) == canonical_columns(campaign.config)
    assert log_path.with_name("campaign.csv.manifest.json").exists()

def test_create_campaign_files_validates_before_writing(tmp_path: Path) -> None:
    config_path = tmp_path / "nested" / "campaign.yaml"
    log_path = tmp_path / "logs" / "campaign.csv"

    with pytest.raises(Exception, match="objective"):
        create_campaign_files(
            config_text="campaign_name: invalid\n",
            config_path=config_path,
            log_path=log_path,
        )

    assert not config_path.exists()
    assert not log_path.exists()
    assert not config_path.parent.exists()

def test_create_campaign_files_refuses_overwrite(tmp_path: Path) -> None:
    config_path = tmp_path / "campaign.yaml"
    log_path = tmp_path / "campaign.csv"
    config_path.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Config file already exists"):
        create_campaign_files(
            config_text=build_campaign_yaml_text(
                campaign_name="campaign",
                objective_name="activity",
                objective_direction="maximize",
                variables=[{"name": "x", "type": "continuous", "lower": 0.0, "upper": 1.0}],
                batch_size=1,
                initial_design_size=4,
                initial_design_method="sobol",
                random_seed=0,
            ),
            config_path=config_path,
            log_path=log_path,
        )

    assert config_path.read_text(encoding="utf-8") == "existing"
    assert not log_path.exists()

def test_create_campaign_files_rolls_back_config_if_log_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "campaign.yaml"
    log_path = tmp_path / "campaign.csv"
    config_text = build_campaign_yaml_text(
        campaign_name="campaign",
        objective_name="activity",
        objective_direction="maximize",
        variables=[{"name": "x", "type": "continuous", "lower": 0.0, "upper": 1.0}],
        batch_size=1,
        initial_design_size=4,
        initial_design_method="sobol",
        random_seed=0,
    )

    def fail_log_write(config_path: Path, log_path: Path) -> None:
        raise OSError("no log write")

    monkeypatch.setattr(
        streamlit_helpers.CampaignSession,
        "initialize",
        staticmethod(fail_log_write),
    )

    with pytest.raises(OSError, match="no log write"):
        create_campaign_files(
            config_text=config_text,
            config_path=config_path,
            log_path=log_path,
        )

    assert not config_path.exists()
    assert not log_path.exists()
    assert not log_path.with_name("campaign.csv.manifest.json").exists()

def test_make_staged_suggestion_bundle_records_context(tmp_path: Path) -> None:
    config_path = tmp_path / "campaign.yaml"
    log_path = tmp_path / "campaign.csv"
    config_path.write_text("campaign", encoding="utf-8")
    log_path.write_text("log", encoding="utf-8")

    bundle = make_staged_suggestion_bundle(simple_suggestions(), config_path, log_path)

    assert isinstance(bundle["suggestions"], pd.DataFrame)
    assert bundle["config_path"] == str(config_path.resolve())
    assert bundle["log_path"] == str(log_path.resolve())
    assert bundle["appended"] is False

def test_stage_aware_staged_bundle_records_and_validates_stage(tmp_path: Path) -> None:
    config_path = tmp_path / "campaign.yaml"
    log_path = tmp_path / "campaign.csv"
    config_path.write_text("campaign", encoding="utf-8")
    log_path.write_text("log", encoding="utf-8")

    bundle = make_staged_suggestion_bundle(
        simple_suggestions(),
        config_path,
        log_path,
        stage="screen",
    )

    assert bundle["stage"] == "screen"
    assert staged_bundle_invalidation_reason(
        bundle,
        config_path,
        log_path,
        stage="screen",
    ) is None
    assert staged_bundle_invalidation_reason(
        bundle,
        config_path,
        log_path,
    ) == "Stage selection changed after suggestions were staged."
    assert append_disabled_reason(
        bundle,
        config_path,
        log_path,
    ) == "Append disabled: the selected stage changed after these suggestions were generated."
    assert staged_bundle_invalidation_reason(
        bundle,
        config_path,
        log_path,
        stage="refine",
    ) == "Stage selection changed after suggestions were staged."
    assert append_disabled_reason(
        bundle,
        config_path,
        log_path,
        stage="refine",
    ) == "Append disabled: the selected stage changed after these suggestions were generated."
    assert not staged_bundle_is_appendable(
        bundle,
        config_path,
        log_path,
        stage="refine",
    )
    assert streamlit_app._should_clear_staged_bundle(
        "Stage selection changed after suggestions were staged."
    )

def test_staged_bundle_is_appendable_for_matching_context(tmp_path: Path) -> None:
    config_path = tmp_path / "campaign.yaml"
    log_path = tmp_path / "campaign.csv"
    config_path.write_text("campaign", encoding="utf-8")
    log_path.write_text("log", encoding="utf-8")
    bundle = make_staged_suggestion_bundle(simple_suggestions(), config_path, log_path)

    assert staged_bundle_is_appendable(bundle, config_path, log_path)
    assert staged_bundle_invalidation_reason(bundle, config_path, log_path) is None

def test_staged_bundle_invalidates_for_changed_config_path(tmp_path: Path) -> None:
    config_path = tmp_path / "campaign.yaml"
    other_config_path = tmp_path / "other.yaml"
    log_path = tmp_path / "campaign.csv"
    config_path.write_text("campaign", encoding="utf-8")
    other_config_path.write_text("campaign", encoding="utf-8")
    log_path.write_text("log", encoding="utf-8")
    bundle = make_staged_suggestion_bundle(simple_suggestions(), config_path, log_path)

    reason = staged_bundle_invalidation_reason(bundle, other_config_path, log_path)

    assert reason == "Config path changed after suggestions were staged."

def test_staged_bundle_invalidates_for_changed_config_content(tmp_path: Path) -> None:
    config_path = tmp_path / "campaign.yaml"
    log_path = tmp_path / "campaign.csv"
    config_path.write_text("campaign", encoding="utf-8")
    log_path.write_text("log", encoding="utf-8")
    bundle = make_staged_suggestion_bundle(simple_suggestions(), config_path, log_path)

    config_path.write_text("updated", encoding="utf-8")

    reason = staged_bundle_invalidation_reason(bundle, config_path, log_path)
    assert reason == "Config file changed after suggestions were staged."

def test_staged_bundle_invalidates_for_changed_log_path(tmp_path: Path) -> None:
    config_path = tmp_path / "campaign.yaml"
    log_path = tmp_path / "campaign.csv"
    other_log_path = tmp_path / "other.csv"
    config_path.write_text("campaign", encoding="utf-8")
    log_path.write_text("log", encoding="utf-8")
    other_log_path.write_text("log", encoding="utf-8")
    bundle = make_staged_suggestion_bundle(simple_suggestions(), config_path, log_path)

    reason = staged_bundle_invalidation_reason(bundle, config_path, other_log_path)

    assert reason == "Log path changed after suggestions were staged."

def test_staged_bundle_invalidates_for_changed_log_content(tmp_path: Path) -> None:
    config_path = tmp_path / "campaign.yaml"
    log_path = tmp_path / "campaign.csv"
    config_path.write_text("campaign", encoding="utf-8")
    log_path.write_text("log", encoding="utf-8")
    bundle = make_staged_suggestion_bundle(simple_suggestions(), config_path, log_path)

    log_path.write_text("updated", encoding="utf-8")

    reason = staged_bundle_invalidation_reason(bundle, config_path, log_path)
    assert reason == "Log file changed after suggestions were staged."

def test_staged_bundle_invalidates_for_already_appended_fingerprint(tmp_path: Path) -> None:
    config_path = tmp_path / "campaign.yaml"
    log_path = tmp_path / "campaign.csv"
    config_path.write_text("campaign", encoding="utf-8")
    log_path.write_text("log", encoding="utf-8")
    bundle = make_staged_suggestion_bundle(simple_suggestions(), config_path, log_path)

    reason = staged_bundle_invalidation_reason(
        bundle,
        config_path,
        log_path,
        last_appended_fingerprint=str(bundle["suggestions_fingerprint"]),
    )

    assert reason == "Staged suggestions were already appended."

def test_staged_bundle_invalidates_for_mutated_payload(tmp_path: Path) -> None:
    config_path = tmp_path / "campaign.yaml"
    log_path = tmp_path / "campaign.csv"
    config_path.write_text("campaign", encoding="utf-8")
    log_path.write_text("log", encoding="utf-8")
    bundle = make_staged_suggestion_bundle(simple_suggestions(), config_path, log_path)
    suggestions = bundle["suggestions"]
    assert isinstance(suggestions, pd.DataFrame)

    suggestions.loc[0, "row_id"] = "tampered"

    reason = staged_bundle_invalidation_reason(bundle, config_path, log_path)
    assert reason == "Staged suggestions changed after they were staged."
    assert append_disabled_reason(bundle, config_path, log_path) == (
        "Append disabled: the staged suggestion payload changed after staging."
    )

def test_tampered_staged_bundle_reason_clears_app_bundle() -> None:
    assert streamlit_app._should_clear_staged_bundle(
        "Staged suggestions changed after they were staged."
    )

def test_append_disabled_reason_maps_to_user_facing_text(tmp_path: Path) -> None:
    config_path = tmp_path / "campaign.yaml"
    log_path = tmp_path / "campaign.csv"
    config_path.write_text("campaign", encoding="utf-8")
    log_path.write_text("log", encoding="utf-8")
    bundle = make_staged_suggestion_bundle(simple_suggestions(), config_path, log_path)

    assert append_disabled_reason(bundle, config_path, log_path) is None
    assert append_disabled_reason(None, config_path, log_path) == (
        "Append disabled: no staged suggestions."
    )
    assert append_disabled_reason(
        bundle,
        config_path,
        log_path,
        last_appended_fingerprint=str(bundle["suggestions_fingerprint"]),
    ) == "Append disabled: this staged batch has already been appended."

def test_staged_bundle_rejects_missing_or_empty_suggestions(tmp_path: Path) -> None:
    config_path = tmp_path / "campaign.yaml"
    log_path = tmp_path / "campaign.csv"
    config_path.write_text("campaign", encoding="utf-8")
    log_path.write_text("log", encoding="utf-8")
    empty_bundle = make_staged_suggestion_bundle(pd.DataFrame(), config_path, log_path)

    assert not staged_bundle_is_appendable(None, config_path, log_path)
    assert not staged_bundle_is_appendable(empty_bundle, config_path, log_path)

def test_observable_rows_returns_all_suggested_rows_without_review() -> None:
    config = CampaignConfig.from_yaml("configs/01_simple_2d_maximise_logei.yaml")
    df = pd.DataFrame(
        [
            {"row_id": "observed_0", "status": "observed"},
            {"row_id": "suggested_0", "status": "suggested"},
            {"row_id": "suggested_1", "status": "suggested"},
        ]
    )

    rows = observable_rows(config, df)

    assert rows["row_id"].tolist() == ["suggested_0", "suggested_1"]

def test_observable_rows_returns_only_accepted_review_suggestions() -> None:
    config = CampaignConfig.from_yaml("configs/07_cost_aware_human_review_logei.yaml")
    df = pd.DataFrame(
        [
            {"row_id": "pending_0", "status": "suggested", "review_status": "pending"},
            {"row_id": "accepted_0", "status": "suggested", "review_status": "accepted"},
            {"row_id": "rejected_0", "status": "suggested", "review_status": "rejected"},
            {"row_id": "deferred_0", "status": "suggested", "review_status": "deferred"},
            {"row_id": "observed_0", "status": "observed", "review_status": "accepted"},
        ]
    )

    rows = observable_rows(config, df)

    assert rows["row_id"].tolist() == ["accepted_0"]

def test_observable_row_options_use_short_ids_and_design_values() -> None:
    config = CampaignConfig.from_yaml("configs/01_simple_2d_maximise_logei.yaml")
    row_id = "8a69540f7cb847e9b2a4acb56b3a67ed"
    df = pd.DataFrame(
        [
            {
                "row_id": row_id,
                "status": "suggested",
                "precursor_ratio": 0.167123456,
                "annealing_temperature": 540.0,
            }
        ]
    )

    options = observable_row_options(config, df)

    assert list(options.values()) == [row_id]
    assert "8a69540f...3a67ed" in next(iter(options))
    assert "precursor_ratio=0.1671" in next(iter(options))

def test_export_staged_suggestions_csv_is_non_mutating(tmp_path: Path) -> None:
    log_path = copy_example_log(tmp_path, "01_simple_2d_maximise_logei_campaign_log.csv")
    campaign = load_campaign_session(
        "configs/01_simple_2d_maximise_logei.yaml",
        log_path,
    )
    config_path = Path("configs/01_simple_2d_maximise_logei.yaml")
    suggestions = simple_suggestions()
    bundle = make_staged_suggestion_bundle(suggestions, config_path, log_path)
    before_bundle_fingerprint = str(bundle["suggestions_fingerprint"])
    before_log_bytes = log_path.read_bytes()
    before_df = campaign.df.copy(deep=True)

    output_path = export_staged_suggestions_csv(
        staged_suggestions_from_bundle(bundle),
        tmp_path / "exports" / "suggestions.csv",
    )

    assert output_path.exists()
    pd.testing.assert_frame_equal(pd.read_csv(output_path, keep_default_na=False), suggestions)
    assert bundle["suggestions_fingerprint"] == before_bundle_fingerprint
    assert bundle["appended"] is False
    assert log_path.read_bytes() == before_log_bytes
    pd.testing.assert_frame_equal(campaign.df, before_df)

def test_campaign_report_text_uses_session_report_formatting(tmp_path: Path) -> None:
    log_path = copy_example_log(tmp_path, "01_simple_2d_maximise_logei_campaign_log.csv")
    campaign = load_campaign_session(
        "configs/01_simple_2d_maximise_logei.yaml",
        log_path,
    )

    text = campaign_report_text(campaign)

    assert "BO Forge Campaign Report" in text
    assert "Summary" in text
    assert "Next Action" in text

def test_report_and_plot_exports_do_not_mutate_campaign_log_or_session(tmp_path: Path) -> None:
    log_path = copy_example_log(tmp_path, "01_simple_2d_maximise_logei_campaign_log.csv")
    campaign = load_campaign_session(
        "configs/01_simple_2d_maximise_logei.yaml",
        log_path,
    )
    before_log_bytes = log_path.read_bytes()
    before_df = campaign.df.copy(deep=True)

    campaign.export_report(tmp_path / "reports" / "campaign.txt")
    plot_result = campaign.plot_progress(save_path=tmp_path / "reports" / "progress.png")
    plt.close(extract_matplotlib_figure(plot_result))

    assert log_path.read_bytes() == before_log_bytes
    pd.testing.assert_frame_equal(campaign.df, before_df)

def test_streamlit_app_clear_staged_suggestions_removes_bundle() -> None:
    class FakeStreamlit:
        session_state = {"bo_forge_staged_suggestion_bundle": {"suggestions": simple_suggestions()}}

    streamlit_app._clear_staged_suggestions(FakeStreamlit)

    assert "bo_forge_staged_suggestion_bundle" not in FakeStreamlit.session_state

def test_create_campaign_from_inputs_sets_session_state_and_clears_staged(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "campaign.yaml"
    log_path = tmp_path / "campaign.csv"
    config_text = build_campaign_yaml_text(
        campaign_name="app_created",
        objective_name="activity",
        objective_direction="maximize",
        variables=[{"name": "x", "type": "continuous", "lower": 0.0, "upper": 1.0}],
        batch_size=1,
        initial_design_size=4,
        initial_design_method="sobol",
        random_seed=0,
    )

    class FakeStreamlit:
        session_state = {
            "bo_forge_staged_suggestion_bundle": {"suggestions": simple_suggestions()}
        }
        success_messages: list[str] = []
        error_messages: list[str] = []
        markdown_messages: list[str] = []

        @classmethod
        def success(cls, message: str) -> None:
            cls.success_messages.append(message)

        @classmethod
        def error(cls, message: str) -> None:
            cls.error_messages.append(message)

        @classmethod
        def markdown(cls, message: str, unsafe_allow_html: bool = False) -> None:
            cls.markdown_messages.append(message)

    streamlit_app._create_campaign_from_inputs(
        FakeStreamlit,
        config_text,
        str(config_path),
        str(log_path),
    )

    assert FakeStreamlit.error_messages == []
    assert FakeStreamlit.session_state["bo_forge_config_path"] == str(config_path)
    assert FakeStreamlit.session_state["bo_forge_log_path"] == str(log_path)
    assert FakeStreamlit.session_state["bo_forge_campaign_session"].config.campaign_name == (
        "app_created"
    )
    assert "bo_forge_staged_suggestion_bundle" not in FakeStreamlit.session_state
    assert "Campaign created and loaded" in "\n".join(FakeStreamlit.success_messages)

@pytest.mark.parametrize(
    ("config_path", "expected"),
    [
        (
            "configs/01_simple_2d_maximise_logei.yaml",
            {
                "has_constraints": False,
                "has_cost": False,
                "has_review": False,
                "has_replicates": False,
            },
        ),
        (
            "configs/07_cost_aware_human_review_logei.yaml",
            {
                "has_constraints": True,
                "has_cost": True,
                "has_review": True,
                "has_replicates": False,
            },
        ),
        (
            "configs/08_replicate_aware_logei.yaml",
            {
                "has_constraints": False,
                "has_cost": False,
                "has_review": False,
                "has_replicates": True,
            },
        ),
    ],
)
def test_feature_flags(config_path: str, expected: dict[str, bool]) -> None:
    config = CampaignConfig.from_yaml(config_path)

    assert feature_flags(config) == expected

def test_available_plot_kinds_follow_config_features() -> None:
    plain = CampaignConfig.from_yaml("configs/01_simple_2d_maximise_logei.yaml")
    cost = CampaignConfig.from_yaml("configs/07_cost_aware_human_review_logei.yaml")
    replicate = CampaignConfig.from_yaml("configs/08_replicate_aware_logei.yaml")
    multi = CampaignConfig.from_yaml("configs/10_multi_objective_mixed_constrained_qlogehvi.yaml")
    four_objective = CampaignConfig.from_yaml(
        "configs/11_four_objective_mixed_constrained_qlogehvi.yaml"
    )
    multi_cost = CampaignConfig.from_yaml("configs/12_cost_aware_multi_objective_qlogehvi.yaml")
    fidelity = CampaignConfig.from_yaml("configs/15_multi_fidelity_qmfkg.yaml")
    context = CampaignConfig.from_yaml("configs/16_contextual_logei.yaml")
    qlog_nei = CampaignConfig.from_yaml("configs/18_noisy_pending_qlognei.yaml")

    assert available_plot_kinds(plain) == [
        "progress",
        "diagnostics",
        "model_diagnostics",
        "model_comparison",
    ]
    assert available_plot_kinds(cost) == [
        "progress",
        "diagnostics",
        "model_diagnostics",
        "model_comparison",
        "cost_progress",
    ]
    assert available_plot_kinds(replicate) == [
        "progress",
        "diagnostics",
        "model_diagnostics",
        "model_comparison",
        "replicates",
    ]
    assert available_plot_kinds(multi) == ["pareto", "hypervolume"]
    assert available_plot_kinds(four_objective) == [
        "pareto",
        "hypervolume",
        "pareto_parallel",
    ]
    assert available_plot_kinds(multi_cost) == [
        "pareto",
        "hypervolume",
        "pareto_parallel",
        "cost_progress",
    ]
    assert available_plot_kinds(fidelity) == [
        "progress",
        "diagnostics",
        "fidelity_diagnostics",
        "fidelity_progress",
    ]
    assert available_plot_kinds(context) == [
        "progress",
        "diagnostics",
        "model_diagnostics",
        "model_comparison",
        "context_diagnostics",
    ]
    assert available_plot_kinds(qlog_nei) == [
        "progress",
        "diagnostics",
        "model_diagnostics",
        "model_comparison",
        "qlog_nei_diagnostics",
    ]
