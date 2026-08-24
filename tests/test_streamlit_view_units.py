"""Streamlit view-unit, state-cache, and lazy-render tests."""

from tests._streamlit_support import (
    FORGE_SUITE_CSS,
    BOConfig,
    CampaignConfig,
    CampaignSession,
    CostConfig,
    ObjectiveConfig,
    Path,
    SimpleNamespace,
    VariableConfig,
    active_variables_display,
    available_plot_kinds,
    canonical_columns,
    copy_example_log,
    default_export_path,
    extract_matplotlib_figure,
    feature_flags,
    forge_action_label,
    forge_status_label,
    load_campaign_session,
    make_staged_suggestion_bundle,
    pd,
    plt,
    pytest,
    shutil,
    simple_suggestions,
    staged_suggestions_from_bundle,
    streamlit_app,
    streamlit_helpers,
    streamlit_style,
    structured_stage_config_table,
    structured_stage_options,
)


def test_structured_stage_display_helpers_show_configured_activity() -> None:
    config = CampaignConfig.from_yaml("configs/13_structured_campaign_core.yaml")

    table = structured_stage_config_table(config)

    assert structured_stage_options(config) == ["screen", "refine"]
    assert active_variables_display(config, "screen") == "precursor_ratio, electrolyte"
    assert table["stage"].tolist() == ["screen", "refine"]
    assert table.loc[table["stage"] == "screen", "active_variables"].iloc[0] == (
        "precursor_ratio, electrolyte"
    )
    assert table.loc[table["stage"] == "screen", "inactive_variables"].iloc[0] == (
        "annealing_temperature"
    )
    assert available_plot_kinds(config) == [
        "progress",
        "diagnostics",
        "stage_diagnostics",
    ]

def test_non_structured_stage_display_helpers_are_empty() -> None:
    config = CampaignConfig.from_yaml("configs/01_simple_2d_maximise_logei.yaml")

    assert structured_stage_options(config) == []
    assert structured_stage_config_table(config).empty
    assert available_plot_kinds(config) == [
        "progress",
        "diagnostics",
        "model_diagnostics",
        "model_comparison",
    ]

def test_default_export_path_uses_reports_directory() -> None:
    path = default_export_path(Path("examples/my_campaign_log.csv"), "progress", "png")

    assert path == Path("reports/my_campaign_log_progress.png")

def test_extract_matplotlib_figure_from_figure_and_tuple() -> None:
    fig, ax = plt.subplots()

    assert extract_matplotlib_figure(fig) is fig
    assert extract_matplotlib_figure((fig, ax)) is fig
    plt.close(fig)

def test_app_modules_import_without_streamlit_runtime() -> None:
    assert hasattr(streamlit_helpers, "make_staged_suggestion_bundle")
    assert hasattr(streamlit_style, "apply_forge_suite_style")
    assert hasattr(streamlit_helpers, "create_campaign_files")
    assert hasattr(streamlit_app, "main")
    assert hasattr(streamlit_app, "render_app")
    assert hasattr(streamlit_app, "_render_workbench_header")
    assert hasattr(streamlit_app, "_render_campaign_source_bar")
    assert hasattr(streamlit_app, "_render_campaign_files_panel")
    assert hasattr(streamlit_app, "_render_create_new_campaign")
    assert hasattr(streamlit_app, "_render_campaign_state_blocks")
    assert hasattr(streamlit_app, "_render_data")

def test_streamlit_resolve_panel_marks_multi_objective_rows_observed(
    tmp_path: Path,
) -> None:
    cfg = CampaignConfig(
        campaign_name="mo_app",
        objective=ObjectiveConfig("yield_score", "maximize", 0.0),
        objectives=(
            ObjectiveConfig("yield_score", "maximize", 0.0),
            ObjectiveConfig("waste_score", "minimize", 1.0),
        ),
        variables=(VariableConfig("x", "continuous", 0.0, 1.0),),
        bo=BOConfig(batch_size=1, initial_design_size=1, acquisition="qlog_ehvi"),
        cost=CostConfig(expression="1.0 + x", budget=10.0),
    )
    df = pd.DataFrame(
        [
            {
                "row_id": "suggested_1",
                "iteration": 1,
                "status": "suggested",
                "source": "qlog_ehvi",
                "x": 0.5,
                "yield_score": "",
                "waste_score": "",
                "cost_estimate": 1.5,
                "cost_actual": "",
                "predicted_mean_yield_score": 0.6,
                "predicted_std_yield_score": 0.1,
                "predicted_mean_waste_score": 0.4,
                "predicted_std_waste_score": 0.1,
                "acquisition": 0.2,
                "utility": "",
            }
        ],
        columns=canonical_columns(cfg),
    )
    log_path = tmp_path / "campaign.csv"
    df.to_csv(log_path, index=False)
    campaign = CampaignSession(
        config_path=Path("config.yaml"),
        log_path=log_path,
        config=cfg,
        df=df,
    )

    class _Context:
        def __enter__(self) -> "_Context":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    class FakeStreamlit:
        session_state: dict[str, object] = {
            "bo_forge_config_path": "config.yaml",
            "bo_forge_log_path": str(log_path),
        }
        markdown_messages: list[str] = []
        subheaders: list[str] = []
        form_submit_labels: list[str] = []
        text_values = {
            "Observed yield_score": "0.8",
            "Observed waste_score": "0.35",
            "Actual cost (optional)": "1.7",
        }

        @classmethod
        def markdown(cls, body: str, unsafe_allow_html: bool = False) -> None:
            cls.markdown_messages.append(body)

        @classmethod
        def subheader(cls, label: str) -> None:
            cls.subheaders.append(label)

        @classmethod
        def dataframe(cls, *_args: object, **_kwargs: object) -> None:
            return None

        @classmethod
        def expander(cls, *_args: object, **_kwargs: object) -> _Context:
            return _Context()

        @classmethod
        def form(cls, *_args: object, **_kwargs: object) -> _Context:
            return _Context()

        @classmethod
        def selectbox(cls, _label: str, options: list[str], **_kwargs: object) -> str:
            return options[0]

        @classmethod
        def text_input(cls, label: str, *_args: object, **_kwargs: object) -> str:
            return cls.text_values.get(label, "")

        @classmethod
        def form_submit_button(cls, label: str, *_args: object, **_kwargs: object) -> bool:
            cls.form_submit_labels.append(label)
            return label == "Record coupled objectives"

        @classmethod
        def success(cls, *_args: object, **_kwargs: object) -> None:
            return None

        @classmethod
        def error(cls, message: str, *_args: object, **_kwargs: object) -> None:
            raise AssertionError(message)

    streamlit_app._render_resolve(FakeStreamlit, campaign, feature_flags(cfg))

    refreshed = pd.read_csv(log_path, keep_default_na=False)
    row = refreshed.loc[refreshed["row_id"] == "suggested_1"].iloc[0]
    assert row["status"] == "observed"
    assert float(row["yield_score"]) == pytest.approx(0.8)
    assert float(row["waste_score"]) == pytest.approx(0.35)
    assert float(row["cost_actual"]) == pytest.approx(1.7)
    assert "Record coupled objectives" in FakeStreamlit.form_submit_labels

def test_streamlit_resolve_panel_rejects_incomplete_multi_objective_entry(
    tmp_path: Path,
) -> None:
    cfg = CampaignConfig(
        campaign_name="mo_app",
        objective=ObjectiveConfig("yield_score", "maximize", 0.0),
        objectives=(
            ObjectiveConfig("yield_score", "maximize", 0.0),
            ObjectiveConfig("waste_score", "minimize", 1.0),
        ),
        variables=(VariableConfig("x", "continuous", 0.0, 1.0),),
        bo=BOConfig(batch_size=1, initial_design_size=1, acquisition="qlog_ehvi"),
    )
    df = pd.DataFrame(
        [
            {
                "row_id": "suggested_1",
                "iteration": 1,
                "status": "suggested",
                "source": "qlog_ehvi",
                "x": 0.5,
                "yield_score": "",
                "waste_score": "",
                "predicted_mean_yield_score": 0.6,
                "predicted_std_yield_score": 0.1,
                "predicted_mean_waste_score": 0.4,
                "predicted_std_waste_score": 0.1,
                "acquisition": 0.2,
            }
        ],
        columns=canonical_columns(cfg),
    )
    log_path = tmp_path / "campaign.csv"
    df.to_csv(log_path, index=False)
    before = log_path.read_bytes()
    campaign = CampaignSession(
        config_path=Path("config.yaml"),
        log_path=log_path,
        config=cfg,
        df=df,
    )

    class _Context:
        def __enter__(self) -> "_Context":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    class FakeStreamlit:
        session_state: dict[str, object] = {}
        errors: list[str] = []

        @classmethod
        def markdown(cls, *_args: object, **_kwargs: object) -> None:
            return None

        @classmethod
        def subheader(cls, *_args: object, **_kwargs: object) -> None:
            return None

        @classmethod
        def dataframe(cls, *_args: object, **_kwargs: object) -> None:
            return None

        @classmethod
        def expander(cls, *_args: object, **_kwargs: object) -> _Context:
            return _Context()

        @classmethod
        def form(cls, *_args: object, **_kwargs: object) -> _Context:
            return _Context()

        @classmethod
        def selectbox(cls, _label: str, options: list[str], **_kwargs: object) -> str:
            return options[0]

        @classmethod
        def text_input(cls, label: str, *_args: object, **_kwargs: object) -> str:
            return "0.8" if label == "Observed yield_score" else ""

        @classmethod
        def form_submit_button(cls, label: str, *_args: object, **_kwargs: object) -> bool:
            return label == "Record coupled objectives"

        @classmethod
        def success(cls, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("Incomplete entry should not be recorded.")

        @classmethod
        def error(cls, message: str, *_args: object, **_kwargs: object) -> None:
            cls.errors.append(message)

    streamlit_app._render_resolve(FakeStreamlit, campaign, feature_flags(cfg))

    assert log_path.read_bytes() == before
    assert FakeStreamlit.errors == ["Observed waste_score is required."]

def test_workbench_header_uses_compact_technical_identity() -> None:
    class FakeStreamlit:
        markdown_calls: list[str] = []

        @classmethod
        def markdown(cls, body: str, unsafe_allow_html: bool = False) -> None:
            cls.markdown_calls.append(body)
            assert unsafe_allow_html is True

    streamlit_app._render_workbench_header(FakeStreamlit, campaign_loaded=True)

    rendered = "\n".join(FakeStreamlit.markdown_calls)
    assert 'class="bf-title">BO Forge</h1>' in rendered
    assert "Scientific campaign workbench" in rendered
    assert "Campaign loaded" in rendered

def test_forge_suite_css_contains_expected_palette_tokens() -> None:
    assert "#f2efe8" in FORGE_SUITE_CSS
    assert "#292d2f" in FORGE_SUITE_CSS
    assert "#b96f45" in FORGE_SUITE_CSS
    assert "bf-workbench-header" in FORGE_SUITE_CSS
    assert "bf-source-bar" in FORGE_SUITE_CSS
    assert "--bf-canvas" in FORGE_SUITE_CSS
    assert "forge-card" in FORGE_SUITE_CSS
    assert "forge-empty" in FORGE_SUITE_CSS
    assert "forge-artifact" in FORGE_SUITE_CSS
    assert "forge-callout" in FORGE_SUITE_CSS
    assert '[data-testid="stHeader"]' in FORGE_SUITE_CSS
    assert '[data-testid="stToolbar"]' in FORGE_SUITE_CSS
    assert '[data-testid="stDecoration"]' in FORGE_SUITE_CSS

def test_forge_status_labels_are_stable() -> None:
    assert forge_status_label("has_pending_suggestions") == "Pending suggestions"
    assert forge_status_label("ready_for_initial_design") == "Ready for initial design"
    assert forge_status_label("ready_for_bo") == "Ready for BO"

def test_forge_action_labels_are_stable() -> None:
    assert forge_action_label("review_pending_suggestions") == "Review pending suggestions"
    assert forge_action_label("run_accepted_suggestions") == "Run accepted suggestions"
    assert forge_action_label("resolve_pending_suggestions") == "Resolve pending suggestions"
    assert forge_action_label("suggest_initial_design") == "Suggest initial design"
    assert forge_action_label("suggest_bo") == "Suggest BO candidates"

def test_active_panel_dispatch_renders_only_selected_panel(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class FakeCampaign:
        config = CampaignConfig.from_yaml("configs/01_simple_2d_maximise_logei.yaml")

    def fake_collect(_campaign: object, panel: str) -> dict[str, object]:
        calls.append(f"collect:{panel}")
        return {}

    monkeypatch.setattr(streamlit_app, "_collect_panel_view_data", fake_collect)
    monkeypatch.setattr(
        streamlit_app,
        "_render_run_area",
        lambda *_args, **_kwargs: calls.append("Run"),
    )

    streamlit_app._render_active_workflow_panel(object(), FakeCampaign(), {}, "Resolve")

    assert calls == ["collect:Run", "Run"]

def test_collect_panel_view_data_requires_service_boundary() -> None:
    with pytest.raises(TypeError, match="must provide collect_view_data"):
        streamlit_app._collect_panel_view_data(object(), "Overview")

def test_source_bar_does_not_fingerprint_staged_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Context:
        def __enter__(self) -> "_Context":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    class FakeStreamlit:
        session_state = {
            "bo_forge_config_path": "configs/campaign.yaml",
            "bo_forge_log_path": "examples/campaign.csv",
            "bo_forge_staged_suggestion_bundle": {"suggestions": simple_suggestions()},
        }

        @classmethod
        def markdown(cls, *_args: object, **_kwargs: object) -> None:
            return None

        @classmethod
        def success(cls, *_args: object, **_kwargs: object) -> None:
            return None

        @classmethod
        def radio(cls, *_args: object, **_kwargs: object) -> str:
            return "Load Existing"

        @classmethod
        def expander(cls, *_args: object, **_kwargs: object) -> _Context:
            return _Context()

    monkeypatch.setattr(
        streamlit_app,
        "_current_invalidation_reason",
        lambda *_args, **_kwargs: pytest.fail("source bar should not hash files"),
    )
    monkeypatch.setattr(streamlit_app, "_render_load_existing_campaign", lambda *_args: None)

    streamlit_app._render_campaign_source_bar(FakeStreamlit)

def test_source_bar_uses_cached_validation_without_validating(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "campaign.yaml"
    log_path = tmp_path / "campaign.csv"
    config_path.write_text("campaign_name: cached\n", encoding="utf-8")
    log_path.write_text("row_id\n", encoding="utf-8")
    signature = streamlit_app._validation_cache_signature(config_path, log_path)

    class _Context:
        def __enter__(self) -> "_Context":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    class FakeCampaign:
        def validate(self) -> None:
            raise AssertionError("source bar should not validate during render")

    class FakeStreamlit:
        session_state = {
            "bo_forge_campaign_session": FakeCampaign(),
            "bo_forge_config_path": str(config_path),
            "bo_forge_log_path": str(log_path),
            "bo_forge_validation_cache": {"signature": signature, "label": "Valid"},
        }
        markdown_messages: list[str] = []

        @classmethod
        def markdown(cls, body: str, **_kwargs: object) -> None:
            cls.markdown_messages.append(body)

        @classmethod
        def success(cls, *_args: object, **_kwargs: object) -> None:
            return None

        @classmethod
        def radio(cls, *_args: object, **_kwargs: object) -> str:
            return "Load Existing"

        @classmethod
        def expander(cls, *_args: object, **_kwargs: object) -> _Context:
            return _Context()

    monkeypatch.setattr(streamlit_app, "_render_load_existing_campaign", lambda *_args: None)

    streamlit_app._render_campaign_source_bar(FakeStreamlit)

    assert "Valid" in "\n".join(FakeStreamlit.markdown_messages)

def test_cached_validation_status_detects_changed_file_metadata(tmp_path: Path) -> None:
    config_path = tmp_path / "campaign.yaml"
    log_path = tmp_path / "campaign.csv"
    config_path.write_text("campaign_name: cached\n", encoding="utf-8")
    log_path.write_text("row_id\n", encoding="utf-8")

    class FakeStreamlit:
        session_state = {
            "bo_forge_config_path": str(config_path),
            "bo_forge_log_path": str(log_path),
            "bo_forge_validation_cache": {
                "signature": streamlit_app._validation_cache_signature(config_path, log_path),
                "label": "Validation issue",
            },
        }

    assert streamlit_app._cached_validation_label(FakeStreamlit, object()) == (
        "Validation issue"
    )

    log_path.write_text("row_id\nchanged\n", encoding="utf-8")

    assert streamlit_app._cached_validation_label(FakeStreamlit, object()) == (
        "Reload to validate"
    )

def test_overview_uses_cached_validation_without_validating(tmp_path: Path) -> None:
    log_path = copy_example_log(tmp_path, "01_simple_2d_maximise_logei_campaign_log.csv")
    campaign = load_campaign_session("configs/01_simple_2d_maximise_logei.yaml", log_path)
    view_data = {
        "summary": campaign.summary(),
        "next_action": campaign.next_action(),
        "observed": campaign.observed_data(),
        "pending": campaign.pending_suggestions(),
    }
    campaign.validate = lambda: pytest.fail("Overview should use cached validation state")
    signature = streamlit_app._validation_cache_signature(
        "configs/01_simple_2d_maximise_logei.yaml",
        log_path,
    )

    class _Context:
        def __enter__(self) -> "_Context":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    class FakeStreamlit:
        session_state = {
            "bo_forge_config_path": "configs/01_simple_2d_maximise_logei.yaml",
            "bo_forge_log_path": str(log_path),
            "bo_forge_validation_cache": {"signature": signature, "label": "Valid", "error": ""},
        }
        markdown_messages: list[str] = []
        errors: list[str] = []

        @classmethod
        def markdown(cls, body: str, **_kwargs: object) -> None:
            cls.markdown_messages.append(body)

        @classmethod
        def error(cls, message: str, *_args: object, **_kwargs: object) -> None:
            cls.errors.append(message)

        @classmethod
        def columns(cls, count: int, *_args: object, **_kwargs: object) -> list[_Context]:
            return [_Context() for _ in range(count)]

        @classmethod
        def subheader(cls, *_args: object, **_kwargs: object) -> None:
            return None

        @classmethod
        def dataframe(cls, *_args: object, **_kwargs: object) -> None:
            return None

        @classmethod
        def expander(cls, *_args: object, **_kwargs: object) -> _Context:
            return _Context()

    streamlit_app._render_overview(
        FakeStreamlit,
        campaign,
        view_data,
    )

    assert FakeStreamlit.errors == []
    assert "Campaign log is valid" in "\n".join(FakeStreamlit.markdown_messages)

def test_overview_renders_cached_validation_error(tmp_path: Path) -> None:
    config_path = tmp_path / "campaign.yaml"
    log_path = tmp_path / "campaign.csv"
    config_path.write_text("campaign_name: cached\n", encoding="utf-8")
    log_path.write_text("row_id\n", encoding="utf-8")
    signature = streamlit_app._validation_cache_signature(config_path, log_path)

    class _Context:
        def __enter__(self) -> "_Context":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    class FakeCampaign:
        df = pd.DataFrame([{"row_id": "bad"}])

        def validate(self) -> None:
            raise AssertionError("Overview should not validate during render")

    class FakeStreamlit:
        session_state = {
            "bo_forge_config_path": str(config_path),
            "bo_forge_log_path": str(log_path),
            "bo_forge_validation_cache": {
                "signature": signature,
                "label": "Validation issue",
                "error": "bad CSV",
            },
        }
        errors: list[str] = []

        @classmethod
        def markdown(cls, *_args: object, **_kwargs: object) -> None:
            return None

        @classmethod
        def error(cls, message: str, *_args: object, **_kwargs: object) -> None:
            cls.errors.append(message)

        @classmethod
        def subheader(cls, *_args: object, **_kwargs: object) -> None:
            return None

        @classmethod
        def dataframe(cls, *_args: object, **_kwargs: object) -> None:
            return None

        @classmethod
        def expander(cls, *_args: object, **_kwargs: object) -> _Context:
            return _Context()

    streamlit_app._render_overview(FakeStreamlit, FakeCampaign(), {})

    assert FakeStreamlit.errors == ["Validation failed: bad CSV"]

def test_successful_dry_run_clears_stale_staged_freshness(tmp_path: Path) -> None:
    config_path = tmp_path / "campaign.yaml"
    log_path = tmp_path / "campaign.csv"
    config_path.write_text("config", encoding="utf-8")
    log_path.write_text("log", encoding="utf-8")

    class _Context:
        def __enter__(self) -> "_Context":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    class FakeCampaign:
        config = CampaignConfig.from_yaml("configs/01_simple_2d_maximise_logei.yaml")

        def suggest_next(self, batch_size: int) -> pd.DataFrame:
            assert batch_size == 1
            return simple_suggestions()

        def suggestion_quality(self, _suggestions: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame([{"check": "ok"}])

    class FakeStreamlit:
        session_state = {
            "bo_forge_config_path": str(config_path),
            "bo_forge_log_path": str(log_path),
            "bo_forge_staged_freshness_message": "Log file changed after suggestions were staged.",
        }

        @classmethod
        def markdown(cls, *_args: object, **_kwargs: object) -> None:
            return None

        @classmethod
        def number_input(cls, *_args: object, **_kwargs: object) -> int:
            return 1

        @classmethod
        def form_submit_button(cls, label: str, *_args: object, **_kwargs: object) -> bool:
            return label == "Generate suggestions (dry run)"

        @classmethod
        def form(cls, *_args: object, **_kwargs: object) -> _Context:
            return _Context()

        @classmethod
        def text_input(cls, *_args: object, **_kwargs: object) -> str:
            return str(tmp_path / "staged.csv")

        @classmethod
        def columns(cls, count: int, *_args: object, **_kwargs: object) -> list[_Context]:
            return [_Context() for _ in range(count)]

        @classmethod
        def subheader(cls, *_args: object, **_kwargs: object) -> None:
            return None

        @classmethod
        def dataframe(cls, *_args: object, **_kwargs: object) -> None:
            return None

        @classmethod
        def expander(cls, *_args: object, **_kwargs: object) -> _Context:
            return _Context()

        @classmethod
        def success(cls, *_args: object, **_kwargs: object) -> None:
            return None

        @classmethod
        def error(cls, message: str, *_args: object, **_kwargs: object) -> None:
            raise AssertionError(message)

        @classmethod
        def warning(cls, message: str, *_args: object, **_kwargs: object) -> None:
            raise AssertionError(message)

    streamlit_app._render_suggest(FakeStreamlit, FakeCampaign())

    assert "bo_forge_staged_freshness_message" not in FakeStreamlit.session_state

def test_structured_suggest_panel_stages_selected_stage(tmp_path: Path) -> None:
    config_path = Path("configs/13_structured_campaign_core.yaml")
    log_path = tmp_path / "structured.csv"
    shutil.copyfile("examples/13_structured_campaign_core_campaign_log.csv", log_path)
    structured_config = CampaignConfig.from_yaml(config_path)
    calls: list[tuple[int, str | None]] = []

    class _Context:
        def __enter__(self) -> "_Context":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    class _FormContext:
        def __enter__(self) -> "_FormContext":
            FakeStreamlit.form_depth += 1
            return self

        def __exit__(self, *_args: object) -> None:
            FakeStreamlit.form_depth -= 1

    class FakeCampaign:
        config = structured_config

        def suggest_dry_run(self, batch_size: int, stage: str | None = None) -> object:
            calls.append((batch_size, stage))
            suggestions = pd.DataFrame(
                [
                    {
                        "row_id": "suggested_screen",
                        "iteration": 2,
                        "status": "suggested",
                        "source": "sobol",
                        "stage": str(stage),
                        "precursor_ratio": 0.25,
                        "electrolyte": "KPF6",
                        "annealing_temperature": "",
                        "activity": "",
                        "predicted_mean": "",
                        "predicted_std": "",
                        "acquisition": "",
                    }
                ]
            )
            bundle = make_staged_suggestion_bundle(
                suggestions,
                config_path,
                log_path,
                stage=stage,
            )
            return SimpleNamespace(
                suggestions=suggestions,
                bundle=bundle,
                quality=pd.DataFrame([{"check": "ok"}]),
            )

        def suggestion_quality(self, _suggestions: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame([{"check": "ok"}])

    class FakeStreamlit:
        session_state = {
            "bo_forge_config_path": str(config_path),
            "bo_forge_log_path": str(log_path),
        }
        selectbox_labels: list[str] = []
        selectbox_called_inside_form = False
        form_depth = 0

        @classmethod
        def markdown(cls, *_args: object, **_kwargs: object) -> None:
            return None

        @classmethod
        def number_input(cls, *_args: object, **_kwargs: object) -> int:
            return 1

        @classmethod
        def selectbox(cls, label: str, options: list[str], **_kwargs: object) -> str:
            cls.selectbox_labels.append(label)
            cls.selectbox_called_inside_form = cls.form_depth > 0
            assert options == ["screen", "refine"]
            return "screen"

        @classmethod
        def form_submit_button(cls, label: str, *_args: object, **_kwargs: object) -> bool:
            return label == "Generate suggestions (dry run)"

        @classmethod
        def form(cls, *_args: object, **_kwargs: object) -> _FormContext:
            return _FormContext()

        @classmethod
        def text_input(cls, *_args: object, **_kwargs: object) -> str:
            return str(tmp_path / "staged.csv")

        @classmethod
        def columns(cls, count: int, *_args: object, **_kwargs: object) -> list[_Context]:
            return [_Context() for _ in range(count)]

        @classmethod
        def subheader(cls, *_args: object, **_kwargs: object) -> None:
            return None

        @classmethod
        def dataframe(cls, *_args: object, **_kwargs: object) -> None:
            return None

        @classmethod
        def expander(cls, *_args: object, **_kwargs: object) -> _Context:
            return _Context()

        @classmethod
        def success(cls, *_args: object, **_kwargs: object) -> None:
            return None

        @classmethod
        def error(cls, message: str, *_args: object, **_kwargs: object) -> None:
            raise AssertionError(message)

        @classmethod
        def warning(cls, message: str, *_args: object, **_kwargs: object) -> None:
            raise AssertionError(message)

    streamlit_app._render_suggest(FakeStreamlit, FakeCampaign())

    bundle = FakeStreamlit.session_state["bo_forge_staged_suggestion_bundle"]
    assert calls == [(1, "screen")]
    assert FakeStreamlit.selectbox_labels == ["Suggestion stage"]
    assert not FakeStreamlit.selectbox_called_inside_form
    assert bundle["stage"] == "screen"
    staged = staged_suggestions_from_bundle(bundle)
    assert staged.loc[0, "stage"] == "screen"
    assert log_path.read_bytes() == Path(
        "examples/13_structured_campaign_core_campaign_log.csv"
    ).read_bytes()

def test_structured_selected_row_preview_handles_invalid_stage() -> None:
    config = CampaignConfig.from_yaml("configs/13_structured_campaign_core.yaml")
    campaign = SimpleNamespace(config=config)

    class FakeStreamlit:
        markdown_calls: list[str] = []

        @classmethod
        def markdown(cls, content: str, *_args: object, **_kwargs: object) -> None:
            cls.markdown_calls.append(content)

    streamlit_app._render_selected_row_preview(
        FakeStreamlit,
        campaign,
        pd.Series({"stage": "unknown_stage", "precursor_ratio": 0.25}),
    )

    rendered = "\n".join(FakeStreamlit.markdown_calls)
    assert "unknown_stage" in rendered

def test_valid_staged_bundle_clears_old_freshness_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "campaign.yaml"
    log_path = tmp_path / "campaign.csv"
    config_path.write_text("config", encoding="utf-8")
    log_path.write_text("log", encoding="utf-8")
    bundle = make_staged_suggestion_bundle(simple_suggestions(), config_path, log_path)

    class _Context:
        def __enter__(self) -> "_Context":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    class FakeCampaign:
        config = CampaignConfig.from_yaml("configs/01_simple_2d_maximise_logei.yaml")

        def suggestion_quality(self, _suggestions: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame([{"check": "ok"}])

    class FakeStreamlit:
        session_state = {
            "bo_forge_config_path": str(config_path),
            "bo_forge_log_path": str(log_path),
            "bo_forge_staged_suggestion_bundle": bundle,
            "bo_forge_staged_freshness_message": "Log file changed after suggestions were staged.",
        }

        @classmethod
        def markdown(cls, *_args: object, **_kwargs: object) -> None:
            return None

        @classmethod
        def number_input(cls, *_args: object, **_kwargs: object) -> int:
            return 1

        @classmethod
        def form_submit_button(cls, *_args: object, **_kwargs: object) -> bool:
            return False

        @classmethod
        def form(cls, *_args: object, **_kwargs: object) -> _Context:
            return _Context()

        @classmethod
        def text_input(cls, *_args: object, **_kwargs: object) -> str:
            return str(tmp_path / "staged.csv")

        @classmethod
        def columns(cls, count: int, *_args: object, **_kwargs: object) -> list[_Context]:
            return [_Context() for _ in range(count)]

        @classmethod
        def subheader(cls, *_args: object, **_kwargs: object) -> None:
            return None

        @classmethod
        def dataframe(cls, *_args: object, **_kwargs: object) -> None:
            return None

        @classmethod
        def expander(cls, *_args: object, **_kwargs: object) -> _Context:
            return _Context()

    monkeypatch.setattr(streamlit_app, "_current_invalidation_reason", lambda *_args: None)

    streamlit_app._render_suggest(FakeStreamlit, FakeCampaign())

    assert "bo_forge_staged_freshness_message" not in FakeStreamlit.session_state

def test_reports_are_lazy_and_render_only_selected_plot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class _Context:
        def __enter__(self) -> "_Context":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    class FakeStreamlit:
        session_state = {
            "bo_forge_config_path": "configs/10_multi_objective_mixed_constrained_qlogehvi.yaml",
            "bo_forge_log_path": "examples/10_multi_objective_mixed_constrained_campaign_log.csv",
        }
        text_labels: list[str] = []

        @classmethod
        def markdown(cls, *_args: object, **_kwargs: object) -> None:
            return None

        @classmethod
        def text_input(cls, label: str, *_args: object, **_kwargs: object) -> str:
            cls.text_labels.append(label)
            return "/tmp/plot.png"

        @classmethod
        def selectbox(cls, _label: str, options: list[str], **_kwargs: object) -> str:
            return "Hypervolume"

        @classmethod
        def form(cls, *_args: object, **_kwargs: object) -> _Context:
            return _Context()

        @classmethod
        def columns(cls, count: int, *_args: object, **_kwargs: object) -> list[_Context]:
            return [_Context() for _ in range(count)]

        @classmethod
        def form_submit_button(cls, *_args: object, **_kwargs: object) -> bool:
            return False

        @classmethod
        def text_area(cls, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("Report text should not render until preview is requested.")

    class FakeCampaign:
        config = CampaignConfig.from_yaml(
            "configs/10_multi_objective_mixed_constrained_qlogehvi.yaml"
        )

        def summary(self) -> pd.DataFrame:
            raise AssertionError("summary should come from view data")

        def plot_pareto(self, *_args: object, **_kwargs: object) -> None:
            calls.append("pareto")

        def plot_hypervolume(self, *_args: object, **_kwargs: object) -> None:
            calls.append("hypervolume")

    monkeypatch.setattr(
        streamlit_app,
        "campaign_report_text",
        lambda *_args, **_kwargs: pytest.fail("report preview should be lazy"),
    )
    summary = pd.DataFrame(
        [
            {"field": "campaign_status", "value": "ready_for_bo"},
            {"field": "observed_rows", "value": 3},
            {"field": "pending_suggestions", "value": 0},
            {"field": "hypervolume", "value": 1.2},
        ]
    )

    streamlit_app._render_reports(
        FakeStreamlit,
        FakeCampaign(),
        {"has_cost": False, "has_replicates": False},
        {"summary": summary},
    )

    assert calls == []
    assert "Hypervolume export path" in FakeStreamlit.text_labels
    assert "Pareto export path" not in FakeStreamlit.text_labels
