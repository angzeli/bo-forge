"""Streamlit AppTest loading, switching, and campaign workflow tests."""

import json

import bo_forge._campaign.provenance as provenance_module
from bo_forge.errors import ProvenanceRecoveryRequired
from tests._streamlit_support import (
    PROJECT_ROOT,
    BOConfig,
    CampaignConfig,
    CampaignSession,
    CostConfig,
    LogBusyError,
    LogConflictError,
    ObjectiveConfig,
    Path,
    SimpleNamespace,
    VariableConfig,
    canonical_columns,
    copy_example_log,
    empty_campaign_log,
    feature_flags,
    pd,
    pytest,
    streamlit_app,
)


def test_streamlit_strict_provenance_loading_and_explicit_recovery(
    tmp_path: Path,
) -> None:
    from streamlit.testing.v1 import AppTest

    config_path = tmp_path / "campaign.yaml"
    config_path.write_text(
        (PROJECT_ROOT / "configs" / "01_simple_2d_maximise_logei.yaml").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
    legacy_log = copy_example_log(
        tmp_path,
        "01_simple_2d_maximise_logei_campaign_log.csv",
    )
    app = AppTest.from_file(PROJECT_ROOT / "bo_forge_app" / "streamlit_app.py")
    app.run(timeout=10)
    next(input_ for input_ in app.text_input if input_.label == "YAML config path").set_value(
        str(config_path)
    )
    next(input_ for input_ in app.text_input if input_.label == "CSV log path").set_value(
        str(legacy_log)
    )
    next(
        item for item in app.checkbox if item.label == "Require provenance manifest"
    ).set_value(True)
    next(button for button in app.button if button.label == "Load campaign").click()
    app.run(timeout=10)
    assert any("provenance manifest is required" in error.value for error in app.error)
    assert streamlit_app.SESSION_KEY not in app.session_state

    managed_log = tmp_path / "managed.csv"
    CampaignSession.initialize(config_path, managed_log)
    manifest_path = provenance_module.manifest_path_for_log(managed_log)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pending = provenance_module._manifest_with_pending_transaction(
        manifest,
        config_file=config_path,
        operation="append_suggestions",
        affected_row_ids=["row_1"],
        metadata={"appended_row_count": 1},
        resulting_hash="1" * 64,
        resulting_row_count=1,
    )
    provenance_module._write_json_atomic(manifest_path, pending)
    before_log = managed_log.read_bytes()

    next(input_ for input_ in app.text_input if input_.label == "CSV log path").set_value(
        str(managed_log)
    )
    next(button for button in app.button if button.label == "Load campaign").click()
    app.run(timeout=10)
    assert any("interrupted transaction" in error.value for error in app.error)
    confirmation = next(
        item
        for item in app.checkbox
        if item.label == "I understand recovery changes the provenance manifest"
    )
    confirmation.set_value(True)
    app.run(timeout=10)
    next(button for button in app.button if button.label == "Recover provenance").click()
    app.run(timeout=10)

    assert len(app.exception) == 0
    assert streamlit_app.SESSION_KEY in app.session_state
    assert managed_log.read_bytes() == before_log
    recovered = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert recovered["pending_transaction"] is None


def test_reports_model_comparison_is_lazy(
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
            "bo_forge_config_path": "configs/17_model_profile_logei.yaml",
            "bo_forge_log_path": "examples/17_model_profile_campaign_log.csv",
        }

        @classmethod
        def markdown(cls, *_args: object, **_kwargs: object) -> None:
            return None

        @classmethod
        def text_input(cls, *_args: object, **_kwargs: object) -> str:
            return "/tmp/plot.png"

        @classmethod
        def selectbox(cls, _label: str, options: list[str], **_kwargs: object) -> str:
            return "Progress"

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
        config = CampaignConfig.from_yaml("configs/17_model_profile_logei.yaml")

        def summary(self) -> pd.DataFrame:
            raise AssertionError("summary should come from view data")

        def model_profile_comparison(self) -> pd.DataFrame:
            raise AssertionError("model comparison should be lazy")

        def available_plot_kinds(self) -> list[str]:
            return ["progress"]

        def plot_progress(self, *_args: object, **_kwargs: object) -> None:
            calls.append("progress")

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
            {"field": "best_objective_value", "value": 1.2},
        ]
    )

    streamlit_app._render_reports(
        FakeStreamlit,
        FakeCampaign(),
        {"has_cost": False, "has_replicates": False},
        {"summary": summary},
    )

    assert calls == []

def test_multi_objective_observation_keys_are_row_scoped(tmp_path: Path) -> None:
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
    rows = [
        {
            "row_id": row_id,
            "iteration": 1,
            "status": "suggested",
            "source": "qlog_ehvi",
            "x": x_value,
            "yield_score": "",
            "waste_score": "",
            "cost_estimate": 1.0 + x_value,
            "cost_actual": "",
            "predicted_mean_yield_score": 0.6,
            "predicted_std_yield_score": 0.1,
            "predicted_mean_waste_score": 0.4,
            "predicted_std_waste_score": 0.1,
            "acquisition": 0.2,
            "utility": "",
        }
        for row_id, x_value in [("suggested_1", 0.2), ("suggested_2", 0.8)]
    ]
    df = pd.DataFrame(rows, columns=canonical_columns(cfg))
    log_path = tmp_path / "campaign.csv"
    df.to_csv(log_path, index=False)
    before = log_path.read_bytes()
    campaign = CampaignSession(
        config_path=Path("config.yaml"),
        log_path=log_path,
        config=cfg,
        df=df,
    )
    input_scope = streamlit_app._campaign_widget_key_scope(
        cfg,
        config_path=campaign.config_path,
        log_path=campaign.log_path,
    )
    first_yield_key = streamlit_app._stable_widget_key(
        "observed_objective",
        input_scope,
        "suggested_1",
        "yield_score",
    )
    first_waste_key = streamlit_app._stable_widget_key(
        "observed_objective",
        input_scope,
        "suggested_1",
        "waste_score",
    )
    first_cost_key = streamlit_app._stable_widget_key(
        "actual_cost",
        f"{input_scope}|suggested_1",
    )

    class _Context:
        def __enter__(self) -> "_Context":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    class FakeStreamlit:
        session_state: dict[str, object] = {
            first_yield_key: "0.8",
            first_waste_key: "0.2",
            first_cost_key: "1.4",
        }
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
            return options[1]

        @classmethod
        def text_input(cls, _label: str, *_args: object, **kwargs: object) -> str:
            key = str(kwargs.get("key", ""))
            return str(cls.session_state.get(key, ""))

        @classmethod
        def form_submit_button(cls, label: str, *_args: object, **_kwargs: object) -> bool:
            return label == "Record coupled objectives"

        @classmethod
        def success(cls, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("Second row should not reuse first-row values.")

        @classmethod
        def error(cls, message: str, *_args: object, **_kwargs: object) -> None:
            cls.errors.append(message)

    streamlit_app._render_resolve(FakeStreamlit, campaign, feature_flags(cfg))

    assert log_path.read_bytes() == before
    assert FakeStreamlit.errors == ["Observed yield_score is required."]

def test_streamlit_app_smoke_runs_without_exceptions() -> None:
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(PROJECT_ROOT / "bo_forge_app" / "streamlit_app.py")
    app.run(timeout=10)

    assert len(app.exception) == 0
    assert any(radio.label == "Campaign file action" for radio in app.radio)
    assert any("Nothing loaded yet" in markdown.value for markdown in app.markdown)

def test_campaign_scoped_observation_keys_do_not_cross_campaigns() -> None:
    config = CampaignConfig.from_yaml("configs/20_contextual_cost_review_logei.yaml")
    first_scope = streamlit_app._campaign_widget_key_scope(
        config,
        config_path="configs/first.yaml",
        log_path="logs/first.csv",
    )
    second_scope = streamlit_app._campaign_widget_key_scope(
        config,
        config_path="configs/second.yaml",
        log_path="logs/second.csv",
    )

    first_key = streamlit_app._stable_widget_key(
        "observed_objective",
        first_scope,
        "shared_row_id",
        "yield_score",
    )
    second_key = streamlit_app._stable_widget_key(
        "observed_objective",
        second_scope,
        "shared_row_id",
        "yield_score",
    )

    assert first_scope != second_scope
    assert first_key != second_key

def test_clear_observation_inputs_removes_only_row_scoped_values() -> None:
    state = {
        "observed_objective_abc": 0.8,
        "actual_cost_def": "2.5",
        streamlit_app.STAGED_SUGGESTION_BUNDLE_KEY: {"suggestions": "keep"},
        "unrelated": "keep",
    }
    fake_streamlit = SimpleNamespace(session_state=state)

    streamlit_app._clear_observation_inputs(fake_streamlit)

    assert "observed_objective_abc" not in state
    assert "actual_cost_def" not in state
    assert streamlit_app.STAGED_SUGGESTION_BUNDLE_KEY in state
    assert state["unrelated"] == "keep"

def test_loading_campaign_clears_previous_observation_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "campaign.yaml"
    log_path = tmp_path / "campaign.csv"
    config_path.write_text("campaign_name: placeholder\n", encoding="utf-8")
    log_path.write_text("row_id\n", encoding="utf-8")
    loaded_service = SimpleNamespace(config_path=config_path, log_path=log_path)

    class FakeStreamlit:
        session_state = {
            "observed_objective_old": 0.7,
            "actual_cost_old": "4.2",
            "unrelated": "keep",
        }

        @staticmethod
        def error(message: str) -> None:
            raise AssertionError(message)

    monkeypatch.setattr(
        streamlit_app.CampaignAppService,
        "load",
        lambda *_args, **_kwargs: loaded_service,
    )
    monkeypatch.setattr(
        streamlit_app,
        "_refresh_validation_cache",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        streamlit_app,
        "_flash_and_rerun",
        lambda *_args, **_kwargs: None,
    )

    streamlit_app._load_campaign_from_inputs(
        FakeStreamlit,
        str(config_path),
        str(log_path),
    )

    assert "observed_objective_old" not in FakeStreamlit.session_state
    assert "actual_cost_old" not in FakeStreamlit.session_state
    assert FakeStreamlit.session_state["unrelated"] == "keep"
    assert FakeStreamlit.session_state[streamlit_app.SESSION_KEY] is loaded_service

def test_streamlit_log_conflict_clears_stale_state_and_reloads_campaign(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "campaign.yaml"
    log_path = tmp_path / "campaign.csv"
    refreshed = SimpleNamespace(config_path=config_path, log_path=log_path)
    messages: list[str] = []
    loaded_policies: list[str] = []

    class FakeStreamlit:
        session_state = {
            streamlit_app.CONFIG_PATH_KEY: str(config_path),
            streamlit_app.LOG_PATH_KEY: str(log_path),
            streamlit_app.STAGED_SUGGESTION_BUNDLE_KEY: {"suggestions": "stale"},
            "observed_objective_old": 0.7,
            "actual_cost_old": "4.2",
            streamlit_app.REPORT_PREVIEW_KEY: "stale report",
        }

        @staticmethod
        def error(message: str) -> None:
            raise AssertionError(message)

    def load_with_policy(*_args: object, **kwargs: object) -> object:
        loaded_policies.append(str(kwargs["provenance_policy"]))
        return refreshed

    monkeypatch.setattr(streamlit_app.CampaignAppService, "load", load_with_policy)
    monkeypatch.setattr(
        streamlit_app,
        "_refresh_validation_cache",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        streamlit_app,
        "_flash_and_rerun",
        lambda _st, message: messages.append(message),
    )

    handled = streamlit_app._handle_log_mutation_error(
        FakeStreamlit,
        SimpleNamespace(session=object(), provenance_policy="required"),
        LogConflictError("stale"),
    )

    assert handled is True
    assert FakeStreamlit.session_state[streamlit_app.SESSION_KEY] is refreshed
    assert streamlit_app.STAGED_SUGGESTION_BUNDLE_KEY not in FakeStreamlit.session_state
    assert streamlit_app.REPORT_PREVIEW_KEY not in FakeStreamlit.session_state
    assert "observed_objective_old" not in FakeStreamlit.session_state
    assert "actual_cost_old" not in FakeStreamlit.session_state
    assert messages == [
        "Campaign log changed in another process. The latest log was reloaded; retry the action."
    ]
    assert loaded_policies == ["required"]


def test_streamlit_recovery_conflict_preserves_policy_and_exposes_action(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "campaign.yaml"
    log_path = tmp_path / "campaign.csv"
    config_path.write_text("campaign_name: managed\n", encoding="utf-8")
    log_path.write_text("row_id\n", encoding="utf-8")
    campaign = SimpleNamespace(provenance_policy="required")
    errors: list[str] = []

    class FakeStreamlit:
        session_state = {
            streamlit_app.CONFIG_PATH_KEY: str(config_path),
            streamlit_app.LOG_PATH_KEY: str(log_path),
            streamlit_app.SESSION_KEY: campaign,
            streamlit_app.STAGED_SUGGESTION_BUNDLE_KEY: {"suggestions": "stale"},
        }

        @staticmethod
        def error(message: str) -> None:
            errors.append(message)

    recovery_error = ProvenanceRecoveryRequired(
        "recovery required",
        reason_code="pending_previous_state",
        recovery_action="Run provenance recovery.",
    )

    handled = streamlit_app._handle_log_mutation_error(
        FakeStreamlit,
        campaign,
        recovery_error,
    )

    recovery = FakeStreamlit.session_state[streamlit_app.PROVENANCE_RECOVERY_KEY]
    assert handled is True
    assert recovery["require_provenance"] is True
    assert recovery["reason_code"] == "pending_previous_state"
    assert FakeStreamlit.session_state[streamlit_app.SESSION_KEY] is campaign
    assert streamlit_app.STAGED_SUGGESTION_BUNDLE_KEY not in FakeStreamlit.session_state
    assert errors == ["recovery required"]

def test_streamlit_log_busy_keeps_retryable_staged_state() -> None:
    errors: list[str] = []

    class FakeStreamlit:
        session_state = {
            streamlit_app.STAGED_SUGGESTION_BUNDLE_KEY: {"suggestions": "keep"},
        }

        @staticmethod
        def error(message: str) -> None:
            errors.append(message)

    handled = streamlit_app._handle_log_mutation_error(
        FakeStreamlit,
        SimpleNamespace(),
        LogBusyError("busy"),
    )

    assert handled is True
    assert streamlit_app.STAGED_SUGGESTION_BUNDLE_KEY in FakeStreamlit.session_state
    assert "another process" in errors[0]

def test_streamlit_load_refreshes_source_bar_and_does_not_leak_metric_html() -> None:
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(PROJECT_ROOT / "bo_forge_app" / "streamlit_app.py")
    app.run(timeout=10)

    next(input_ for input_ in app.text_input if input_.label == "YAML config path").set_value(
        "configs/01_simple_2d_maximise_logei.yaml"
    )
    next(input_ for input_ in app.text_input if input_.label == "CSV log path").set_value(
        "examples/01_simple_2d_maximise_logei_campaign_log.csv"
    )
    next(button for button in app.button if button.label == "Load campaign").click()
    app.run(timeout=10)

    markdown_text = "\n".join(markdown.value for markdown in app.markdown)
    code_text = "\n".join(code.value for code in getattr(app, "code", []))
    assert len(app.exception) == 0
    assert "configs/01_simple_2d_maximise_logei.yaml" in markdown_text
    assert "examples/01_simple_2d_maximise_logei_campaign_log.csv" in markdown_text
    assert "Valid" in markdown_text
    assert "forge-metric" not in code_text

def test_streamlit_loaded_contextual_campaign_shows_context_inputs() -> None:
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(PROJECT_ROOT / "bo_forge_app" / "streamlit_app.py")
    app.run(timeout=10)

    next(input_ for input_ in app.text_input if input_.label == "YAML config path").set_value(
        "configs/16_contextual_logei.yaml"
    )
    next(input_ for input_ in app.text_input if input_.label == "CSV log path").set_value(
        "examples/16_contextual_logei_campaign_log.csv"
    )
    next(button for button in app.button if button.label == "Load campaign").click()
    app.run(timeout=10)

    assert any(subheader.value == "Context Summary" for subheader in app.subheader)

    next(radio for radio in app.radio if radio.label == "Workbench area").set_value("Run")
    app.run(timeout=10)

    markdown_text = "\n".join(markdown.value for markdown in app.markdown)
    number_labels = {input_.label for input_ in app.number_input}
    assert len(app.exception) == 0
    assert "Suggestion context values are used only" in markdown_text
    assert "Suggestion context: feedstock_acidity" in number_labels

    next(radio for radio in app.radio if radio.label == "Workbench area").set_value("Analyze")
    app.run(timeout=10)
    plot_select = next(selectbox for selectbox in app.selectbox if selectbox.label == "Plot kind")
    assert len(app.exception) == 0
    assert "Context Diagnostics" in list(plot_select.options)

def test_streamlit_switch_from_fidelity_campaign_clears_fidelity_tables() -> None:
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(PROJECT_ROOT / "bo_forge_app" / "streamlit_app.py")
    app.run(timeout=10)
    next(input_ for input_ in app.text_input if input_.label == "YAML config path").set_value(
        "configs/15_multi_fidelity_qmfkg.yaml"
    )
    next(input_ for input_ in app.text_input if input_.label == "CSV log path").set_value(
        "examples/15_multi_fidelity_qmfkg_campaign_log.csv"
    )
    next(button for button in app.button if button.label == "Load campaign").click()
    app.run(timeout=10)
    next(radio for radio in app.radio if radio.label == "Workbench area").set_value("Campaign")
    app.run(timeout=10)
    assert any(subheader.value == "Fidelity Coverage" for subheader in app.subheader)

    next(input_ for input_ in app.text_input if input_.label == "YAML config path").set_value(
        "configs/01_simple_2d_maximise_logei.yaml"
    )
    next(input_ for input_ in app.text_input if input_.label == "CSV log path").set_value(
        "examples/01_simple_2d_maximise_logei_campaign_log.csv"
    )
    next(button for button in app.button if button.label == "Load campaign").click()
    app.run(timeout=10)
    next(radio for radio in app.radio if radio.label == "Workbench area").set_value("Campaign")
    app.run(timeout=10)

    assert not any(subheader.value == "Fidelity Coverage" for subheader in app.subheader)
    assert len(app.exception) == 0

def test_streamlit_context_change_clears_staged_bundle_without_mutation(
    tmp_path: Path,
) -> None:
    from streamlit.testing.v1 import AppTest

    log_path = copy_example_log(tmp_path, "16_contextual_logei_campaign_log.csv")
    before_bytes = log_path.read_bytes()
    app = AppTest.from_file(PROJECT_ROOT / "bo_forge_app" / "streamlit_app.py")
    app.run(timeout=10)

    next(input_ for input_ in app.text_input if input_.label == "YAML config path").set_value(
        "configs/16_contextual_logei.yaml"
    )
    next(input_ for input_ in app.text_input if input_.label == "CSV log path").set_value(
        str(log_path)
    )
    next(button for button in app.button if button.label == "Load campaign").click()
    app.run(timeout=10)
    next(radio for radio in app.radio if radio.label == "Workbench area").set_value("Run")
    app.run(timeout=10)

    context_input = next(
        input_
        for input_ in app.number_input
        if input_.label == "Suggestion context: feedstock_acidity"
    )
    context_input.set_value(0.25)
    app.run(timeout=10)
    next(
        button for button in app.button if button.label == "Generate suggestions (dry run)"
    ).click()
    app.run(timeout=20)

    bundle = app.session_state[streamlit_app.STAGED_SUGGESTION_BUNDLE_KEY]
    assert bundle["context_values"] == {"feedstock_acidity": 0.25}

    context_input = next(
        input_
        for input_ in app.number_input
        if input_.label == "Suggestion context: feedstock_acidity"
    )
    context_input.set_value(0.75)
    app.run(timeout=10)

    assert streamlit_app.STAGED_SUGGESTION_BUNDLE_KEY not in app.session_state
    markdown_text = "\n".join(markdown.value for markdown in app.markdown)
    assert "Cleared stale staged suggestions." in markdown_text
    assert log_path.read_bytes() == before_bytes
    assert len(app.exception) == 0

def test_streamlit_context_inputs_reset_after_campaign_switch(tmp_path: Path) -> None:
    from streamlit.testing.v1 import AppTest

    first_log_path = copy_example_log(tmp_path, "16_contextual_logei_campaign_log.csv")
    second_config_path = tmp_path / "contextual_second.yaml"
    second_log_path = tmp_path / "contextual_second.csv"
    second_config_path.write_text(
        """
campaign_name: contextual_second
objective:
  name: yield_score
  direction: maximize
variables:
  - name: catalyst_loading
    type: continuous
    lower: 0.0
    upper: 1.0
  - name: feedstock_acidity
    type: continuous
    lower: 0.0
    upper: 1.0
context:
  variables: [feedstock_acidity]
  default_values:
    feedstock_acidity: 0.8
bo:
  batch_size: 1
  initial_design_size: 2
  acquisition: log_ei
  initial_design_method: sobol
  random_seed: 7
""",
        encoding="utf-8",
    )
    second_config = CampaignConfig.from_yaml(second_config_path)
    empty_campaign_log(second_config).to_csv(second_log_path, index=False)

    app = AppTest.from_file(PROJECT_ROOT / "bo_forge_app" / "streamlit_app.py")
    app.run(timeout=10)
    next(input_ for input_ in app.text_input if input_.label == "YAML config path").set_value(
        "configs/16_contextual_logei.yaml"
    )
    next(input_ for input_ in app.text_input if input_.label == "CSV log path").set_value(
        str(first_log_path)
    )
    next(button for button in app.button if button.label == "Load campaign").click()
    app.run(timeout=10)
    next(radio for radio in app.radio if radio.label == "Workbench area").set_value("Run")
    app.run(timeout=10)
    next(
        input_
        for input_ in app.number_input
        if input_.label == "Suggestion context: feedstock_acidity"
    ).set_value(0.25)
    app.run(timeout=10)

    next(input_ for input_ in app.text_input if input_.label == "YAML config path").set_value(
        str(second_config_path)
    )
    next(input_ for input_ in app.text_input if input_.label == "CSV log path").set_value(
        str(second_log_path)
    )
    next(button for button in app.button if button.label == "Load campaign").click()
    app.run(timeout=10)
    next(radio for radio in app.radio if radio.label == "Workbench area").set_value("Run")
    app.run(timeout=10)

    context_input = next(
        input_
        for input_ in app.number_input
        if input_.label == "Suggestion context: feedstock_acidity"
    )
    assert context_input.value == pytest.approx(0.8)
    assert len(app.exception) == 0

def test_streamlit_contextual_bundle_clears_when_loading_non_contextual_campaign(
    tmp_path: Path,
) -> None:
    from streamlit.testing.v1 import AppTest

    contextual_log_path = copy_example_log(tmp_path, "16_contextual_logei_campaign_log.csv")
    before_bytes = contextual_log_path.read_bytes()
    app = AppTest.from_file(PROJECT_ROOT / "bo_forge_app" / "streamlit_app.py")
    app.run(timeout=10)

    next(input_ for input_ in app.text_input if input_.label == "YAML config path").set_value(
        "configs/16_contextual_logei.yaml"
    )
    next(input_ for input_ in app.text_input if input_.label == "CSV log path").set_value(
        str(contextual_log_path)
    )
    next(button for button in app.button if button.label == "Load campaign").click()
    app.run(timeout=10)
    next(radio for radio in app.radio if radio.label == "Workbench area").set_value("Run")
    app.run(timeout=10)
    next(
        button for button in app.button if button.label == "Generate suggestions (dry run)"
    ).click()
    app.run(timeout=20)

    assert streamlit_app.STAGED_SUGGESTION_BUNDLE_KEY in app.session_state

    next(input_ for input_ in app.text_input if input_.label == "YAML config path").set_value(
        "configs/01_simple_2d_maximise_logei.yaml"
    )
    next(input_ for input_ in app.text_input if input_.label == "CSV log path").set_value(
        "examples/01_simple_2d_maximise_logei_campaign_log.csv"
    )
    next(button for button in app.button if button.label == "Load campaign").click()
    app.run(timeout=10)

    assert streamlit_app.STAGED_SUGGESTION_BUNDLE_KEY not in app.session_state
    assert contextual_log_path.read_bytes() == before_bytes
    assert len(app.exception) == 0

def test_streamlit_loads_cost_aware_multi_objective_reports_panel() -> None:
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(PROJECT_ROOT / "bo_forge_app" / "streamlit_app.py")
    app.run(timeout=10)

    next(input_ for input_ in app.text_input if input_.label == "YAML config path").set_value(
        "configs/12_cost_aware_multi_objective_qlogehvi.yaml"
    )
    next(input_ for input_ in app.text_input if input_.label == "CSV log path").set_value(
        "examples/12_cost_aware_multi_objective_campaign_log.csv"
    )
    next(button for button in app.button if button.label == "Load campaign").click()
    app.run(timeout=10)
    next(radio for radio in app.radio if radio.label == "Workbench area").set_value("Analyze")
    app.run(timeout=10)

    assert len(app.exception) == 0
    assert any(selectbox.label == "Plot kind" for selectbox in app.selectbox)

def test_streamlit_structured_stage_change_clears_staged_bundle(tmp_path: Path) -> None:
    from streamlit.testing.v1 import AppTest

    log_path = tmp_path / "structured.csv"
    pd.read_csv(
        "examples/13_structured_campaign_core_campaign_log.csv",
        keep_default_na=False,
    ).query("status == 'observed'").to_csv(log_path, index=False)
    before_bytes = log_path.read_bytes()

    app = AppTest.from_file(PROJECT_ROOT / "bo_forge_app" / "streamlit_app.py")
    app.run(timeout=10)

    next(input_ for input_ in app.text_input if input_.label == "YAML config path").set_value(
        "configs/13_structured_campaign_core.yaml"
    )
    next(input_ for input_ in app.text_input if input_.label == "CSV log path").set_value(
        str(log_path)
    )
    next(button for button in app.button if button.label == "Load campaign").click()
    app.run(timeout=10)
    next(radio for radio in app.radio if radio.label == "Workbench area").set_value("Run")
    app.run(timeout=10)

    stage_select = next(
        selectbox for selectbox in app.selectbox if selectbox.label == "Suggestion stage"
    )
    assert stage_select.value == "screen"
    next(
        button for button in app.button if button.label == "Generate suggestions (dry run)"
    ).click()
    app.run(timeout=20)

    bundle = app.session_state[streamlit_app.STAGED_SUGGESTION_BUNDLE_KEY]
    assert bundle["stage"] == "screen"
    stage_select = next(
        selectbox for selectbox in app.selectbox if selectbox.label == "Suggestion stage"
    )
    stage_select.set_value("refine")
    app.run(timeout=10)

    assert streamlit_app.STAGED_SUGGESTION_BUNDLE_KEY not in app.session_state
    markdown_text = "\n".join(markdown.value for markdown in app.markdown)
    assert "Cleared stale staged suggestions." in markdown_text
    assert log_path.read_bytes() == before_bytes

    next(radio for radio in app.radio if radio.label == "Workbench area").set_value("Analyze")
    app.run(timeout=10)
    plot_select = next(selectbox for selectbox in app.selectbox if selectbox.label == "Plot kind")
    assert "Stage Diagnostics" in list(plot_select.options)
    assert len(app.exception) == 0

def test_streamlit_app_can_create_minimal_campaign(tmp_path: Path) -> None:
    from streamlit.testing.v1 import AppTest

    config_path = tmp_path / "configs" / "campaign.yaml"
    log_path = tmp_path / "logs" / "campaign.csv"
    app = AppTest.from_file(PROJECT_ROOT / "bo_forge_app" / "streamlit_app.py")
    app.run(timeout=10)

    app.radio[0].set_value("Create Campaign")
    app.run(timeout=10)

    next(selectbox for selectbox in app.selectbox if selectbox.label == "Model profile").set_value(
        "smooth"
    )
    app.run(timeout=10)
    next(button for button in app.button if button.label == "Update YAML preview from form").click()
    app.run(timeout=10)
    app.text_input[1].set_value(str(config_path))
    app.text_input[2].set_value(str(log_path))
    create_button = next(button for button in app.button if button.label == "Create campaign")
    create_button.click()
    app.run(timeout=10)

    assert len(app.exception) == 0
    assert config_path.exists()
    assert log_path.exists()
    assert log_path.with_name("campaign.csv.manifest.json").exists()
    config = CampaignConfig.from_yaml(config_path)
    assert config.model.profile == "smooth"
    markdown_text = "\n".join(markdown.value for markdown in app.markdown)
    success_text = "\n".join(success.value for success in app.success)
    assert str(config_path) in markdown_text
    assert str(log_path) in markdown_text
    assert "Valid" in markdown_text
    assert "Campaign created and loaded" in success_text
    assert any(subheader.value == "Model Summary" for subheader in app.subheader)
    assert any(subheader.value == "Provenance" for subheader in app.subheader)

def test_streamlit_app_can_create_qlog_nei_campaign(tmp_path: Path) -> None:
    from streamlit.testing.v1 import AppTest

    config_path = tmp_path / "configs" / "qlog_nei.yaml"
    log_path = tmp_path / "logs" / "qlog_nei.csv"
    app = AppTest.from_file(PROJECT_ROOT / "bo_forge_app" / "streamlit_app.py")
    app.run(timeout=10)

    next(radio for radio in app.radio if radio.label == "Campaign file action").set_value(
        "Create Campaign"
    )
    app.run(timeout=10)
    next(radio for radio in app.radio if radio.label == "Campaign kind").set_value(
        "Single-objective qLogNEI"
    )
    app.run(timeout=10)

    next(
        input_
        for input_ in app.text_input
        if input_.label == "New YAML config output path"
    ).set_value(str(config_path))
    next(
        input_
        for input_ in app.text_input
        if input_.label == "New CSV log output path"
    ).set_value(str(log_path))
    next(button for button in app.button if button.label == "Update YAML preview from form").click()
    app.run(timeout=10)
    next(button for button in app.button if button.label == "Create campaign").click()
    app.run(timeout=10)

    config = CampaignConfig.from_yaml(config_path)
    assert len(app.exception) == 0
    assert config.bo.acquisition == "qlog_nei"
    assert config.review.enabled
    assert list(pd.read_csv(log_path, keep_default_na=False).columns) == canonical_columns(config)
    assert any(subheader.value == "qLogNEI Summary" for subheader in app.subheader)

    next(radio for radio in app.radio if radio.label == "Workbench area").set_value("Run")
    app.run(timeout=10)
    markdown_text = "\n".join(markdown.value for markdown in app.markdown)
    assert "qLogNEI pending semantics" in markdown_text

def test_streamlit_app_can_create_multi_fidelity_qmfkg_campaign(tmp_path: Path) -> None:
    from streamlit.testing.v1 import AppTest

    config_path = tmp_path / "configs" / "fidelity.yaml"
    log_path = tmp_path / "logs" / "fidelity.csv"
    app = AppTest.from_file(PROJECT_ROOT / "bo_forge_app" / "streamlit_app.py")
    app.run(timeout=10)

    next(radio for radio in app.radio if radio.label == "Campaign file action").set_value(
        "Create Campaign"
    )
    app.run(timeout=10)
    next(radio for radio in app.radio if radio.label == "Campaign kind").set_value(
        "Multi-fidelity qMFKG"
    )
    app.run(timeout=10)
    next(
        input_
        for input_ in app.number_input
        if input_.key == "new_bo_batch_size_multi_fidelity"
    ).set_value(2)
    next(radio for radio in app.radio if radio.label == "Fidelity mode").set_value(
        "Ordered discrete levels"
    )
    app.run(timeout=10)
    next(
        input_
        for input_ in app.number_input
        if input_.label == "Max optimizer iterations"
    ).set_value(85)
    next(
        checkbox
        for checkbox in app.checkbox
        if checkbox.label == "Limit acquisition runtime"
    ).check()
    app.run(timeout=10)
    next(
        input_
        for input_ in app.number_input
        if input_.label == "Acquisition timeout (seconds)"
    ).set_value(12.0)
    next(checkbox for checkbox in app.checkbox if checkbox.label == "Enable review").check()
    app.run(timeout=10)

    next(
        input_
        for input_ in app.text_input
        if input_.label == "New YAML config output path"
    ).set_value(str(config_path))
    next(
        input_
        for input_ in app.text_input
        if input_.label == "New CSV log output path"
    ).set_value(str(log_path))
    next(button for button in app.button if button.label == "Update YAML preview from form").click()
    app.run(timeout=10)
    next(button for button in app.button if button.label == "Create campaign").click()
    app.run(timeout=10)

    config = CampaignConfig.from_yaml(config_path)
    assert len(app.exception) == 0
    assert config.fidelity is not None
    assert config.fidelity.variable == "fidelity"
    assert config.fidelity.target == pytest.approx(1.0)
    assert config.fidelity.levels == (0.0, 0.5, 1.0)
    assert config.fidelity.optimizer_maxiter == 85
    assert config.fidelity.optimizer_timeout_seconds == pytest.approx(12.0)
    assert config.bo.acquisition == "qmf_kg"
    assert config.bo.batch_size == 2
    assert config.review.enabled
    assert list(pd.read_csv(log_path, keep_default_na=False).columns) == canonical_columns(config)

    assert any(subheader.value == "Fidelity Summary" for subheader in app.subheader)

    next(radio for radio in app.radio if radio.label == "Workbench area").set_value("Campaign")
    app.run(timeout=10)
    assert any(subheader.value == "Fidelity Coverage" for subheader in app.subheader)
    assert any(subheader.value == "Fidelity Summary" for subheader in app.subheader)

    next(radio for radio in app.radio if radio.label == "Workbench area").set_value("Analyze")
    app.run(timeout=10)
    assert any(subheader.value == "Fidelity Coverage" for subheader in app.subheader)
    plot_select = next(selectbox for selectbox in app.selectbox if selectbox.label == "Plot kind")
    assert "Fidelity Diagnostics" in list(plot_select.options)
    assert "Fidelity Progress" in list(plot_select.options)

    next(radio for radio in app.radio if radio.label == "Workbench area").set_value("Run")
    app.run(timeout=10)
    suggest_markdown = "\n".join(markdown.value for markdown in app.markdown)
    batch_inputs = [
        number_input
        for number_input in app.number_input
        if number_input.label == "Batch size"
    ]
    assert "qMFKG suggestions" in suggest_markdown
    assert "Optimizer max iterations: 85" in suggest_markdown
    assert "Acquisition timeout: 12 seconds" in suggest_markdown
    assert batch_inputs[-1].value == 2
    assert batch_inputs[-1].min == 1
    assert batch_inputs[-1].max == 4
    assert not batch_inputs[-1].proto.disabled

def test_streamlit_app_can_create_contextual_logei_campaign(tmp_path: Path) -> None:
    from streamlit.testing.v1 import AppTest

    config_path = tmp_path / "configs" / "contextual.yaml"
    log_path = tmp_path / "logs" / "contextual.csv"
    app = AppTest.from_file(PROJECT_ROOT / "bo_forge_app" / "streamlit_app.py")
    app.run(timeout=10)

    next(radio for radio in app.radio if radio.label == "Campaign file action").set_value(
        "Create Campaign"
    )
    app.run(timeout=10)
    next(radio for radio in app.radio if radio.label == "Campaign kind").set_value(
        "Contextual LogEI"
    )
    app.run(timeout=10)

    assert any(input_.label == "Default context: x2" for input_ in app.number_input)

    next(
        input_
        for input_ in app.text_input
        if input_.label == "New YAML config output path"
    ).set_value(str(config_path))
    next(
        input_
        for input_ in app.text_input
        if input_.label == "New CSV log output path"
    ).set_value(str(log_path))
    next(button for button in app.button if button.label == "Update YAML preview from form").click()
    app.run(timeout=10)
    next(button for button in app.button if button.label == "Create campaign").click()
    app.run(timeout=10)

    config = CampaignConfig.from_yaml(config_path)
    assert len(app.exception) == 0
    assert config.context is not None
    assert config.context_variable_names == ["x2"]
    assert config.context.default_values == {"x2": 0.0}
    assert config.bo.acquisition == "log_ei"
    assert list(pd.read_csv(log_path, keep_default_na=False).columns) == canonical_columns(config)
    assert any(subheader.value == "Context Summary" for subheader in app.subheader)

    next(radio for radio in app.radio if radio.label == "Workbench area").set_value("Run")
    app.run(timeout=10)
    markdown_text = "\n".join(markdown.value for markdown in app.markdown)
    assert "Suggestion context values are used only" in markdown_text
    assert any(input_.label == "Suggestion context: x2" for input_ in app.number_input)

    next(
        button for button in app.button if button.label == "Generate suggestions (dry run)"
    ).click()
    app.run(timeout=20)

    bundle = app.session_state[streamlit_app.STAGED_SUGGESTION_BUNDLE_KEY]
    suggestions = bundle["suggestions"]
    assert bundle["context_values"] == {"x2": 0.0}
    assert suggestions["x2"].astype(float).tolist() == [pytest.approx(0.0)]

    next(button for button in app.button if button.label == "Append staged suggestions").click()
    app.run(timeout=10)
    assert streamlit_app.STAGED_SUGGESTION_BUNDLE_KEY not in app.session_state
    appended = pd.read_csv(log_path, keep_default_na=False)
    assert appended["status"].tolist() == ["suggested"]
    assert appended["x2"].astype(float).tolist() == [pytest.approx(0.0)]

    app = AppTest.from_file(PROJECT_ROOT / "bo_forge_app" / "streamlit_app.py")
    app.run(timeout=10)
    next(input_ for input_ in app.text_input if input_.label == "YAML config path").set_value(
        str(config_path)
    )
    next(input_ for input_ in app.text_input if input_.label == "CSV log path").set_value(
        str(log_path)
    )
    next(button for button in app.button if button.label == "Load campaign").click()
    app.run(timeout=10)
    next(radio for radio in app.radio if radio.label == "Workbench area").set_value("Run")
    app.run(timeout=10)
    next(input_ for input_ in app.number_input if input_.label == "Observed activity").set_value(
        0.42
    )
    next(button for button in app.button if button.label == "Mark row observed").click()
    app.run(timeout=10)

    reloaded = CampaignSession.from_files(config_path, log_path)
    reloaded.validate()
    observed = reloaded.observed_data()
    assert observed["status"].tolist() == ["observed"]
    assert observed["activity"].astype(float).tolist() == [pytest.approx(0.42)]
    assert observed["x2"].astype(float).tolist() == [pytest.approx(0.0)]

def test_streamlit_app_can_create_contextual_review_cost_campaign(tmp_path: Path) -> None:
    from streamlit.testing.v1 import AppTest

    config_path = tmp_path / "configs" / "contextual_cost_review.yaml"
    log_path = tmp_path / "logs" / "contextual_cost_review.csv"
    app = AppTest.from_file(PROJECT_ROOT / "bo_forge_app" / "streamlit_app.py")
    app.run(timeout=10)

    next(radio for radio in app.radio if radio.label == "Campaign file action").set_value(
        "Create Campaign"
    )
    app.run(timeout=10)
    next(radio for radio in app.radio if radio.label == "Campaign kind").set_value(
        "Contextual LogEI"
    )
    app.run(timeout=10)

    next(
        checkbox
        for checkbox in app.checkbox
        if checkbox.key == "new_campaign_review_enabled_contextual"
    ).check()
    next(
        checkbox
        for checkbox in app.checkbox
        if checkbox.key == "new_campaign_cost_enabled_contextual"
    ).check()
    app.run(timeout=10)
    next(
        input_
        for input_ in app.text_input
        if input_.key == "new_campaign_contextual_cost_expression"
    ).set_value("1.0 + x2")
    next(
        input_
        for input_ in app.text_input
        if input_.label == "New YAML config output path"
    ).set_value(str(config_path))
    next(
        input_
        for input_ in app.text_input
        if input_.label == "New CSV log output path"
    ).set_value(str(log_path))
    next(button for button in app.button if button.label == "Update YAML preview from form").click()
    app.run(timeout=10)
    next(button for button in app.button if button.label == "Create campaign").click()
    app.run(timeout=10)

    config = CampaignConfig.from_yaml(config_path)
    assert len(app.exception) == 0
    assert config.context is not None
    assert config.review.enabled
    assert config.cost is not None
    assert config.cost.expression == "1.0 + x2"
    assert list(pd.read_csv(log_path, keep_default_na=False).columns) == canonical_columns(config)
    assert any(subheader.value == "Context Summary" for subheader in app.subheader)

    next(radio for radio in app.radio if radio.label == "Workbench area").set_value("Campaign")
    app.run(timeout=10)
    assert any(subheader.value == "Cost Summary" for subheader in app.subheader)
