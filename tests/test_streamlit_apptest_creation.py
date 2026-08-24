"""Streamlit AppTest campaign-creation and advanced-form tests."""

from tests._streamlit_support import (
    PROJECT_ROOT,
    CampaignConfig,
    CampaignSession,
    Path,
    canonical_columns,
    copy_example_log,
    pd,
    pytest,
    streamlit_app,
    suggestions_module,
    torch,
)


def test_streamlit_app_can_create_contextual_replicate_campaign(tmp_path: Path) -> None:
    from streamlit.testing.v1 import AppTest

    config_path = tmp_path / "configs" / "contextual_replicates.yaml"
    log_path = tmp_path / "logs" / "contextual_replicates.csv"
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
        if checkbox.key == "new_campaign_replicates_enabled_contextual"
    ).check()
    app.run(timeout=10)
    next(
        selectbox
        for selectbox in app.selectbox
        if selectbox.label == "Replicate suggestion policy"
    ).set_value("new_only")
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
    assert config.replicates.enabled
    assert config.replicates.suggestion_policy == "new_only"
    assert list(pd.read_csv(log_path, keep_default_na=False).columns) == canonical_columns(config)
    assert any(subheader.value == "Context Summary" for subheader in app.subheader)
    assert any(subheader.value == "Replicate Summary" for subheader in app.subheader)

def test_streamlit_loaded_contextual_replicate_campaign_stages_context_matched_repeat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from streamlit.testing.v1 import AppTest

    log_path = copy_example_log(tmp_path, "21_contextual_replicate_campaign_log.csv")
    before = log_path.read_bytes()

    class FakePosterior:
        mean = torch.tensor([[2.0], [1.0], [10.0], [0.0]], dtype=torch.double)
        variance = torch.full((4, 1), 0.04, dtype=torch.double)

    class FakeModel:
        def posterior(self, _x):
            return FakePosterior()

    monkeypatch.setattr(suggestions_module, "fit_gp_model", lambda *_args: FakeModel())
    app = AppTest.from_file(PROJECT_ROOT / "bo_forge_app" / "streamlit_app.py")
    app.run(timeout=10)
    next(input_ for input_ in app.text_input if input_.label == "YAML config path").set_value(
        "configs/21_contextual_replicate_logei.yaml"
    )
    next(input_ for input_ in app.text_input if input_.label == "CSV log path").set_value(
        str(log_path)
    )
    next(button for button in app.button if button.label == "Load campaign").click()
    app.run(timeout=10)

    assert any(subheader.value == "Context Summary" for subheader in app.subheader)
    assert any(subheader.value == "Replicate Summary" for subheader in app.subheader)
    next(radio for radio in app.radio if radio.label == "Workbench area").set_value("Campaign")
    app.run(timeout=10)
    assert any(subheader.value == "Cost Summary" for subheader in app.subheader)
    next(radio for radio in app.radio if radio.label == "Workbench area").set_value("Run")
    app.run(timeout=10)
    next(input_ for input_ in app.number_input if input_.label == "Batch size").set_value(1)
    app.run(timeout=10)
    next(
        button for button in app.button if button.label == "Generate suggestions (dry run)"
    ).click()
    app.run(timeout=20)

    bundle = app.session_state[streamlit_app.STAGED_SUGGESTION_BUNDLE_KEY]
    suggestions = bundle["suggestions"]
    assert bundle["context_values"] == {"feedstock_acidity": 0.25}
    assert suggestions.loc[0, "replicate_group"] == "group_acid25_best"
    assert int(suggestions.loc[0, "replicate_index"]) == 2
    assert log_path.read_bytes() == before

    context_input = next(
        input_
        for input_ in app.number_input
        if input_.label == "Suggestion context: feedstock_acidity"
    )
    context_input.set_value(0.75)
    app.run(timeout=10)
    assert streamlit_app.STAGED_SUGGESTION_BUNDLE_KEY not in app.session_state
    assert log_path.read_bytes() == before
    assert len(app.exception) == 0

    next(radio for radio in app.radio if radio.label == "Workbench area").set_value("Analyze")
    app.run(timeout=10)
    plot_select = next(selectbox for selectbox in app.selectbox if selectbox.label == "Plot kind")
    assert "Context Diagnostics" in list(plot_select.options)
    assert "Replicates" in list(plot_select.options)
    assert "Cost Progress" in list(plot_select.options)

def test_streamlit_contextual_review_cost_suggest_review_observe_round_trip(
    tmp_path: Path,
) -> None:
    from streamlit.testing.v1 import AppTest

    config_path = tmp_path / "configs" / "contextual_cost_review_round_trip.yaml"
    log_path = tmp_path / "logs" / "contextual_cost_review_round_trip.csv"
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

    next(radio for radio in app.radio if radio.label == "Workbench area").set_value("Run")
    app.run(timeout=10)
    next(
        button for button in app.button if button.label == "Generate suggestions (dry run)"
    ).click()
    app.run(timeout=20)
    bundle = app.session_state[streamlit_app.STAGED_SUGGESTION_BUNDLE_KEY]
    assert bundle["context_values"] == {"x2": 0.0}
    suggest_markdown = "\n".join(markdown.value for markdown in app.markdown)
    assert "Context: x2" in suggest_markdown
    assert "Remaining budget" in suggest_markdown
    assert "Staged estimated cost" in suggest_markdown
    assert "Review state" in suggest_markdown

    next(button for button in app.button if button.label == "Append staged suggestions").click()
    app.run(timeout=10)
    appended = pd.read_csv(log_path, keep_default_na=False)
    row_id = str(appended.loc[appended["status"] == "suggested", "row_id"].iloc[0])
    assert appended.loc[appended["row_id"] == row_id, "review_status"].iloc[0] == "pending"

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
    resolve_markdown = "\n".join(markdown.value for markdown in app.markdown)
    assert "Remaining" in resolve_markdown
    next(button for button in app.button if button.label == "Apply review decision").click()
    app.run(timeout=10)
    observable_markdown = "\n".join(markdown.value for markdown in app.markdown)
    assert "Review state" in observable_markdown
    assert "Estimated cost" in observable_markdown

    next(
        input_ for input_ in app.number_input if input_.label == "Observed activity"
    ).set_value(0.42)
    next(
        input_ for input_ in app.text_input if input_.label == "Actual cost (optional)"
    ).set_value("2.5")
    next(button for button in app.button if button.label == "Mark row observed").click()
    app.run(timeout=10)

    reloaded = CampaignSession.from_files(config_path, log_path)
    reloaded.validate()
    observed = reloaded.observed_data()
    assert len(app.exception) == 0
    assert observed["row_id"].tolist() == [row_id]
    assert observed["review_status"].tolist() == ["accepted"]
    assert observed["activity"].astype(float).tolist() == [pytest.approx(0.42)]
    assert observed["cost_actual"].astype(float).tolist() == [pytest.approx(2.5)]
    assert observed["x2"].astype(float).tolist() == [pytest.approx(0.0)]

def test_streamlit_multi_fidelity_target_defaults_to_selected_variable_upper(
    tmp_path: Path,
) -> None:
    from streamlit.testing.v1 import AppTest

    config_path = tmp_path / "configs" / "fidelity_alt.yaml"
    log_path = tmp_path / "logs" / "fidelity_alt.csv"
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
        for input_ in app.text_input
        if input_.key == "new_fidelity_variable_0_name"
    ).set_value("temperature")
    next(
        input_
        for input_ in app.text_input
        if input_.key == "new_fidelity_variable_1_name"
    ).set_value("loading")
    next(
        input_
        for input_ in app.number_input
        if input_.key == "new_fidelity_variable_0_upper"
    ).set_value(2.5)
    app.run(timeout=10)
    next(
        selectbox
        for selectbox in app.selectbox
        if selectbox.label == "Fidelity variable"
    ).set_value("temperature")
    app.run(timeout=10)

    target_input = next(
        input_ for input_ in app.number_input if input_.label == "Target fidelity"
    )
    assert target_input.value == pytest.approx(2.5)
    assert target_input.max == pytest.approx(2.5)

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
    assert config.fidelity.variable == "temperature"
    assert config.fidelity.target == pytest.approx(2.5)
    assert config.fidelity.optimizer_maxiter == 200
    assert config.fidelity.optimizer_timeout_seconds is None
    assert "optimizer_timeout_seconds" not in config_path.read_text(encoding="utf-8")
    assert list(pd.read_csv(log_path, keep_default_na=False).columns) == canonical_columns(config)

def test_streamlit_advanced_create_hides_single_objective_fields() -> None:
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(PROJECT_ROOT / "bo_forge_app" / "streamlit_app.py")
    app.run(timeout=10)
    next(radio for radio in app.radio if radio.label == "Campaign file action").set_value(
        "Create Campaign"
    )
    app.run(timeout=10)
    next(radio for radio in app.radio if radio.label == "Campaign kind").set_value(
        "Multi-objective"
    )
    app.run(timeout=10)

    text_labels = {input_.label for input_ in app.text_input}
    checkbox_labels = {checkbox.label for checkbox in app.checkbox}
    assert "Objective name" not in text_labels
    assert "Objective 1 name" in text_labels
    assert "Advanced multi-objective campaign" not in checkbox_labels

def test_streamlit_create_blocks_stale_yaml_preview(tmp_path: Path) -> None:
    from streamlit.testing.v1 import AppTest

    config_path = tmp_path / "configs" / "campaign.yaml"
    log_path = tmp_path / "logs" / "campaign.csv"
    app = AppTest.from_file(PROJECT_ROOT / "bo_forge_app" / "streamlit_app.py")
    app.run(timeout=10)
    app.radio[0].set_value("Create Campaign")
    app.run(timeout=10)

    next(input_ for input_ in app.text_input if input_.label == "New campaign name").set_value(
        "renamed_campaign"
    )
    config_input = next(
        input_
        for input_ in app.text_input
        if input_.label == "New YAML config output path"
    )
    log_input = next(
        input_
        for input_ in app.text_input
        if input_.label == "New CSV log output path"
    )
    config_input.set_value(str(config_path))
    log_input.set_value(str(log_path))
    app.run(timeout=10)
    next(button for button in app.button if button.label == "Create campaign").click()
    app.run(timeout=10)

    assert len(app.exception) == 0
    assert not config_path.exists()
    assert not log_path.exists()
    assert any("Update YAML preview from form" in error.value for error in app.error)
