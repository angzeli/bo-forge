import shutil
from pathlib import Path

import pandas as pd
import pytest
from matplotlib import pyplot as plt

from bo_forge.config import CampaignConfig
from bo_forge.validation import canonical_columns
from bo_forge_app import streamlit_app, streamlit_helpers, streamlit_style
from bo_forge_app.streamlit_helpers import (
    append_disabled_reason,
    available_plot_kinds,
    build_campaign_yaml_text,
    campaign_report_text,
    compact_dataframe,
    create_campaign_files,
    dataframe_fingerprint,
    default_export_path,
    default_new_campaign_paths,
    drop_all_blank_columns,
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
    resolve_path_input,
    select_display_columns,
    staged_bundle_invalidation_reason,
    staged_bundle_is_appendable,
    staged_suggestions_from_bundle,
    status_tone,
)
from bo_forge_app.streamlit_style import FORGE_SUITE_CSS, forge_action_label, forge_status_label


def copy_example_log(tmp_path: Path, name: str) -> Path:
    source = Path("examples") / name
    destination = tmp_path / name
    shutil.copyfile(source, destination)
    return destination


def simple_suggestions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "row_id": "suggested_1",
                "iteration": 1,
                "status": "suggested",
                "source": "sobol",
                "x": 0.5,
                "y": "",
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            }
        ]
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
    )

    config = parse_campaign_config_text(text)

    assert config.campaign_name == "app_created_campaign"
    assert config.objective.name == "yield"
    assert config.variable_names == ["temperature", "solvent"]
    assert config.bo.batch_size == 2


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

    def fail_log_write(path: Path, df: pd.DataFrame) -> None:
        raise OSError("no log write")

    monkeypatch.setattr(streamlit_helpers, "_write_dataframe_no_overwrite", fail_log_write)

    with pytest.raises(OSError, match="no log write"):
        create_campaign_files(
            config_text=config_text,
            config_path=config_path,
            log_path=log_path,
        )

    assert not config_path.exists()
    assert not log_path.exists()


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
    assert "Campaign created and loaded" in "\n".join(FakeStreamlit.markdown_messages)


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

    assert available_plot_kinds(plain) == ["progress", "diagnostics"]
    assert available_plot_kinds(cost) == ["progress", "diagnostics", "cost_progress"]
    assert available_plot_kinds(replicate) == ["progress", "diagnostics", "replicates"]


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
    assert hasattr(streamlit_app, "_render_campaign_files_panel")
    assert hasattr(streamlit_app, "_render_create_new_campaign")
    assert hasattr(streamlit_app, "_render_campaign_state_blocks")


def test_workbench_header_uses_bo_brand_mark() -> None:
    class FakeStreamlit:
        markdown_calls: list[str] = []

        @classmethod
        def markdown(cls, body: str, unsafe_allow_html: bool = False) -> None:
            cls.markdown_calls.append(body)
            assert unsafe_allow_html is True

    streamlit_app._render_workbench_header(FakeStreamlit, campaign_loaded=True)

    assert 'class="bf-brand-mark">BO</div>' in "\n".join(FakeStreamlit.markdown_calls)
    assert "Campaign loaded" in "\n".join(FakeStreamlit.markdown_calls)


def test_forge_suite_css_contains_expected_palette_tokens() -> None:
    assert "#9f4f32" in FORGE_SUITE_CSS
    assert "#d6a84f" in FORGE_SUITE_CSS
    assert "#7f9a7a" in FORGE_SUITE_CSS
    assert "bf-workbench-header" in FORGE_SUITE_CSS
    assert "bf-file-panel" in FORGE_SUITE_CSS
    assert "--forge-paper" in FORGE_SUITE_CSS
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


def test_streamlit_app_smoke_runs_without_exceptions() -> None:
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file("bo_forge_app/streamlit_app.py")
    app.run(timeout=10)

    assert len(app.exception) == 0
    assert [tab.label for tab in app.tabs[:2]] == ["Load Existing", "Create Campaign"]
    assert len(app.text_area) >= 1


def test_streamlit_app_can_create_minimal_campaign(tmp_path: Path) -> None:
    from streamlit.testing.v1 import AppTest

    config_path = tmp_path / "configs" / "campaign.yaml"
    log_path = tmp_path / "logs" / "campaign.csv"
    app = AppTest.from_file("bo_forge_app/streamlit_app.py")
    app.run(timeout=10)

    app.text_input[3].set_value(str(config_path))
    app.text_input[4].set_value(str(log_path))
    create_button = next(button for button in app.button if button.label == "Create campaign")
    create_button.click()
    app.run(timeout=10)

    assert len(app.exception) == 0
    assert config_path.exists()
    assert log_path.exists()
    assert any("Campaign created and loaded" in markdown.value for markdown in app.markdown)
