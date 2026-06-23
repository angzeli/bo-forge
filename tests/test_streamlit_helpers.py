import shutil
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from matplotlib import pyplot as plt

from bo_forge.config import BOConfig, CampaignConfig, CostConfig, ObjectiveConfig, VariableConfig
from bo_forge.io import empty_campaign_log
from bo_forge.session import CampaignSession
from bo_forge.validation import canonical_columns
from bo_forge_app import streamlit_app, streamlit_helpers, streamlit_style
from bo_forge_app.streamlit_helpers import (
    active_variables_display,
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
    structured_stage_config_table,
    structured_stage_options,
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
    )

    config = parse_campaign_config_text(text)

    assert config.campaign_name == "app_created_campaign"
    assert config.objective.name == "yield"
    assert config.variable_names == ["temperature", "solvent"]
    assert config.bo.batch_size == 2


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
    assert config.bo.acquisition == "qmf_kg"
    assert config.bo.batch_size == 1
    assert config.bo.raw_samples == 8
    assert config.bo.num_restarts == 1
    assert config.bo.mc_samples == 16
    assert config.bo.min_normalized_distance == pytest.approx(0.0)
    assert config.review.enabled
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

    assert available_plot_kinds(plain) == ["progress", "diagnostics"]
    assert available_plot_kinds(cost) == ["progress", "diagnostics", "cost_progress"]
    assert available_plot_kinds(replicate) == ["progress", "diagnostics", "replicates"]
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
    ]


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
    assert available_plot_kinds(config) == ["progress", "diagnostics"]


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
        "_render_overview",
        lambda *_args, **_kwargs: calls.append("Overview"),
    )
    monkeypatch.setattr(
        streamlit_app,
        "_render_suggest",
        lambda *_args, **_kwargs: calls.append("Suggest"),
    )
    monkeypatch.setattr(
        streamlit_app,
        "_render_resolve",
        lambda *_args, **_kwargs: calls.append("Resolve"),
    )
    monkeypatch.setattr(
        streamlit_app,
        "_render_reports",
        lambda *_args, **_kwargs: calls.append("Reports"),
    )
    monkeypatch.setattr(
        streamlit_app,
        "_render_data",
        lambda *_args, **_kwargs: calls.append("Data"),
    )

    streamlit_app._render_active_workflow_panel(object(), FakeCampaign(), {}, "Resolve")

    assert calls == ["collect:Resolve", "Resolve"]


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
    first_yield_key = streamlit_app._stable_widget_key(
        "observed_objective",
        "suggested_1",
        "yield_score",
    )
    first_waste_key = streamlit_app._stable_widget_key(
        "observed_objective",
        "suggested_1",
        "waste_score",
    )
    first_cost_key = streamlit_app._stable_widget_key("actual_cost", "suggested_1")

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

    app = AppTest.from_file("bo_forge_app/streamlit_app.py")
    app.run(timeout=10)

    assert len(app.exception) == 0
    assert any(radio.label == "Campaign file action" for radio in app.radio)
    assert any("Nothing loaded yet" in markdown.value for markdown in app.markdown)


def test_streamlit_load_refreshes_source_bar_and_does_not_leak_metric_html() -> None:
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file("bo_forge_app/streamlit_app.py")
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

    app = AppTest.from_file("bo_forge_app/streamlit_app.py")
    app.run(timeout=10)

    next(input_ for input_ in app.text_input if input_.label == "YAML config path").set_value(
        "configs/16_contextual_logei.yaml"
    )
    next(input_ for input_ in app.text_input if input_.label == "CSV log path").set_value(
        "examples/16_contextual_logei_campaign_log.csv"
    )
    next(button for button in app.button if button.label == "Load campaign").click()
    app.run(timeout=10)
    next(radio for radio in app.radio if radio.label == "Workbench panel").set_value("Suggest")
    app.run(timeout=10)

    markdown_text = "\n".join(markdown.value for markdown in app.markdown)
    number_labels = {input_.label for input_ in app.number_input}
    assert len(app.exception) == 0
    assert "Context variables are fixed" in markdown_text
    assert "Context: feedstock_acidity" in number_labels


def test_streamlit_loads_cost_aware_multi_objective_reports_panel() -> None:
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file("bo_forge_app/streamlit_app.py")
    app.run(timeout=10)

    next(input_ for input_ in app.text_input if input_.label == "YAML config path").set_value(
        "configs/12_cost_aware_multi_objective_qlogehvi.yaml"
    )
    next(input_ for input_ in app.text_input if input_.label == "CSV log path").set_value(
        "examples/12_cost_aware_multi_objective_campaign_log.csv"
    )
    next(button for button in app.button if button.label == "Load campaign").click()
    app.run(timeout=10)
    next(radio for radio in app.radio if radio.label == "Workbench panel").set_value("Reports")
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

    app = AppTest.from_file("bo_forge_app/streamlit_app.py")
    app.run(timeout=10)

    next(input_ for input_ in app.text_input if input_.label == "YAML config path").set_value(
        "configs/13_structured_campaign_core.yaml"
    )
    next(input_ for input_ in app.text_input if input_.label == "CSV log path").set_value(
        str(log_path)
    )
    next(button for button in app.button if button.label == "Load campaign").click()
    app.run(timeout=10)
    next(radio for radio in app.radio if radio.label == "Workbench panel").set_value("Suggest")
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

    next(radio for radio in app.radio if radio.label == "Workbench panel").set_value("Reports")
    app.run(timeout=10)
    plot_select = next(selectbox for selectbox in app.selectbox if selectbox.label == "Plot kind")
    assert "Stage Diagnostics" in list(plot_select.options)
    assert len(app.exception) == 0


def test_streamlit_app_can_create_minimal_campaign(tmp_path: Path) -> None:
    from streamlit.testing.v1 import AppTest

    config_path = tmp_path / "configs" / "campaign.yaml"
    log_path = tmp_path / "logs" / "campaign.csv"
    app = AppTest.from_file("bo_forge_app/streamlit_app.py")
    app.run(timeout=10)

    app.radio[0].set_value("Create Campaign")
    app.run(timeout=10)

    app.text_input[1].set_value(str(config_path))
    app.text_input[2].set_value(str(log_path))
    create_button = next(button for button in app.button if button.label == "Create campaign")
    create_button.click()
    app.run(timeout=10)

    assert len(app.exception) == 0
    assert config_path.exists()
    assert log_path.exists()
    markdown_text = "\n".join(markdown.value for markdown in app.markdown)
    success_text = "\n".join(success.value for success in app.success)
    assert str(config_path) in markdown_text
    assert str(log_path) in markdown_text
    assert "Valid" in markdown_text
    assert "Campaign created and loaded" in success_text


def test_streamlit_app_can_create_multi_fidelity_qmfkg_campaign(tmp_path: Path) -> None:
    from streamlit.testing.v1 import AppTest

    config_path = tmp_path / "configs" / "fidelity.yaml"
    log_path = tmp_path / "logs" / "fidelity.csv"
    app = AppTest.from_file("bo_forge_app/streamlit_app.py")
    app.run(timeout=10)

    next(radio for radio in app.radio if radio.label == "Campaign file action").set_value(
        "Create Campaign"
    )
    app.run(timeout=10)
    next(radio for radio in app.radio if radio.label == "Campaign kind").set_value(
        "Multi-fidelity qMFKG"
    )
    app.run(timeout=10)
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
    assert config.bo.acquisition == "qmf_kg"
    assert config.bo.batch_size == 1
    assert config.review.enabled
    assert list(pd.read_csv(log_path, keep_default_na=False).columns) == canonical_columns(config)

    assert any(subheader.value == "Fidelity Summary" for subheader in app.subheader)

    next(radio for radio in app.radio if radio.label == "Workbench panel").set_value("Suggest")
    app.run(timeout=10)
    suggest_markdown = "\n".join(markdown.value for markdown in app.markdown)
    batch_inputs = [
        number_input
        for number_input in app.number_input
        if number_input.label == "Batch size"
    ]
    assert "qMFKG suggestions" in suggest_markdown
    assert batch_inputs[-1].value == 1
    assert batch_inputs[-1].min == 1
    assert batch_inputs[-1].max == 1
    assert batch_inputs[-1].proto.disabled


def test_streamlit_multi_fidelity_target_defaults_to_selected_variable_upper(
    tmp_path: Path,
) -> None:
    from streamlit.testing.v1 import AppTest

    config_path = tmp_path / "configs" / "fidelity_alt.yaml"
    log_path = tmp_path / "logs" / "fidelity_alt.csv"
    app = AppTest.from_file("bo_forge_app/streamlit_app.py")
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
    assert list(pd.read_csv(log_path, keep_default_na=False).columns) == canonical_columns(config)


def test_streamlit_advanced_create_hides_single_objective_fields() -> None:
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file("bo_forge_app/streamlit_app.py")
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
    app = AppTest.from_file("bo_forge_app/streamlit_app.py")
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
