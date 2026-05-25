import shutil
from pathlib import Path

import pandas as pd
import pytest
from matplotlib import pyplot as plt

from bo_forge.config import CampaignConfig
from bo_forge_app import streamlit_app, streamlit_helpers
from bo_forge_app.streamlit_helpers import (
    campaign_report_text,
    dataframe_fingerprint,
    default_export_path,
    export_staged_suggestions_csv,
    extract_matplotlib_figure,
    feature_flags,
    file_fingerprint,
    load_campaign_session,
    make_staged_suggestion_bundle,
    observable_rows,
    resolve_path_input,
    staged_bundle_invalidation_reason,
    staged_bundle_is_appendable,
    staged_suggestions_from_bundle,
)


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
    assert hasattr(streamlit_app, "main")
    assert hasattr(streamlit_app, "render_app")
