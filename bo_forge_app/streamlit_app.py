"""Streamlit UI for local BO Forge campaign workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from bo_forge.errors import BOForgeError
from bo_forge_app.streamlit_helpers import (
    CONFIG_PATH_KEY,
    LAST_APPENDED_FINGERPRINT_KEY,
    LOG_PATH_KEY,
    SESSION_KEY,
    STAGED_SUGGESTION_BUNDLE_KEY,
    default_export_path,
    extract_matplotlib_figure,
    feature_flags,
    format_dataframe_for_display,
    load_campaign_session,
    make_staged_suggestion_bundle,
    resolve_path_input,
    staged_bundle_invalidation_reason,
    staged_suggestions_from_bundle,
)


def main() -> None:
    """Run the Streamlit app."""
    render_app()


def render_app() -> None:
    """Render the Streamlit page."""
    import streamlit as st

    st.set_page_config(page_title="BO Forge", layout="wide")
    st.title("BO Forge Campaign App")
    st.caption("Local Streamlit wrapper around CampaignSession.")

    _render_sidebar(st)
    campaign = st.session_state.get(SESSION_KEY)
    if campaign is None:
        st.info("Enter a YAML config path and CSV log path, then load a campaign.")
        return

    flags = feature_flags(campaign.config)
    overview_tab, suggest_tab, resolve_tab, reports_tab = st.tabs(
        ["Overview", "Suggest", "Resolve", "Reports & Plots"]
    )
    with overview_tab:
        _render_overview(st, campaign)
    with suggest_tab:
        _render_suggest(st, campaign)
    with resolve_tab:
        _render_resolve(st, campaign, flags)
    with reports_tab:
        _render_reports(st, campaign, flags)


def _render_sidebar(st: Any) -> None:
    with st.sidebar:
        st.header("Campaign Files")
        config_value = st.text_input(
            "YAML config path",
            value=st.session_state.get(CONFIG_PATH_KEY, ""),
            placeholder="configs/01_simple_2d_maximise_logei.yaml",
        )
        log_value = st.text_input(
            "CSV log path",
            value=st.session_state.get(LOG_PATH_KEY, ""),
            placeholder="examples/01_simple_2d_maximise_logei_working_log.csv",
        )

        if _path_changed(config_value, LOG_PATH_KEY, log_value):
            _clear_staged_suggestions(st)

        st.warning(
            "Append, review, and mark-observed actions modify the selected CSV log. "
            "Report and plot exports write files to the selected output path.",
        )

        if st.button("Load campaign", type="primary"):
            _load_campaign_from_inputs(st, config_value, log_value)

        if st.button("Reload from disk"):
            _clear_staged_suggestions(st)
            _load_campaign_from_inputs(st, config_value, log_value)

        if st.session_state.get(CONFIG_PATH_KEY):
            st.text("Current config:")
            st.code(st.session_state[CONFIG_PATH_KEY], language=None)
        if st.session_state.get(LOG_PATH_KEY):
            st.text("Current log:")
            st.code(st.session_state[LOG_PATH_KEY], language=None)


def _path_changed(config_value: str, log_key: str, log_value: str) -> bool:
    return (
        bool(config_value)
        and config_value != ""
        and config_value != str(_session_value(CONFIG_PATH_KEY))
    ) or (
        bool(log_value)
        and log_value != ""
        and log_value != str(_session_value(log_key))
    )


def _session_value(key: str) -> object:
    try:
        import streamlit as st
    except ModuleNotFoundError:
        return None
    return st.session_state.get(key)


def _load_campaign_from_inputs(st: Any, config_value: str, log_value: str) -> None:
    try:
        config_path = resolve_path_input(config_value, "Config")
        log_path = resolve_path_input(log_value, "Log")
        campaign = load_campaign_session(config_path, log_path)
    except (BOForgeError, OSError, ValueError) as exc:
        st.error(str(exc))
        return

    st.session_state[CONFIG_PATH_KEY] = str(config_path)
    st.session_state[LOG_PATH_KEY] = str(log_path)
    st.session_state[SESSION_KEY] = campaign
    _clear_staged_suggestions(st)
    st.success("Campaign loaded.")


def _render_overview(st: Any, campaign: Any) -> None:
    try:
        campaign.validate()
    except BOForgeError as exc:
        st.error(f"Validation failed: {exc}")
    else:
        st.success("Campaign log is valid.")

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Summary")
        st.dataframe(format_dataframe_for_display(campaign.summary()), use_container_width=True)
    with col_right:
        st.subheader("Next Action")
        st.dataframe(format_dataframe_for_display(campaign.next_action()), use_container_width=True)

    st.subheader("Best Observation")
    st.dataframe(
        format_dataframe_for_display(campaign.best_observation()),
        use_container_width=True,
    )
    if campaign.config.replicates.enabled:
        st.subheader("Best Replicate Group")
        st.dataframe(
            format_dataframe_for_display(campaign.best_replicate_group()),
            use_container_width=True,
        )

    st.subheader("Campaign Log")
    st.dataframe(format_dataframe_for_display(campaign.df), use_container_width=True)


def _render_suggest(st: Any, campaign: Any) -> None:
    config_path, log_path = _current_paths(st)
    batch_size = st.number_input(
        "Batch size",
        min_value=1,
        max_value=32,
        value=max(1, int(campaign.config.bo.batch_size)),
        step=1,
    )

    if st.button("Generate suggestions (dry run)", type="primary"):
        try:
            suggestions = campaign.suggest_next(batch_size=int(batch_size))
            bundle = make_staged_suggestion_bundle(suggestions, config_path, log_path)
        except (BOForgeError, OSError, ValueError) as exc:
            st.error(str(exc))
        else:
            st.session_state[STAGED_SUGGESTION_BUNDLE_KEY] = bundle
            st.success("Suggestions staged. Review them before appending.")

    bundle = st.session_state.get(STAGED_SUGGESTION_BUNDLE_KEY)
    suggestions = staged_suggestions_from_bundle(bundle)
    if suggestions.empty:
        st.info("No staged suggestions.")
        return

    reason = _current_invalidation_reason(st, bundle)
    if reason and reason != "No staged suggestions.":
        st.warning(reason)
        if _should_clear_staged_bundle(reason):
            _clear_staged_suggestions(st)
            st.info("Cleared stale staged suggestions. Generate a fresh dry-run batch.")
            return

    st.subheader("Staged Suggestions")
    st.dataframe(format_dataframe_for_display(suggestions), use_container_width=True)
    try:
        quality = campaign.suggestion_quality(suggestions)
    except BOForgeError as exc:
        st.warning(f"Could not compute suggestion quality: {exc}")
    else:
        st.subheader("Suggestion Quality")
        st.dataframe(format_dataframe_for_display(quality), use_container_width=True)

    if st.button("Append staged suggestions", disabled=reason is not None):
        try:
            campaign.append_suggestions(suggestions)
        except BOForgeError as exc:
            st.error(str(exc))
            return
        st.session_state[LAST_APPENDED_FINGERPRINT_KEY] = bundle["suggestions_fingerprint"]
        _clear_staged_suggestions(st)
        st.session_state[SESSION_KEY] = campaign
        st.success("Staged suggestions appended to the campaign log.")


def _render_resolve(st: Any, campaign: Any, flags: dict[str, bool]) -> None:
    st.subheader("Pending Suggestions")
    pending = campaign.pending_suggestions()
    st.dataframe(format_dataframe_for_display(pending), use_container_width=True)

    if flags["has_review"]:
        st.subheader("Review Queue")
        review_queue = campaign.review_queue()
        st.dataframe(format_dataframe_for_display(review_queue), use_container_width=True)
        if not review_queue.empty:
            row_id = st.selectbox("Review row_id", review_queue["row_id"].astype(str).tolist())
            decision = st.selectbox("Decision", ["accept", "reject", "defer"])
            note = st.text_input("Review note", value="")
            if st.button("Apply review decision"):
                try:
                    campaign.review_suggestion(row_id, decision, note)
                except BOForgeError as exc:
                    st.error(str(exc))
                else:
                    _clear_staged_suggestions(st)
                    st.session_state[SESSION_KEY] = campaign
                    st.success("Review decision recorded.")

    if pending.empty:
        st.info("No suggested rows to mark observed.")
        return

    st.subheader("Mark Observed")
    observed_row_id = st.selectbox("Observed row_id", pending["row_id"].astype(str).tolist())
    objective_value = st.number_input("Objective value", value=0.0, format="%.8f")
    actual_cost = None
    if flags["has_cost"]:
        record_actual_cost = st.checkbox("Record actual cost")
        if record_actual_cost:
            actual_cost = st.number_input("Actual cost", min_value=0.0, value=0.0, format="%.8f")

    if st.button("Mark row observed"):
        try:
            campaign.mark_observed(
                row_id=observed_row_id,
                objective_value=float(objective_value),
                actual_cost=None if actual_cost is None else float(actual_cost),
            )
        except BOForgeError as exc:
            st.error(str(exc))
        else:
            _clear_staged_suggestions(st)
            st.session_state[SESSION_KEY] = campaign
            st.success("Observation recorded.")


def _render_reports(st: Any, campaign: Any, flags: dict[str, bool]) -> None:
    _, log_path = _current_paths(st)

    report_path = Path(
        st.text_input(
            "Report export path",
            value=str(default_export_path(log_path, "campaign_report", "txt")),
        )
    )
    if st.button("Export report"):
        try:
            written_path = campaign.export_report(report_path)
        except (BOForgeError, OSError) as exc:
            st.error(str(exc))
        else:
            st.success(f"Wrote report: {written_path}")

    st.subheader("Progress Plot")
    _render_plot_controls(
        st,
        "Progress",
        "progress",
        campaign.plot_progress,
        default_export_path(log_path, "progress", "png"),
    )
    st.subheader("Diagnostics Plot")
    _render_plot_controls(
        st,
        "Diagnostics",
        "diagnostics",
        campaign.plot_diagnostics,
        default_export_path(log_path, "diagnostics", "png"),
    )
    if flags["has_cost"]:
        st.subheader("Cost Summary")
        st.dataframe(
            format_dataframe_for_display(campaign.cost_summary()),
            use_container_width=True,
        )
        _render_plot_controls(
            st,
            "Cost Progress",
            "cost_progress",
            campaign.plot_cost_progress,
            default_export_path(log_path, "cost_progress", "png"),
        )
    if flags["has_replicates"]:
        st.subheader("Replicate Summary")
        st.dataframe(
            format_dataframe_for_display(campaign.replicate_summary()),
            use_container_width=True,
        )
        _render_plot_controls(
            st,
            "Replicates",
            "replicates",
            campaign.plot_replicates,
            default_export_path(log_path, "replicates", "png"),
        )


def _render_plot_controls(
    st: Any,
    label: str,
    key_suffix: str,
    plotter: Any,
    default_path: Path,
) -> None:
    export_path = Path(
        st.text_input(
            f"{label} export path",
            value=str(default_path),
            key=f"{key_suffix}_export_path",
        )
    )
    col_show, col_export = st.columns(2)
    with col_show:
        show_clicked = st.button(f"Show {label.lower()} plot", key=f"{key_suffix}_show")
    with col_export:
        export_clicked = st.button(f"Export {label.lower()} plot", key=f"{key_suffix}_export")

    if show_clicked:
        try:
            fig = extract_matplotlib_figure(plotter())
        except (BOForgeError, ValueError) as exc:
            st.error(str(exc))
        else:
            st.pyplot(fig)
    if export_clicked:
        try:
            plotter(save_path=export_path)
        except (BOForgeError, OSError, ValueError) as exc:
            st.error(str(exc))
        else:
            st.success(f"Wrote plot: {export_path}")


def _current_paths(st: Any) -> tuple[Path, Path]:
    return Path(st.session_state[CONFIG_PATH_KEY]), Path(st.session_state[LOG_PATH_KEY])


def _current_invalidation_reason(st: Any, bundle: dict[str, object] | None) -> str | None:
    config_path, log_path = _current_paths(st)
    try:
        return staged_bundle_invalidation_reason(
            bundle=bundle,
            config_path=config_path,
            log_path=log_path,
            last_appended_fingerprint=st.session_state.get(LAST_APPENDED_FINGERPRINT_KEY),
        )
    except OSError as exc:
        return str(exc)


def _should_clear_staged_bundle(reason: str) -> bool:
    return reason in {
        "Config path changed after suggestions were staged.",
        "Log path changed after suggestions were staged.",
        "Config file changed after suggestions were staged.",
        "Log file changed after suggestions were staged.",
    }


def _clear_staged_suggestions(st: Any) -> None:
    st.session_state.pop(STAGED_SUGGESTION_BUNDLE_KEY, None)


if __name__ == "__main__":
    main()
