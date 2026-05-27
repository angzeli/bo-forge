"""Streamlit UI for local BO Forge campaign workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from bo_forge.errors import BOForgeError
from bo_forge_app.streamlit_helpers import (
    CONFIG_PATH_KEY,
    LAST_APPENDED_FINGERPRINT_KEY,
    LOG_PATH_KEY,
    NEW_CAMPAIGN_YAML_KEY,
    SESSION_KEY,
    STAGED_SUGGESTION_BUNDLE_KEY,
    build_campaign_yaml_text,
    campaign_report_text,
    create_campaign_files,
    default_export_path,
    default_new_campaign_paths,
    export_staged_suggestions_csv,
    extract_matplotlib_figure,
    feature_flags,
    format_dataframe_for_display,
    load_campaign_session,
    make_staged_suggestion_bundle,
    observable_rows,
    parse_categorical_values_text,
    parse_discrete_values_text,
    resolve_path_input,
    staged_bundle_invalidation_reason,
    staged_suggestions_from_bundle,
)
from bo_forge_app.streamlit_style import (
    apply_forge_suite_style,
    forge_action_label,
    forge_status_label,
)


def main() -> None:
    """Run the Streamlit app."""
    render_app()


def render_app() -> None:
    """Render the Streamlit page."""
    import streamlit as st

    st.set_page_config(page_title="BO Forge", layout="wide")
    apply_forge_suite_style(st)
    _render_workbench_header(st)

    _render_campaign_files_panel(st)
    campaign = st.session_state.get(SESSION_KEY)
    if campaign is None:
        st.info("Enter a YAML config path and CSV log path, then load a campaign.")
        return

    flags = feature_flags(campaign.config)
    overview_tab, suggest_tab, resolve_tab, reports_tab = st.tabs(
        ["Campaign", "Suggest", "Resolve", "Reports"]
    )
    with overview_tab:
        _render_overview(st, campaign)
    with suggest_tab:
        _render_suggest(st, campaign)
    with resolve_tab:
        _render_resolve(st, campaign, flags)
    with reports_tab:
        _render_reports(st, campaign, flags)


def _render_workbench_header(st: Any) -> None:
    st.markdown(
        """
        <section class="bf-workbench-header">
          <div class="bf-brand-row">
            <div class="bf-brand-mark">BO</div>
            <div>
              <p class="bf-kicker">Forge Suite workbench</p>
              <h1 class="bf-title">BO Forge</h1>
            </div>
          </div>
          <p class="bf-subtitle">
            Local campaign control for CSV-backed Bayesian optimisation workflows.
            Load files, stage suggestions, record outcomes, and export diagnostics without
            moving BO logic into the app layer.
          </p>
          <div class="bf-chip-row">
            <span class="bf-chip">Local CSV</span>
            <span class="bf-chip">Dry-run suggestions</span>
            <span class="bf-chip">CampaignSession backend</span>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _render_campaign_files_panel(st: Any) -> None:
    st.markdown(
        """
        <section class="bf-panel bf-file-panel">
          <div class="bf-panel-header">
            <div>
              <p class="bf-kicker">Local campaign files</p>
              <h2 class="bf-panel-title">Campaign Files</h2>
              <p class="bf-panel-note">
                Select the YAML config and CSV log that define the active campaign.
              </p>
            </div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    load_tab, create_tab = st.tabs(["Load Existing", "Create New"])
    with load_tab:
        _render_load_existing_campaign(st)
    with create_tab:
        _render_create_new_campaign(st)


def _render_load_existing_campaign(st: Any) -> None:
    config_col, log_col = st.columns(2)
    with config_col:
        config_value = st.text_input(
            "YAML config path",
            value=st.session_state.get(CONFIG_PATH_KEY, ""),
            placeholder="configs/01_simple_2d_maximise_logei.yaml",
        )
    with log_col:
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

    action_col, reload_col = st.columns([1, 1])
    with action_col:
        if st.button("Load campaign", type="primary"):
            _load_campaign_from_inputs(st, config_value, log_value)
    with reload_col:
        if st.button("Reload from disk"):
            _clear_staged_suggestions(st)
            _load_campaign_from_inputs(st, config_value, log_value)

    current_config = st.session_state.get(CONFIG_PATH_KEY)
    current_log = st.session_state.get(LOG_PATH_KEY)
    if current_config or current_log:
        st.markdown("**Current file paths**")
        path_col, log_path_col = st.columns(2)
        if current_config:
            with path_col:
                st.caption("Current config")
                st.code(current_config, language=None)
        if current_log:
            with log_path_col:
                st.caption("Current log")
                st.code(current_log, language=None)


def _render_create_new_campaign(st: Any) -> None:
    st.caption("Create a validated YAML config and empty canonical CSV log.")
    campaign_name = st.text_input(
        "New campaign name",
        value="my_campaign",
        key="new_campaign_name",
    )
    suggested_config_path, suggested_log_path = default_new_campaign_paths(campaign_name)
    path_col, log_path_col = st.columns(2)
    with path_col:
        config_output = st.text_input(
            "New YAML config output path",
            value=str(suggested_config_path),
            key="new_campaign_config_output_path",
        )
    with log_path_col:
        log_output = st.text_input(
            "New CSV log output path",
            value=str(suggested_log_path),
            key="new_campaign_log_output_path",
        )

    objective_col, direction_col = st.columns(2)
    with objective_col:
        objective_name = st.text_input(
            "Objective name",
            value="activity",
            key="new_campaign_objective_name",
        )
    with direction_col:
        objective_direction = st.selectbox(
            "Objective direction",
            ["maximize", "minimize"],
            key="new_campaign_objective_direction",
        )

    bo_col_1, bo_col_2, bo_col_3, bo_col_4 = st.columns(4)
    with bo_col_1:
        batch_size = st.number_input("Batch size", min_value=1, value=1, key="new_bo_batch_size")
    with bo_col_2:
        initial_design_size = st.number_input(
            "Initial design size",
            min_value=1,
            value=8,
            key="new_bo_initial_design_size",
        )
    with bo_col_3:
        initial_design_method = st.selectbox(
            "Initial design method",
            ["sobol", "random"],
            key="new_bo_initial_design_method",
        )
    with bo_col_4:
        random_seed = st.number_input("Random seed", min_value=0, value=0, key="new_bo_seed")

    variable_count = st.number_input(
        "Number of variables",
        min_value=1,
        max_value=12,
        value=2,
        key="new_campaign_variable_count",
    )

    generated_yaml = ""
    try:
        variables = _collect_new_campaign_variables(st, int(variable_count))
        generated_yaml = build_campaign_yaml_text(
            campaign_name=campaign_name,
            objective_name=objective_name,
            objective_direction=str(objective_direction),
            variables=variables,
            batch_size=int(batch_size),
            initial_design_size=int(initial_design_size),
            initial_design_method=str(initial_design_method),
            random_seed=int(random_seed),
        )
    except ValueError as exc:
        st.error(f"Could not build YAML preview: {exc}")

    if NEW_CAMPAIGN_YAML_KEY not in st.session_state:
        st.session_state[NEW_CAMPAIGN_YAML_KEY] = generated_yaml
    if st.button("Regenerate YAML from structured fields"):
        st.session_state[NEW_CAMPAIGN_YAML_KEY] = generated_yaml

    st.warning(
        "Advanced edits are allowed, but the YAML must pass BO Forge config validation "
        "before files are written."
    )
    edited_yaml = st.text_area(
        "Campaign YAML",
        height=360,
        key=NEW_CAMPAIGN_YAML_KEY,
    )

    if st.button("Create and load campaign", type="primary"):
        _create_campaign_from_inputs(st, edited_yaml, config_output, log_output)


def _collect_new_campaign_variables(st: Any, variable_count: int) -> list[dict[str, object]]:
    variables: list[dict[str, object]] = []
    for index in range(variable_count):
        with st.expander(f"Variable {index + 1}", expanded=index < 2):
            name_col, type_col = st.columns(2)
            with name_col:
                name = st.text_input(
                    "Variable name",
                    value=f"x{index + 1}",
                    key=f"new_variable_{index}_name",
                )
            with type_col:
                variable_type = st.selectbox(
                    "Variable type",
                    ["continuous", "integer", "discrete", "categorical"],
                    key=f"new_variable_{index}_type",
                )
            variable: dict[str, object] = {"name": name, "type": variable_type}
            if variable_type in {"continuous", "integer"}:
                lower_col, upper_col = st.columns(2)
                with lower_col:
                    lower = st.number_input(
                        "Lower",
                        value=0.0,
                        key=f"new_variable_{index}_lower",
                    )
                with upper_col:
                    upper = st.number_input(
                        "Upper",
                        value=1.0,
                        key=f"new_variable_{index}_upper",
                    )
                variable["lower"] = int(lower) if variable_type == "integer" else float(lower)
                variable["upper"] = int(upper) if variable_type == "integer" else float(upper)
            elif variable_type == "discrete":
                values_text = st.text_input(
                    "Discrete values",
                    value="0.0, 0.5, 1.0",
                    key=f"new_variable_{index}_discrete_values",
                )
                variable["values"] = parse_discrete_values_text(values_text, name)
            else:
                values_text = st.text_input(
                    "Categorical labels",
                    value="A, B, C",
                    key=f"new_variable_{index}_categorical_values",
                )
                variable["values"] = parse_categorical_values_text(values_text, name)
            variables.append(variable)
    return variables


def _create_campaign_from_inputs(
    st: Any,
    edited_yaml: str,
    config_output: str,
    log_output: str,
) -> None:
    try:
        config_path = resolve_path_input(config_output, "Config output")
        log_path = resolve_path_input(log_output, "Log output")
        campaign = create_campaign_files(
            config_text=edited_yaml,
            config_path=config_path,
            log_path=log_path,
        )
    except (BOForgeError, OSError, ValueError) as exc:
        st.error(str(exc))
        return

    st.session_state[CONFIG_PATH_KEY] = str(config_path)
    st.session_state[LOG_PATH_KEY] = str(log_path)
    st.session_state[SESSION_KEY] = campaign
    _clear_staged_suggestions(st)
    st.success(f"Created and loaded campaign: {campaign.config.campaign_name}")


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
    _render_panel_intro(
        st,
        "Campaign",
        "Inspect campaign state, next action, observed data, and pending rows.",
    )
    try:
        campaign.validate()
    except BOForgeError as exc:
        st.error(f"Validation failed: {exc}")
        st.subheader("Campaign Log")
        st.dataframe(format_dataframe_for_display(campaign.df), width="stretch")
        return
    else:
        st.success("Campaign log is valid.")

    _render_campaign_state_blocks(st, campaign)

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Summary")
        st.dataframe(format_dataframe_for_display(campaign.summary()), width="stretch")
    with col_right:
        st.subheader("Next Action")
        st.dataframe(format_dataframe_for_display(campaign.next_action()), width="stretch")

    st.subheader("Best Observation")
    st.dataframe(
        format_dataframe_for_display(campaign.best_observation()),
        width="stretch",
    )
    if campaign.config.cost is not None:
        st.subheader("Cost Summary")
        st.dataframe(
            format_dataframe_for_display(campaign.cost_summary()),
            width="stretch",
        )
    if campaign.config.replicates.enabled:
        st.subheader("Best Replicate Group")
        st.dataframe(
            format_dataframe_for_display(campaign.best_replicate_group()),
            width="stretch",
        )
        st.subheader("Replicate Summary")
        st.dataframe(
            format_dataframe_for_display(campaign.replicate_summary()),
            width="stretch",
        )

    with st.expander("Observed Rows", expanded=False):
        st.dataframe(
            format_dataframe_for_display(campaign.observed_data()),
            width="stretch",
        )

    with st.expander("Pending Suggestions", expanded=True):
        st.dataframe(
            format_dataframe_for_display(campaign.pending_suggestions()),
            width="stretch",
        )

    with st.expander("Full Campaign Log", expanded=False):
        st.dataframe(format_dataframe_for_display(campaign.df), width="stretch")


def _render_campaign_state_blocks(st: Any, campaign: Any) -> None:
    status = campaign.campaign_status()
    next_action = campaign.next_action()
    action = ""
    reason = ""
    if not next_action.empty:
        action = str(next_action.loc[0, "action"])
        reason = str(next_action.loc[0, "reason"])

    status_col, action_col = st.columns(2)
    with status_col:
        _render_status_block(
            st,
            "Campaign status",
            forge_status_label(status),
            status,
        )
    with action_col:
        _render_status_block(
            st,
            "Next action",
            forge_action_label(action),
            reason,
        )


def _render_suggest(st: Any, campaign: Any) -> None:
    _render_panel_intro(
        st,
        "Suggest",
        "Generate candidates as a dry run, inspect quality, then append explicitly.",
    )
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
    st.dataframe(format_dataframe_for_display(suggestions), width="stretch")

    export_path = Path(
        st.text_input(
            "Staged suggestions CSV export path",
            value=str(default_export_path(log_path, "staged_suggestions", "csv")),
            key="staged_suggestions_export_path",
        )
    )
    if st.button("Export staged suggestions CSV"):
        try:
            written_path = export_staged_suggestions_csv(suggestions, export_path)
        except OSError as exc:
            st.error(str(exc))
        else:
            st.success(f"Wrote staged suggestions CSV: {written_path}")

    try:
        quality = campaign.suggestion_quality(suggestions)
    except BOForgeError as exc:
        st.warning(f"Could not compute suggestion quality: {exc}")
    else:
        st.subheader("Suggestion Quality")
        st.dataframe(format_dataframe_for_display(quality), width="stretch")

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
    _render_panel_intro(
        st,
        "Resolve",
        "Review suggested rows and record experimental outcomes.",
    )
    pending = campaign.pending_suggestions()
    with st.expander("Pending Suggestions", expanded=True):
        st.dataframe(format_dataframe_for_display(pending), width="stretch")
    observable = observable_rows(campaign.config, campaign.df)

    if flags["has_review"]:
        st.subheader("Review Queue")
        review_queue = campaign.review_queue()
        st.dataframe(format_dataframe_for_display(review_queue), width="stretch")
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

    st.subheader("Rows Ready To Mark Observed")
    st.dataframe(format_dataframe_for_display(observable), width="stretch")
    if observable.empty:
        st.info("No suggested rows are ready to mark observed.")
        return

    st.subheader("Mark Observed")
    observed_row_id = st.selectbox(
        "Observed row_id",
        observable["row_id"].astype(str).tolist(),
    )
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
    _render_panel_intro(
        st,
        "Reports",
        "Preview reports and export campaign figures.",
    )
    _, log_path = _current_paths(st)

    st.subheader("Report Preview")
    try:
        report_text = campaign_report_text(campaign)
    except BOForgeError as exc:
        st.error(str(exc))
    else:
        st.text_area("Campaign report", value=report_text, height=360)

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
            width="stretch",
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
            width="stretch",
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


def _render_panel_intro(st: Any, title: str, note: str) -> None:
    st.markdown(
        f"""
        <section class="bf-panel">
          <div class="bf-panel-header">
            <div>
              <p class="bf-kicker">Campaign workbench</p>
              <h2 class="bf-panel-title">{title}</h2>
              <p class="bf-panel-note">{note}</p>
            </div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _render_status_block(st: Any, label: str, value: str, detail: str) -> None:
    st.markdown(
        f"""
        <div class="bf-status-block">
          <p class="bf-status-label">{label}</p>
          <p class="bf-status-value">{value}</p>
          <p class="bf-status-detail">{detail}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
