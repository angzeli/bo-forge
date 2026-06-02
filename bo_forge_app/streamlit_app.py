"""Streamlit UI for local BO Forge campaign workflows."""

from __future__ import annotations

from html import escape
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
    append_disabled_reason,
    available_plot_kinds,
    build_campaign_yaml_text,
    campaign_report_text,
    compact_dataframe,
    create_campaign_files,
    default_export_path,
    default_new_campaign_paths,
    empty_state_message,
    export_staged_suggestions_csv,
    extract_matplotlib_figure,
    feature_flags,
    format_dataframe_for_display,
    format_number_for_display,
    humanize_campaign_status,
    load_campaign_session,
    make_staged_suggestion_bundle,
    observable_row_options,
    observable_rows,
    parse_campaign_config_text,
    parse_categorical_values_text,
    parse_discrete_values_text,
    resolve_path_input,
    staged_bundle_invalidation_reason,
    staged_suggestions_from_bundle,
    status_tone,
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
    campaign = st.session_state.get(SESSION_KEY)
    _render_workbench_header(st, campaign_loaded=campaign is not None)

    _render_campaign_files_panel(st)
    campaign = st.session_state.get(SESSION_KEY)
    if campaign is None:
        _render_empty_state(
            st,
            "Nothing loaded yet.",
            "Enter a YAML config path and CSV log path, or create a campaign in the "
            "workbench above.",
        )
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


def _render_workbench_header(st: Any, *, campaign_loaded: bool) -> None:
    campaign_chip = "Campaign loaded" if campaign_loaded else "No campaign loaded"
    campaign_chip_class = "bf-chip-success" if campaign_loaded else "bf-chip-warning"
    st.markdown(
        f"""
        <section class="bf-workbench-header">
          <div class="bf-brand-row">
            <div class="bf-brand-mark">BO</div>
            <div>
              <p class="bf-kicker">Forge Suite workbench</p>
              <h1 class="bf-title">BO Forge</h1>
            </div>
          </div>
          <p class="bf-subtitle">
            Local campaign control for CSV-backed Bayesian optimisation. Load files,
            stage suggestions, record outcomes, and export diagnostics while BO logic
            stays in the backend.
          </p>
          <div class="bf-chip-row">
            <span class="bf-chip">Local CSV</span>
            <span class="bf-chip">Staged suggestions</span>
            <span class="bf-chip">CampaignSession backend</span>
            <span class="bf-chip {campaign_chip_class}">{escape(campaign_chip)}</span>
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
                Select or create the YAML config and CSV log that define the active campaign.
                Relative paths are resolved from the project working directory.
              </p>
            </div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    load_tab, create_tab = st.tabs(["Load Existing", "Create Campaign"])
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

    _render_callout(
        st,
        "Write actions modify local files",
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
        _render_file_cards(st, str(current_config or ""), str(current_log or ""))


def _render_create_new_campaign(st: Any) -> None:
    _render_section_label(st, "Campaign identity")
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

    _render_section_label(st, "Objective")
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

    _render_section_label(st, "BO settings")
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

    _render_section_label(st, "Variables")
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
    if st.button("Regenerate YAML from form"):
        st.session_state[NEW_CAMPAIGN_YAML_KEY] = generated_yaml

    _render_section_label(st, "Generated YAML Preview")
    _render_artifact_note(
        st,
        "Editable before writing",
        "Advanced edits are allowed, but the YAML must pass BO Forge config validation "
        "before files are written.",
    )
    edited_yaml = st.text_area(
        "Campaign YAML",
        height=360,
        key=NEW_CAMPAIGN_YAML_KEY,
    )

    validate_col, create_col = st.columns([1, 1])
    with validate_col:
        if st.button("Validate YAML"):
            try:
                parse_campaign_config_text(edited_yaml)
            except BOForgeError as exc:
                _render_result_card(st, "Could not validate YAML", str(exc), success=False)
            else:
                _render_result_card(
                    st,
                    "YAML is valid",
                    "The preview passes BO Forge config validation.",
                )

    _render_callout(
        st,
        "Creation safety checks",
        "YAML must validate; config and log paths must not already exist; the empty CSV log "
        "is validated before loading; staged suggestions are cleared after creation.",
    )
    with create_col:
        if st.button("Create campaign", type="primary"):
            _create_campaign_from_inputs(st, edited_yaml, config_output, log_output)


def _render_file_cards(st: Any, config_path: str, log_path: str) -> None:
    config_card = ""
    if config_path:
        config_card = f"""
        <div class="forge-file-card">
          <span class="forge-pill">YAML</span>
          <p class="forge-file-path">{escape(config_path)}</p>
        </div>
        """
    log_card = ""
    if log_path:
        log_card = f"""
        <div class="forge-file-card">
          <span class="forge-pill">CSV</span>
          <p class="forge-file-path">{escape(log_path)}</p>
        </div>
        """
    st.markdown(
        f"""
        <div class="forge-file-grid">
          {config_card}
          {log_card}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_section_label(st: Any, label: str) -> None:
    st.markdown(f'<p class="bf-kicker">{escape(label)}</p>', unsafe_allow_html=True)


def _render_artifact_note(st: Any, title: str, detail: str) -> None:
    st.markdown(
        f"""
        <div class="forge-artifact">
          <p class="forge-card-title">{escape(title)}</p>
          <p class="forge-card-value">{escape(detail)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _collect_new_campaign_variables(st: Any, variable_count: int) -> list[dict[str, object]]:
    variables: list[dict[str, object]] = []
    for index in range(variable_count):
        with st.expander(f"Variable {index + 1}", expanded=index < 2):
            st.markdown(
                f'<span class="forge-pill">Variable {index + 1}</span>',
                unsafe_allow_html=True,
            )
            name_col, type_col = st.columns(2)
            with name_col:
                name = st.text_input(
                    "Variable name",
                    value=f"x{index + 1}",
                    placeholder="temperature",
                    key=f"new_variable_{index}_name",
                )
            with type_col:
                variable_type = st.selectbox(
                    "Variable type",
                    ["continuous", "integer", "discrete", "categorical"],
                    key=f"new_variable_{index}_type",
                )
            _render_variable_type_badge(st, str(variable_type))
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
                    placeholder="0.1, 0.2, 0.5",
                    key=f"new_variable_{index}_discrete_values",
                    help="Discrete values must be comma-separated numbers.",
                )
                variable["values"] = parse_discrete_values_text(values_text, name)
            else:
                values_text = st.text_input(
                    "Categorical labels",
                    value="A, B, C",
                    placeholder="MeCN, DMF, Water",
                    key=f"new_variable_{index}_categorical_values",
                    help=(
                        "Categorical labels are case-sensitive. Empty or duplicate "
                        "labels are rejected."
                    ),
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
        _render_result_card(st, "Could not create campaign", str(exc), success=False)
        return

    st.session_state[CONFIG_PATH_KEY] = str(config_path)
    st.session_state[LOG_PATH_KEY] = str(log_path)
    st.session_state[SESSION_KEY] = campaign
    _clear_staged_suggestions(st)
    _render_result_card(
        st,
        "Campaign created and loaded",
        f"Config: {config_path} | Log: {log_path}. The campaign is now active.",
    )


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
        _render_table_section(
            st,
            "Campaign Log",
            campaign.df,
            empty_kind="pending_suggestions",
            expanded_raw=True,
        )
        return
    else:
        _render_result_card(
            st,
            "Campaign log is valid",
            "The selected CSV matches the active config.",
        )

    _render_campaign_state_blocks(st, campaign)
    summary = campaign.summary()
    _render_metric_grid(
        st,
        [
            ("Total rows", _summary_value(summary, "total_rows")),
            ("Observed", _summary_value(summary, "observed_rows")),
            ("Pending", _summary_value(summary, "pending_suggestions")),
            ("Initial left", _summary_value(summary, "initial_design_remaining")),
            ("Next iteration", _summary_value(summary, "next_iteration")),
        ],
    )

    col_left, col_right = st.columns(2)
    with col_left:
        with st.expander("Raw summary table", expanded=False):
            st.dataframe(format_dataframe_for_display(summary), width="stretch")
    with col_right:
        with st.expander("Raw next-action table", expanded=False):
            st.dataframe(format_dataframe_for_display(campaign.next_action()), width="stretch")

    if campaign.config.is_multi_objective:
        _render_table_section(
            st,
            "Pareto Summary",
            campaign.pareto_summary(),
            empty_kind="report_preview",
            expanded_raw=False,
        )
        _render_table_section(
            st,
            "Pareto Front",
            compact_dataframe(campaign.pareto_front()),
            empty_kind="observed_rows",
            raw_df=campaign.pareto_front(),
            expanded_raw=False,
        )
    else:
        _render_table_section(
            st,
            "Best Observation",
            campaign.best_observation(),
            empty_kind="best_observation",
            expanded_raw=False,
        )
    if campaign.config.cost is not None:
        cost_summary = campaign.cost_summary()
        _render_metric_grid(
            st,
            [
                ("Observed cost", _summary_value(cost_summary, "total_observed_cost")),
                ("Accepted pending", _summary_value(cost_summary, "accepted_pending_cost")),
                ("Budget", _summary_value(cost_summary, "budget")),
                ("Remaining", _summary_value(cost_summary, "budget_remaining")),
                ("Best objective", _summary_value(cost_summary, "best_observed_objective")),
            ],
        )
        with st.expander("Raw cost summary table", expanded=False):
            st.dataframe(format_dataframe_for_display(cost_summary), width="stretch")
    if campaign.config.replicates.enabled:
        if not campaign.config.is_multi_objective:
            _render_table_section(
                st,
                "Best Replicate Group",
                campaign.best_replicate_group(),
                empty_kind="replicate_summary",
                expanded_raw=False,
            )
        _render_table_section(
            st,
            "Replicate Summary",
            _compact_replicate_summary(campaign.replicate_summary()),
            empty_kind="replicate_summary",
            raw_df=campaign.replicate_summary(),
            expanded_raw=False,
        )

    _render_table_section(
        st,
        "Observed Rows",
        campaign.observed_data(),
        empty_kind="observed_rows",
        expanded_raw=False,
    )
    _render_table_section(
        st,
        "Pending Suggestions",
        campaign.pending_suggestions(),
        empty_kind="pending_suggestions",
        expanded_raw=True,
    )
    with st.expander("Show full raw campaign log", expanded=False):
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
            tone=status_tone(status),
        )
    with action_col:
        _render_status_block(
            st,
            "Next action",
            forge_action_label(action),
            reason,
            tone="neutral",
        )


def _render_suggest(st: Any, campaign: Any) -> None:
    _render_panel_intro(
        st,
        "Suggest",
        "Generate candidates as a dry run, inspect quality, then append explicitly.",
    )
    _render_step_flow(
        st,
        ["1. Generate dry-run suggestions", "2. Inspect quality", "3. Append explicitly"],
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
        _render_empty_state(st, *empty_state_message("staged_suggestions"))
        return

    raw_reason = _current_invalidation_reason(st, bundle)
    disabled_reason = append_disabled_reason(
        bundle,
        config_path,
        log_path,
        st.session_state.get(LAST_APPENDED_FINGERPRINT_KEY),
    )
    if raw_reason and raw_reason != "No staged suggestions.":
        _render_callout(st, "Append state", disabled_reason or raw_reason)
        if _should_clear_staged_bundle(raw_reason):
            _clear_staged_suggestions(st)
            _render_empty_state(
                st,
                "Cleared stale staged suggestions.",
                "Generate a fresh dry-run batch before appending.",
            )
            return

    _render_metric_grid(
        st,
        [
            ("Staged rows", len(suggestions)),
            ("Status", "Ready" if disabled_reason is None else "Blocked"),
        ],
    )
    _render_table_section(
        st,
        "Staged Suggestions",
        suggestions,
        empty_kind="staged_suggestions",
        expanded_raw=False,
    )

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
        _render_artifact_note(
            st,
            "Suggestion Quality",
            "Read-only checks for feasibility, duplicates, and distance threshold.",
        )
        _render_table_section(
            st,
            "Suggestion Quality",
            quality,
            empty_kind="staged_suggestions",
            expanded_raw=False,
        )

    if disabled_reason is not None:
        _render_callout(st, "Append disabled", disabled_reason)

    if st.button("Append staged suggestions", disabled=disabled_reason is not None):
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
    with st.expander("Pending Suggestions", expanded=False):
        if pending.empty:
            _render_empty_state(st, *empty_state_message("pending_suggestions"))
        else:
            st.dataframe(compact_dataframe(pending), width="stretch")
            with st.expander("Show full raw pending suggestions", expanded=False):
                st.dataframe(format_dataframe_for_display(pending), width="stretch")
    observable = observable_rows(campaign.config, campaign.df)

    if flags["has_review"]:
        st.subheader("Review Queue")
        review_queue = campaign.review_queue()
        if review_queue.empty:
            _render_empty_state(st, *empty_state_message("review_queue"))
        else:
            st.dataframe(compact_dataframe(review_queue), width="stretch")
            with st.expander("Show full raw review queue", expanded=False):
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
    else:
        _render_empty_state(
            st,
            "Review is not enabled.",
            "This campaign can mark suggested rows observed without a review decision.",
        )

    _render_table_section(
        st,
        "Observable Suggestions",
        observable,
        empty_kind="pending_suggestions",
        expanded_raw=False,
    )
    if campaign.config.is_multi_objective:
        _render_empty_state(
            st,
            "Multi-objective observation entry is not supported in the app yet.",
            "Use the CLI or CampaignSession.mark_observed(..., objective_values={...}) "
            "to record coupled objective values.",
        )
        return
    if observable.empty:
        return

    st.subheader("Mark Observed")
    option_map = observable_row_options(campaign.config, campaign.df)
    selected_label = st.selectbox("Observed suggestion", list(option_map))
    observed_row_id = option_map[selected_label]
    selected_row = observable.loc[observable["row_id"].astype(str) == observed_row_id]
    if not selected_row.empty:
        _render_selected_row_preview(st, campaign, selected_row.iloc[0])
    objective_name = campaign.config.objective.name
    objective_value = st.number_input(f"Observed {objective_name}", value=0.0, format="%.8f")
    actual_cost = None
    if flags["has_cost"]:
        record_actual_cost = st.checkbox("Record actual cost")
        if record_actual_cost:
            actual_cost = st.number_input(
                "Actual cost (optional)",
                min_value=0.0,
                value=0.0,
                format="%.8f",
                help="Leave blank by not enabling this checkbox to use estimated cost.",
            )

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
        summary = campaign.summary()
        _render_metric_grid(
            st,
            [
                (
                    "Status",
                    humanize_campaign_status(str(_summary_value(summary, "campaign_status"))),
                ),
                ("Observed", _summary_value(summary, "observed_rows")),
                ("Pending", _summary_value(summary, "pending_suggestions")),
                ("Best objective", _summary_value(summary, "best_objective_value")),
            ],
        )
        with st.expander("Raw report text", expanded=False):
            st.text_area("Campaign report", value=report_text, height=360)

    with st.expander("Report export settings", expanded=True):
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

    plot_kinds = available_plot_kinds(campaign.config)
    if "progress" in plot_kinds:
        _render_plot_controls(
            st,
            "Progress",
            "progress",
            campaign.plot_progress,
            default_export_path(log_path, "progress", "png"),
        )
    if "diagnostics" in plot_kinds:
        _render_plot_controls(
            st,
            "Diagnostics",
            "diagnostics",
            campaign.plot_diagnostics,
            default_export_path(log_path, "diagnostics", "png"),
        )
    if flags["has_cost"]:
        cost_summary = campaign.cost_summary()
        _render_metric_grid(
            st,
            [
                ("Observed cost", _summary_value(cost_summary, "total_observed_cost")),
                ("Accepted pending", _summary_value(cost_summary, "accepted_pending_cost")),
                ("Budget", _summary_value(cost_summary, "budget")),
                ("Remaining", _summary_value(cost_summary, "budget_remaining")),
                ("Best objective", _summary_value(cost_summary, "best_observed_objective")),
            ],
        )
        with st.expander("Raw cost summary table", expanded=False):
            st.dataframe(format_dataframe_for_display(cost_summary), width="stretch")
        _render_plot_controls(
            st,
            "Cost Progress",
            "cost_progress",
            campaign.plot_cost_progress,
            default_export_path(log_path, "cost_progress", "png"),
        )
    if flags["has_replicates"]:
        _render_table_section(
            st,
            "Replicate Summary",
            _compact_replicate_summary(campaign.replicate_summary()),
            empty_kind="replicate_summary",
            raw_df=campaign.replicate_summary(),
            expanded_raw=False,
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
    st.markdown(
        f"""
        <div class="forge-card">
          <p class="forge-card-title">{escape(label)} plot</p>
          <p class="forge-card-value">Render in the app or export the figure to a local file.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander(f"{label} plot settings", expanded=True):
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


def _render_status_block(
    st: Any,
    label: str,
    value: str,
    detail: str,
    *,
    tone: str = "neutral",
) -> None:
    tone_class = f" bf-status-block-{tone}" if tone != "neutral" else ""
    st.markdown(
        f"""
        <div class="bf-status-block{tone_class}">
          <p class="bf-status-label">{escape(label)}</p>
          <p class="bf-status-value">{escape(value)}</p>
          <p class="bf-status-detail">{escape(detail)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_callout(st: Any, title: str, detail: str) -> None:
    st.markdown(
        f"""
        <div class="forge-callout">
          <p class="forge-callout-title">{escape(title)}</p>
          <p class="forge-callout-detail">{escape(detail)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_result_card(st: Any, title: str, detail: str, *, success: bool = True) -> None:
    class_name = "forge-success" if success else "forge-warning"
    st.markdown(
        f"""
        <div class="{class_name}">
          <p class="forge-card-title">{escape(title)}</p>
          <p class="forge-card-value">{escape(detail)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_empty_state(st: Any, title: str, detail: str) -> None:
    st.markdown(
        f"""
        <div class="forge-empty">
          <p class="forge-empty-title">{escape(title)}</p>
          <p class="forge-empty-detail">{escape(detail)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_metric_grid(st: Any, metrics: list[tuple[str, object]]) -> None:
    cards = []
    for label, value in metrics:
        display_value = format_number_for_display(value)
        cards.append(
            f"""
            <div class="forge-metric">
              <p class="forge-metric-label">{escape(str(label))}</p>
              <p class="forge-metric-value">{escape(str(display_value))}</p>
            </div>
            """
        )
    st.markdown(
        f'<div class="forge-metric-grid">{"".join(cards)}</div>',
        unsafe_allow_html=True,
    )


def _render_step_flow(st: Any, steps: list[str]) -> None:
    chips = "".join(f'<span class="forge-step">{escape(step)}</span>' for step in steps)
    st.markdown(f'<div class="forge-step-row">{chips}</div>', unsafe_allow_html=True)


def _render_table_section(
    st: Any,
    title: str,
    df: Any,
    *,
    empty_kind: str,
    raw_df: Any | None = None,
    expanded_raw: bool = False,
) -> None:
    st.subheader(title)
    table = df.copy(deep=True) if hasattr(df, "copy") else df
    raw_table = raw_df if raw_df is not None else df
    if getattr(table, "empty", False):
        _render_empty_state(st, *empty_state_message(empty_kind))
        return
    st.dataframe(compact_dataframe(table), width="stretch")
    with st.expander(f"Show full raw {title.lower()}", expanded=expanded_raw):
        st.dataframe(format_dataframe_for_display(raw_table), width="stretch")


def _render_selected_row_preview(st: Any, campaign: Any, row: Any) -> None:
    metrics = []
    for variable in campaign.config.variables[:6]:
        metrics.append((variable.name, row.get(variable.name, "")))
    _render_metric_grid(st, metrics)


def _render_variable_type_badge(st: Any, variable_type: str) -> None:
    tones = {
        "continuous": "forge-pill-sage",
        "integer": "forge-pill-gold",
        "discrete": "forge-pill-copper",
        "categorical": "forge-pill-blue",
    }
    tone = tones.get(variable_type, "")
    st.markdown(
        f'<span class="forge-pill {tone}">{escape(variable_type)}</span>',
        unsafe_allow_html=True,
    )


def _summary_value(df: Any, field: str) -> object:
    if getattr(df, "empty", True) or "field" not in df.columns or "value" not in df.columns:
        return ""
    values = df.loc[df["field"] == field, "value"]
    if values.empty:
        return ""
    return values.iloc[0]


def _compact_replicate_summary(df: Any) -> Any:
    columns = [
        "replicate_group",
        "n_replicates",
        "objective_mean",
        "objective_std",
        "objective_sem",
        "objective_min",
        "objective_max",
    ]
    if getattr(df, "empty", True):
        return df
    columns.extend(
        column
        for column in df.columns
        if column.endswith(("_mean", "_std", "_sem", "_min", "_max"))
        and column not in columns
    )
    return df.loc[:, [column for column in columns if column in df.columns]]


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
