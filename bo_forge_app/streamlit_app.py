"""Streamlit UI for local BO Forge campaign workflows."""

from __future__ import annotations

import math
import os
import time
from hashlib import sha1
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bo_forge.errors import BOForgeError
from bo_forge_app.service import CampaignAppService
from bo_forge_app.streamlit_helpers import (
    CONFIG_PATH_KEY,
    LAST_APPENDED_FINGERPRINT_KEY,
    LOG_PATH_KEY,
    NEW_CAMPAIGN_YAML_KEY,
    SESSION_KEY,
    STAGED_SUGGESTION_BUNDLE_KEY,
    active_variables_display,
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
    structured_stage_config_table,
    structured_stage_options,
)
from bo_forge_app.streamlit_style import (
    apply_forge_suite_style,
    forge_action_label,
    forge_status_label,
)

if TYPE_CHECKING:
    from bo_forge_app.service import CampaignViewData

    ViewDataLike = CampaignViewData | dict[str, Any]
else:
    ViewDataLike = dict[str, Any]

ACTIVE_PANEL_KEY = "bo_forge_active_panel"
CAMPAIGN_FILE_MODE_KEY = "bo_forge_campaign_file_mode"
FLASH_MESSAGE_KEY = "bo_forge_flash_message"
NEW_CAMPAIGN_FORM_YAML_KEY = "bo_forge_new_campaign_form_yaml"
NEW_CAMPAIGN_KIND_KEY = "bo_forge_new_campaign_kind"
REPORT_PREVIEW_KEY = "bo_forge_report_preview_text"
STAGED_FRESHNESS_MESSAGE_KEY = "bo_forge_staged_freshness_message"
SUGGEST_STAGE_KEY = "bo_forge_suggest_stage"
VALIDATION_CACHE_KEY = "bo_forge_validation_cache"
WORKFLOW_PANELS = ["Overview", "Suggest", "Resolve", "Reports", "Data"]


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

    _render_campaign_source_bar(st)
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
    active_panel = st.radio(
        "Workbench panel",
        WORKFLOW_PANELS,
        horizontal=True,
        key=ACTIVE_PANEL_KEY,
    )
    _render_active_workflow_panel(st, campaign, flags, str(active_panel))


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


def _render_campaign_source_bar(st: Any) -> None:
    campaign = st.session_state.get(SESSION_KEY)
    current_config = str(st.session_state.get(CONFIG_PATH_KEY, ""))
    current_log = str(st.session_state.get(LOG_PATH_KEY, ""))
    validation_label = _cached_validation_label(st, campaign)

    bundle = st.session_state.get(STAGED_SUGGESTION_BUNDLE_KEY)
    staged_label = "Staged batch present" if bundle is not None else "No staged batch"
    last_freshness_message = st.session_state.get(STAGED_FRESHNESS_MESSAGE_KEY)
    if bundle is not None and last_freshness_message:
        staged_label = str(last_freshness_message)

    st.markdown(
        f"""
        <section class="bf-source-bar">
          <div class="bf-panel-header">
            <div>
              <p class="bf-kicker">Campaign source</p>
              <h2 class="bf-panel-title">Local YAML + CSV</h2>
              <p class="bf-panel-note">
                Config: {escape(current_config or "not selected")}<br>
                Log: {escape(current_log or "not selected")}
              </p>
            </div>
            <div class="bf-chip-row">
              <span class="bf-chip">{escape(validation_label)}</span>
              <span class="bf-chip">{escape(staged_label)}</span>
            </div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    _render_flash_message(st)

    mode = st.radio(
        "Campaign file action",
        ["Load Existing", "Create Campaign"],
        horizontal=True,
        key=CAMPAIGN_FILE_MODE_KEY,
    )
    with st.expander(str(mode), expanded=campaign is None):
        if mode == "Load Existing":
            _render_load_existing_campaign(st)
        else:
            _render_create_new_campaign(st)


def _render_campaign_files_panel(st: Any) -> None:
    """Backward-compatible wrapper for tests and imports."""
    _render_campaign_source_bar(st)


def _render_active_workflow_panel(
    st: Any,
    campaign: Any,
    flags: dict[str, bool],
    active_panel: str,
) -> None:
    panel = active_panel if active_panel in WORKFLOW_PANELS else WORKFLOW_PANELS[0]
    view_data = _collect_panel_view_data(campaign, panel)
    renderers = {
        "Overview": lambda: _render_overview(st, campaign, view_data),
        "Suggest": lambda: _render_suggest(st, campaign),
        "Resolve": lambda: _render_resolve(st, campaign, flags, view_data),
        "Reports": lambda: _render_reports(st, campaign, flags, view_data),
        "Data": lambda: _render_data(st, campaign, flags, view_data),
    }
    renderers[panel]()


def _collect_panel_view_data(campaign: Any, panel: str) -> ViewDataLike:
    if hasattr(campaign, "collect_view_data"):
        return campaign.collect_view_data(panel)
    view_data: dict[str, Any] = {}
    with _TimedBlock(f"collect:{panel}"):
        if panel in {"Overview", "Data", "Reports"}:
            view_data["summary"] = campaign.summary()
            view_data["next_action"] = campaign.next_action()
        if panel in {"Overview", "Data"}:
            view_data["observed"] = campaign.observed_data()
            view_data["pending"] = campaign.pending_suggestions()
        if panel == "Resolve":
            view_data["pending"] = campaign.pending_suggestions()
            view_data["observable"] = observable_rows(campaign.config, campaign.df)
            if campaign.config.review.enabled:
                view_data["review_queue"] = campaign.review_queue()
        if panel in {"Overview", "Data"} and campaign.config.is_multi_objective:
            view_data["pareto_summary"] = campaign.pareto_summary()
            if panel == "Data":
                view_data["pareto_front"] = campaign.pareto_front()
        if panel in {"Overview", "Data"} and campaign.config.cost is not None:
            view_data["cost_summary"] = campaign.cost_summary()
        if panel in {"Overview", "Data"} and campaign.config.replicates.enabled:
            view_data["replicate_summary"] = campaign.replicate_summary()
        if panel in {"Overview", "Data", "Reports"} and campaign.config.is_structured_campaign:
            view_data["stage_summary"] = campaign.stage_summary()
        if panel in {"Overview", "Data", "Reports"} and campaign.config.fidelity is not None:
            view_data["fidelity_summary"] = campaign.fidelity_summary()
    return view_data


def _view_data_value(view_data: ViewDataLike, key: str, fallback: Any) -> Any:
    if key in view_data:
        return view_data[key]
    return fallback()


class _TimedBlock:
    def __init__(self, label: str) -> None:
        self.label = label
        self.started = 0.0

    def __enter__(self) -> None:
        self.started = time.perf_counter()

    def __exit__(self, *_args: object) -> None:
        if os.environ.get("BO_FORGE_STREAMLIT_DEBUG_TIMINGS"):
            elapsed_ms = (time.perf_counter() - self.started) * 1000.0
            print(f"[bo_forge_app] {self.label}: {elapsed_ms:.1f} ms")


def _render_load_existing_campaign(st: Any) -> None:
    _render_callout(
        st,
        "Write actions modify local files",
        "Append, review, and mark-observed actions modify the selected CSV log. "
        "Report and plot exports write files to the selected output path.",
    )

    with st.form("load_existing_campaign_form"):
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
        action_col, reload_col = st.columns([1, 1])
        with action_col:
            load_clicked = st.form_submit_button("Load campaign", type="primary")
        with reload_col:
            reload_clicked = st.form_submit_button("Reload from disk")

    if _path_changed(config_value, LOG_PATH_KEY, log_value):
        _clear_staged_suggestions(st)

    if load_clicked:
        _load_campaign_from_inputs(st, config_value, log_value)
    if reload_clicked:
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

    campaign_kind = st.radio(
        "Campaign kind",
        ["Single-objective", "Multi-objective", "Multi-fidelity qMFKG"],
        horizontal=True,
        key=NEW_CAMPAIGN_KIND_KEY,
        help=(
            "Multi-fidelity qMFKG creates a single-objective config with one "
            "continuous fidelity variable."
        ),
    )
    is_multi_objective = campaign_kind == "Multi-objective"
    is_multi_fidelity = campaign_kind == "Multi-fidelity qMFKG"
    if is_multi_fidelity:
        _render_callout(
            st,
            "Multi-fidelity qMFKG",
            "App-created multi-fidelity campaigns are single-objective, continuous-variable "
            "qMFKG campaigns. Advanced qMFKG defaults stay editable in the YAML preview.",
        )

    _render_section_label(st, "Objective")
    objective_name = "activity"
    objective_direction = "maximize"
    objectives: list[dict[str, object]] | None = None
    if is_multi_objective:
        objective_count = st.number_input(
            "Objective count",
            min_value=2,
            max_value=4,
            value=2,
            key="new_campaign_objective_count",
        )
        objectives = _collect_new_campaign_objectives(st, int(objective_count))
    else:
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
        if is_multi_fidelity:
            batch_size = st.number_input(
                "Batch size",
                min_value=1,
                max_value=1,
                value=1,
                key="new_bo_batch_size_multi_fidelity",
                disabled=True,
                help="qMFKG model-based suggestions are sequential in v1.4.",
            )
        else:
            batch_size = st.number_input(
                "Batch size",
                min_value=1,
                value=1,
                key="new_bo_batch_size",
            )
    with bo_col_2:
        initial_design_size = st.number_input(
            "Initial design size",
            min_value=1,
            value=4 if is_multi_fidelity else 8,
            key=(
                "new_bo_initial_design_size_multi_fidelity"
                if is_multi_fidelity
                else "new_bo_initial_design_size"
            ),
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
        variables = _collect_new_campaign_variables(
            st,
            int(variable_count),
            continuous_only=is_multi_fidelity,
        )
        review_enabled = False
        replicates_enabled = False
        cost_settings = None
        fidelity_settings = None
        bo_overrides = None
        if is_multi_fidelity:
            _render_section_label(st, "Fidelity")
            fidelity_settings = _collect_new_campaign_fidelity_settings(st, variables)
            bo_overrides = {
                "acquisition": "qmf_kg",
                "batch_size": 1,
                "raw_samples": 8,
                "num_restarts": 1,
                "mc_samples": 16,
                "min_normalized_distance": 0.0,
            }
            _render_artifact_note(
                st,
                "qMFKG defaults",
                "Generated YAML uses num_fantasies=8, raw_samples=8, num_restarts=1, "
                "mc_samples=16, and min_normalized_distance=0.0.",
            )
            review_enabled = st.checkbox(
                "Enable review",
                value=False,
                key="new_campaign_review_enabled_multi_fidelity",
            )
        elif is_multi_objective:
            _render_section_label(st, "Advanced sections")
            review_enabled = st.checkbox(
                "Enable review",
                value=False,
                key="new_campaign_review_enabled",
            )
            replicates_enabled = st.checkbox(
                "Enable replicates",
                value=False,
                key="new_campaign_replicates_enabled",
            )
            cost_enabled = st.checkbox(
                "Enable deterministic cost",
                value=False,
                key="new_campaign_cost_enabled",
            )
            if cost_enabled:
                cost_col_1, cost_col_2, cost_col_3 = st.columns(3)
                with cost_col_1:
                    cost_expression = st.text_input(
                        "Cost expression",
                        value="1.0",
                        key="new_campaign_cost_expression",
                    )
                with cost_col_2:
                    cost_weight = st.number_input(
                        "Cost weight",
                        min_value=0.0,
                        value=1.0,
                        key="new_campaign_cost_weight",
                    )
                with cost_col_3:
                    cost_budget = st.number_input(
                        "Budget",
                        min_value=0.0,
                        value=100.0,
                        key="new_campaign_cost_budget",
                    )
                cost_settings = {
                    "expression": cost_expression,
                    "weight": float(cost_weight),
                    "budget": float(cost_budget),
                }
        generated_yaml = build_campaign_yaml_text(
            campaign_name=campaign_name,
            objective_name=objective_name,
            objective_direction=str(objective_direction),
            variables=variables,
            batch_size=int(batch_size),
            initial_design_size=int(initial_design_size),
            initial_design_method=str(initial_design_method),
            random_seed=int(random_seed),
            objectives=objectives,
            review_enabled=review_enabled,
            replicates_enabled=replicates_enabled,
            cost=cost_settings,
            fidelity=fidelity_settings,
            bo_overrides=bo_overrides,
        )
    except ValueError as exc:
        st.error(f"Could not build YAML preview: {exc}")

    if NEW_CAMPAIGN_YAML_KEY not in st.session_state:
        st.session_state[NEW_CAMPAIGN_YAML_KEY] = generated_yaml
        st.session_state[NEW_CAMPAIGN_FORM_YAML_KEY] = generated_yaml
    if st.button("Update YAML preview from form"):
        st.session_state[NEW_CAMPAIGN_YAML_KEY] = generated_yaml
        st.session_state[NEW_CAMPAIGN_FORM_YAML_KEY] = generated_yaml

    preview_is_stale = st.session_state.get(NEW_CAMPAIGN_FORM_YAML_KEY) != generated_yaml

    _render_section_label(st, "Generated YAML Preview")
    _render_artifact_note(
        st,
        "Editable before writing",
        "Advanced edits are allowed, but the YAML must pass BO Forge config validation "
        "before files are written. Create campaign writes this editable YAML preview.",
    )
    edited_yaml = st.text_area(
        "Campaign YAML",
        height=360,
        key=NEW_CAMPAIGN_YAML_KEY,
    )
    if preview_is_stale:
        st.warning(
            "Structured form values changed after this YAML preview was generated. "
            "Use Update YAML preview from form before creating the campaign."
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
            if preview_is_stale:
                st.error("Update YAML preview from form before creating the campaign.")
                return
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


def _collect_new_campaign_objectives(st: Any, objective_count: int) -> list[dict[str, object]]:
    objectives: list[dict[str, object]] = []
    for index in range(objective_count):
        name_col, direction_col, reference_col = st.columns(3)
        with name_col:
            name = st.text_input(
                f"Objective {index + 1} name",
                value=["yield", "selectivity", "waste", "energy_use"][index],
                key=f"new_objective_{index}_name",
            )
        with direction_col:
            direction = st.selectbox(
                f"Objective {index + 1} direction",
                ["maximize", "minimize"],
                key=f"new_objective_{index}_direction",
            )
        with reference_col:
            reference_point = st.number_input(
                f"Objective {index + 1} reference point",
                value=0.0,
                key=f"new_objective_{index}_reference_point",
            )
        objectives.append(
            {
                "name": name,
                "direction": str(direction),
                "reference_point": float(reference_point),
            }
        )
    return objectives


def _collect_new_campaign_variables(
    st: Any,
    variable_count: int,
    *,
    continuous_only: bool = False,
) -> list[dict[str, object]]:
    variables: list[dict[str, object]] = []
    key_prefix = "new_fidelity_variable" if continuous_only else "new_variable"
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
                    value=(
                        "fidelity"
                        if continuous_only and index == variable_count - 1
                        else f"x{index + 1}"
                    ),
                    placeholder="temperature",
                    key=f"{key_prefix}_{index}_name",
                )
            with type_col:
                if continuous_only:
                    variable_type = "continuous"
                    st.markdown("**Variable type**  \ncontinuous")
                else:
                    variable_type = st.selectbox(
                        "Variable type",
                        ["continuous", "integer", "discrete", "categorical"],
                        key=f"{key_prefix}_{index}_type",
                    )
            _render_variable_type_badge(st, str(variable_type))
            variable: dict[str, object] = {"name": name, "type": variable_type}
            if variable_type in {"continuous", "integer"}:
                lower_col, upper_col = st.columns(2)
                with lower_col:
                    lower = st.number_input(
                        "Lower",
                        value=0.0,
                        key=f"{key_prefix}_{index}_lower",
                    )
                with upper_col:
                    upper = st.number_input(
                        "Upper",
                        value=1.0,
                        key=f"{key_prefix}_{index}_upper",
                    )
                variable["lower"] = int(lower) if variable_type == "integer" else float(lower)
                variable["upper"] = int(upper) if variable_type == "integer" else float(upper)
            elif variable_type == "discrete":
                values_text = st.text_input(
                    "Discrete values",
                    value="0.0, 0.5, 1.0",
                    placeholder="0.1, 0.2, 0.5",
                    key=f"{key_prefix}_{index}_discrete_values",
                    help="Discrete values must be comma-separated numbers.",
                )
                variable["values"] = parse_discrete_values_text(values_text, name)
            else:
                values_text = st.text_input(
                    "Categorical labels",
                    value="A, B, C",
                    placeholder="MeCN, DMF, Water",
                    key=f"{key_prefix}_{index}_categorical_values",
                    help=(
                        "Categorical labels are case-sensitive. Empty or duplicate "
                        "labels are rejected."
                    ),
                )
                variable["values"] = parse_categorical_values_text(values_text, name)
            variables.append(variable)
    return variables


def _collect_new_campaign_fidelity_settings(
    st: Any,
    variables: list[dict[str, object]],
) -> dict[str, object]:
    continuous_variables = [
        variable
        for variable in variables
        if variable.get("type") == "continuous"
        and "lower" in variable
        and "upper" in variable
    ]
    if not continuous_variables:
        raise ValueError("Multi-fidelity qMFKG campaigns require a continuous variable.")

    variable_names = [str(variable["name"]) for variable in continuous_variables]
    default_index = next(
        (index for index, name in enumerate(variable_names) if name == "fidelity"),
        len(variable_names) - 1,
    )
    fidelity_variable = st.selectbox(
        "Fidelity variable",
        variable_names,
        index=default_index,
        key="new_campaign_fidelity_variable",
    )
    selected_variable = continuous_variables[variable_names.index(str(fidelity_variable))]
    lower = float(selected_variable["lower"])
    upper = float(selected_variable["upper"])
    target = st.number_input(
        "Target fidelity",
        min_value=lower,
        max_value=upper,
        value=upper,
        key=f"new_campaign_fidelity_target_{fidelity_variable}",
        help="Defaults to the selected fidelity variable's upper bound.",
    )
    return {
        "variable": str(fidelity_variable),
        "target": float(target),
        "fixed_cost": 0.01,
        "fidelity_cost_weight": 1.0,
        "num_fantasies": 8,
    }


def _create_campaign_from_inputs(
    st: Any,
    edited_yaml: str,
    config_output: str,
    log_output: str,
) -> None:
    try:
        config_path = resolve_path_input(config_output, "Config output")
        log_path = resolve_path_input(log_output, "Log output")
        session = create_campaign_files(
            config_text=edited_yaml,
            config_path=config_path,
            log_path=log_path,
        )
    except (BOForgeError, OSError, ValueError) as exc:
        _render_result_card(st, "Could not create campaign", str(exc), success=False)
        return

    st.session_state[CONFIG_PATH_KEY] = str(config_path)
    st.session_state[LOG_PATH_KEY] = str(log_path)
    campaign = CampaignAppService.from_session(session)
    st.session_state[SESSION_KEY] = campaign
    _clear_staged_suggestions(st)
    _clear_report_preview(st)
    _refresh_validation_cache(st, campaign, config_path, log_path)
    _flash_and_rerun(
        st,
        f"Campaign created and loaded. Config: {config_path} | Log: {log_path}.",
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
        campaign = CampaignAppService.load(config_path, log_path)
    except (BOForgeError, OSError, ValueError) as exc:
        st.error(str(exc))
        return

    st.session_state[CONFIG_PATH_KEY] = str(config_path)
    st.session_state[LOG_PATH_KEY] = str(log_path)
    st.session_state[SESSION_KEY] = campaign
    _clear_staged_suggestions(st)
    _clear_report_preview(st)
    _refresh_validation_cache(st, campaign, config_path, log_path)
    _flash_and_rerun(st, "Campaign loaded.")


def _render_overview(st: Any, campaign: Any, view_data: ViewDataLike) -> None:
    _render_panel_intro(
        st,
        "Overview",
        "Inspect campaign status, next action, and compact decision summaries.",
    )
    validation_state = _cached_validation_state(st, campaign)
    if validation_state["label"] == "Validation issue":
        st.error(f"Validation failed: {validation_state['error']}")
        _render_table_section(
            st,
            "Campaign Log",
            campaign.df,
            empty_kind="pending_suggestions",
            expanded_raw=True,
        )
        return
    if validation_state["label"] == "Valid":
        _render_result_card(
            st,
            "Campaign log is valid",
            "The selected CSV matches the active config.",
        )
    else:
        _render_callout(
            st,
            str(validation_state["label"]),
            "The config or log file metadata changed. Reload from disk to refresh validation.",
        )

    summary = _view_data_value(view_data, "summary", campaign.summary)
    _render_campaign_state_blocks(st, campaign, view_data)
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

    if campaign.config.is_multi_objective:
        _render_table_section(
            st,
            "Pareto Summary",
            _view_data_value(view_data, "pareto_summary", campaign.pareto_summary),
            empty_kind="report_preview",
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
        _render_cost_metric_cards(st, campaign, view_data.get("cost_summary"))
    if campaign.config.replicates.enabled:
        replicate_summary = _view_data_value(
            view_data,
            "replicate_summary",
            campaign.replicate_summary,
        )
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
            _compact_replicate_summary(replicate_summary),
            empty_kind="replicate_summary",
            raw_df=replicate_summary,
            expanded_raw=False,
        )
    if campaign.config.fidelity is not None:
        _render_table_section(
            st,
            "Fidelity Summary",
            _view_data_value(view_data, "fidelity_summary", campaign.fidelity_summary),
            empty_kind="fidelity_summary",
            expanded_raw=False,
        )
    if campaign.config.is_structured_campaign:
        _render_table_section(
            st,
            "Stage Summary",
            _view_data_value(view_data, "stage_summary", campaign.stage_summary),
            empty_kind="report_preview",
            expanded_raw=False,
        )

    observed = view_data.get("observed")
    pending = view_data.get("pending")
    _render_metric_grid(
        st,
        [
            ("Observed preview rows", min(len(observed), 8) if observed is not None else ""),
            ("Pending preview rows", min(len(pending), 8) if pending is not None else ""),
        ],
    )


def _render_data(
    st: Any,
    campaign: Any,
    flags: dict[str, bool],
    view_data: ViewDataLike,
) -> None:
    _render_panel_intro(
        st,
        "Data",
        "Inspect full raw tables and backend summaries.",
    )
    summary = _view_data_value(view_data, "summary", campaign.summary)
    next_action = _view_data_value(view_data, "next_action", campaign.next_action)
    observed = _view_data_value(view_data, "observed", campaign.observed_data)
    pending = _view_data_value(view_data, "pending", campaign.pending_suggestions)

    _render_table_section(
        st,
        "Summary",
        summary,
        empty_kind="report_preview",
        expanded_raw=True,
    )
    _render_table_section(
        st,
        "Next Action",
        next_action,
        empty_kind="pending_suggestions",
        expanded_raw=False,
    )
    _render_table_section(
        st,
        "Observed Rows",
        observed,
        empty_kind="observed_rows",
        expanded_raw=False,
    )
    _render_table_section(
        st,
        "Pending Suggestions",
        pending,
        empty_kind="pending_suggestions",
        expanded_raw=True,
    )
    if campaign.config.is_multi_objective:
        _render_table_section(
            st,
            "Pareto Summary",
            _view_data_value(view_data, "pareto_summary", campaign.pareto_summary),
            empty_kind="report_preview",
            expanded_raw=False,
        )
        _render_table_section(
            st,
            "Pareto Front",
            _view_data_value(view_data, "pareto_front", campaign.pareto_front),
            empty_kind="observed_rows",
            expanded_raw=False,
        )
    if flags["has_cost"]:
        _render_table_section(
            st,
            "Cost Summary",
            _view_data_value(view_data, "cost_summary", campaign.cost_summary),
            empty_kind="cost_summary",
            expanded_raw=False,
        )
    if flags["has_replicates"]:
        replicate_summary = _view_data_value(
            view_data,
            "replicate_summary",
            campaign.replicate_summary,
        )
        _render_table_section(
            st,
            "Replicate Summary",
            replicate_summary,
            empty_kind="replicate_summary",
            expanded_raw=False,
        )
    if campaign.config.fidelity is not None:
        _render_table_section(
            st,
            "Fidelity Summary",
            _view_data_value(view_data, "fidelity_summary", campaign.fidelity_summary),
            empty_kind="fidelity_summary",
            expanded_raw=False,
        )
    if campaign.config.is_structured_campaign:
        _render_table_section(
            st,
            "Stage Summary",
            _view_data_value(view_data, "stage_summary", campaign.stage_summary),
            empty_kind="report_preview",
            expanded_raw=False,
        )
    with st.expander("Show full raw campaign log", expanded=False):
        st.dataframe(format_dataframe_for_display(campaign.df), width="stretch")


def _render_campaign_state_blocks(
    st: Any,
    campaign: Any,
    view_data: ViewDataLike | None = None,
) -> None:
    view_data = view_data or {}
    summary = view_data.get("summary")
    status = str(_summary_value(summary, "campaign_status")) if summary is not None else ""
    if not status:
        status = campaign.campaign_status()
    next_action = _view_data_value(view_data, "next_action", campaign.next_action)
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
    stage_options = structured_stage_options(campaign.config)
    if stage_options:
        _render_table_section(
            st,
            "Configured Stages",
            structured_stage_config_table(campaign.config),
            empty_kind="report_preview",
            expanded_raw=False,
        )
    selected_stage = None
    if stage_options:
        selected_stage = str(
            st.selectbox(
                "Suggestion stage",
                stage_options,
                key=SUGGEST_STAGE_KEY,
                help="Structured campaigns require an explicit stage for suggestions.",
            )
        )
        _render_artifact_note(
            st,
            "Active variables",
            active_variables_display(campaign.config, selected_stage),
        )
    is_multi_fidelity = campaign.config.fidelity is not None
    if is_multi_fidelity:
        _render_artifact_note(
            st,
            "qMFKG suggestions",
            "Multi-fidelity qMFKG suggestions are sequential in v1.4, so the "
            "dry-run batch size is capped at 1.",
        )
    with st.form("suggest_dry_run_form"):
        batch_size = st.number_input(
            "Batch size",
            min_value=1,
            max_value=1 if is_multi_fidelity else 32,
            value=1 if is_multi_fidelity else max(1, int(campaign.config.bo.batch_size)),
            step=1,
            disabled=is_multi_fidelity,
        )
        generate_clicked = st.form_submit_button(
            "Generate suggestions (dry run)",
            type="primary",
        )

    if generate_clicked:
        try:
            if hasattr(campaign, "suggest_dry_run"):
                result = campaign.suggest_dry_run(int(batch_size), stage=selected_stage)
                suggestions = result.suggestions
                bundle = result.bundle
            else:
                if selected_stage is None:
                    suggestions = campaign.suggest_next(batch_size=int(batch_size))
                else:
                    suggestions = campaign.suggest_next(
                        batch_size=int(batch_size),
                        stage=selected_stage,
                    )
                bundle = make_staged_suggestion_bundle(
                    suggestions,
                    config_path,
                    log_path,
                    stage=selected_stage,
                )
        except (BOForgeError, OSError, ValueError) as exc:
            st.error(str(exc))
        else:
            st.session_state[STAGED_SUGGESTION_BUNDLE_KEY] = bundle
            st.session_state.pop(STAGED_FRESHNESS_MESSAGE_KEY, None)
            st.success("Suggestions staged. Review them before appending.")

    bundle = st.session_state.get(STAGED_SUGGESTION_BUNDLE_KEY)
    suggestions = staged_suggestions_from_bundle(bundle)
    if suggestions.empty:
        _render_empty_state(st, *empty_state_message("staged_suggestions"))
        return

    if selected_stage is None:
        raw_reason = _current_invalidation_reason(st, bundle)
    else:
        raw_reason = _current_invalidation_reason(st, bundle, stage=selected_stage)
    if raw_reason is None:
        st.session_state.pop(STAGED_FRESHNESS_MESSAGE_KEY, None)
    disabled_reason = append_disabled_reason(
        bundle,
        config_path,
        log_path,
        st.session_state.get(LAST_APPENDED_FINGERPRINT_KEY),
        stage=selected_stage,
    )
    if raw_reason and raw_reason != "No staged suggestions.":
        st.session_state[STAGED_FRESHNESS_MESSAGE_KEY] = raw_reason
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

    with st.form("staged_suggestions_export_form"):
        export_path = Path(
            st.text_input(
                "Staged suggestions CSV export path",
                value=str(default_export_path(log_path, "staged_suggestions", "csv")),
                key="staged_suggestions_export_path",
            )
        )
        export_clicked = st.form_submit_button("Export staged suggestions CSV")
    if export_clicked:
        try:
            if hasattr(campaign, "export_staged_suggestions"):
                written_path = campaign.export_staged_suggestions(bundle, export_path)
            else:
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

    with st.form("append_staged_suggestions_form"):
        append_clicked = st.form_submit_button(
            "Append staged suggestions",
            disabled=disabled_reason is not None,
        )
    if append_clicked:
        try:
            if hasattr(campaign, "append_staged"):
                result = campaign.append_staged(
                    bundle,
                    st.session_state.get(LAST_APPENDED_FINGERPRINT_KEY),
                    stage=selected_stage,
                )
                campaign = result.service
                appended_fingerprint = result.appended_fingerprint
            else:
                campaign.append_suggestions(suggestions)
                appended_fingerprint = str(bundle["suggestions_fingerprint"])
        except (BOForgeError, ValueError) as exc:
            st.error(str(exc))
            return
        st.session_state[LAST_APPENDED_FINGERPRINT_KEY] = appended_fingerprint
        _clear_staged_suggestions(st)
        _clear_report_preview(st)
        st.session_state[SESSION_KEY] = campaign
        _refresh_validation_cache(st, campaign, config_path, log_path)
        _flash_and_rerun(st, "Staged suggestions appended to the campaign log.")


def _render_resolve(
    st: Any,
    campaign: Any,
    flags: dict[str, bool],
    view_data: ViewDataLike | None = None,
) -> None:
    _render_panel_intro(
        st,
        "Resolve",
        "Review suggested rows and record experimental outcomes.",
    )
    view_data = view_data or {}
    pending = _view_data_value(view_data, "pending", campaign.pending_suggestions)
    with st.expander("Pending Suggestions", expanded=False):
        if pending.empty:
            _render_empty_state(st, *empty_state_message("pending_suggestions"))
        else:
            st.dataframe(compact_dataframe(pending), width="stretch")
            with st.expander("Show full raw pending suggestions", expanded=False):
                st.dataframe(format_dataframe_for_display(pending), width="stretch")
    observable = _view_data_value(
        view_data,
        "observable",
        lambda: observable_rows(campaign.config, campaign.df),
    )

    if flags["has_review"]:
        st.subheader("Review Queue")
        review_queue = _view_data_value(view_data, "review_queue", campaign.review_queue)
        if review_queue.empty:
            _render_empty_state(st, *empty_state_message("review_queue"))
        else:
            st.dataframe(compact_dataframe(review_queue), width="stretch")
            with st.expander("Show full raw review queue", expanded=False):
                st.dataframe(format_dataframe_for_display(review_queue), width="stretch")
        if not review_queue.empty:
            with st.form("review_decision_form"):
                row_id = st.selectbox("Review row_id", review_queue["row_id"].astype(str).tolist())
                decision = st.selectbox("Decision", ["accept", "reject", "defer"])
                note = st.text_input("Review note", value="")
                review_clicked = st.form_submit_button("Apply review decision")
            if review_clicked:
                try:
                    if hasattr(campaign, "review"):
                        result = campaign.review(row_id, decision, note)
                        campaign = result.service
                    else:
                        campaign.review_suggestion(row_id, decision, note)
                except BOForgeError as exc:
                    st.error(str(exc))
                else:
                    _clear_staged_suggestions(st)
                    _clear_report_preview(st)
                    st.session_state[SESSION_KEY] = campaign
                    config_path, log_path = _current_paths(st)
                    _refresh_validation_cache(st, campaign, config_path, log_path)
                    _flash_and_rerun(st, "Review decision recorded.")
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
    if observable.empty:
        return

    st.subheader(
        "Record Coupled Objectives" if campaign.config.is_multi_objective else "Mark Observed"
    )
    option_map = observable_row_options(campaign.config, campaign.df)
    with st.form("mark_observed_form"):
        selected_label = st.selectbox("Observed suggestion", list(option_map))
        observed_row_id = option_map[selected_label]
        selected_row = observable.loc[observable["row_id"].astype(str) == observed_row_id]
        if not selected_row.empty:
            _render_selected_row_preview(st, campaign, selected_row.iloc[0])
        if campaign.config.is_multi_objective:
            objective_inputs = {
                objective.name: st.text_input(
                    f"Observed {objective.name}",
                    value="",
                    key=_stable_widget_key(
                        "observed_objective",
                        observed_row_id,
                        objective.name,
                    ),
                    help="Required. Enter a finite numeric value.",
                )
                for objective in campaign.config.objectives
            }
            actual_cost_text = _render_actual_cost_input(
                st,
                flags,
                key_suffix=observed_row_id,
            )
            mark_clicked = st.form_submit_button("Record coupled objectives")
        else:
            objective_name = campaign.config.objective.name
            objective_value = st.number_input(
                f"Observed {objective_name}",
                value=0.0,
                format="%.8f",
                key=_stable_widget_key("observed_objective", observed_row_id, objective_name),
            )
            actual_cost_text = _render_actual_cost_input(
                st,
                flags,
                key_suffix=observed_row_id,
            )
            mark_clicked = st.form_submit_button("Mark row observed")

    if not mark_clicked:
        return

    selected_row = observable.loc[observable["row_id"].astype(str) == observed_row_id]
    if campaign.config.is_multi_objective:
        try:
            objective_values = _parse_multi_objective_inputs(
                objective_inputs,
                campaign.config.objective_names,
            )
            actual_cost = _parse_actual_cost_input(actual_cost_text)
            if hasattr(campaign, "mark_observed"):
                result = campaign.mark_observed(
                    row_id=observed_row_id,
                    objective_values=objective_values,
                    actual_cost=actual_cost,
                )
                if hasattr(result, "service"):
                    campaign = result.service
        except (BOForgeError, ValueError) as exc:
            st.error(str(exc))
        else:
            _clear_staged_suggestions(st)
            _clear_report_preview(st)
            st.session_state[SESSION_KEY] = campaign
            config_path, log_path = _current_paths(st)
            _refresh_validation_cache(st, campaign, config_path, log_path)
            _flash_and_rerun(st, "Coupled objective values recorded.")
        return

    try:
        actual_cost = _parse_actual_cost_input(actual_cost_text)
        result = campaign.mark_observed(
            row_id=observed_row_id,
            objective_value=float(objective_value),
            actual_cost=None if actual_cost is None else float(actual_cost),
        )
        if hasattr(result, "service"):
            campaign = result.service
    except (BOForgeError, ValueError) as exc:
        st.error(str(exc))
    else:
        _clear_staged_suggestions(st)
        _clear_report_preview(st)
        st.session_state[SESSION_KEY] = campaign
        config_path, log_path = _current_paths(st)
        _refresh_validation_cache(st, campaign, config_path, log_path)
        _flash_and_rerun(st, "Observation recorded.")


def _render_actual_cost_input(
    st: Any,
    flags: dict[str, bool],
    *,
    key_suffix: str | None = None,
) -> str | None:
    if not flags["has_cost"]:
        return None
    return st.text_input(
        "Actual cost (optional)",
        value="",
        key=_stable_widget_key("actual_cost", key_suffix or "default"),
        help="Leave blank to use the estimated cost.",
    )


def _parse_actual_cost_input(actual_cost_text: str | None) -> float | None:
    if actual_cost_text is None or not actual_cost_text.strip():
        return None
    try:
        actual_cost = float(actual_cost_text)
    except ValueError as exc:
        raise ValueError("Actual cost must be numeric when provided.") from exc
    if not math.isfinite(actual_cost) or actual_cost < 0:
        raise ValueError("Actual cost must be finite and nonnegative when provided.")
    return actual_cost


def _parse_multi_objective_inputs(
    values: dict[str, str],
    objective_names: list[str],
) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for name in objective_names:
        raw_value = values.get(name, "").strip()
        if not raw_value:
            raise ValueError(f"Observed {name} is required.")
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ValueError(f"Observed {name} must be numeric.") from exc
        if not math.isfinite(value):
            raise ValueError(f"Observed {name} must be finite.")
        parsed[name] = value
    return parsed


def _stable_widget_key(namespace: str, *parts: object) -> str:
    raw = "|".join(str(part) for part in (namespace, *parts))
    digest = sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"{namespace}_{digest}"


def _render_reports(
    st: Any,
    campaign: Any,
    flags: dict[str, bool],
    view_data: ViewDataLike | None = None,
) -> None:
    _render_panel_intro(
        st,
        "Reports",
        "Preview reports and export campaign figures.",
    )
    view_data = view_data or {}
    _, log_path = _current_paths(st)

    summary = _view_data_value(view_data, "summary", campaign.summary)
    _render_metric_grid(
        st,
        [
            ("Status", humanize_campaign_status(str(_summary_value(summary, "campaign_status")))),
            ("Observed", _summary_value(summary, "observed_rows")),
            ("Pending", _summary_value(summary, "pending_suggestions")),
            (
                "Hypervolume"
                if campaign.config.is_multi_objective
                else "Best objective",
                _summary_value(summary, "hypervolume")
                if campaign.config.is_multi_objective
                else _summary_value(summary, "best_objective_value"),
            ),
        ],
    )

    with st.form("report_actions_form"):
        report_path = Path(
            st.text_input(
                "Report export path",
                value=str(default_export_path(log_path, "campaign_report", "txt")),
            )
        )
        preview_clicked = st.form_submit_button("Preview report")
        export_clicked = st.form_submit_button("Export report")
    if preview_clicked:
        try:
            if hasattr(campaign, "report_text"):
                st.session_state[REPORT_PREVIEW_KEY] = campaign.report_text()
            else:
                st.session_state[REPORT_PREVIEW_KEY] = campaign_report_text(campaign)
        except BOForgeError as exc:
            st.error(str(exc))
    report_text = st.session_state.get(REPORT_PREVIEW_KEY)
    if report_text:
        with st.expander("Raw report text", expanded=True):
            st.text_area("Campaign report", value=str(report_text), height=360)
    if export_clicked:
        try:
            written_path = campaign.export_report(report_path)
        except (BOForgeError, OSError) as exc:
            st.error(str(exc))
        else:
            st.success(f"Wrote report: {written_path}")

    plot_options = _available_plot_options(campaign, flags, log_path)
    if not plot_options:
        _render_empty_state(st, *empty_state_message("plots"))
        return
    labels = [option["label"] for option in plot_options]
    selected_label = st.selectbox("Plot kind", labels, key="reports_plot_kind")
    selected_plot = next(option for option in plot_options if option["label"] == selected_label)
    _render_plot_controls(
        st,
        str(selected_plot["label"]),
        str(selected_plot["key"]),
        selected_plot["plotter"],
        selected_plot["path"],
    )


def _available_plot_options(
    campaign: Any,
    flags: dict[str, bool],
    log_path: Path,
) -> list[dict[str, Any]]:
    plot_kinds = (
        campaign.available_plot_kinds()
        if hasattr(campaign, "available_plot_kinds")
        else available_plot_kinds(campaign.config)
    )
    options: list[dict[str, Any]] = []
    mapping = {
        "progress": ("Progress", "plot_progress"),
        "diagnostics": ("Diagnostics", "plot_diagnostics"),
        "pareto": ("Pareto", "plot_pareto"),
        "pareto_parallel": ("Pareto Parallel", "plot_pareto_parallel"),
        "hypervolume": ("Hypervolume", "plot_hypervolume"),
        "stage_diagnostics": ("Stage Diagnostics", "plot_stage_diagnostics"),
        "fidelity_diagnostics": ("Fidelity Diagnostics", "plot_fidelity_diagnostics"),
    }
    for kind, (label, plotter_name) in mapping.items():
        if kind in plot_kinds:
            plotter = (
                _service_plotter(campaign, kind)
                if hasattr(campaign, "plot")
                else getattr(campaign, plotter_name)
            )
            options.append(
                {
                    "label": label,
                    "key": kind,
                    "plotter": plotter,
                    "path": default_export_path(log_path, kind, "png"),
                }
            )
    if flags["has_cost"]:
        plotter = (
            _service_plotter(campaign, "cost_progress")
            if hasattr(campaign, "plot")
            else campaign.plot_cost_progress
        )
        options.append(
            {
                "label": "Cost Progress",
                "key": "cost_progress",
                "plotter": plotter,
                "path": default_export_path(log_path, "cost_progress", "png"),
            }
        )
    if flags["has_replicates"]:
        plotter = (
            _service_plotter(campaign, "replicates")
            if hasattr(campaign, "plot")
            else campaign.plot_replicates
        )
        options.append(
            {
                "label": "Replicates",
                "key": "replicates",
                "plotter": plotter,
                "path": default_export_path(log_path, "replicates", "png"),
            }
        )
    return options


def _service_plotter(campaign: Any, kind: str) -> Any:
    def plotter(*, save_path: Path | None = None) -> object:
        return campaign.plot(kind, save_path=save_path)

    return plotter


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
    with st.form(f"{key_suffix}_plot_form"):
        export_path = Path(
            st.text_input(
                f"{label} export path",
                value=str(default_path),
                key=f"{key_suffix}_export_path",
            )
        )
        col_show, col_export = st.columns(2)
        with col_show:
            show_clicked = st.form_submit_button(f"Show {label.lower()} plot")
        with col_export:
            export_clicked = st.form_submit_button(f"Export {label.lower()} plot")

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
            '<div class="forge-metric">'
            f'<p class="forge-metric-label">{escape(str(label))}</p>'
            f'<p class="forge-metric-value">{escape(str(display_value))}</p>'
            "</div>"
        )
    st.markdown(
        f'<div class="forge-metric-grid">{"".join(cards)}</div>',
        unsafe_allow_html=True,
    )


def _render_cost_metric_cards(st: Any, campaign: Any, cost_summary: Any | None = None) -> None:
    summary = cost_summary if cost_summary is not None else campaign.cost_summary()
    if campaign.config.is_multi_objective:
        metrics = [
            ("Budget", _summary_value(summary, "budget")),
            ("Remaining", _summary_value(summary, "budget_remaining")),
            ("Current hypervolume", _summary_value(summary, "current_hypervolume")),
            ("Pareto count", _summary_value(summary, "pareto_count")),
        ]
    else:
        metrics = [
            ("Observed cost", _summary_value(summary, "total_observed_cost")),
            ("Accepted pending", _summary_value(summary, "accepted_pending_cost")),
            ("Budget", _summary_value(summary, "budget")),
            ("Remaining", _summary_value(summary, "budget_remaining")),
            ("Best objective", _summary_value(summary, "best_observed_objective")),
        ]
    _render_metric_grid(st, metrics)


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
    variables = campaign.config.variables
    if campaign.config.is_structured_campaign:
        stage_name = str(row.get("stage", ""))
        metrics.append(("Stage", stage_name))
        try:
            active_names = set(campaign.config.active_variable_names_for_stage(stage_name))
        except BOForgeError:
            active_names = set()
        else:
            variables = tuple(
                variable
                for variable in campaign.config.variables
                if variable.name in active_names
            )
    for variable in variables[:6]:
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


def _cached_validation_label(st: Any, campaign: Any | None) -> str:
    return str(_cached_validation_state(st, campaign)["label"])


def _cached_validation_state(st: Any, campaign: Any | None) -> dict[str, str]:
    if campaign is None:
        return {"label": "Not loaded", "error": ""}
    cache = st.session_state.get(VALIDATION_CACHE_KEY)
    expected = _validation_cache_signature(
        st.session_state.get(CONFIG_PATH_KEY, ""),
        st.session_state.get(LOG_PATH_KEY, ""),
    )
    if not isinstance(cache, dict):
        return {"label": "Reload to validate", "error": ""}
    if cache.get("signature") != expected:
        return {"label": "Reload to validate", "error": ""}
    return {
        "label": str(cache.get("label", "Reload to validate")),
        "error": str(cache.get("error", "")),
    }


def _refresh_validation_cache(
    st: Any,
    campaign: Any,
    config_path: Path,
    log_path: Path,
) -> None:
    try:
        result = campaign.validate()
    except BOForgeError as exc:
        label = "Validation issue"
        error = str(exc)
    else:
        if hasattr(result, "label"):
            label = str(result.label)
            error = str(getattr(result, "message", ""))
        else:
            label = "Valid"
            error = ""
    st.session_state[VALIDATION_CACHE_KEY] = {
        "signature": _validation_cache_signature(config_path, log_path),
        "label": label,
        "error": error,
    }


def _validation_cache_signature(config_path: object, log_path: object) -> tuple[object, object]:
    return (_file_metadata_signature(config_path), _file_metadata_signature(log_path))


def _file_metadata_signature(path_value: object) -> tuple[str, int | None, int | None]:
    path = Path(str(path_value)).expanduser()
    resolved = path.resolve(strict=False)
    try:
        stat_result = path.stat()
    except OSError:
        return (str(resolved), None, None)
    return (str(resolved), int(stat_result.st_size), int(stat_result.st_mtime_ns))


def _render_flash_message(st: Any) -> None:
    message = st.session_state.pop(FLASH_MESSAGE_KEY, None)
    if message:
        st.success(str(message))


def _flash_and_rerun(st: Any, message: str) -> None:
    st.session_state[FLASH_MESSAGE_KEY] = message
    rerun = getattr(st, "rerun", None)
    if callable(rerun):
        rerun()
    elif hasattr(st, "success"):
        st.success(message)


def _current_invalidation_reason(
    st: Any,
    bundle: dict[str, object] | None,
    *,
    stage: str | None = None,
) -> str | None:
    config_path, log_path = _current_paths(st)
    try:
        return staged_bundle_invalidation_reason(
            bundle=bundle,
            config_path=config_path,
            log_path=log_path,
            last_appended_fingerprint=st.session_state.get(LAST_APPENDED_FINGERPRINT_KEY),
            stage=stage,
        )
    except OSError as exc:
        return str(exc)


def _should_clear_staged_bundle(reason: str) -> bool:
    return reason in {
        "Config path changed after suggestions were staged.",
        "Log path changed after suggestions were staged.",
        "Stage selection changed after suggestions were staged.",
        "Config file changed after suggestions were staged.",
        "Log file changed after suggestions were staged.",
        "Staged suggestions changed after they were staged.",
    }


def _clear_staged_suggestions(st: Any) -> None:
    st.session_state.pop(STAGED_SUGGESTION_BUNDLE_KEY, None)
    st.session_state.pop(STAGED_FRESHNESS_MESSAGE_KEY, None)


def _clear_report_preview(st: Any) -> None:
    st.session_state.pop(REPORT_PREVIEW_KEY, None)


if __name__ == "__main__":
    main()
