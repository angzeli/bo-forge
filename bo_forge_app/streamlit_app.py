"""Streamlit UI for local BO Forge campaign workflows."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from hashlib import sha1
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from bo_forge.errors import BOForgeError
from bo_forge.plot_registry import _PLOT_ROUTES
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


@dataclass(frozen=True)
class _NewCampaignSections:
    review_enabled: bool = False
    replicates_enabled: bool = False
    replicate_settings: dict[str, object] | None = None
    cost_settings: dict[str, object] | None = None
    fidelity_settings: dict[str, object] | None = None
    context_settings: dict[str, object] | None = None
    bo_overrides: dict[str, object] | None = None


@dataclass(frozen=True)
class _SuggestionRequestState:
    config_path: Path
    log_path: Path
    selected_stage: str | None
    context_values: dict[str, object] | None


@dataclass(frozen=True)
class _ObservationFormValues:
    row_id: str
    objective_inputs: dict[str, str] | None
    objective_value: object | None
    actual_cost_text: str | None
    submitted: bool


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
        "Suggest": lambda: _render_suggest(st, campaign, view_data),
        "Resolve": lambda: _render_resolve(st, campaign, flags, view_data),
        "Reports": lambda: _render_reports(st, campaign, flags, view_data),
        "Data": lambda: _render_data(st, campaign, flags, view_data),
    }
    renderers[panel]()


def _collect_panel_view_data(campaign: Any, panel: str) -> ViewDataLike:
    collector = getattr(campaign, "collect_view_data", None)
    if not callable(collector):
        raise TypeError("Streamlit campaigns must provide collect_view_data(panel).")
    with _TimedBlock(f"collect:{panel}"):
        return collector(panel)


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
        [
            "Single-objective",
            "Single-objective qLogNEI",
            "Multi-objective",
            "Multi-fidelity qMFKG",
            "Contextual LogEI",
        ],
        horizontal=True,
        key=NEW_CAMPAIGN_KIND_KEY,
        help=(
            "Choose a backend campaign template. qLogNEI adds noisy/pending-aware "
            "single-objective suggestions; Contextual LogEI creates a config with "
            "one or more fixed context variables."
        ),
    )
    is_multi_objective = campaign_kind == "Multi-objective"
    is_multi_fidelity = campaign_kind == "Multi-fidelity qMFKG"
    is_contextual = campaign_kind == "Contextual LogEI"
    _render_new_campaign_kind_callout(st, campaign_kind)

    _render_section_label(st, "Model profile")
    if is_multi_objective or is_multi_fidelity:
        model_profile = st.selectbox(
            "Model profile",
            ["default"],
            key="new_campaign_model_profile_default_only",
            disabled=True,
            help=(
                "Non-default model profiles require a single-objective config with "
                "bo.acquisition: log_ei or qlog_nei."
            ),
        )
    else:
        model_profile = st.selectbox(
            "Model profile",
            ["default", "smooth", "rough", "robust"],
            key="new_campaign_model_profile",
            help=(
                "default preserves BO Forge's current GP path; smooth uses an RBF/ARD "
                "kernel; rough uses a Matern-1.5/ARD kernel; robust records fitting "
                "warnings for diagnostics."
            ),
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
                max_value=4,
                value=1,
                key="new_bo_batch_size_multi_fidelity",
                help="qMFKG supports batches from 1 through 4.",
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
        min_value=2 if is_contextual else 1,
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
        sections = _collect_new_campaign_sections(st, campaign_kind, variables)
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
            review_enabled=sections.review_enabled,
            replicates_enabled=sections.replicates_enabled,
            replicates=sections.replicate_settings,
            cost=sections.cost_settings,
            fidelity=sections.fidelity_settings,
            context=sections.context_settings,
            model={"profile": str(model_profile)},
            bo_overrides=sections.bo_overrides,
        )
    except ValueError as exc:
        st.error(f"Could not build YAML preview: {exc}")

    _render_new_campaign_preview(
        st,
        generated_yaml=generated_yaml,
        config_output=config_output,
        log_output=log_output,
    )


def _render_new_campaign_kind_callout(st: Any, campaign_kind: str) -> None:
    callouts = {
        "Single-objective qLogNEI": (
            "Single-objective qLogNEI",
            "App-created qLogNEI campaigns support single-objective noisy/pending-aware "
            "suggestions. Review rows marked pending block; accepted review rows become "
            "active pending candidates.",
        ),
        "Multi-fidelity qMFKG": (
            "Multi-fidelity qMFKG",
            "App-created multi-fidelity campaigns are single-objective qMFKG campaigns "
            "with continuous or ordered discrete fidelity. Advanced defaults stay editable "
            "in the YAML preview.",
        ),
        "Contextual LogEI": (
            "Contextual LogEI",
            "App-created contextual campaigns are single-objective LogEI campaigns. "
            "Context variables stay ordinary CSV columns but are fixed at suggestion time.",
        ),
    }
    callout = callouts.get(campaign_kind)
    if callout is not None:
        _render_callout(st, *callout)


def _collect_new_campaign_sections(
    st: Any,
    campaign_kind: str,
    variables: list[dict[str, object]],
) -> _NewCampaignSections:
    collectors = {
        "Multi-fidelity qMFKG": _collect_multi_fidelity_sections,
        "Multi-objective": _collect_multi_objective_sections,
        "Single-objective qLogNEI": _collect_qlog_nei_sections,
        "Contextual LogEI": _collect_contextual_sections,
    }
    collector = collectors.get(campaign_kind)
    return collector(st, variables) if collector is not None else _NewCampaignSections()


def _collect_multi_fidelity_sections(
    st: Any,
    variables: list[dict[str, object]],
) -> _NewCampaignSections:
    _render_section_label(st, "Fidelity")
    fidelity_settings = _collect_new_campaign_fidelity_settings(st, variables)
    _render_artifact_note(
        st,
        "qMFKG defaults",
        "Generated YAML uses num_fantasies=8, raw_samples=8, num_restarts=1, "
        "mc_samples=16, min_normalized_distance=0.0, optimizer_maxiter=200, "
        "and no acquisition timeout unless enabled.",
    )
    review_enabled = st.checkbox(
        "Enable review",
        value=False,
        key="new_campaign_review_enabled_multi_fidelity",
    )
    return _NewCampaignSections(
        review_enabled=review_enabled,
        fidelity_settings=fidelity_settings,
        bo_overrides={
            "acquisition": "qmf_kg",
            "raw_samples": 8,
            "num_restarts": 1,
            "mc_samples": 16,
            "min_normalized_distance": 0.0,
        },
    )


def _collect_multi_objective_sections(
    st: Any,
    variables: list[dict[str, object]],
) -> _NewCampaignSections:
    del variables
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
    cost_settings = (
        _collect_new_campaign_cost_settings(st, key_prefix="new_campaign")
        if cost_enabled
        else None
    )
    return _NewCampaignSections(
        review_enabled=review_enabled,
        replicates_enabled=replicates_enabled,
        cost_settings=cost_settings,
    )


def _collect_qlog_nei_sections(
    st: Any,
    variables: list[dict[str, object]],
) -> _NewCampaignSections:
    del variables
    _render_section_label(st, "qLogNEI review semantics")
    review_enabled = st.checkbox(
        "Enable review",
        value=True,
        key="new_campaign_review_enabled_qlog_nei",
        help=(
            "Review-pending rows block qLogNEI. Accepted rows are treated as "
            "active pending experiments and passed as X_pending."
        ),
    )
    _render_artifact_note(
        st,
        "qLogNEI scope",
        "Generated YAML uses bo.acquisition=qlog_nei. Cost-aware, contextual, "
        "structured, multi-fidelity, and multi-objective qLogNEI remain deferred.",
    )
    return _NewCampaignSections(
        review_enabled=review_enabled,
        bo_overrides={"acquisition": "qlog_nei"},
    )


def _collect_contextual_sections(
    st: Any,
    variables: list[dict[str, object]],
) -> _NewCampaignSections:
    _render_section_label(st, "Context")
    context_settings = _collect_new_campaign_context_settings(st, variables)
    _render_section_label(st, "Contextual review, cost, and replicates")
    review_enabled = st.checkbox(
        "Enable review",
        value=False,
        key="new_campaign_review_enabled_contextual",
        help="Optional review metadata is supported for contextual LogEI campaigns.",
    )
    cost_enabled = st.checkbox(
        "Enable deterministic cost",
        value=False,
        key="new_campaign_cost_enabled_contextual",
        help=(
            "Optional deterministic cost uses the full candidate, including "
            "fixed context values, and keeps a campaign-global budget."
        ),
    )
    cost_settings = (
        _collect_new_campaign_cost_settings(st, key_prefix="new_campaign_contextual")
        if cost_enabled
        else None
    )
    replicates_enabled = st.checkbox(
        "Enable replicates",
        value=False,
        key="new_campaign_replicates_enabled_contextual",
        help=(
            "Replicate groups retain their context. Active uncertain-best "
            "repeats only target groups matching the suggestion context."
        ),
    )
    replicate_settings = (
        _collect_new_campaign_contextual_replicates(st)
        if replicates_enabled
        else None
    )
    _render_artifact_note(
        st,
        "Contextual scope",
        "Generated YAML uses bo.acquisition=log_ei. Contextual review, "
        "deterministic cost, replicates, and their combinations are supported. "
        "Contextual multi-objective, structured, multi-fidelity, and "
        "qLogNEI/qLogNEHVI workflows remain out of scope.",
    )
    return _NewCampaignSections(
        review_enabled=review_enabled,
        replicates_enabled=replicates_enabled,
        replicate_settings=replicate_settings,
        cost_settings=cost_settings,
        context_settings=context_settings,
    )


def _collect_new_campaign_cost_settings(
    st: Any,
    *,
    key_prefix: str,
) -> dict[str, object]:
    cost_col_1, cost_col_2, cost_col_3 = st.columns(3)
    with cost_col_1:
        cost_expression = st.text_input(
            "Cost expression",
            value="1.0",
            key=f"{key_prefix}_cost_expression",
        )
    with cost_col_2:
        cost_weight = st.number_input(
            "Cost weight",
            min_value=0.0,
            value=1.0,
            key=f"{key_prefix}_cost_weight",
        )
    with cost_col_3:
        cost_budget = st.number_input(
            "Budget",
            min_value=0.0,
            value=100.0,
            key=f"{key_prefix}_cost_budget",
        )
    return {
        "expression": cost_expression,
        "weight": float(cost_weight),
        "budget": float(cost_budget),
    }


def _collect_new_campaign_contextual_replicates(st: Any) -> dict[str, object]:
    policy_col, threshold_col = st.columns(2)
    with policy_col:
        replicate_policy = st.selectbox(
            "Replicate suggestion policy",
            ["uncertain_best", "new_only"],
            key="new_campaign_contextual_replicate_policy",
        )
    with threshold_col:
        replicate_threshold = st.number_input(
            "Replicate uncertainty threshold",
            min_value=1.0e-12,
            value=0.1,
            key="new_campaign_contextual_replicate_threshold",
        )
    repeat_col, max_repeat_col = st.columns(2)
    with repeat_col:
        min_repeats = st.number_input(
            "Minimum repeats at best",
            min_value=1,
            value=2,
            step=1,
            key="new_campaign_contextual_min_repeats",
        )
    with max_repeat_col:
        max_repeats = st.number_input(
            "Maximum repeats per group",
            min_value=1,
            value=5,
            step=1,
            key="new_campaign_contextual_max_repeats",
        )
    noise_floor = st.number_input(
        "Replicate noise floor",
        min_value=1.0e-12,
        value=1.0e-8,
        format="%.2e",
        key="new_campaign_contextual_noise_floor",
    )
    return {
        "enabled": True,
        "suggestion_policy": str(replicate_policy),
        "replicate_threshold": float(replicate_threshold),
        "min_repeats_at_best": int(min_repeats),
        "max_repeats_per_group": int(max_repeats),
        "noise_floor": float(noise_floor),
    }


def _render_new_campaign_preview(
    st: Any,
    *,
    generated_yaml: str,
    config_output: str,
    log_output: str,
) -> None:
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
    edited_yaml = st.text_area("Campaign YAML", height=360, key=NEW_CAMPAIGN_YAML_KEY)
    if preview_is_stale:
        st.warning(
            "Structured form values changed after this YAML preview was generated. "
            "Use Update YAML preview from form before creating the campaign."
        )

    validate_col, create_col = st.columns([1, 1])
    with validate_col:
        if st.button("Validate YAML"):
            _validate_new_campaign_yaml(st, edited_yaml)
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


def _validate_new_campaign_yaml(st: Any, edited_yaml: str) -> None:
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
    mode = st.radio(
        "Fidelity mode",
        ["Continuous", "Ordered discrete levels"],
        horizontal=True,
        key="new_campaign_fidelity_mode",
    )
    fidelity: dict[str, object] = {
        "variable": str(fidelity_variable),
        "fixed_cost": 0.01,
        "fidelity_cost_weight": 1.0,
        "num_fantasies": 8,
        "optimizer_maxiter": int(
            st.number_input(
                "Max optimizer iterations",
                min_value=1,
                value=200,
                step=1,
                key="new_campaign_fidelity_optimizer_maxiter",
            )
        ),
    }
    limit_runtime = st.checkbox(
        "Limit acquisition runtime",
        value=False,
        key="new_campaign_fidelity_limit_runtime",
        help=(
            "Applies one safety deadline to target-value optimization and candidate "
            "retries after model fitting. It does not guarantee candidate quality or "
            "immediate cancellation."
        ),
    )
    if limit_runtime:
        fidelity["optimizer_timeout_seconds"] = float(
            st.number_input(
                "Acquisition timeout (seconds)",
                min_value=0.1,
                value=60.0,
                step=1.0,
                key="new_campaign_fidelity_optimizer_timeout_seconds",
            )
        )
    if mode == "Ordered discrete levels":
        levels_text = st.text_input(
            "Fidelity levels",
            value=f"{lower:g}, {(lower + upper) / 2:g}, {upper:g}",
            key=f"new_campaign_fidelity_levels_{fidelity_variable}",
            help=(
                "Enter at least two strictly increasing numeric levels. The highest "
                "level becomes the target fidelity."
            ),
        )
        levels = parse_discrete_values_text(levels_text, str(fidelity_variable))
        if len(levels) < 2:
            raise ValueError("Discrete fidelity requires at least two levels.")
        if any(
            current <= previous
            for previous, current in zip(levels, levels[1:], strict=False)
        ):
            raise ValueError("Fidelity levels must be strictly increasing.")
        if levels[0] < lower or levels[-1] > upper:
            raise ValueError(
                f"Fidelity levels must stay within [{lower:g}, {upper:g}]."
            )
        fidelity["levels"] = levels
        fidelity["target"] = levels[-1]
    else:
        fidelity["target"] = float(
            st.number_input(
                "Target fidelity",
                min_value=lower,
                max_value=upper,
                value=upper,
                key=f"new_campaign_fidelity_target_{fidelity_variable}",
                help="Defaults to the selected fidelity variable's upper bound.",
            )
        )
    return fidelity


def _collect_new_campaign_context_settings(
    st: Any,
    variables: list[dict[str, object]],
) -> dict[str, object]:
    variable_names = [str(variable["name"]) for variable in variables]
    if len(variable_names) < 2:
        raise ValueError(
            "Contextual LogEI campaigns require at least one decision variable and one "
            "context variable."
        )
    default_context_variables = [variable_names[-1]]
    selected_names = st.multiselect(
        "Context variables",
        variable_names,
        default=default_context_variables,
        key="new_campaign_context_variables",
        help=(
            "Selected variables remain normal CSV columns but are fixed at suggestion "
            "time. Enabled defaults are written to YAML context.default_values."
        ),
    )
    selected_names = [str(name) for name in selected_names]
    if not selected_names:
        raise ValueError("Contextual LogEI campaigns require at least one context variable.")
    if len(set(selected_names)) == len(variable_names):
        raise ValueError(
            "Context variables cannot include every variable; at least one decision "
            "variable is required."
        )

    variable_by_name = {str(variable["name"]): variable for variable in variables}
    default_values: dict[str, object] = {}
    for name in selected_names:
        variable = variable_by_name[name]
        use_default = st.checkbox(
            f"Set default for context: {name}",
            value=True,
            key=_stable_widget_key("new_context_default_enabled", name),
            help=(
                "Defaults are optional and are written to YAML context.default_values. "
                "Users can still override context values in Suggest."
            ),
        )
        if use_default:
            default_values[name] = _collect_new_campaign_context_default(st, variable)

    context: dict[str, object] = {"variables": selected_names}
    if default_values:
        context["default_values"] = default_values
    return context


def _collect_new_campaign_context_default(
    st: Any,
    variable: dict[str, object],
) -> object:
    name = str(variable["name"])
    variable_type = str(variable.get("type", "continuous"))
    key = _stable_widget_key("new_context_default", name)
    label = f"Default context: {name}"
    help_text = "Written to YAML context.default_values for app-created contextual configs."
    if variable_type == "categorical":
        values = [str(value) for value in variable.get("values", [])]
        return st.selectbox(label, values, key=key, help=help_text)
    if variable_type == "discrete":
        values = [float(value) for value in variable.get("values", [])]
        return float(st.selectbox(label, values, key=key, help=help_text))
    if variable_type == "integer":
        lower = int(variable["lower"])
        upper = int(variable["upper"])
        return int(
            st.number_input(
                label,
                min_value=lower,
                max_value=upper,
                value=lower,
                step=1,
                key=key,
                help=help_text,
            )
        )
    lower = float(variable["lower"])
    upper = float(variable["upper"])
    return float(
        st.number_input(
            label,
            min_value=lower,
            max_value=upper,
            value=lower,
            key=key,
            help=help_text,
        )
    )


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
    _clear_observation_inputs(st)
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
    _clear_observation_inputs(st)
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
    if campaign.config.context is not None:
        context_summary = _view_data_value(
            view_data,
            "context_summary",
            campaign.context_summary,
        )
        _render_table_section(
            st,
            "Context Summary",
            _compact_context_summary(context_summary),
            empty_kind="context_summary",
            raw_df=context_summary,
            expanded_raw=False,
        )
    if campaign.config.bo.acquisition == "qlog_nei":
        _render_table_section(
            st,
            "qLogNEI Summary",
            _view_data_value(view_data, "qlog_nei_summary", campaign.qlog_nei_summary),
            empty_kind="qlog_nei_summary",
            expanded_raw=False,
        )
    _render_table_section(
        st,
        "Model Summary",
        _view_data_value(view_data, "model_summary", campaign.model_summary),
        empty_kind="report_preview",
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
            "Fidelity Coverage",
            _view_data_value(view_data, "fidelity_coverage", campaign.fidelity_coverage),
            empty_kind="fidelity_coverage",
            expanded_raw=False,
        )
    if campaign.config.context is not None:
        context_summary = _view_data_value(
            view_data,
            "context_summary",
            campaign.context_summary,
        )
        _render_table_section(
            st,
            "Context Summary",
            _compact_context_summary(context_summary),
            empty_kind="context_summary",
            raw_df=context_summary,
            expanded_raw=False,
        )
    if campaign.config.bo.acquisition == "qlog_nei":
        _render_table_section(
            st,
            "qLogNEI Summary",
            _view_data_value(view_data, "qlog_nei_summary", campaign.qlog_nei_summary),
            empty_kind="qlog_nei_summary",
            expanded_raw=False,
        )
    _render_table_section(
        st,
        "Model Summary",
        _view_data_value(view_data, "model_summary", campaign.model_summary),
        empty_kind="report_preview",
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


def _render_suggest(
    st: Any,
    campaign: Any,
    view_data: ViewDataLike | None = None,
) -> None:
    _render_panel_intro(
        st,
        "Suggest",
        "Generate candidates as a dry run, inspect quality, then append explicitly.",
    )
    _render_step_flow(
        st,
        ["1. Generate dry-run suggestions", "2. Inspect quality", "3. Append explicitly"],
    )
    request_state = _render_suggestion_request_controls(st, campaign, view_data)
    batch_size, generate_clicked = _render_suggestion_dry_run_form(st, campaign)
    if generate_clicked:
        _generate_staged_suggestions(st, campaign, request_state, int(batch_size))

    bundle = st.session_state.get(STAGED_SUGGESTION_BUNDLE_KEY)
    suggestions = staged_suggestions_from_bundle(bundle)
    if suggestions.empty:
        _render_empty_state(st, *empty_state_message("staged_suggestions"))
        return

    disabled_reason, cleared = _resolve_staged_suggestion_state(
        st,
        bundle,
        request_state,
    )
    if cleared:
        return
    _render_staged_suggestion_summary(st, campaign, suggestions, disabled_reason)
    _render_staged_suggestion_export(
        st,
        campaign,
        bundle,
        suggestions,
        request_state.log_path,
    )
    _render_suggestion_quality(st, campaign, suggestions)
    _render_append_staged_suggestions(
        st,
        campaign,
        bundle,
        suggestions,
        request_state,
        disabled_reason,
    )


def _render_suggestion_request_controls(
    st: Any,
    campaign: Any,
    view_data: ViewDataLike | None,
) -> _SuggestionRequestState:
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
    selected_stage = _render_suggestion_stage(st, campaign, stage_options)
    context_values = _render_context_inputs(
        st,
        campaign.config,
        config_path=config_path,
        log_path=log_path,
    )
    _render_suggestion_mode_notes(st, campaign, view_data, context_values)
    return _SuggestionRequestState(
        config_path=config_path,
        log_path=log_path,
        selected_stage=selected_stage,
        context_values=context_values,
    )


def _render_suggestion_stage(
    st: Any,
    campaign: Any,
    stage_options: list[str],
) -> str | None:
    if not stage_options:
        return None
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
    return selected_stage


def _render_suggestion_mode_notes(
    st: Any,
    campaign: Any,
    view_data: ViewDataLike | None,
    context_values: dict[str, object] | None,
) -> None:
    if campaign.config.context is not None:
        contextual_cost_summary = None
        if campaign.config.cost is not None:
            contextual_cost_summary = _view_data_value(
                view_data or {},
                "cost_summary",
                campaign.cost_summary,
            )
        _render_contextual_workflow_state(
            st,
            campaign,
            context_values=context_values,
            cost_summary=contextual_cost_summary,
        )
        _render_contextual_replicate_note(st, campaign)
    if campaign.config.fidelity is not None:
        mode = "discrete-level" if campaign.config.fidelity.levels is not None else "continuous"
        detail = (
            "Discrete-level qMFKG constructs conditioned greedy batches from 1 through 4 "
            "and reports one joint post-selection acquisition value."
            if campaign.config.fidelity.levels is not None
            else "Continuous qMFKG jointly optimizes batches from 1 through 4."
        )
        timeout = campaign.config.fidelity.optimizer_timeout_seconds
        runtime = (
            f"Optimizer max iterations: {campaign.config.fidelity.optimizer_maxiter}. "
            + (
                "No acquisition timeout is configured."
                if timeout is None
                else f"Acquisition timeout: {timeout:g} seconds."
            )
        )
        _render_artifact_note(
            st,
            "qMFKG suggestions",
            f"{mode.capitalize()} mode. {detail} {runtime} Larger batches increase KG runtime.",
        )
    if campaign.config.bo.acquisition == "qlog_nei":
        _render_artifact_note(
            st,
            "qLogNEI pending semantics",
            "Review-pending rows must be resolved first. Accepted pending suggestions "
            "can stay in the log and are accounted for as X_pending.",
        )


def _render_contextual_replicate_note(st: Any, campaign: Any) -> None:
    if not campaign.config.replicates.enabled:
        return
    policy = campaign.config.replicates.suggestion_policy
    detail = (
        "Uncertain-best repeats are restricted to replicate groups matching "
        "every active suggestion context value."
        if policy == "uncertain_best"
        else "New-only mode models replicate-derived variance but proposes new designs."
    )
    _render_artifact_note(st, f"Context-matched replicates: {policy}", detail)


def _render_suggestion_dry_run_form(st: Any, campaign: Any) -> tuple[object, bool]:
    is_multi_fidelity = campaign.config.fidelity is not None
    configured_batch_size = max(1, int(campaign.config.bo.batch_size))
    default_batch_size = (
        min(4, configured_batch_size) if is_multi_fidelity else configured_batch_size
    )
    with st.form("suggest_dry_run_form"):
        batch_size = st.number_input(
            "Batch size",
            min_value=1,
            max_value=4 if is_multi_fidelity else 32,
            value=default_batch_size,
            step=1,
        )
        generate_clicked = st.form_submit_button(
            "Generate suggestions (dry run)",
            type="primary",
        )
    return batch_size, bool(generate_clicked)


def _generate_staged_suggestions(
    st: Any,
    campaign: Any,
    request_state: _SuggestionRequestState,
    batch_size: int,
) -> None:
    try:
        if hasattr(campaign, "suggest_dry_run"):
            kwargs = _suggestion_request_kwargs(request_state)
            result = campaign.suggest_dry_run(batch_size, **kwargs)
            bundle = result.bundle
        else:
            kwargs = _suggestion_request_kwargs(request_state)
            suggestions = campaign.suggest_next(batch_size=batch_size, **kwargs)
            bundle = make_staged_suggestion_bundle(
                suggestions,
                request_state.config_path,
                request_state.log_path,
                stage=request_state.selected_stage,
                context_values=request_state.context_values,
            )
    except (BOForgeError, OSError, ValueError) as exc:
        st.error(str(exc))
        return
    st.session_state[STAGED_SUGGESTION_BUNDLE_KEY] = bundle
    st.session_state.pop(STAGED_FRESHNESS_MESSAGE_KEY, None)
    st.success("Suggestions staged. Review them before appending.")


def _suggestion_request_kwargs(
    request_state: _SuggestionRequestState,
) -> dict[str, object]:
    kwargs: dict[str, object] = {}
    if request_state.selected_stage is not None:
        kwargs["stage"] = request_state.selected_stage
    if request_state.context_values is not None:
        kwargs["context_values"] = request_state.context_values
    return kwargs


def _resolve_staged_suggestion_state(
    st: Any,
    bundle: Any,
    request_state: _SuggestionRequestState,
) -> tuple[str | None, bool]:
    kwargs = _suggestion_request_kwargs(request_state)
    raw_reason = _current_invalidation_reason(st, bundle, **kwargs)
    if raw_reason is None:
        st.session_state.pop(STAGED_FRESHNESS_MESSAGE_KEY, None)
    disabled_reason = append_disabled_reason(
        bundle,
        request_state.config_path,
        request_state.log_path,
        st.session_state.get(LAST_APPENDED_FINGERPRINT_KEY),
        **kwargs,
    )
    if not raw_reason or raw_reason == "No staged suggestions.":
        return disabled_reason, False
    st.session_state[STAGED_FRESHNESS_MESSAGE_KEY] = raw_reason
    _render_callout(st, "Append state", disabled_reason or raw_reason)
    if not _should_clear_staged_bundle(raw_reason):
        return disabled_reason, False
    _clear_staged_suggestions(st)
    _render_empty_state(
        st,
        "Cleared stale staged suggestions.",
        "Generate a fresh dry-run batch before appending.",
    )
    return disabled_reason, True


def _render_staged_suggestion_summary(
    st: Any,
    campaign: Any,
    suggestions: pd.DataFrame,
    disabled_reason: str | None,
) -> None:
    staged_metrics: list[tuple[str, object]] = [
        ("Staged rows", len(suggestions)),
        ("Status", "Ready" if disabled_reason is None else "Blocked"),
    ]
    if campaign.config.cost is not None and "cost_estimate" in suggestions.columns:
        estimates = pd.to_numeric(suggestions["cost_estimate"], errors="coerce")
        staged_metrics.append(("Staged estimated cost", float(estimates.fillna(0.0).sum())))
    if campaign.config.review.enabled and "review_status" in suggestions.columns:
        states = sorted(set(suggestions["review_status"].astype(str)))
        staged_metrics.append(("Review state", ", ".join(states)))
    _render_metric_grid(st, staged_metrics)
    _render_table_section(
        st,
        "Staged Suggestions",
        suggestions,
        empty_kind="staged_suggestions",
        expanded_raw=False,
    )


def _render_staged_suggestion_export(
    st: Any,
    campaign: Any,
    bundle: Any,
    suggestions: pd.DataFrame,
    log_path: Path,
) -> None:
    with st.form("staged_suggestions_export_form"):
        export_path = Path(
            st.text_input(
                "Staged suggestions CSV export path",
                value=str(default_export_path(log_path, "staged_suggestions", "csv")),
                key="staged_suggestions_export_path",
            )
        )
        export_clicked = st.form_submit_button("Export staged suggestions CSV")
    if not export_clicked:
        return
    try:
        written_path = (
            campaign.export_staged_suggestions(bundle, export_path)
            if hasattr(campaign, "export_staged_suggestions")
            else export_staged_suggestions_csv(suggestions, export_path)
        )
    except OSError as exc:
        st.error(str(exc))
    else:
        st.success(f"Wrote staged suggestions CSV: {written_path}")


def _render_suggestion_quality(st: Any, campaign: Any, suggestions: pd.DataFrame) -> None:
    try:
        quality = campaign.suggestion_quality(suggestions)
    except BOForgeError as exc:
        st.warning(f"Could not compute suggestion quality: {exc}")
        return
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


def _render_append_staged_suggestions(
    st: Any,
    campaign: Any,
    bundle: Any,
    suggestions: pd.DataFrame,
    request_state: _SuggestionRequestState,
    disabled_reason: str | None,
) -> None:
    if disabled_reason is not None:
        _render_callout(st, "Append disabled", disabled_reason)
    with st.form("append_staged_suggestions_form"):
        append_clicked = st.form_submit_button(
            "Append staged suggestions",
            disabled=disabled_reason is not None,
        )
    if not append_clicked:
        return
    try:
        campaign, appended_fingerprint = _append_staged_suggestion_bundle(
            st,
            campaign,
            bundle,
            suggestions,
            request_state,
        )
    except (BOForgeError, ValueError) as exc:
        st.error(str(exc))
        return
    st.session_state[LAST_APPENDED_FINGERPRINT_KEY] = appended_fingerprint
    _clear_staged_suggestions(st)
    _clear_report_preview(st)
    st.session_state[SESSION_KEY] = campaign
    _refresh_validation_cache(
        st,
        campaign,
        request_state.config_path,
        request_state.log_path,
    )
    _flash_and_rerun(st, "Staged suggestions appended to the campaign log.")


def _append_staged_suggestion_bundle(
    st: Any,
    campaign: Any,
    bundle: Any,
    suggestions: pd.DataFrame,
    request_state: _SuggestionRequestState,
) -> tuple[Any, str]:
    if hasattr(campaign, "append_staged"):
        result = campaign.append_staged(
            bundle,
            st.session_state.get(LAST_APPENDED_FINGERPRINT_KEY),
            **_suggestion_request_kwargs(request_state),
        )
        return result.service, result.appended_fingerprint
    campaign.append_suggestions(suggestions)
    return campaign, str(bundle["suggestions_fingerprint"])


def _render_context_inputs(
    st: Any,
    config: Any,
    *,
    config_path: object | None = None,
    log_path: object | None = None,
) -> dict[str, object] | None:
    if config.context is None:
        return None
    _render_artifact_note(
        st,
        "Context",
        "Suggestion context values are used only for this dry-run batch and remain "
        "normal CSV variable columns.",
    )
    context_values: dict[str, object] = {}
    variables_by_name = {variable.name: variable for variable in config.variables}
    key_scope = _context_widget_key_scope(config, config_path=config_path, log_path=log_path)
    columns = st.columns(min(len(config.context_variable_names), 3) or 1)
    for index, name in enumerate(config.context_variable_names):
        variable = variables_by_name[name]
        default = config.context.default_values.get(name, _default_context_input(variable))
        key = f"context_input_{key_scope}_{sha1(name.encode('utf-8')).hexdigest()[:10]}"
        with columns[index % len(columns)]:
            if variable.type == "categorical":
                options = [str(value) for value in variable.values]
                default_index = options.index(str(default)) if str(default) in options else 0
                context_values[name] = st.selectbox(
                    f"Suggestion context: {name}",
                    options,
                    index=default_index,
                    key=key,
                    help="Used only for this dry-run suggestion batch.",
                )
            elif variable.type == "discrete":
                options = [float(value) for value in variable.values]
                default_float = float(default)
                default_index = (
                    options.index(default_float) if default_float in options else 0
                )
                context_values[name] = st.selectbox(
                    f"Suggestion context: {name}",
                    options,
                    index=default_index,
                    key=key,
                    help="Used only for this dry-run suggestion batch.",
                )
            elif variable.type == "integer":
                context_values[name] = int(
                    st.number_input(
                        f"Suggestion context: {name}",
                        min_value=int(variable.lower),
                        max_value=int(variable.upper),
                        value=int(default),
                        step=1,
                        key=key,
                        help="Used only for this dry-run suggestion batch.",
                    )
                )
            else:
                context_values[name] = float(
                    st.number_input(
                        f"Suggestion context: {name}",
                        min_value=float(variable.lower),
                        max_value=float(variable.upper),
                        value=float(default),
                        key=key,
                        help="Used only for this dry-run suggestion batch.",
                    )
                )
    return context_values


def _context_widget_key_scope(
    config: Any,
    *,
    config_path: object | None = None,
    log_path: object | None = None,
) -> str:
    variable_payload = []
    for variable in config.variables:
        variable_payload.append(
            {
                "name": variable.name,
                "type": variable.type,
                "lower": getattr(variable, "lower", None),
                "upper": getattr(variable, "upper", None),
                "values": list(getattr(variable, "values", ()) or ()),
            }
        )
    payload = {
        "config_path": str(Path(str(config_path)).expanduser().resolve(strict=False))
        if config_path is not None
        else "",
        "log_path": str(Path(str(log_path)).expanduser().resolve(strict=False))
        if log_path is not None
        else "",
        "campaign_name": getattr(config, "campaign_name", ""),
        "context_variables": list(getattr(config, "context_variable_names", [])),
        "context_defaults": dict(getattr(config.context, "default_values", {})),
        "variables": variable_payload,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return sha1(encoded.encode("utf-8")).hexdigest()[:10]


def _default_context_input(variable: Any) -> object:
    if variable.type in {"continuous", "integer"}:
        return variable.lower
    if variable.type == "discrete":
        return float(variable.values[0])
    if variable.type == "categorical":
        return str(variable.values[0])
    return ""


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
    if flags["has_cost"]:
        _render_cost_metric_cards(
            st,
            campaign,
            _view_data_value(view_data, "cost_summary", campaign.cost_summary),
        )
    pending = _view_data_value(view_data, "pending", campaign.pending_suggestions)
    _render_pending_suggestions(st, pending)
    observable = _view_data_value(
        view_data,
        "observable",
        lambda: observable_rows(campaign.config, campaign.df),
    )
    _render_review_workflow(st, campaign, flags, view_data)

    _render_table_section(
        st,
        "Observable Suggestions",
        observable,
        empty_kind="pending_suggestions",
        expanded_raw=False,
    )
    if observable.empty:
        return
    form_values = _render_observation_form(st, campaign, flags, observable)
    if form_values.submitted:
        _record_observation(st, campaign, form_values)


def _render_pending_suggestions(st: Any, pending: pd.DataFrame) -> None:
    with st.expander("Pending Suggestions", expanded=False):
        if pending.empty:
            _render_empty_state(st, *empty_state_message("pending_suggestions"))
            return
        st.dataframe(compact_dataframe(pending), width="stretch")
        with st.expander("Show full raw pending suggestions", expanded=False):
            st.dataframe(format_dataframe_for_display(pending), width="stretch")


def _render_review_workflow(
    st: Any,
    campaign: Any,
    flags: dict[str, bool],
    view_data: ViewDataLike,
) -> None:
    if not flags["has_review"]:
        _render_empty_state(
            st,
            "Review is not enabled.",
            "This campaign can mark suggested rows observed without a review decision.",
        )
        return
    st.subheader("Review Queue")
    review_queue = _view_data_value(view_data, "review_queue", campaign.review_queue)
    if review_queue.empty:
        _render_empty_state(st, *empty_state_message("review_queue"))
        return
    st.dataframe(compact_dataframe(review_queue), width="stretch")
    with st.expander("Show full raw review queue", expanded=False):
        st.dataframe(format_dataframe_for_display(review_queue), width="stretch")
    with st.form("review_decision_form"):
        row_id = st.selectbox("Review row_id", review_queue["row_id"].astype(str).tolist())
        decision = st.selectbox("Decision", ["accept", "reject", "defer"])
        note = st.text_input("Review note", value="")
        review_clicked = st.form_submit_button("Apply review decision")
    if not review_clicked:
        return
    try:
        if hasattr(campaign, "review"):
            campaign = campaign.review(row_id, decision, note).service
        else:
            campaign.review_suggestion(row_id, decision, note)
    except BOForgeError as exc:
        st.error(str(exc))
        return
    _complete_resolve_mutation(st, campaign, "Review decision recorded.")


def _render_observation_form(
    st: Any,
    campaign: Any,
    flags: dict[str, bool],
    observable: pd.DataFrame,
) -> _ObservationFormValues:
    st.subheader(
        "Record Coupled Objectives" if campaign.config.is_multi_objective else "Mark Observed"
    )
    option_map = observable_row_options(campaign.config, campaign.df)
    config_path = st.session_state.get(
        CONFIG_PATH_KEY,
        getattr(campaign, "config_path", ""),
    )
    log_path = st.session_state.get(
        LOG_PATH_KEY,
        getattr(campaign, "log_path", ""),
    )
    input_scope = _campaign_widget_key_scope(
        campaign.config,
        config_path=config_path,
        log_path=log_path,
    )
    with st.form("mark_observed_form"):
        selected_label = st.selectbox("Observed suggestion", list(option_map))
        observed_row_id = option_map[selected_label]
        selected_row = observable.loc[observable["row_id"].astype(str) == observed_row_id]
        if not selected_row.empty:
            _render_selected_row_preview(st, campaign, selected_row.iloc[0])
        objective_inputs, objective_value = _render_observation_objective_inputs(
            st,
            campaign,
            input_scope,
            observed_row_id,
        )
        actual_cost_text = _render_actual_cost_input(
            st,
            flags,
            key_suffix=f"{input_scope}|{observed_row_id}",
        )
        button_label = (
            "Record coupled objectives"
            if campaign.config.is_multi_objective
            else "Mark row observed"
        )
        submitted = st.form_submit_button(button_label)
    return _ObservationFormValues(
        row_id=observed_row_id,
        objective_inputs=objective_inputs,
        objective_value=objective_value,
        actual_cost_text=actual_cost_text,
        submitted=bool(submitted),
    )


def _render_observation_objective_inputs(
    st: Any,
    campaign: Any,
    input_scope: str,
    observed_row_id: str,
) -> tuple[dict[str, str] | None, object | None]:
    if campaign.config.is_multi_objective:
        objective_inputs = {
            objective.name: st.text_input(
                f"Observed {objective.name}",
                value="",
                key=_stable_widget_key(
                    "observed_objective",
                    input_scope,
                    observed_row_id,
                    objective.name,
                ),
                help="Required. Enter a finite numeric value.",
            )
            for objective in campaign.config.objectives
        }
        return objective_inputs, None
    objective_name = campaign.config.objective.name
    objective_value = st.number_input(
        f"Observed {objective_name}",
        value=0.0,
        format="%.8f",
        key=_stable_widget_key(
            "observed_objective",
            input_scope,
            observed_row_id,
            objective_name,
        ),
    )
    return None, objective_value


def _record_observation(
    st: Any,
    campaign: Any,
    form_values: _ObservationFormValues,
) -> None:
    try:
        if campaign.config.is_multi_objective:
            campaign = _record_multi_objective_observation(campaign, form_values)
            success_message = "Coupled objective values recorded."
        else:
            campaign = _record_single_objective_observation(campaign, form_values)
            success_message = "Observation recorded."
    except (BOForgeError, ValueError) as exc:
        st.error(str(exc))
        return
    _complete_resolve_mutation(st, campaign, success_message)


def _record_multi_objective_observation(
    campaign: Any,
    form_values: _ObservationFormValues,
) -> Any:
    objective_values = _parse_multi_objective_inputs(
        form_values.objective_inputs or {},
        campaign.config.objective_names,
    )
    actual_cost = _parse_actual_cost_input(form_values.actual_cost_text)
    result = campaign.mark_observed(
        row_id=form_values.row_id,
        objective_values=objective_values,
        actual_cost=actual_cost,
    )
    return result.service if hasattr(result, "service") else campaign


def _record_single_objective_observation(
    campaign: Any,
    form_values: _ObservationFormValues,
) -> Any:
    actual_cost = _parse_actual_cost_input(form_values.actual_cost_text)
    result = campaign.mark_observed(
        row_id=form_values.row_id,
        objective_value=float(form_values.objective_value),
        actual_cost=None if actual_cost is None else float(actual_cost),
    )
    return result.service if hasattr(result, "service") else campaign


def _complete_resolve_mutation(st: Any, campaign: Any, message: str) -> None:
    _clear_staged_suggestions(st)
    _clear_report_preview(st)
    st.session_state[SESSION_KEY] = campaign
    config_path, log_path = _current_paths(st)
    _refresh_validation_cache(st, campaign, config_path, log_path)
    _flash_and_rerun(st, message)


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


def _campaign_widget_key_scope(
    config: Any,
    *,
    config_path: object,
    log_path: object,
) -> str:
    """Return a stable campaign identity for row-scoped mutation widgets."""
    payload = {
        "config_path": str(Path(str(config_path)).expanduser().resolve(strict=False)),
        "log_path": str(Path(str(log_path)).expanduser().resolve(strict=False)),
        "campaign_name": getattr(config, "campaign_name", ""),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return sha1(encoded.encode("utf-8")).hexdigest()[:10]


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
    if campaign.config.fidelity is not None:
        _render_table_section(
            st,
            "Fidelity Coverage",
            _view_data_value(view_data, "fidelity_coverage", campaign.fidelity_coverage),
            empty_kind="fidelity_coverage",
            expanded_raw=False,
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

    if _supports_model_profile_comparison(campaign.config):
        with st.form("model_comparison_form"):
            run_comparison = st.form_submit_button("Run model comparison")
        if run_comparison:
            try:
                comparison = campaign.model_profile_comparison()
            except (BOForgeError, ValueError) as exc:
                st.error(str(exc))
            else:
                _render_table_section(
                    st,
                    "Model Profile Comparison",
                    comparison,
                    empty_kind="report_preview",
                    expanded_raw=False,
                )

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
    for kind, route in _PLOT_ROUTES.items():
        if kind in {"cost_progress", "replicates"}:
            continue
        if kind in plot_kinds:
            plotter = (
                _service_plotter(campaign, kind)
                if hasattr(campaign, "plot")
                else getattr(campaign, route.session_method)
            )
            options.append(
                {
                    "label": route.label,
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


def _supports_model_profile_comparison(config: Any) -> bool:
    return (
        not config.is_multi_objective
        and config.fidelity is None
        and not config.is_structured_campaign
    )


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
    context_names = set(getattr(campaign.config, "context_variable_names", []))
    variables = tuple(
        variable for variable in variables if variable.name in context_names
    ) + tuple(variable for variable in variables if variable.name not in context_names)
    for variable in variables[:6]:
        metrics.append((variable.name, row.get(variable.name, "")))
    if campaign.config.review.enabled:
        metrics.append(("Review state", row.get("review_status", "")))
    if campaign.config.cost is not None:
        metrics.append(("Estimated cost", row.get("cost_estimate", "")))
    _render_metric_grid(st, metrics)


def _render_contextual_workflow_state(
    st: Any,
    campaign: Any,
    *,
    context_values: dict[str, object] | None,
    cost_summary: Any | None = None,
) -> None:
    """Render compact context, budget, and review state for one dry run."""
    metrics = [
        (f"Context: {name}", value)
        for name, value in (context_values or {}).items()
    ]
    if campaign.config.cost is not None:
        summary = cost_summary if cost_summary is not None else campaign.cost_summary()
        metrics.append(("Remaining budget", _summary_value(summary, "budget_remaining")))
    if campaign.config.review.enabled:
        metrics.append(("Review state", "Required before observation"))
    if metrics:
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


def _compact_context_summary(df: Any) -> Any:
    if getattr(df, "empty", True):
        return df
    fixed_columns = [
        "context_key",
        "observed_rows",
        "pending_suggestions",
        "best_row_id",
        "best_objective",
    ]
    context_columns = [
        column
        for column in df.columns
        if column not in fixed_columns
    ]
    columns = ["context_key", *context_columns, *fixed_columns[1:]]
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
    context_values: dict[str, object] | None = None,
) -> str | None:
    config_path, log_path = _current_paths(st)
    try:
        return staged_bundle_invalidation_reason(
            bundle=bundle,
            config_path=config_path,
            log_path=log_path,
            last_appended_fingerprint=st.session_state.get(LAST_APPENDED_FINGERPRINT_KEY),
            stage=stage,
            context_values=context_values,
        )
    except OSError as exc:
        return str(exc)


def _should_clear_staged_bundle(reason: str) -> bool:
    return reason in {
        "Config path changed after suggestions were staged.",
        "Log path changed after suggestions were staged.",
        "Stage selection changed after suggestions were staged.",
        "Context values changed after suggestions were staged.",
        "Config file changed after suggestions were staged.",
        "Log file changed after suggestions were staged.",
        "Staged suggestions changed after they were staged.",
    }


def _clear_staged_suggestions(st: Any) -> None:
    st.session_state.pop(STAGED_SUGGESTION_BUNDLE_KEY, None)
    st.session_state.pop(STAGED_FRESHNESS_MESSAGE_KEY, None)


def _clear_observation_inputs(st: Any) -> None:
    """Clear row-scoped observation values when the loaded campaign changes."""
    prefixes = ("observed_objective_", "actual_cost_")
    for key in list(st.session_state):
        if str(key).startswith(prefixes):
            st.session_state.pop(key, None)


def _clear_report_preview(st: Any) -> None:
    st.session_state.pop(REPORT_PREVIEW_KEY, None)


if __name__ == "__main__":
    main()
