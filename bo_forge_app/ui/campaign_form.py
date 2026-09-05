"""Form UI ownership for BO Forge."""


from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Any

from bo_forge.errors import BOForgeError
from bo_forge_app.streamlit_helpers import (
    CONFIG_PATH_KEY,
    LOG_PATH_KEY,
    NEW_CAMPAIGN_YAML_KEY,
    SESSION_KEY,
    build_campaign_yaml_text,
    default_new_campaign_paths,
    parse_campaign_config_text,
)
from bo_forge_app.ui.components import (
    _render_callout,
    _render_result_card,
)
from bo_forge_app.ui.form_fields import (
    _collect_new_campaign_context_settings,
    _collect_new_campaign_fidelity_settings,
    _collect_new_campaign_objectives,
    _collect_new_campaign_variables,
    _create_campaign_from_inputs,
    _load_campaign_from_inputs,
    _path_changed,
    _render_artifact_note,
    _render_section_label,
)
from bo_forge_app.ui.state import (
    NEW_CAMPAIGN_FORM_YAML_KEY,
    NEW_CAMPAIGN_KIND_KEY,
    PROVENANCE_POLICY_KEY,
    PROVENANCE_RECOVERY_KEY,
    VALIDATION_CACHE_KEY,
    _clear_observation_inputs,
    _clear_report_preview,
    _clear_staged_suggestions,
    _flash_and_rerun,
    _refresh_validation_cache,
)


@dataclass(frozen=True)
class _NewCampaignSections:
    review_enabled: bool = False
    replicates_enabled: bool = False
    replicate_settings: dict[str, object] | None = None
    cost_settings: dict[str, object] | None = None
    fidelity_settings: dict[str, object] | None = None
    context_settings: dict[str, object] | None = None
    bo_overrides: dict[str, object] | None = None


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
        require_provenance = st.checkbox(
            "Require provenance manifest",
            value=bool(st.session_state.get(PROVENANCE_POLICY_KEY, False)),
            key=PROVENANCE_POLICY_KEY,
            help="Reject legacy CSV campaigns that do not have a managed manifest.",
        )
        action_col, reload_col = st.columns([1, 1])
        with action_col:
            load_clicked = st.form_submit_button("Load campaign", type="primary")
        with reload_col:
            reload_clicked = st.form_submit_button("Reload from disk")

    if _path_changed(config_value, LOG_PATH_KEY, log_value):
        _clear_staged_suggestions(st)
    loaded = st.session_state.get(SESSION_KEY)
    loaded_policy = getattr(
        loaded,
        "provenance_policy",
        getattr(getattr(loaded, "session", loaded), "_provenance_policy", None),
    )
    selected_policy = "required" if require_provenance else "compatible"
    if loaded_policy is not None and loaded_policy != selected_policy:
        _clear_staged_suggestions(st)
        st.session_state.pop(VALIDATION_CACHE_KEY, None)

    if load_clicked:
        _load_campaign_from_inputs(
            st,
            config_value,
            log_value,
            require_provenance=require_provenance,
        )
    if reload_clicked:
        _clear_staged_suggestions(st)
        _load_campaign_from_inputs(
            st,
            config_value,
            log_value,
            require_provenance=require_provenance,
        )

    _render_provenance_recovery_action(st)

    current_config = st.session_state.get(CONFIG_PATH_KEY)
    current_log = st.session_state.get(LOG_PATH_KEY)
    if current_config or current_log:
        _render_file_cards(st, str(current_config or ""), str(current_log or ""))


def _render_provenance_recovery_action(st: Any) -> None:
    recovery = st.session_state.get(PROVENANCE_RECOVERY_KEY)
    if not isinstance(recovery, dict) or recovery.get("reason_code") not in {
        "pending_previous_state",
        "pending_resulting_state",
    }:
        return
    _render_callout(
        st,
        "Provenance recovery required",
        str(recovery["recovery_action"]),
    )
    confirmed = st.checkbox(
        "I understand recovery changes the provenance manifest",
        key="bo_forge_confirm_provenance_recovery",
    )
    if not st.button("Recover provenance", disabled=not confirmed):
        return
    from pathlib import Path

    from bo_forge.application import CampaignAppService

    config_path = Path(str(recovery["config_path"]))
    log_path = Path(str(recovery["log_path"]))
    try:
        CampaignAppService.recover_provenance(
            config_path,
            log_path,
            expected_log_fingerprint=str(recovery["expected_log_fingerprint"]),
        )
        policy = "required" if recovery.get("require_provenance") else "compatible"
        campaign = CampaignAppService.load(
            config_path,
            log_path,
            provenance_policy=policy,
        )
    except (BOForgeError, OSError, ValueError) as exc:
        st.error(str(exc))
        return
    st.session_state[CONFIG_PATH_KEY] = str(config_path)
    st.session_state[LOG_PATH_KEY] = str(log_path)
    st.session_state[SESSION_KEY] = campaign
    st.session_state.pop(PROVENANCE_RECOVERY_KEY, None)
    _clear_staged_suggestions(st)
    _clear_observation_inputs(st)
    _clear_report_preview(st)
    _refresh_validation_cache(st, campaign, config_path, log_path)
    _flash_and_rerun(st, "Provenance recovered and campaign reloaded.")


def _render_create_new_campaign(st: Any) -> None:
    campaign_name, config_output, log_output = _collect_campaign_identity(st)
    campaign_kind = _collect_campaign_kind(st)
    is_multi_objective = campaign_kind == "Multi-objective"
    is_multi_fidelity = campaign_kind == "Multi-fidelity qMFKG"
    is_contextual = campaign_kind == "Contextual LogEI"
    model_profile = _collect_model_profile(st, is_multi_objective, is_multi_fidelity)
    objective_name, objective_direction, objectives = _collect_campaign_objective(
        st,
        is_multi_objective,
    )
    bo_settings = _collect_campaign_bo_settings(st, is_multi_fidelity)
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
            objective_direction=objective_direction,
            variables=variables,
            objectives=objectives,
            model={"profile": model_profile},
            review_enabled=sections.review_enabled,
            replicates_enabled=sections.replicates_enabled,
            replicates=sections.replicate_settings,
            cost=sections.cost_settings,
            fidelity=sections.fidelity_settings,
            context=sections.context_settings,
            bo_overrides=sections.bo_overrides,
            **bo_settings,
        )
    except ValueError as exc:
        st.error(f"Could not build YAML preview: {exc}")
    _render_new_campaign_preview(
        st,
        generated_yaml=generated_yaml,
        config_output=config_output,
        log_output=log_output,
    )


def _collect_campaign_identity(st: Any) -> tuple[str, str, str]:
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
    return campaign_name, config_output, log_output


def _collect_campaign_kind(st: Any) -> str:
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
    _render_new_campaign_kind_callout(st, campaign_kind)
    return str(campaign_kind)


def _collect_model_profile(
    st: Any,
    is_multi_objective: bool,
    is_multi_fidelity: bool,
) -> str:
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
    return str(model_profile)


def _collect_campaign_objective(
    st: Any,
    is_multi_objective: bool,
) -> tuple[str, str, list[dict[str, object]] | None]:
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
    return objective_name, str(objective_direction), objectives


def _collect_campaign_bo_settings(
    st: Any,
    is_multi_fidelity: bool,
) -> dict[str, object]:
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
    return {
        "batch_size": int(batch_size),
        "initial_design_size": int(initial_design_size),
        "initial_design_method": str(initial_design_method),
        "random_seed": int(random_seed),
    }


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
