"""Form Fields UI ownership for BO Forge."""


from __future__ import annotations

from html import escape
from typing import Any

from bo_forge.application import CampaignAppService
from bo_forge.errors import BOForgeError, ProvenanceRecoveryRequired
from bo_forge_app.streamlit_helpers import (
    CONFIG_PATH_KEY,
    LOG_PATH_KEY,
    SESSION_KEY,
    create_campaign_files,
    parse_categorical_values_text,
    parse_discrete_values_text,
    resolve_path_input,
)
from bo_forge_app.ui.components import (
    _render_result_card,
    _render_variable_type_badge,
)
from bo_forge_app.ui.state import (
    PROVENANCE_RECOVERY_KEY,
    _clear_observation_inputs,
    _clear_report_preview,
    _clear_staged_suggestions,
    _flash_and_rerun,
    _refresh_validation_cache,
)
from bo_forge_app.views.resolve import (
    _stable_widget_key,
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


def _load_campaign_from_inputs(
    st: Any,
    config_value: str,
    log_value: str,
    require_provenance: bool = False,
) -> None:
    try:
        config_path = resolve_path_input(config_value, "Config")
        log_path = resolve_path_input(log_value, "Log")
        campaign = CampaignAppService.load(
            config_path,
            log_path,
            provenance_policy="required" if require_provenance else "compatible",
        )
    except ProvenanceRecoveryRequired as exc:
        from bo_forge.application import file_fingerprint

        st.session_state[PROVENANCE_RECOVERY_KEY] = {
            "config_path": str(config_path),
            "log_path": str(log_path),
            "expected_log_fingerprint": file_fingerprint(log_path),
            "reason_code": exc.reason_code,
            "recovery_action": exc.recovery_action,
            "require_provenance": require_provenance,
        }
        st.error(str(exc))
        return
    except (BOForgeError, OSError, ValueError) as exc:
        st.session_state.pop(PROVENANCE_RECOVERY_KEY, None)
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
    _flash_and_rerun(st, "Campaign loaded.")
