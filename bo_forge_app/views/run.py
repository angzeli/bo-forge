"""Run UI ownership for BO Forge."""


from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Any

import pandas as pd

from bo_forge.errors import BOForgeError
from bo_forge_app.streamlit_helpers import (
    LAST_APPENDED_FINGERPRINT_KEY,
    SESSION_KEY,
    STAGED_SUGGESTION_BUNDLE_KEY,
    active_variables_display,
    append_disabled_reason,
    default_export_path,
    empty_state_message,
    export_staged_suggestions_csv,
    make_staged_suggestion_bundle,
    staged_suggestions_from_bundle,
    structured_stage_config_table,
    structured_stage_options,
)
from bo_forge_app.ui.components import (
    ViewDataLike,
    _render_callout,
    _render_contextual_workflow_state,
    _render_empty_state,
    _render_metric_grid,
    _render_panel_intro,
    _render_step_flow,
    _render_table_section,
    _view_data_value,
)
from bo_forge_app.ui.form_fields import (
    _render_artifact_note,
)
from bo_forge_app.ui.state import (
    STAGED_FRESHNESS_MESSAGE_KEY,
    SUGGEST_STAGE_KEY,
    _clear_report_preview,
    _clear_staged_suggestions,
    _current_invalidation_reason,
    _current_paths,
    _flash_and_rerun,
    _refresh_validation_cache,
    _should_clear_staged_bundle,
)
from bo_forge_app.views.resolve import (
    _handle_log_mutation_error,
    _render_resolve,
)


def _render_run_area(
    st: Any,
    campaign: Any,
    flags: dict[str, bool],
    view_data: ViewDataLike,
) -> None:
    """Render suggestion staging and explicit resolution in one workflow."""
    _render_suggest(st, campaign, view_data)
    st.divider()
    _render_resolve(st, campaign, flags, view_data)


@dataclass(frozen=True)
class _SuggestionRequestState:
    config_path: Path
    log_path: Path
    selected_stage: str | None
    context_values: dict[str, object] | None


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
        if _handle_log_mutation_error(st, campaign, exc):
            return
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
