"""Resolve UI ownership for BO Forge."""


from __future__ import annotations

import json
import math
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Any

import pandas as pd

from bo_forge.application import CampaignAppService
from bo_forge.errors import BOForgeError, LogBusyError, LogConflictError
from bo_forge_app.streamlit_helpers import (
    CONFIG_PATH_KEY,
    LOG_PATH_KEY,
    SESSION_KEY,
    compact_dataframe,
    empty_state_message,
    format_dataframe_for_display,
    observable_row_options,
    observable_rows,
)
from bo_forge_app.ui.components import (
    ViewDataLike,
    _render_cost_metric_cards,
    _render_empty_state,
    _render_panel_intro,
    _render_selected_row_preview,
    _render_table_section,
    _view_data_value,
)
from bo_forge_app.ui.state import (
    _clear_observation_inputs,
    _clear_report_preview,
    _clear_staged_suggestions,
    _current_paths,
    _flash_and_rerun,
    _refresh_validation_cache,
)


@dataclass(frozen=True)
class _ObservationFormValues:
    row_id: str
    objective_inputs: dict[str, str] | None
    objective_value: object | None
    actual_cost_text: str | None
    submitted: bool


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
        if _handle_log_mutation_error(st, campaign, exc):
            return
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
        if _handle_log_mutation_error(st, campaign, exc):
            return
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


def _handle_log_mutation_error(st: Any, campaign: Any, exc: Exception) -> bool:
    if isinstance(exc, LogBusyError):
        st.error(
            "The campaign log is busy because another process is writing it. "
            "Wait briefly, then retry."
        )
        return True
    if not isinstance(exc, LogConflictError):
        return False

    _clear_staged_suggestions(st)
    _clear_observation_inputs(st)
    _clear_report_preview(st)
    config_path, log_path = _current_paths(st)
    try:
        if isinstance(campaign, CampaignAppService) or hasattr(campaign, "session"):
            campaign = CampaignAppService.load(config_path, log_path)
        else:
            campaign.reload()
    except (BOForgeError, OSError, ValueError) as reload_exc:
        st.error(f"Campaign log changed in another process, and reload failed: {reload_exc}")
        return True
    st.session_state[SESSION_KEY] = campaign
    _refresh_validation_cache(st, campaign, config_path, log_path)
    _flash_and_rerun(
        st,
        "Campaign log changed in another process. The latest log was reloaded; retry the action.",
    )
    return True


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
