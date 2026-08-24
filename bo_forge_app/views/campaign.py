"""Campaign UI ownership for BO Forge."""


from __future__ import annotations

from typing import Any

from bo_forge_app.streamlit_helpers import (
    format_dataframe_for_display,
    status_tone,
)
from bo_forge_app.streamlit_style import (
    forge_action_label,
    forge_status_label,
)
from bo_forge_app.ui.components import (
    ViewDataLike,
    _compact_context_summary,
    _compact_replicate_summary,
    _render_callout,
    _render_cost_metric_cards,
    _render_metric_grid,
    _render_panel_intro,
    _render_result_card,
    _render_status_block,
    _render_table_section,
    _summary_value,
    _view_data_value,
)
from bo_forge_app.ui.state import (
    _cached_validation_state,
)


def _render_campaign_area(
    st: Any,
    campaign: Any,
    flags: dict[str, bool],
    view_data: ViewDataLike,
) -> None:
    """Render campaign state and the canonical CSV source of truth."""
    _render_overview(st, campaign, view_data)
    st.divider()
    st.subheader("Campaign Data")
    observed = _view_data_value(view_data, "observed", campaign.observed_data)
    pending = _view_data_value(view_data, "pending", campaign.pending_suggestions)
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
    _render_data_feature_tables(st, campaign, flags, view_data)
    with st.expander("Show full raw campaign log", expanded=False):
        st.dataframe(format_dataframe_for_display(campaign.df), width="stretch")


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

    _render_overview_feature_summaries(st, campaign, view_data)
    observed = view_data.get("observed")
    pending = view_data.get("pending")
    _render_metric_grid(
        st,
        [
            ("Observed preview rows", min(len(observed), 8) if observed is not None else ""),
            ("Pending preview rows", min(len(pending), 8) if pending is not None else ""),
        ],
    )



def _render_overview_feature_summaries(
    st: Any,
    campaign: Any,
    view_data: ViewDataLike,
) -> None:
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
    _render_data_feature_tables(st, campaign, flags, view_data)
    with st.expander("Show full raw campaign log", expanded=False):
        st.dataframe(format_dataframe_for_display(campaign.df), width="stretch")



def _render_data_feature_tables(
    st: Any,
    campaign: Any,
    flags: dict[str, bool],
    view_data: ViewDataLike,
) -> None:
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
