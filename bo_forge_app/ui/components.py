"""Components UI ownership for BO Forge."""


from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING, Any, TypeAlias

from bo_forge.errors import BOForgeError
from bo_forge_app.streamlit_helpers import (
    compact_dataframe,
    empty_state_message,
    format_dataframe_for_display,
    format_number_for_display,
)

if TYPE_CHECKING:
    from bo_forge.application import CampaignViewData

    ViewDataLike: TypeAlias = CampaignViewData | dict[str, Any]
else:
    ViewDataLike: TypeAlias = dict[str, Any]



def _view_data_value(view_data: ViewDataLike, key: str, fallback: Any) -> Any:
    if key in view_data:
        return view_data[key]
    return fallback()


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
