"""Analyze UI ownership for BO Forge."""


from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

from bo_forge.errors import BOForgeError
from bo_forge.plot_registry import _PLOT_ROUTES
from bo_forge_app.streamlit_helpers import (
    available_plot_kinds,
    campaign_report_text,
    default_export_path,
    empty_state_message,
    extract_matplotlib_figure,
    humanize_campaign_status,
)
from bo_forge_app.ui.components import (
    ViewDataLike,
    _render_empty_state,
    _render_metric_grid,
    _render_panel_intro,
    _render_table_section,
    _summary_value,
    _view_data_value,
)
from bo_forge_app.ui.state import (
    REPORT_PREVIEW_KEY,
    _current_paths,
)


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

    _render_report_text_actions(st, campaign, log_path)
    _render_model_comparison_action(st, campaign)
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




def _render_model_comparison_action(st: Any, campaign: Any) -> None:
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


def _render_report_text_actions(st: Any, campaign: Any, log_path: Path) -> None:
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
