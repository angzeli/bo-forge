"""Entry UI ownership for BO Forge."""


from __future__ import annotations

import os
import time
from html import escape
from typing import Any

from bo_forge_app.streamlit_helpers import (
    CONFIG_PATH_KEY,
    LOG_PATH_KEY,
    SESSION_KEY,
    STAGED_SUGGESTION_BUNDLE_KEY,
    feature_flags,
)
from bo_forge_app.streamlit_style import (
    apply_forge_suite_style,
    render_theme_control,
)
from bo_forge_app.ui.campaign_form import (
    _render_create_new_campaign,
    _render_load_existing_campaign,
)
from bo_forge_app.ui.components import (
    ViewDataLike,
    _render_empty_state,
)
from bo_forge_app.ui.state import (
    ACTIVE_PANEL_KEY,
    CAMPAIGN_FILE_MODE_KEY,
    LEGACY_PANEL_MAP,
    STAGED_FRESHNESS_MESSAGE_KEY,
    WORKFLOW_PANELS,
    _cached_validation_label,
    _render_flash_message,
)
from bo_forge_app.views.analyze import (
    _render_reports,
)
from bo_forge_app.views.campaign import (
    _render_campaign_area,
)
from bo_forge_app.views.run import (
    _render_run_area,
)


def main() -> None:
    """Run the Streamlit app."""
    render_app()


def render_app() -> None:
    """Render the Streamlit page."""
    import streamlit as st

    st.set_page_config(page_title="BO Forge", layout="wide")
    theme = render_theme_control(st)
    apply_forge_suite_style(st, theme)
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
    stored_panel = str(st.session_state.get(ACTIVE_PANEL_KEY, WORKFLOW_PANELS[0]))
    active_area = LEGACY_PANEL_MAP.get(stored_panel, stored_panel)
    if active_area not in WORKFLOW_PANELS:
        active_area = WORKFLOW_PANELS[0]
    st.session_state[ACTIVE_PANEL_KEY] = active_area
    active_panel = st.radio(
        "Workbench area",
        WORKFLOW_PANELS,
        horizontal=True,
        key=ACTIVE_PANEL_KEY,
    )
    _render_active_workflow_panel(st, campaign, flags, str(active_panel))


def _render_workbench_header(st: Any, *, campaign_loaded: bool) -> None:
    campaign_chip = "Campaign loaded" if campaign_loaded else "No campaign loaded"
    campaign_chip_class = "bf-chip-success" if campaign_loaded else "bf-chip-warning"
    st.markdown(
        (
            '<header class="bf-workbench-header">'
            '<p class="bf-kicker">Scientific campaign workbench</p>'
            '<div class="bf-panel-header"><h1 class="bf-title">BO Forge</h1>'
            f'<span class="bf-chip {campaign_chip_class}">{escape(campaign_chip)}</span>'
            "</div></header>"
        ),
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
    panel = LEGACY_PANEL_MAP.get(active_panel, active_panel)
    panel = panel if panel in WORKFLOW_PANELS else WORKFLOW_PANELS[0]
    view_data = _collect_panel_view_data(campaign, panel)
    renderers = {
        "Campaign": lambda: _render_campaign_area(st, campaign, flags, view_data),
        "Run": lambda: _render_run_area(st, campaign, flags, view_data),
        "Analyze": lambda: _render_reports(st, campaign, flags, view_data),
    }
    renderers[panel]()


def _collect_panel_view_data(campaign: Any, panel: str) -> ViewDataLike:
    collector = getattr(campaign, "collect_view_data", None)
    if not callable(collector):
        raise TypeError("Streamlit campaigns must provide collect_view_data(panel).")
    with _TimedBlock(f"collect:{panel}"):
        return collector(panel)


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
