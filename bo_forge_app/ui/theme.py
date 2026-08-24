"""Day/Night theme state and technical Streamlit styling."""

from __future__ import annotations

from typing import Any

from bo_forge_app.streamlit_helpers import humanize_campaign_status, humanize_next_action

THEME_STATE_KEY = "bo_forge_theme"
THEME_CONTROL_KEY = f"{THEME_STATE_KEY}_control"
THEME_QUERY_SYNC_KEY = f"{THEME_STATE_KEY}_query_sync"
THEMES = ("day", "night")

DAY_TOKENS = {
    "canvas": "#f2efe8",
    "surface": "#f8f5ef",
    "surface_strong": "#ece8df",
    "surface_subtle": "#eeebe4",
    "text": "#292d2f",
    "cool": "#263746",
    "muted": "#686967",
    "muted_cool": "#62717a",
    "line": "#c9c2b8",
    "line_soft": "rgba(187, 178, 166, 0.72)",
    "line_strong": "rgba(38, 55, 70, 0.58)",
    "accent": "#b96f45",
    "accent_soft": "#a65f3c",
    "accent_faint": "rgba(185, 111, 69, 0.055)",
}

NIGHT_TOKENS = {
    "canvas": "#11181d",
    "surface": "#172027",
    "surface_strong": "#1b252c",
    "surface_subtle": "#141c21",
    "text": "#e8dfd1",
    "cool": "#d7e0e7",
    "muted": "#a99f92",
    "muted_cool": "#9aa8b0",
    "line": "#34424b",
    "line_soft": "rgba(52, 66, 75, 0.7)",
    "line_strong": "rgba(215, 224, 231, 0.5)",
    "accent": "#c9855c",
    "accent_soft": "#d0936d",
    "accent_faint": "rgba(201, 133, 92, 0.075)",
}


def initial_theme(st: Any) -> str:
    """Resolve theme from URL, explicit state, browser context, then Day fallback."""
    query_theme = _normalise_theme(st.query_params.get("theme"))
    if query_theme is not None:
        st.session_state[THEME_STATE_KEY] = query_theme
        return query_theme
    stored_theme = _normalise_theme(st.session_state.get(THEME_STATE_KEY))
    if stored_theme is not None:
        return stored_theme
    context_type = getattr(getattr(st.context, "theme", None), "type", None)
    context_theme = {"light": "day", "dark": "night"}.get(str(context_type).lower())
    resolved = context_theme or "day"
    st.session_state[THEME_STATE_KEY] = resolved
    return resolved


def render_theme_control(st: Any) -> str:
    """Render the native theme selector and persist an explicit URL choice."""
    current = initial_theme(st)
    query_theme = _normalise_theme(st.query_params.get("theme"))
    synced_query = _normalise_theme(st.session_state.get(THEME_QUERY_SYNC_KEY))

    # Synchronize a new URL choice into the keyed widget before Streamlit renders it.
    if query_theme is not None and query_theme != synced_query:
        st.session_state[THEME_CONTROL_KEY] = query_theme.title()
    elif THEME_CONTROL_KEY not in st.session_state:
        st.session_state[THEME_CONTROL_KEY] = current.title()

    selected = st.segmented_control(
        "Theme",
        options=["Day", "Night"],
        key=THEME_CONTROL_KEY,
    )
    theme = str(selected or current).lower()
    theme = theme if theme in THEMES else current
    st.session_state[THEME_STATE_KEY] = theme
    if _normalise_theme(st.query_params.get("theme")) != theme:
        st.query_params["theme"] = theme
    st.session_state[THEME_QUERY_SYNC_KEY] = theme
    return theme


def apply_forge_suite_style(st: Any, theme: str = "day") -> None:
    """Apply the scoped technical theme without external assets or JavaScript."""
    st.markdown(_theme_css(theme), unsafe_allow_html=True)


def forge_status_label(status: str) -> str:
    """Return a readable label for a campaign status value."""
    return humanize_campaign_status(status)


def forge_action_label(action: str) -> str:
    """Return a readable label for a next-action value."""
    return humanize_next_action(action)


def _normalise_theme(value: object) -> str | None:
    if isinstance(value, list):
        value = value[-1] if value else None
    candidate = str(value).lower() if value is not None else ""
    return candidate if candidate in THEMES else None


def _theme_css(theme: str) -> str:
    tokens = NIGHT_TOKENS if theme == "night" else DAY_TOKENS
    variables = "\n".join(
        f"  --bf-{name.replace('_', '-')}: {value};" for name, value in tokens.items()
    )
    return f"""
<style>
:root {{
{variables}
  --bf-mono: "IBM Plex Mono", "SFMono-Regular", Consolas, monospace;
  --bf-sans: Arial, Helvetica, sans-serif;
  color-scheme: {"dark" if theme == "night" else "light"};
}}
html, body, [data-testid="stAppViewContainer"] {{
  background: var(--bf-canvas);
  color: var(--bf-text);
  font-family: var(--bf-mono) !important;
  letter-spacing: 0 !important;
}}
[data-testid="stMainBlockContainer"] {{
  max-width: 1120px;
  padding-top: 1.4rem;
}}
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] {{
  background: var(--bf-canvas);
}}
[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3,
[data-testid="stAppViewContainer"] h4,
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] button,
[data-testid="stAppViewContainer"] input,
[data-testid="stAppViewContainer"] textarea {{
  font-family: var(--bf-mono) !important;
  letter-spacing: 0 !important;
}}
[data-testid="stAppViewContainer"] h1 {{ font-size: 32px !important; }}
[data-testid="stAppViewContainer"] h2 {{ font-size: 22px !important; }}
[data-testid="stAppViewContainer"] h3 {{ font-size: 18px !important; }}
[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3,
[data-testid="stAppViewContainer"] h4 {{ color: var(--bf-cool); }}
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] label,
[data-testid="stCaptionContainer"] {{ color: var(--bf-muted); }}
a {{ color: var(--bf-accent-soft); }}
:focus-visible {{ outline: 1px solid var(--bf-accent); outline-offset: 4px; }}
hr {{ border-color: var(--bf-line); }}
.bf-workbench-header, .bf-source-bar, .forge-card, .forge-callout,
.forge-empty, .forge-artifact, .forge-result, .forge-status {{
  background: var(--bf-surface);
  border: 1px solid var(--bf-line);
  border-radius: 3px;
  color: var(--bf-text);
}}
.bf-workbench-header, .bf-source-bar {{ padding: 0.75rem 0.9rem; margin-bottom: 0.75rem; }}
.bf-title, .bf-panel-title {{ margin: 0; color: var(--bf-cool); }}
.bf-kicker, .forge-eyebrow {{ color: var(--bf-accent); margin: 0; }}
.bf-panel-note, .bf-subtitle, .forge-note {{ color: var(--bf-muted); }}
.bf-chip-row, .forge-step-row {{ display: flex; flex-wrap: wrap; gap: 0.4rem; }}
.bf-chip, .forge-pill, .forge-step {{
  border: 1px solid var(--bf-line);
  border-radius: 2px;
  color: var(--bf-muted-cool);
  padding: 0.22rem 0.4rem;
}}
.bf-chip-success, .forge-pill-copper {{ color: var(--bf-accent); border-color: var(--bf-accent); }}
.bf-chip-warning {{ color: var(--bf-muted); }}
.forge-metric-grid {{
  display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 1px;
}}
.forge-metric {{
  background: var(--bf-surface-strong); border: 1px solid var(--bf-line); padding: 0.55rem;
}}
.forge-metric-label {{ color: var(--bf-muted); }}
.forge-metric-value {{ color: var(--bf-cool); font-weight: 700; }}
.forge-card, .forge-callout, .forge-empty, .forge-artifact,
.forge-result, .forge-status {{ padding: 0.7rem; }}
.stButton > button, [data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-primary"] {{
  border: 1px solid var(--bf-line-strong);
  border-radius: 3px;
  background: var(--bf-surface) !important;
  color: var(--bf-cool) !important;
  font-family: var(--bf-mono) !important;
}}
.stButton > button[kind="primary"], [data-testid="stBaseButton-primary"] {{
  border-color: var(--bf-accent);
  color: var(--bf-accent-soft);
}}
[data-testid="stDataFrame"], [data-testid="stTable"], textarea, input,
[data-baseweb="select"] > div {{ border-radius: 2px; }}
[data-testid="stExpander"] summary,
[data-testid="stTextInputRootElement"],
[data-testid="stNumberInputContainer"],
[data-baseweb="select"] > div {{
  background: var(--bf-surface-strong) !important;
  color: var(--bf-text) !important;
  border-color: var(--bf-line) !important;
}}
[data-testid="stTextInputRootElement"] input,
[data-testid="stNumberInputContainer"] input,
textarea {{
  color: var(--bf-text) !important;
  caret-color: var(--bf-accent) !important;
}}
@media (max-width: 720px) {{
  [data-testid="stMainBlockContainer"] {{ padding-left: 0.75rem; padding-right: 0.75rem; }}
  .forge-metric-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
}}
</style>
"""


FORGE_SUITE_CSS = _theme_css("day")
