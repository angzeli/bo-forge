"""Forge Suite-inspired styling helpers for the BO Forge Streamlit app."""

from __future__ import annotations

from bo_forge_app.streamlit_helpers import humanize_campaign_status, humanize_next_action

FORGE_SUITE_CSS = """
<style>
:root {
  --forge-paper: #fff8ed;
  --forge-panel: rgba(255, 250, 241, 0.86);
  --forge-ink: #2a1d16;
  --forge-muted: #77685f;
  --forge-copper: #9f4f32;
  --forge-gold: #d6a84f;
  --forge-sage: #7f9a7a;
  --forge-border: rgba(87, 60, 40, 0.16);
  --bf-ink: #2a1d16;
  --bf-muted: #77685f;
  --bf-paper: #fff8ed;
  --bf-paper-strong: #fffaf1;
  --bf-card: rgba(255, 250, 241, 0.86);
  --bf-card-strong: rgba(255, 253, 247, 0.96);
  --bf-line: rgba(87, 60, 40, 0.16);
  --bf-accent: #9f4f32;
  --bf-accent-strong: #7d3524;
  --bf-gold: #d6a84f;
  --bf-sage: #7f9a7a;
  --bf-shadow-soft: 0 14px 36px rgba(72, 45, 24, 0.10);
  --bf-radius-xl: 32px;
  --bf-radius-lg: 22px;
  --bf-radius-md: 16px;
  --bf-radius-sm: 12px;
  --bf-ui: Optima, Candara, "Avenir Next", "Segoe UI", system-ui, sans-serif;
  --bf-serif: Charter, "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
}

.stApp {
  color: var(--bf-ink);
  background:
    radial-gradient(circle at 14% 8%, rgba(214, 168, 79, 0.34), transparent 27rem),
    radial-gradient(circle at 86% 12%, rgba(127, 154, 122, 0.24), transparent 29rem),
    linear-gradient(135deg, #fff7e8 0%, #f8efe1 45%, #efe3d1 100%);
}

.stApp::before {
  position: fixed;
  inset: 0;
  z-index: -1;
  content: "";
  pointer-events: none;
  opacity: 0.24;
  background-image:
    linear-gradient(rgba(42, 29, 22, 0.045) 1px, transparent 1px),
    linear-gradient(90deg, rgba(42, 29, 22, 0.035) 1px, transparent 1px);
  background-size: 46px 46px;
  mask-image: linear-gradient(to bottom, black, transparent 80%);
}

.block-container {
  max-width: 1160px;
  padding-top: 1.15rem;
  padding-bottom: 3rem;
}

[data-testid="stHeader"] {
  height: 0 !important;
  min-height: 0 !important;
  background: transparent !important;
}

[data-testid="stToolbar"],
[data-testid="stDecoration"] {
  display: none !important;
}

[data-testid="stSidebar"] {
  background: rgba(255, 250, 241, 0.78);
  border-right: 1px solid var(--bf-line);
}

[data-testid="stSidebar"] * {
  font-family: var(--bf-ui);
}

h1, h2, h3 {
  color: var(--bf-ink);
  font-family: var(--bf-serif);
  letter-spacing: 0;
}

p, label, span, div {
  font-family: var(--bf-ui);
}

.bf-workbench-header,
.bf-panel,
.bf-status-block,
.forge-card,
.forge-empty,
.forge-artifact,
.forge-callout,
.forge-success,
.forge-warning {
  border: 1px solid var(--bf-line);
  background: var(--bf-card);
  box-shadow: var(--bf-shadow-soft);
  backdrop-filter: blur(16px);
}

.bf-workbench-header {
  position: relative;
  overflow: hidden;
  margin-bottom: 0.85rem;
  padding: clamp(1rem, 2.2vw, 1.55rem);
  border-radius: var(--bf-radius-xl);
}

.bf-workbench-header::after {
  position: absolute;
  right: -92px;
  bottom: -162px;
  width: 260px;
  height: 260px;
  content: "";
  pointer-events: none;
  border: 1px solid rgba(87, 60, 40, 0.12);
  border-radius: 50%;
  opacity: 0.62;
  background: repeating-conic-gradient(
    from 12deg,
    rgba(159, 79, 50, 0.09) 0deg 10deg,
    transparent 10deg 24deg
  );
}

.bf-brand-row,
.bf-chip-row,
.bf-panel-header {
  position: relative;
  z-index: 1;
  display: flex;
  align-items: center;
}

.bf-brand-row {
  gap: 0.9rem;
}

.bf-brand-mark {
  display: grid;
  width: 44px;
  height: 44px;
  place-items: center;
  color: var(--bf-paper-strong);
  border: 1px solid rgba(255, 250, 241, 0.52);
  border-radius: 50%;
  background:
    radial-gradient(circle at 34% 26%, rgba(255, 250, 241, 0.68), transparent 0.54rem),
    radial-gradient(circle at 70% 78%, rgba(42, 29, 22, 0.20), transparent 1.15rem),
    conic-gradient(from 212deg, var(--bf-accent), var(--bf-gold), var(--bf-sage), var(--bf-accent));
  box-shadow:
    inset 0 1px 10px rgba(255, 250, 241, 0.32),
    0 12px 28px rgba(125, 53, 36, 0.18);
  font-family: var(--bf-serif);
  font-size: 1.04rem;
  font-weight: 800;
  line-height: 1;
  text-shadow: 0 1px 7px rgba(42, 29, 22, 0.28);
}

.bf-kicker {
  margin: 0 0 0.2rem;
  color: var(--bf-accent-strong);
  font-family: var(--bf-ui);
  font-size: 0.75rem;
  font-weight: 800;
  letter-spacing: 0.14em;
  text-transform: uppercase;
}

.bf-title {
  margin: 0;
  color: var(--bf-ink);
  font-family: var(--bf-serif);
  font-size: clamp(2rem, 3.3vw, 3.3rem);
  font-weight: 700;
  line-height: 0.92;
}

.bf-subtitle {
  position: relative;
  z-index: 1;
  max-width: 56rem;
  margin: 0.7rem 0 0;
  color: var(--bf-muted);
  font-size: 0.98rem;
  line-height: 1.5;
}

.bf-chip-row {
  flex-wrap: wrap;
  gap: 0.55rem;
  margin-top: 0.8rem;
}

.bf-chip {
  display: inline-flex;
  align-items: center;
  min-height: 32px;
  padding: 0.45rem 0.7rem;
  color: var(--bf-accent-strong);
  border: 1px solid rgba(87, 60, 40, 0.11);
  border-radius: 999px;
  background: rgba(255, 253, 247, 0.66);
  font-size: 0.74rem;
  font-weight: 800;
  letter-spacing: 0.08em;
  line-height: 1.1;
  text-transform: uppercase;
}

.bf-chip-warning::before {
  background: var(--bf-gold);
}

.bf-chip-success::before {
  background: var(--bf-sage);
}

.bf-chip::before {
  width: 7px;
  height: 7px;
  margin-right: 7px;
  content: "";
  border-radius: 50%;
  background: var(--bf-sage);
}

.bf-panel {
  margin: 0.6rem 0 0.85rem;
  padding: 1.05rem 1.1rem;
  border-radius: var(--bf-radius-lg);
}

.bf-file-panel {
  margin-bottom: 0.55rem;
  background:
    linear-gradient(145deg, rgba(255, 253, 248, 0.92), rgba(255, 246, 229, 0.72)),
    radial-gradient(circle at 92% 14%, rgba(214, 168, 79, 0.18), transparent 17rem);
}

.bf-panel-header {
  justify-content: space-between;
  gap: 1rem;
  margin-bottom: 0.75rem;
}

.bf-panel-title {
  margin: 0;
  font-family: var(--bf-serif);
  font-size: 1.55rem;
  font-weight: 700;
  line-height: 1.05;
}

.bf-panel-note {
  margin: 0.3rem 0 0;
  color: var(--bf-muted);
  font-size: 0.92rem;
  line-height: 1.4;
}

.bf-status-block {
  margin: 0.65rem 0 1rem;
  padding: 1rem;
  border-radius: var(--bf-radius-md);
  background: var(--bf-card-strong);
}

.bf-status-block-sage {
  border-color: rgba(127, 154, 122, 0.35);
}

.bf-status-block-success {
  border-color: rgba(87, 130, 82, 0.38);
}

.bf-status-block-warning {
  border-color: rgba(214, 168, 79, 0.42);
}

.bf-status-label {
  margin: 0;
  color: var(--bf-accent-strong);
  font-size: 0.72rem;
  font-weight: 800;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}

.bf-status-value {
  margin: 0.25rem 0 0;
  color: var(--bf-ink);
  font-family: var(--bf-serif);
  font-size: 1.55rem;
  font-weight: 700;
}

.bf-status-detail {
  margin: 0.35rem 0 0;
  color: var(--bf-muted);
  font-size: 0.94rem;
  line-height: 1.45;
}

.forge-card,
.forge-empty,
.forge-artifact,
.forge-callout,
.forge-success,
.forge-warning {
  margin: 0.55rem 0 0.85rem;
  padding: 0.9rem 1rem;
  border-radius: var(--bf-radius-md);
}

.forge-card {
  background: var(--bf-card-strong);
}

.forge-callout {
  border-left: 4px solid var(--bf-gold);
  background: rgba(255, 246, 204, 0.56);
}

.forge-empty {
  background: rgba(255, 253, 247, 0.66);
  box-shadow: none;
}

.forge-artifact {
  border-color: rgba(159, 79, 50, 0.22);
  background: rgba(255, 250, 241, 0.72);
}

.forge-success {
  border-color: rgba(127, 154, 122, 0.38);
  background: rgba(239, 247, 231, 0.74);
}

.forge-warning {
  border-color: rgba(159, 79, 50, 0.30);
  background: rgba(255, 240, 229, 0.72);
}

.forge-card-title,
.forge-empty-title,
.forge-callout-title {
  margin: 0;
  color: var(--bf-accent-strong);
  font-size: 0.72rem;
  font-weight: 800;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}

.forge-card-value,
.forge-empty-detail,
.forge-callout-detail {
  margin: 0.32rem 0 0;
  color: var(--bf-ink);
  font-size: 0.96rem;
  line-height: 1.45;
}

.forge-metric-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 0.75rem;
  margin: 0.6rem 0 0.85rem;
}

.forge-metric {
  padding: 0.8rem 0.9rem;
  border: 1px solid var(--bf-line);
  border-radius: var(--bf-radius-md);
  background: rgba(255, 253, 247, 0.76);
}

.forge-metric-label {
  margin: 0;
  color: var(--bf-muted);
  font-size: 0.72rem;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.forge-metric-value {
  margin: 0.25rem 0 0;
  color: var(--bf-ink);
  font-family: var(--bf-serif);
  font-size: 1.45rem;
  font-weight: 700;
}

.forge-file-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 0.75rem;
}

.forge-file-card {
  padding: 0.78rem 0.85rem;
  border: 1px solid var(--bf-line);
  border-radius: var(--bf-radius-md);
  background: rgba(255, 253, 247, 0.72);
}

.forge-pill {
  display: inline-flex;
  align-items: center;
  padding: 0.2rem 0.45rem;
  color: var(--bf-accent-strong);
  border: 1px solid rgba(87, 60, 40, 0.14);
  border-radius: 999px;
  background: rgba(255, 253, 247, 0.75);
  font-size: 0.66rem;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.forge-pill-sage {
  color: #3e633c;
}

.forge-pill-gold {
  color: #7a5b13;
}

.forge-pill-copper {
  color: var(--bf-accent-strong);
}

.forge-pill-blue {
  color: #42536c;
}

.forge-file-path {
  margin: 0.35rem 0 0;
  color: var(--bf-ink);
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.83rem;
  overflow-wrap: anywhere;
}

.forge-step-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.55rem;
  margin: 0.45rem 0 0.85rem;
}

.forge-step {
  padding: 0.45rem 0.62rem;
  border: 1px solid var(--bf-line);
  border-radius: 999px;
  background: rgba(255, 253, 247, 0.7);
  color: var(--bf-accent-strong);
  font-size: 0.74rem;
  font-weight: 800;
}

.forge-artifact pre,
.forge-artifact textarea {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}

div[data-testid="stTabs"] button[aria-selected="true"] {
  color: var(--bf-accent-strong);
  border-bottom-color: var(--bf-accent);
}

.stButton > button {
  border: 1px solid rgba(125, 53, 36, 0.18);
  border-radius: 14px;
  font-family: var(--bf-ui);
  font-weight: 800;
}

.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, var(--bf-accent), var(--bf-accent-strong));
}

[data-testid="stDataFrame"],
[data-testid="stTable"],
textarea,
input {
  border-radius: var(--bf-radius-sm);
}
</style>
"""


def apply_forge_suite_style(st: object) -> None:
    """Inject Forge Suite-inspired CSS into the Streamlit page."""
    st.markdown(FORGE_SUITE_CSS, unsafe_allow_html=True)


def forge_status_label(status: str) -> str:
    """Return a readable label for a campaign status value."""
    return humanize_campaign_status(status)


def forge_action_label(action: str) -> str:
    """Return a readable label for a next-action value."""
    return humanize_next_action(action)
