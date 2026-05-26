"""Forge Suite-inspired styling helpers for the BO Forge Streamlit app."""

from __future__ import annotations

FORGE_SUITE_CSS = """
<style>
:root {
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
  padding-top: 2rem;
  padding-bottom: 3rem;
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
.bf-status-block {
  border: 1px solid var(--bf-line);
  background: var(--bf-card);
  box-shadow: var(--bf-shadow-soft);
  backdrop-filter: blur(16px);
}

.bf-workbench-header {
  position: relative;
  overflow: hidden;
  margin-bottom: 1.2rem;
  padding: clamp(1.2rem, 3vw, 2rem);
  border-radius: var(--bf-radius-xl);
}

.bf-workbench-header::after {
  position: absolute;
  right: -72px;
  bottom: -150px;
  width: 280px;
  height: 280px;
  content: "";
  pointer-events: none;
  border: 1px solid rgba(87, 60, 40, 0.12);
  border-radius: 50%;
  background: repeating-conic-gradient(
    from 12deg,
    rgba(159, 79, 50, 0.12) 0deg 10deg,
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
  width: 48px;
  height: 48px;
  place-items: center;
  color: var(--bf-paper-strong);
  border: 1px solid rgba(255, 250, 241, 0.52);
  border-radius: 50%;
  background:
    radial-gradient(circle at 34% 30%, rgba(255, 250, 241, 0.56), transparent 0.62rem),
    conic-gradient(from 212deg, var(--bf-accent), var(--bf-gold), var(--bf-sage), var(--bf-accent));
  box-shadow: 0 12px 28px rgba(125, 53, 36, 0.2);
  font-family: var(--bf-serif);
  font-size: 1.22rem;
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
  font-size: clamp(2.2rem, 4vw, 4.25rem);
  font-weight: 700;
  line-height: 0.92;
}

.bf-subtitle {
  position: relative;
  z-index: 1;
  max-width: 56rem;
  margin: 0.85rem 0 0;
  color: var(--bf-muted);
  font-size: 1.04rem;
  line-height: 1.5;
}

.bf-chip-row {
  flex-wrap: wrap;
  gap: 0.55rem;
  margin-top: 1rem;
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

.bf-chip::before {
  width: 7px;
  height: 7px;
  margin-right: 7px;
  content: "";
  border-radius: 50%;
  background: var(--bf-sage);
}

.bf-panel {
  margin: 0.75rem 0 1rem;
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
    labels = {
        "has_pending_suggestions": "Pending suggestions",
        "ready_for_initial_design": "Ready for initial design",
        "ready_for_bo": "Ready for BO",
    }
    return labels.get(status, status.replace("_", " ").title())


def forge_action_label(action: str) -> str:
    """Return a readable label for a next-action value."""
    labels = {
        "review_pending_suggestions": "Review pending suggestions",
        "run_accepted_suggestions": "Run accepted suggestions",
        "resolve_pending_suggestions": "Resolve pending suggestions",
        "suggest_initial_design": "Suggest initial design",
        "suggest_bo": "Suggest BO candidates",
    }
    return labels.get(action, action.replace("_", " ").title())
