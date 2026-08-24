"""Navigation, theme, and CSS contracts for the v3 workbench."""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from bo_forge.application import CampaignAppService
from bo_forge_app.streamlit_helpers import (
    CONFIG_PATH_KEY,
    LOG_PATH_KEY,
    SESSION_KEY,
    STAGED_SUGGESTION_BUNDLE_KEY,
)
from bo_forge_app.ui.state import LEGACY_PANEL_MAP, WORKFLOW_PANELS
from bo_forge_app.ui.theme import (
    DAY_TOKENS,
    NIGHT_TOKENS,
    THEME_CONTROL_KEY,
    THEME_QUERY_SYNC_KEY,
    THEME_STATE_KEY,
    _theme_css,
    initial_theme,
    render_theme_control,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _FakeStreamlit:
    def __init__(
        self,
        *,
        query_theme: object = None,
        stored_theme: object = None,
        browser_theme: object = None,
        selection: str | None = None,
    ) -> None:
        self.query_params = {} if query_theme is None else {"theme": query_theme}
        self.session_state = {}
        if stored_theme is not None:
            self.session_state[THEME_STATE_KEY] = stored_theme
        if selection is not None:
            self.session_state[THEME_CONTROL_KEY] = selection
        self.context = SimpleNamespace(theme=SimpleNamespace(type=browser_theme))

    def segmented_control(self, *_args, **kwargs):
        return self.session_state.get(kwargs["key"])


@pytest.mark.parametrize(
    ("legacy", "current"),
    [
        ("Overview", "Campaign"),
        ("Data", "Campaign"),
        ("Suggest", "Run"),
        ("Resolve", "Run"),
        ("Reports", "Analyze"),
    ],
)
def test_legacy_panel_state_maps_to_v3_area(legacy: str, current: str) -> None:
    assert LEGACY_PANEL_MAP[legacy] == current
    assert current in WORKFLOW_PANELS


def test_v3_workbench_has_three_task_oriented_areas() -> None:
    assert WORKFLOW_PANELS == ["Campaign", "Run", "Analyze"]


@pytest.mark.parametrize(("legacy", "expected"), LEGACY_PANEL_MAP.items())
def test_legacy_panel_state_maps_through_real_workbench_navigation(
    legacy: str,
    expected: str,
) -> None:
    from streamlit.testing.v1 import AppTest

    config_path = Path("configs/01_simple_2d_maximise_logei.yaml")
    log_path = Path("examples/01_simple_2d_maximise_logei_campaign_log.csv")
    app = AppTest.from_file(PROJECT_ROOT / "bo_forge_app" / "streamlit_app.py")
    app.session_state[SESSION_KEY] = CampaignAppService.load(config_path, log_path)
    app.session_state[CONFIG_PATH_KEY] = str(config_path)
    app.session_state[LOG_PATH_KEY] = str(log_path)
    app.session_state["bo_forge_active_panel"] = legacy
    app.run(timeout=10)

    area = next(radio for radio in app.radio if radio.label == "Workbench area")
    assert area.value == expected
    assert len(app.exception) == 0


def test_theme_resolution_precedence_and_fallback() -> None:
    assert initial_theme(
        _FakeStreamlit(query_theme="night", stored_theme="day", browser_theme="light")
    ) == "night"
    assert initial_theme(_FakeStreamlit(stored_theme="night", browser_theme="light")) == "night"
    assert initial_theme(_FakeStreamlit(browser_theme="dark")) == "night"
    assert initial_theme(_FakeStreamlit(browser_theme=None)) == "day"


def test_explicit_theme_persists_without_clearing_staged_work() -> None:
    st = _FakeStreamlit(stored_theme="day", selection="Night")
    bundle = {"suggestions_fingerprint": "unchanged"}
    st.session_state[STAGED_SUGGESTION_BUNDLE_KEY] = bundle

    assert render_theme_control(st) == "night"
    assert st.query_params["theme"] == "night"
    assert st.session_state[THEME_STATE_KEY] == "night"
    assert st.session_state[STAGED_SUGGESTION_BUNDLE_KEY] is bundle


def test_new_query_theme_overrides_stale_control_state() -> None:
    st = _FakeStreamlit(query_theme="night", stored_theme="day", selection="Day")

    assert render_theme_control(st) == "night"
    assert st.query_params["theme"] == "night"
    assert st.session_state[THEME_CONTROL_KEY] == "Night"
    assert st.session_state[THEME_QUERY_SYNC_KEY] == "night"


def test_explicit_control_change_updates_an_already_synchronized_query() -> None:
    st = _FakeStreamlit(query_theme="night", stored_theme="night", selection="Day")
    st.session_state[THEME_QUERY_SYNC_KEY] = "night"

    assert render_theme_control(st) == "day"
    assert st.query_params["theme"] == "day"


@pytest.mark.parametrize("theme", ["day", "night"])
def test_theme_css_obeys_technical_visual_contract(theme: str) -> None:
    css = _theme_css(theme)
    token_colors = DAY_TOKENS if theme == "day" else NIGHT_TOKENS

    assert "max-width: 1120px" in css
    assert "gradient" not in css.lower()
    assert "box-shadow" not in css.lower()
    letter_spacings = re.findall(r"letter-spacing:\s*([^;]+);", css)
    assert letter_spacings
    assert {value.strip().split()[0] for value in letter_spacings} == {"0"}
    radii = [float(value) for value in re.findall(r"border-radius:\s*([0-9.]+)px", css)]
    assert radii and max(radii) <= 4
    assert set(re.findall(r"#[0-9a-fA-F]{6}", css)) == {
        value for value in token_colors.values() if str(value).startswith("#")
    }
