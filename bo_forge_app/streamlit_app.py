"""Streamlit orchestration facade for the BO Forge workbench."""

from __future__ import annotations

import sys
from types import ModuleType

from bo_forge_app import streamlit_entry
from bo_forge_app.streamlit_entry import main, render_app
from bo_forge_app.ui import campaign_form, components, form_fields, state
from bo_forge_app.views import analyze, campaign, resolve, run

__all__ = ["main", "render_app"]
_COMPATIBILITY_MODULES = (
    streamlit_entry,
    campaign_form,
    form_fields,
    campaign,
    run,
    resolve,
    analyze,
    components,
    state,
)
_PATCH_TARGETS = {
    "_collect_panel_view_data": (streamlit_entry,),
    "_render_load_existing_campaign": (streamlit_entry, campaign_form),
    "_current_invalidation_reason": (run, state),
}

class _StreamlitFacade(ModuleType):
    """Expose v2 implementation helpers while v3 uses focused UI modules."""

    def __getattr__(self, name: str) -> object:
        for module in _COMPATIBILITY_MODULES:
            if name in vars(module):
                return vars(module)[name]
        raise AttributeError(f"module {self.__name__!r} has no attribute {name!r}")

    def __setattr__(self, name: str, value: object) -> None:
        super().__setattr__(name, value)
        targets = _PATCH_TARGETS.get(
            name, tuple(module for module in _COMPATIBILITY_MODULES if name in vars(module))
        )
        for module in targets:
            setattr(module, name, value)

sys.modules[__name__].__class__ = _StreamlitFacade

if __name__ == "__main__":
    main()
