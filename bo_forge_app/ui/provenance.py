"""Explicit provenance lifecycle preview, confirmation and reload controls."""

from __future__ import annotations

import json

from bo_forge.application import CampaignAppService
from bo_forge.errors import BOForgeError
from bo_forge_app.streamlit_helpers import CONFIG_PATH_KEY, LOG_PATH_KEY, SESSION_KEY
from bo_forge_app.ui.state import (
    PROVENANCE_POLICY_KEY,
    PROVENANCE_RECOVERY_KEY,
    _clear_observation_inputs,
    _clear_report_preview,
    _clear_staged_suggestions,
    _flash_and_rerun,
    _refresh_validation_cache,
)

_PREVIEW_KEY = "bo_forge_lifecycle_preview"


def render_lifecycle_actions(st, config_path, log_path):
    if not config_path or not log_path:
        return
    with st.expander("Provenance lifecycle", expanded=False):
        actions = _inspect_actions(st, config_path, log_path)
        if not actions:
            return
        operation = st.selectbox(
            "Lifecycle action",
            actions,
            key="bo_forge_lifecycle_action",
        )
        if operation == "adopt":
            st.warning("Adoption starts tracking now. Earlier campaign history remains unknown.")
        destination, changes = None, {}
        if operation == "fork":
            destination = st.text_input("New campaign directory", key="bo_forge_fork_directory")
            changes_text = st.text_area(
                "Fork config changes (JSON)", value="{}", key="bo_forge_fork_config_changes"
            )
            try:
                changes = json.loads(changes_text)
            except ValueError:
                st.error("Fork changes must be a JSON mapping.")
                return
        selection = (str(config_path), str(log_path), operation, destination, changes)
        stored = st.session_state.get(_PREVIEW_KEY)
        if stored is not None and stored["selection"] != selection:
            st.session_state.pop(_PREVIEW_KEY, None)
            stored = None
        if st.button("Preview lifecycle action"):
            try:
                preview = CampaignAppService.provenance_lifecycle(
                    operation,
                    config_path,
                    log_path,
                    destination=destination,
                    config_changes=changes,
                )
            except (BOForgeError, OSError, ValueError) as exc:
                st.session_state.pop(_PREVIEW_KEY, None)
                st.error(str(exc))
                return
            stored = {"selection": selection, "preview": preview}
            st.session_state[_PREVIEW_KEY] = stored
            st.session_state["bo_forge_lifecycle_confirm"] = False
        if stored is None:
            return
        st.json(stored["preview"])
        if stored["preview"]["no_op"]:
            st.info("The campaign already satisfies this lifecycle operation.")
            return
        reason = st.text_input("Lifecycle reason", key="bo_forge_lifecycle_reason")
        confirmed = st.checkbox(
            "Confirm this provenance lifecycle change", key="bo_forge_lifecycle_confirm"
        )
        if st.button("Apply lifecycle action", disabled=not confirmed or not reason.strip()):
            _apply(st, stored, reason)


def _inspect_actions(st, config_path, log_path):
    source = (str(config_path), str(log_path))
    inspected = st.session_state.get("bo_forge_lifecycle_actions")
    if st.button("Inspect lifecycle actions"):
        try:
            actions = CampaignAppService.provenance_lifecycle_actions(config_path, log_path)
        except (BOForgeError, OSError, ValueError) as exc:
            st.session_state.pop("bo_forge_lifecycle_actions", None)
            st.error(str(exc))
            return []
        inspected = {"source": source, "actions": actions}
        st.session_state["bo_forge_lifecycle_actions"] = inspected
        st.session_state.pop(_PREVIEW_KEY, None)
    if inspected is None or inspected["source"] != source:
        return []
    if not inspected["actions"]:
        st.info("No lifecycle action is available for this campaign state.")
    return inspected["actions"]


def _apply(st, stored, reason):
    config_path, log_path, operation, destination, changes = stored["selection"]
    try:
        CampaignAppService.provenance_lifecycle(
            operation,
            config_path,
            log_path,
            apply=True,
            reason=reason,
            expected_identities=stored["preview"]["expected_identities"],
            destination=destination,
            config_changes=changes,
        )
        if operation == "fork":
            from pathlib import Path

            config_path = str(Path(destination) / "campaign.yaml")
            log_path = str(Path(destination) / "campaign.csv")
        policy = "required" if st.session_state.get(PROVENANCE_POLICY_KEY, False) else "compatible"
        campaign = CampaignAppService.load(config_path, log_path, provenance_policy=policy)
    except (BOForgeError, OSError, ValueError) as exc:
        st.session_state.pop(_PREVIEW_KEY, None)
        _clear_staged_suggestions(st)
        st.error(str(exc))
        return
    st.session_state[CONFIG_PATH_KEY] = str(config_path)
    st.session_state[LOG_PATH_KEY] = str(log_path)
    st.session_state[SESSION_KEY] = campaign
    st.session_state.pop(_PREVIEW_KEY, None)
    st.session_state.pop("bo_forge_lifecycle_actions", None)
    st.session_state.pop(PROVENANCE_RECOVERY_KEY, None)
    _clear_staged_suggestions(st)
    _clear_observation_inputs(st)
    _clear_report_preview(st)
    _refresh_validation_cache(st, campaign, config_path, log_path)
    _flash_and_rerun(st, "Provenance updated and campaign reloaded.")
