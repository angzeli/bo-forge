"""State UI ownership for BO Forge."""


from __future__ import annotations

from pathlib import Path
from typing import Any

from bo_forge.errors import BOForgeError
from bo_forge_app.streamlit_helpers import (
    CONFIG_PATH_KEY,
    LAST_APPENDED_FINGERPRINT_KEY,
    LOG_PATH_KEY,
    STAGED_SUGGESTION_BUNDLE_KEY,
    staged_bundle_invalidation_reason,
)

ACTIVE_PANEL_KEY = "bo_forge_active_panel"
CAMPAIGN_FILE_MODE_KEY = "bo_forge_campaign_file_mode"
FLASH_MESSAGE_KEY = "bo_forge_flash_message"
NEW_CAMPAIGN_FORM_YAML_KEY = "bo_forge_new_campaign_form_yaml"
NEW_CAMPAIGN_KIND_KEY = "bo_forge_new_campaign_kind"
PROVENANCE_POLICY_KEY = "bo_forge_require_provenance"
PROVENANCE_RECOVERY_KEY = "bo_forge_provenance_recovery"
REPORT_PREVIEW_KEY = "bo_forge_report_preview_text"
STAGED_FRESHNESS_MESSAGE_KEY = "bo_forge_staged_freshness_message"
SUGGEST_STAGE_KEY = "bo_forge_suggest_stage"
VALIDATION_CACHE_KEY = "bo_forge_validation_cache"
WORKFLOW_PANELS = ["Campaign", "Run", "Analyze"]
LEGACY_PANEL_MAP = {
    "Overview": "Campaign",
    "Data": "Campaign",
    "Suggest": "Run",
    "Resolve": "Run",
    "Reports": "Analyze",
}

def _current_paths(st: Any) -> tuple[Path, Path]:
    return Path(st.session_state[CONFIG_PATH_KEY]), Path(st.session_state[LOG_PATH_KEY])


def _cached_validation_label(st: Any, campaign: Any | None) -> str:
    return str(_cached_validation_state(st, campaign)["label"])


def _cached_validation_state(st: Any, campaign: Any | None) -> dict[str, str]:
    if campaign is None:
        return {"label": "Not loaded", "error": ""}
    active_policy = getattr(
        campaign,
        "provenance_policy",
        getattr(getattr(campaign, "session", campaign), "_provenance_policy", "compatible"),
    )
    cache = st.session_state.get(VALIDATION_CACHE_KEY)
    expected = _validation_cache_signature(
        st.session_state.get(CONFIG_PATH_KEY, ""),
        st.session_state.get(LOG_PATH_KEY, ""),
        require_provenance=active_policy == "required",
    )
    if not isinstance(cache, dict):
        return {"label": "Reload to validate", "error": ""}
    if cache.get("signature") != expected:
        return {"label": "Reload to validate", "error": ""}
    return {
        "label": str(cache.get("label", "Reload to validate")),
        "error": str(cache.get("error", "")),
    }


def _refresh_validation_cache(
    st: Any,
    campaign: Any,
    config_path: Path,
    log_path: Path,
) -> None:
    try:
        result = campaign.validate()
    except BOForgeError as exc:
        label = "Validation issue"
        error = str(exc)
    else:
        if hasattr(result, "label"):
            label = str(result.label)
            error = str(getattr(result, "message", ""))
        else:
            label = "Valid"
            error = ""
    st.session_state[VALIDATION_CACHE_KEY] = {
        "signature": _validation_cache_signature(
            config_path,
            log_path,
            require_provenance=(
                getattr(campaign, "provenance_policy", "compatible") == "required"
            ),
        ),
        "label": label,
        "error": error,
    }


def _validation_cache_signature(
    config_path: object,
    log_path: object,
    *,
    require_provenance: bool = False,
) -> tuple[object, object, object, str]:
    log_file = Path(str(log_path)).expanduser().resolve(strict=False)
    manifest_path = log_file.with_name(f"{log_file.name}.manifest.json")
    return (
        _file_metadata_signature(config_path),
        _file_metadata_signature(log_path),
        _file_metadata_signature(manifest_path),
        "required" if require_provenance else "compatible",
    )


def _file_metadata_signature(path_value: object) -> tuple[str, int | None, int | None]:
    path = Path(str(path_value)).expanduser()
    resolved = path.resolve(strict=False)
    try:
        stat_result = path.stat()
    except OSError:
        return (str(resolved), None, None)
    return (str(resolved), int(stat_result.st_size), int(stat_result.st_mtime_ns))


def _render_flash_message(st: Any) -> None:
    message = st.session_state.pop(FLASH_MESSAGE_KEY, None)
    if message:
        st.success(str(message))


def _flash_and_rerun(st: Any, message: str) -> None:
    st.session_state[FLASH_MESSAGE_KEY] = message
    rerun = getattr(st, "rerun", None)
    if callable(rerun):
        rerun()
    elif hasattr(st, "success"):
        st.success(message)


def _current_invalidation_reason(
    st: Any,
    bundle: dict[str, object] | None,
    *,
    stage: str | None = None,
    context_values: dict[str, object] | None = None,
) -> str | None:
    config_path, log_path = _current_paths(st)
    try:
        return staged_bundle_invalidation_reason(
            bundle=bundle,
            config_path=config_path,
            log_path=log_path,
            last_appended_fingerprint=st.session_state.get(LAST_APPENDED_FINGERPRINT_KEY),
            stage=stage,
            context_values=context_values,
        )
    except OSError as exc:
        return str(exc)


def _should_clear_staged_bundle(reason: str) -> bool:
    return reason in {
        "Config path changed after suggestions were staged.",
        "Log path changed after suggestions were staged.",
        "Stage selection changed after suggestions were staged.",
        "Context values changed after suggestions were staged.",
        "Config file changed after suggestions were staged.",
        "Log file changed after suggestions were staged.",
        "Staged suggestions changed after they were staged.",
    }


def _clear_staged_suggestions(st: Any) -> None:
    st.session_state.pop(STAGED_SUGGESTION_BUNDLE_KEY, None)
    st.session_state.pop(STAGED_FRESHNESS_MESSAGE_KEY, None)


def _clear_observation_inputs(st: Any) -> None:
    """Clear row-scoped observation values when the loaded campaign changes."""
    prefixes = ("observed_objective_", "actual_cost_")
    for key in list(st.session_state):
        if str(key).startswith(prefixes):
            st.session_state.pop(key, None)


def _clear_report_preview(st: Any) -> None:
    st.session_state.pop(REPORT_PREVIEW_KEY, None)
