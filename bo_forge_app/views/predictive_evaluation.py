"""Explicit, identity-bound predictive evaluation in the Analyze view."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from bo_forge._campaign.provenance import config_semantic_sha256, manifest_fingerprint
from bo_forge.application import CampaignAppService, dataframe_fingerprint
from bo_forge.errors import BOForgeError, LogConflictError
from bo_forge_app.streamlit_helpers import default_export_path
from bo_forge_app.ui.state import PROVENANCE_POLICY_KEY, _current_paths

EVALUATION_CACHE_KEY = "bo_forge_predictive_evaluation"
_OPTIONS_STATE_KEY = "bo_forge_predictive_options"
_PROFILES = ("default", "smooth", "rough", "robust")


def _supports_predictive_evaluation(config: Any) -> bool:
    return (
        not config.is_multi_objective
        and not config.is_contextual_campaign
        and not config.replicates.enabled
        and not config.is_structured_campaign
        and config.fidelity is None
    )


def _evaluation_identity(st: Any, campaign: Any, options: tuple) -> tuple:
    """Use source guards as well as content identities, including unsaved observations."""
    service = campaign if isinstance(campaign, CampaignAppService) else (
        CampaignAppService.from_session(campaign)
    )
    paths = tuple(path.expanduser().resolve() for path in _current_paths(st))
    loaded_paths = tuple(
        Path(path).expanduser().resolve() for path in (service.config_path, service.log_path)
    )
    policy = service.provenance_policy
    selected_required = st.session_state.get(PROVENANCE_POLICY_KEY, policy == "required")
    if paths != loaded_paths or selected_required != (policy == "required"):
        raise LogConflictError("Campaign paths or provenance policy changed. Reload the campaign.")
    service.session._assert_provenance_resumable()
    sources = service._verified_source_fingerprints()
    return (
        paths, policy, sources, manifest_fingerprint(service.log_path),
        config_semantic_sha256(service.config), dataframe_fingerprint(service.df), options,
    )


def _evaluation_options(st: Any, campaign: Any) -> tuple:
    """Retain options outside widget state, which Streamlit deletes during navigation."""
    owner = (
        str(Path(campaign.config_path).expanduser().resolve()),
        str(Path(campaign.log_path).expanduser().resolve()), campaign.config.model.profile,
    )
    saved = st.session_state.get(_OPTIONS_STATE_KEY)
    if saved is None or saved["owner"] != owner:
        saved = {"owner": owner, "profiles": [campaign.config.model.profile], "folds": 5, "seed": 0}
        for name in ("profiles", "folds", "seed"):
            st.session_state.pop(f"evaluation_{name}", None)
    profiles = st.multiselect(
        "Evaluation profiles (in order)", _PROFILES, default=saved["profiles"],
        key="evaluation_profiles",
    )
    folds = int(st.number_input(
        "Evaluation folds", min_value=2, max_value=5, value=saved["folds"], step=1,
        key="evaluation_folds",
    ))
    seed = int(st.number_input(
        "Evaluation seed", min_value=0, value=saved["seed"], step=1, key="evaluation_seed",
    ))
    st.session_state[_OPTIONS_STATE_KEY] = {
        "owner": owner, "profiles": list(profiles), "folds": folds, "seed": seed,
    }
    return profiles, folds, seed


def render_predictive_evaluation(st: Any, campaign: Any) -> None:
    """Fit only on request; discard stale results before exposing any result actions."""
    if not _supports_predictive_evaluation(campaign.config):
        st.session_state.pop(EVALUATION_CACHE_KEY, None)
        st.session_state.pop(_OPTIONS_STATE_KEY, None)
        return
    st.subheader("Predictive Evaluation")
    profiles, folds, seed = _evaluation_options(st, campaign)
    st.caption("Profile order: " + ", ".join(profiles))
    st.metric("Requested fits", len(profiles) * folds)
    options = (tuple(profiles), folds, seed)
    try:
        identity = _evaluation_identity(st, campaign, options)
    except (BOForgeError, OSError, ValueError) as exc:
        st.session_state.pop(EVALUATION_CACHE_KEY, None)
        st.warning(str(exc))
        return
    cached = st.session_state.get(EVALUATION_CACHE_KEY)
    if cached is not None and cached["identity"] != identity:
        st.session_state.pop(EVALUATION_CACHE_KEY, None)
        cached = None
        st.info("Predictive evaluation is stale. Run evaluation again for the current inputs.")
    if st.button("Run predictive evaluation", disabled=not profiles, key="evaluation_run"):
        st.session_state.pop(EVALUATION_CACHE_KEY, None)
        cached = None
        try:
            result = campaign.model_predictive_evaluation(profiles=profiles, folds=folds, seed=seed)
            if _evaluation_identity(st, campaign, options) != identity:
                raise LogConflictError("Campaign changed during predictive evaluation. Run again.")
        except (BOForgeError, OSError, ValueError) as exc:
            st.error(str(exc))
        else:
            cached = {"identity": identity, "result": result}
            st.session_state[EVALUATION_CACHE_KEY] = cached
    if cached is not None:
        _render_evaluation_result(st, cached["result"], Path(campaign.log_path))


def _render_evaluation_result(st: Any, result: Any, log_path: Path) -> None:
    from bo_forge_app.views.analyze import _render_plot_controls

    incomplete = result.summary.loc[result.summary.fit_status.ne("complete")]
    if not incomplete.empty:
        st.warning(
            "Incomplete predictive evaluation; aggregate metrics withheld for: "
            + ", ".join(incomplete.model_profile) + ". Inspect summary and fold messages."
        )
    warning_count = int(result.fold_outcomes.fit_warning_count.sum())
    if warning_count:
        st.warning(f"Captured {warning_count} fit warning(s). Inspect the fold evidence below.")
    st.dataframe(result.summary, hide_index=True, width="stretch")
    with st.expander("Held-out predictions and fold outcomes"):
        st.dataframe(result.predictions, hide_index=True, width="stretch")
        st.dataframe(result.fold_outcomes, hide_index=True, width="stretch")
        st.json(result.metadata)
    kind = st.selectbox(
        "Evaluation plot", ["Predictions", "Residuals"], key="evaluation_plot_kind",
    )
    suffix = kind.lower()
    _render_plot_controls(
        st, f"Evaluation {suffix}", f"evaluation_{suffix}",
        getattr(result, f"plot_{suffix}"),
        default_export_path(log_path, f"evaluation_{suffix}", "png"),
    )
    with st.form("evaluation_export_form"):
        output_dir = st.text_input(
            "Evaluation output directory",
            value=str(log_path.parent / f"{log_path.stem}_evaluation"),
            key="evaluation_output_dir",
        )
        export = st.form_submit_button("Export predictive evaluation")
    if export:
        try:
            result.export(Path(output_dir))
        except (BOForgeError, OSError, ValueError, TypeError) as exc:
            st.error(
                f"Could not export predictive evaluation: {exc}. The result is retained for retry."
            )
        else:
            label = (
                "incomplete predictive evaluation"
                if not incomplete.empty else "predictive evaluation"
            )
            st.success(f"Wrote {label}: {output_dir}")
