"""Pure helpers for the local BO Forge Streamlit app."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from bo_forge.config import CampaignConfig
from bo_forge.session import CampaignSession, _format_campaign_report

if TYPE_CHECKING:
    from matplotlib.figure import Figure

CONFIG_PATH_KEY = "bo_forge_config_path"
LOG_PATH_KEY = "bo_forge_log_path"
SESSION_KEY = "bo_forge_campaign_session"
STAGED_SUGGESTION_BUNDLE_KEY = "bo_forge_staged_suggestion_bundle"
LAST_APPENDED_FINGERPRINT_KEY = "bo_forge_last_appended_fingerprint"


def resolve_path_input(value: str, label: str) -> Path:
    """Return a Path from a nonblank text input."""
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{label} path is required.")
    return Path(stripped).expanduser()


def load_campaign_session(config_path: str | Path, log_path: str | Path) -> CampaignSession:
    """Load a BO Forge campaign session from config and log paths."""
    return CampaignSession.from_files(config_path=config_path, log_path=log_path)


def file_fingerprint(path: str | Path) -> str:
    """Return a SHA256 fingerprint for a file's current bytes."""
    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataframe_fingerprint(df: pd.DataFrame) -> str:
    """Return a stable fingerprint for a DataFrame's display values and column order."""
    normalized = df.copy(deep=True).reset_index(drop=True)
    payload = normalized.to_csv(index=False, lineterminator="\n", na_rep="")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_staged_suggestion_bundle(
    suggestions: pd.DataFrame,
    config_path: str | Path,
    log_path: str | Path,
) -> dict[str, object]:
    """Create a staged suggestion bundle tied to the current config/log files."""
    resolved_config_path = Path(config_path).expanduser().resolve()
    resolved_log_path = Path(log_path).expanduser().resolve()
    staged_suggestions = suggestions.copy(deep=True).reset_index(drop=True)
    return {
        "suggestions": staged_suggestions,
        "suggestions_fingerprint": dataframe_fingerprint(staged_suggestions),
        "config_path": str(resolved_config_path),
        "config_fingerprint": file_fingerprint(resolved_config_path),
        "log_path": str(resolved_log_path),
        "log_fingerprint": file_fingerprint(resolved_log_path),
        "appended": False,
    }


def staged_bundle_invalidation_reason(
    bundle: dict[str, object] | None,
    config_path: str | Path,
    log_path: str | Path,
    last_appended_fingerprint: str | None = None,
) -> str | None:
    """Return a reason staged suggestions cannot be appended, or None."""
    if bundle is None:
        return "No staged suggestions."

    suggestions = bundle.get("suggestions")
    if not isinstance(suggestions, pd.DataFrame) or suggestions.empty:
        return "No staged suggestions."

    suggestions_fingerprint = str(bundle.get("suggestions_fingerprint", ""))
    if bool(bundle.get("appended", False)) or (
        last_appended_fingerprint is not None
        and suggestions_fingerprint == last_appended_fingerprint
    ):
        return "Staged suggestions were already appended."

    resolved_config_path = Path(config_path).expanduser().resolve()
    resolved_log_path = Path(log_path).expanduser().resolve()
    if str(resolved_config_path) != bundle.get("config_path"):
        return "Config path changed after suggestions were staged."
    if str(resolved_log_path) != bundle.get("log_path"):
        return "Log path changed after suggestions were staged."
    if file_fingerprint(resolved_config_path) != bundle.get("config_fingerprint"):
        return "Config file changed after suggestions were staged."
    if file_fingerprint(resolved_log_path) != bundle.get("log_fingerprint"):
        return "Log file changed after suggestions were staged."
    return None


def staged_bundle_is_appendable(
    bundle: dict[str, object] | None,
    config_path: str | Path,
    log_path: str | Path,
    last_appended_fingerprint: str | None = None,
) -> bool:
    """Return True when staged suggestions are still valid for append."""
    return (
        staged_bundle_invalidation_reason(
            bundle=bundle,
            config_path=config_path,
            log_path=log_path,
            last_appended_fingerprint=last_appended_fingerprint,
        )
        is None
    )


def feature_flags(config: CampaignConfig) -> dict[str, bool]:
    """Return feature flags used for conditional app panels."""
    return {
        "has_constraints": bool(config.constraints),
        "has_cost": config.cost is not None,
        "has_review": config.review.enabled,
        "has_replicates": config.replicates.enabled,
    }


def format_dataframe_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy suitable for Streamlit display."""
    return df.copy(deep=True).reset_index(drop=True)


def observable_rows(config: CampaignConfig, df: pd.DataFrame) -> pd.DataFrame:
    """Return suggested rows that can be marked observed from the app."""
    suggested = df["status"] == "suggested"
    if config.review.enabled:
        suggested = suggested & (df["review_status"] == "accepted")
    return df.loc[suggested].copy()


def export_staged_suggestions_csv(suggestions: pd.DataFrame, path: str | Path) -> Path:
    """Write staged suggestions to a standalone CSV without mutating app state."""
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suggestions.copy(deep=True).to_csv(output_path, index=False)
    return output_path


def campaign_report_text(campaign: CampaignSession) -> str:
    """Return the same deterministic report text used by CampaignSession.export_report."""
    return _format_campaign_report(campaign.report())


def default_export_path(log_path: Path, suffix: str, extension: str) -> Path:
    """Construct a deterministic report/figure path under reports/."""
    normalized_extension = extension.lstrip(".")
    stem = log_path.stem
    return Path("reports") / f"{stem}_{suffix}.{normalized_extension}"


def extract_matplotlib_figure(plot_result: object) -> Figure:
    """Extract the matplotlib Figure from a session plot return value."""
    from matplotlib.figure import Figure

    if isinstance(plot_result, Figure):
        return plot_result
    if isinstance(plot_result, (tuple, list)) and plot_result:
        first = plot_result[0]
        if isinstance(first, Figure):
            return first
    figure = getattr(plot_result, "figure", None)
    if isinstance(figure, Figure):
        return figure
    raise ValueError("Could not extract a matplotlib Figure from plot result.")


def staged_suggestions_from_bundle(bundle: dict[str, object] | None) -> pd.DataFrame:
    """Return staged suggestions as a copy, or an empty DataFrame."""
    if bundle is None:
        return pd.DataFrame()
    suggestions = bundle.get("suggestions")
    if not isinstance(suggestions, pd.DataFrame):
        return pd.DataFrame()
    return suggestions.copy(deep=True)


def mark_bundle_appended(bundle: dict[str, object]) -> dict[str, object]:
    """Return a copy of a staged bundle marked as appended."""
    updated = dict(bundle)
    updated["appended"] = True
    return updated
