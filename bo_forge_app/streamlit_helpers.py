"""Pure helpers for the local BO Forge Streamlit app."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING

import pandas as pd
import yaml

from bo_forge.config import CampaignConfig, parse_campaign_config
from bo_forge.errors import ConfigError
from bo_forge.io import empty_campaign_log
from bo_forge.session import CampaignSession, _format_campaign_report

if TYPE_CHECKING:
    from matplotlib.figure import Figure

CONFIG_PATH_KEY = "bo_forge_config_path"
LOG_PATH_KEY = "bo_forge_log_path"
SESSION_KEY = "bo_forge_campaign_session"
STAGED_SUGGESTION_BUNDLE_KEY = "bo_forge_staged_suggestion_bundle"
LAST_APPENDED_FINGERPRINT_KEY = "bo_forge_last_appended_fingerprint"
NEW_CAMPAIGN_YAML_KEY = "bo_forge_new_campaign_yaml"


def resolve_path_input(value: str, label: str) -> Path:
    """Return a Path from a nonblank text input."""
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{label} path is required.")
    return Path(stripped).expanduser()


def load_campaign_session(config_path: str | Path, log_path: str | Path) -> CampaignSession:
    """Load a BO Forge campaign session from config and log paths."""
    return CampaignSession.from_files(config_path=config_path, log_path=log_path)


def default_new_campaign_paths(campaign_name: str) -> tuple[Path, Path]:
    """Return suggested config/log paths for a new campaign name."""
    slug = _campaign_slug(campaign_name)
    return Path("configs") / f"{slug}.yaml", Path("examples") / f"{slug}_campaign_log.csv"


def parse_discrete_values_text(value_text: str, variable_name: str) -> list[float]:
    """Parse comma-separated numeric discrete values from app input."""
    parts = value_text.split(",")
    parsed: list[float] = []
    for index, part in enumerate(parts):
        stripped = part.strip()
        if not stripped:
            raise ValueError(
                f"Discrete variable '{variable_name}' has a blank value at position {index + 1}."
            )
        try:
            parsed.append(float(stripped))
        except ValueError as exc:
            raise ValueError(
                f"Discrete variable '{variable_name}' has non-numeric value {stripped!r}."
            ) from exc
    return parsed


def parse_categorical_values_text(value_text: str, variable_name: str) -> list[str]:
    """Parse comma-separated categorical labels from app input."""
    values: list[str] = []
    seen: set[str] = set()
    for index, part in enumerate(value_text.split(",")):
        label = part.strip()
        if not label:
            raise ValueError(
                f"Categorical variable '{variable_name}' has a blank label at position {index + 1}."
            )
        if label in seen:
            raise ValueError(
                f"Categorical variable '{variable_name}' has duplicate label {label!r}."
            )
        seen.add(label)
        values.append(label)
    return values


def build_campaign_yaml_text(
    *,
    campaign_name: str,
    objective_name: str,
    objective_direction: str,
    variables: list[dict[str, object]],
    batch_size: int,
    initial_design_size: int,
    initial_design_method: str,
    random_seed: int,
) -> str:
    """Build editable YAML text from structured app form values."""
    raw = {
        "campaign_name": campaign_name,
        "objective": {
            "name": objective_name,
            "direction": objective_direction,
        },
        "variables": variables,
        "bo": {
            "batch_size": int(batch_size),
            "initial_design_size": int(initial_design_size),
            "acquisition": "log_ei",
            "initial_design_method": initial_design_method,
            "random_seed": int(random_seed),
        },
    }
    return yaml.safe_dump(raw, sort_keys=False)


def parse_campaign_config_text(config_text: str) -> CampaignConfig:
    """Parse edited YAML text through BO Forge config validation."""
    try:
        raw = yaml.safe_load(config_text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse campaign YAML: {exc}") from exc
    return parse_campaign_config(raw)


def create_campaign_files(
    *,
    config_text: str,
    config_path: str | Path,
    log_path: str | Path,
) -> CampaignSession:
    """Create a validated config and empty canonical log, then load the session."""
    config = parse_campaign_config_text(config_text)
    resolved_config_path = Path(config_path).expanduser()
    resolved_log_path = Path(log_path).expanduser()
    if resolved_config_path.exists():
        raise FileExistsError(f"Config file already exists: {resolved_config_path}")
    if resolved_log_path.exists():
        raise FileExistsError(f"Campaign log already exists: {resolved_log_path}")

    resolved_config_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_log_path.parent.mkdir(parents=True, exist_ok=True)

    config_written = False
    try:
        _write_text_no_overwrite(resolved_config_path, config_text)
        config_written = True
        empty_log = empty_campaign_log(config)
        _write_dataframe_no_overwrite(resolved_log_path, empty_log)
    except Exception:
        if config_written:
            resolved_config_path.unlink(missing_ok=True)
        raise
    return CampaignSession.from_files(resolved_config_path, resolved_log_path)


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
    display_df = df.copy(deep=True).reset_index(drop=True)
    for column in display_df.columns:
        values = display_df[column].dropna()
        value_types = {type(value) for value in values}
        if len(value_types) > 1:
            display_df[column] = display_df[column].map(
                lambda value: "" if pd.isna(value) else str(value)
            )
    return display_df


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


def _campaign_slug(campaign_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", campaign_name.strip().lower()).strip("_")
    return slug or "new_campaign"


def _write_text_no_overwrite(path: Path, text: str) -> None:
    temp_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(text)
        os.link(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _write_dataframe_no_overwrite(path: Path, df: pd.DataFrame) -> None:
    temp_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            df.to_csv(temp_file, index=False)
        os.link(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
