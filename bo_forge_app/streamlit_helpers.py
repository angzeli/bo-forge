"""Pure helpers for the local BO Forge Streamlit app."""

from __future__ import annotations

import hashlib
import os
import re
from math import isfinite
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
    objectives: list[dict[str, object]] | None = None,
    review_enabled: bool = False,
    replicates_enabled: bool = False,
    cost: dict[str, object] | None = None,
    fidelity: dict[str, object] | None = None,
    bo_overrides: dict[str, object] | None = None,
) -> str:
    """Build editable YAML text from structured app form values."""
    raw = {
        "campaign_name": campaign_name,
        "variables": variables,
        "bo": {
            "batch_size": int(batch_size),
            "initial_design_size": int(initial_design_size),
            "acquisition": "log_ei",
            "initial_design_method": initial_design_method,
            "random_seed": int(random_seed),
        },
    }
    if objectives:
        if fidelity is not None:
            raise ValueError("Multi-fidelity app-created campaigns are single-objective only.")
        raw["objectives"] = objectives
        raw["bo"]["acquisition"] = "qlog_ehvi"
    else:
        raw["objective"] = {
            "name": objective_name,
            "direction": objective_direction,
        }
    if fidelity is not None:
        raw["fidelity"] = fidelity
        raw["bo"]["acquisition"] = "qmf_kg"
    if bo_overrides:
        raw["bo"].update(bo_overrides)
    if review_enabled:
        raw["review"] = {"enabled": True}
    if replicates_enabled:
        raw["replicates"] = {"enabled": True}
    if cost is not None:
        raw["cost"] = cost
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
    *,
    stage: str | None = None,
) -> dict[str, object]:
    """Create a staged suggestion bundle tied to the current config/log files."""
    resolved_config_path = Path(config_path).expanduser().resolve()
    resolved_log_path = Path(log_path).expanduser().resolve()
    staged_suggestions = suggestions.copy(deep=True).reset_index(drop=True)
    bundle: dict[str, object] = {
        "suggestions": staged_suggestions,
        "suggestions_fingerprint": dataframe_fingerprint(staged_suggestions),
        "config_path": str(resolved_config_path),
        "config_fingerprint": file_fingerprint(resolved_config_path),
        "log_path": str(resolved_log_path),
        "log_fingerprint": file_fingerprint(resolved_log_path),
        "appended": False,
    }
    if stage is not None:
        bundle["stage"] = stage
    return bundle


def staged_bundle_invalidation_reason(
    bundle: dict[str, object] | None,
    config_path: str | Path,
    log_path: str | Path,
    last_appended_fingerprint: str | None = None,
    stage: str | None = None,
) -> str | None:
    """Return a reason staged suggestions cannot be appended, or None."""
    if bundle is None:
        return "No staged suggestions."

    suggestions = bundle.get("suggestions")
    if not isinstance(suggestions, pd.DataFrame) or suggestions.empty:
        return "No staged suggestions."

    suggestions_fingerprint = str(bundle.get("suggestions_fingerprint", ""))
    if dataframe_fingerprint(suggestions) != suggestions_fingerprint:
        return "Staged suggestions changed after they were staged."
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
    if "stage" in bundle and stage != bundle.get("stage"):
        return "Stage selection changed after suggestions were staged."
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
    stage: str | None = None,
) -> bool:
    """Return True when staged suggestions are still valid for append."""
    return (
        staged_bundle_invalidation_reason(
            bundle=bundle,
            config_path=config_path,
            log_path=log_path,
            last_appended_fingerprint=last_appended_fingerprint,
            stage=stage,
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


def structured_stage_options(config: CampaignConfig) -> list[str]:
    """Return configured stage names for app selectors."""
    return config.stage_names if config.is_structured_campaign else []


def active_variables_display(config: CampaignConfig, stage_name: str) -> str:
    """Return a compact active-variable label for one configured stage."""
    return ", ".join(config.active_variable_names_for_stage(stage_name))


def structured_stage_config_table(config: CampaignConfig) -> pd.DataFrame:
    """Return configured stages and active/inactive variables for app display."""
    rows: list[dict[str, str]] = []
    for stage_name in structured_stage_options(config):
        active = config.active_variable_names_for_stage(stage_name)
        inactive = [name for name in config.variable_names if name not in set(active)]
        rows.append(
            {
                "stage": stage_name,
                "active_variables": ", ".join(active),
                "inactive_variables": ", ".join(inactive),
            }
        )
    return pd.DataFrame(
        rows,
        columns=["stage", "active_variables", "inactive_variables"],
    )


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


def humanize_campaign_status(status: str) -> str:
    """Return a concise user-facing campaign status label."""
    labels = {
        "has_pending_suggestions": "Pending suggestions",
        "ready_for_initial_design": "Ready for initial design",
        "ready_for_bo": "Ready for BO",
    }
    return labels.get(status, status.replace("_", " ").title())


def humanize_next_action(action: str) -> str:
    """Return a concise user-facing next-action label."""
    labels = {
        "review_pending_suggestions": "Review pending suggestions",
        "run_accepted_suggestions": "Run accepted suggestions",
        "resolve_pending_suggestions": "Resolve pending suggestions",
        "suggest_initial_design": "Suggest initial design",
        "suggest_bo": "Suggest BO candidates",
    }
    return labels.get(action, action.replace("_", " ").title())


def status_tone(status: str) -> str:
    """Return a stable display tone for a campaign status."""
    tones = {
        "has_pending_suggestions": "warning",
        "ready_for_initial_design": "sage",
        "ready_for_bo": "success",
    }
    return tones.get(status, "neutral")


def shorten_identifier(value: str, head: int = 8, tail: int = 6) -> str:
    """Shorten a long identifier for display."""
    if len(value) <= head + tail + 1:
        return value
    return f"{value[:head]}...{value[-tail:]}"


def format_number_for_display(value: object, digits: int = 4) -> object:
    """Format numbers for compact display while leaving non-numbers unchanged."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            return ""
        rounded = round(value, digits)
        if rounded.is_integer():
            return int(rounded)
        return rounded
    return value


def drop_all_blank_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns where every value is blank or missing."""
    display_df = df.copy(deep=True)
    keep_columns: list[str] = []
    for column in display_df.columns:
        series = display_df[column]
        has_value = series.map(lambda value: not _is_blank_display_value(value)).any()
        if bool(has_value):
            keep_columns.append(column)
    return display_df.loc[:, keep_columns]


def select_display_columns(df: pd.DataFrame, preferred: list[str]) -> pd.DataFrame:
    """Move preferred columns to the front when they exist."""
    front = [column for column in preferred if column in df.columns]
    rest = [column for column in df.columns if column not in front]
    return df.loc[:, front + rest]


def compact_dataframe(df: pd.DataFrame, *, max_id_length: int = 16) -> pd.DataFrame:
    """Return a compact display copy with shorter IDs and rounded numbers."""
    display_df = drop_all_blank_columns(df.copy(deep=True).reset_index(drop=True))
    preferred = [
        "row_id",
        "iteration",
        "status",
        "review_status",
        "source",
        "replicate_group",
        "replicate_index",
        "objective_mean",
        "objective_std",
        "objective_min",
        "objective_max",
    ]
    display_df = select_display_columns(display_df, preferred)
    head = max(4, max_id_length // 2)
    tail = max(4, max_id_length - head - 1)
    for column in display_df.columns:
        if column == "row_id" or column.endswith("_id") or column == "replicate_group":
            display_df[column] = display_df[column].map(
                lambda value: ""
                if _is_blank_display_value(value)
                else shorten_identifier(str(value), head=head, tail=tail)
            )
        else:
            display_df[column] = display_df[column].map(format_number_for_display)
    return format_dataframe_for_display(display_df)


def empty_state_message(kind: str) -> tuple[str, str]:
    """Return a standard empty-state title and detail."""
    messages = {
        "staged_suggestions": (
            "No staged suggestions yet.",
            "Generate a dry-run batch to preview candidates before appending them.",
        ),
        "observed_rows": (
            "No observed rows yet.",
            "Run suggested experiments and mark them observed to build campaign history.",
        ),
        "pending_suggestions": (
            "No pending suggestions.",
            "Generate candidates in the Suggest tab when the campaign is ready.",
        ),
        "review_queue": (
            "No rows awaiting review.",
            "Review-enabled campaigns show pending review decisions here.",
        ),
        "cost_summary": (
            "No cost summary available.",
            "Cost summaries appear only when the campaign config includes cost settings.",
        ),
        "replicate_summary": (
            "No replicate summary available.",
            "Replicate summaries appear only when replicates are enabled.",
        ),
        "fidelity_summary": (
            "No fidelity summary available.",
            "Fidelity summaries appear only when a fidelity section is configured.",
        ),
        "report_preview": (
            "No report preview available.",
            "Load a valid campaign before exporting a report.",
        ),
        "plots": (
            "No plot available yet.",
            "Load a valid campaign and use the plot controls to render figures.",
        ),
        "best_observation": (
            "No best observation yet.",
            "Observed objective values are needed before a best row can be shown.",
        ),
    }
    return messages.get(
        kind,
        ("Nothing to show yet.", "This section will update as the campaign changes."),
    )


def append_disabled_reason(
    bundle: dict[str, object] | None,
    config_path: str | Path,
    log_path: str | Path,
    last_appended_fingerprint: str | None = None,
    stage: str | None = None,
) -> str | None:
    """Return a concise append-disabled reason, or None when append is allowed."""
    reason = staged_bundle_invalidation_reason(
        bundle=bundle,
        config_path=config_path,
        log_path=log_path,
        last_appended_fingerprint=last_appended_fingerprint,
        stage=stage,
    )
    if reason is None:
        return None
    messages = {
        "No staged suggestions.": "Append disabled: no staged suggestions.",
        "Staged suggestions were already appended.": (
            "Append disabled: this staged batch has already been appended."
        ),
        "Config path changed after suggestions were staged.": (
            "Append disabled: the active config path changed after staging."
        ),
        "Log path changed after suggestions were staged.": (
            "Append disabled: the active log path changed after staging."
        ),
        "Config file changed after suggestions were staged.": (
            "Append disabled: the campaign config changed after these suggestions were generated."
        ),
        "Log file changed after suggestions were staged.": (
            "Append disabled: the campaign log changed after these suggestions were generated."
        ),
        "Staged suggestions changed after they were staged.": (
            "Append disabled: the staged suggestion payload changed after staging."
        ),
        "Stage selection changed after suggestions were staged.": (
            "Append disabled: the selected stage changed after these suggestions were generated."
        ),
    }
    return messages.get(reason, f"Append disabled: {reason}")


def observable_row_options(config: CampaignConfig, df: pd.DataFrame) -> dict[str, str]:
    """Return display labels mapped to full row IDs for observable suggestions."""
    rows = observable_rows(config, df)
    options: dict[str, str] = {}
    for _, row in rows.iterrows():
        row_id = str(row["row_id"])
        details = []
        for variable in config.variables[:3]:
            value = row.get(variable.name, "")
            if not _is_blank_display_value(value):
                details.append(f"{variable.name}={format_number_for_display(value)}")
        label = shorten_identifier(row_id)
        if details:
            label = f"{label} - {', '.join(details)}"
        options[label] = row_id
    return options


def available_plot_kinds(config: CampaignConfig) -> list[str]:
    """Return plot kinds supported by the current config."""
    if config.is_multi_objective:
        kinds = ["pareto", "hypervolume"]
        if len(config.objectives) >= 3:
            kinds.append("pareto_parallel")
    else:
        kinds = ["progress", "diagnostics"]
    if config.is_structured_campaign:
        kinds.append("stage_diagnostics")
    if config.fidelity is not None:
        kinds.append("fidelity_diagnostics")
    if config.cost is not None:
        kinds.append("cost_progress")
    if config.replicates.enabled:
        kinds.append("replicates")
    return kinds


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


def _is_blank_display_value(value: object) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return isinstance(value, str) and value.strip() == ""


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
