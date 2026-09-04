"""Internal application service shared by Streamlit and optional HTTP adapters."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from bo_forge.config import CampaignConfig
from bo_forge.errors import BOForgeError, LogConflictError
from bo_forge.plot_registry import _PLOT_ROUTES
from bo_forge.session import CampaignSession, _format_campaign_report

if TYPE_CHECKING:
    from matplotlib.figure import Figure


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


def context_values_fingerprint(context_values: dict[str, object]) -> str:
    """Return a stable fingerprint for staged contextual metadata."""
    payload = json.dumps(
        context_values,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_staged_suggestion_bundle(
    suggestions: pd.DataFrame,
    config_path: str | Path,
    log_path: str | Path,
    *,
    stage: str | None = None,
    context_values: dict[str, object] | None = None,
    config_fingerprint: str | None = None,
    log_fingerprint: str | None = None,
) -> dict[str, object]:
    """Create a staged suggestion bundle tied to the current config/log files."""
    resolved_config_path = Path(config_path).expanduser().resolve()
    resolved_log_path = Path(log_path).expanduser().resolve()
    staged_suggestions = suggestions.copy(deep=True).reset_index(drop=True)
    bundle: dict[str, object] = {
        "suggestions": staged_suggestions,
        "suggestions_fingerprint": dataframe_fingerprint(staged_suggestions),
        "config_path": str(resolved_config_path),
        "config_fingerprint": config_fingerprint
        if config_fingerprint is not None
        else file_fingerprint(resolved_config_path),
        "log_path": str(resolved_log_path),
        "log_fingerprint": log_fingerprint
        if log_fingerprint is not None
        else file_fingerprint(resolved_log_path),
        "appended": False,
    }
    if stage is not None:
        bundle["stage"] = stage
    if context_values is not None:
        staged_context_values = dict(context_values)
        bundle["context_values"] = staged_context_values
        bundle["context_values_fingerprint"] = context_values_fingerprint(
            staged_context_values
        )
    return bundle


def staged_bundle_invalidation_reason(
    bundle: dict[str, object] | None,
    config_path: str | Path,
    log_path: str | Path,
    last_appended_fingerprint: str | None = None,
    stage: str | None = None,
    context_values: dict[str, object] | None = None,
) -> str | None:
    """Return a reason staged suggestions cannot be appended, or None."""
    payload_reason = _staged_payload_invalidation_reason(bundle, last_appended_fingerprint)
    if payload_reason is not None or bundle is None:
        return payload_reason
    path_reason, resolved_paths = _staged_path_invalidation_reason(
        bundle,
        config_path,
        log_path,
    )
    if path_reason is not None:
        return path_reason
    selection_reason = _staged_selection_invalidation_reason(bundle, stage, context_values)
    if selection_reason is not None:
        return selection_reason
    resolved_config_path, resolved_log_path = resolved_paths
    if file_fingerprint(resolved_config_path) != bundle.get("config_fingerprint"):
        return "Config file changed after suggestions were staged."
    if file_fingerprint(resolved_log_path) != bundle.get("log_fingerprint"):
        return "Log file changed after suggestions were staged."
    return None


def _staged_payload_invalidation_reason(
    bundle: dict[str, object] | None,
    last_appended_fingerprint: str | None,
) -> str | None:
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
    return None


def _staged_path_invalidation_reason(
    bundle: dict[str, object],
    config_path: str | Path,
    log_path: str | Path,
) -> tuple[str | None, tuple[Path, Path]]:
    resolved_config_path = Path(config_path).expanduser().resolve()
    resolved_log_path = Path(log_path).expanduser().resolve()
    if str(resolved_config_path) != bundle.get("config_path"):
        return "Config path changed after suggestions were staged.", (
            resolved_config_path,
            resolved_log_path,
        )
    if str(resolved_log_path) != bundle.get("log_path"):
        return "Log path changed after suggestions were staged.", (
            resolved_config_path,
            resolved_log_path,
        )
    return None, (resolved_config_path, resolved_log_path)


def _staged_selection_invalidation_reason(
    bundle: dict[str, object],
    stage: str | None,
    context_values: dict[str, object] | None,
) -> str | None:
    if "stage" in bundle and stage != bundle.get("stage"):
        return "Stage selection changed after suggestions were staged."
    if "context_values" in bundle:
        bundled_context_values = bundle.get("context_values")
        if not isinstance(bundled_context_values, dict):
            return "Context values changed after suggestions were staged."
        if (
            context_values_fingerprint(bundled_context_values)
            != bundle.get("context_values_fingerprint")
        ):
            return "Context values changed after suggestions were staged."
    if (
        "context_values" in bundle
        and context_values is not None
        and dict(context_values) != bundle.get("context_values")
    ):
        return "Context values changed after suggestions were staged."
    return None


def available_plot_kinds(config: CampaignConfig) -> list[str]:
    """Return plot kinds supported by the current config."""
    if config.is_multi_objective:
        kinds = ["pareto", "hypervolume"]
        if len(config.objectives) >= 3:
            kinds.append("pareto_parallel")
    else:
        kinds = ["progress", "diagnostics"]
        if config.fidelity is None and not config.is_structured_campaign:
            kinds.append("model_diagnostics")
            kinds.append("model_comparison")
    if config.is_structured_campaign:
        kinds.append("stage_diagnostics")
    if config.fidelity is not None:
        kinds.append("fidelity_diagnostics")
        kinds.append("fidelity_progress")
    if config.context is not None:
        kinds.append("context_diagnostics")
    if config.bo.acquisition == "qlog_nei":
        kinds.append("qlog_nei_diagnostics")
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



# Preserve read-only render compatibility without exposing raw session mutators.
_SESSION_READ_HELPERS = {
    "summary",
    "next_action",
    "observed_data",
    "pending_suggestions",
    "review_queue",
    "cost_summary",
    "replicate_summary",
    "best_observation",
    "best_replicate_group",
    "pareto_summary",
    "pareto_front",
    "campaign_status",
    "suggestion_quality",
    "stage_summary",
    "fidelity_summary",
    "fidelity_coverage",
    "context_summary",
    "qlog_nei_summary",
    "model_summary",
    "model_profile_comparison",
    "provenance_summary",
}

_PANEL_BASE_READERS = {
    "Campaign": (
        ("summary", "summary"),
        ("next_action", "next_action"),
        ("model_summary", "model_summary"),
        ("observed", "observed_data"),
        ("pending", "pending_suggestions"),
    ),
    "Run": (
        ("summary", "summary"),
        ("next_action", "next_action"),
        ("pending", "pending_suggestions"),
        ("observable", None),
    ),
    "Analyze": (
        ("summary", "summary"),
        ("next_action", "next_action"),
        ("model_summary", "model_summary"),
    ),
    "Overview": (
        ("summary", "summary"),
        ("next_action", "next_action"),
        ("model_summary", "model_summary"),
        ("observed", "observed_data"),
        ("pending", "pending_suggestions"),
    ),
    "Data": (
        ("summary", "summary"),
        ("next_action", "next_action"),
        ("model_summary", "model_summary"),
        ("observed", "observed_data"),
        ("pending", "pending_suggestions"),
    ),
    "Reports": (
        ("summary", "summary"),
        ("next_action", "next_action"),
        ("model_summary", "model_summary"),
    ),
    "Resolve": (
        ("pending", "pending_suggestions"),
        ("observable", None),
    ),
}


def _optional_panel_readers(
    config: CampaignConfig,
    panel: str,
) -> list[tuple[str, str | None]]:
    """Return feature-gated readers for one compatibility panel name."""
    campaign_panels = {"Overview", "Data", "Campaign"}
    run_panels = {"Suggest", "Resolve", "Run"}
    analyze_panels = {"Reports", "Analyze"}
    report_panels = campaign_panels | analyze_panels
    rules = (
        (panel in run_panels and config.review.enabled, "review_queue", "review_queue"),
        (panel in run_panels and config.cost is not None, "cost_summary", "cost_summary"),
        (
            panel in run_panels and config.context is not None and config.cost is not None,
            "cost_summary",
            "cost_summary",
        ),
        (
            panel in campaign_panels and config.is_multi_objective,
            "pareto_summary",
            "pareto_summary",
        ),
        (
            panel in {"Data", "Campaign"} and config.is_multi_objective,
            "pareto_front",
            "pareto_front",
        ),
        (panel in campaign_panels and config.cost is not None, "cost_summary", "cost_summary"),
        (
            panel in campaign_panels and config.replicates.enabled,
            "replicate_summary",
            "replicate_summary",
        ),
        (
            panel in report_panels and config.is_structured_campaign,
            "stage_summary",
            "stage_summary",
        ),
        (
            panel in report_panels and config.fidelity is not None,
            "fidelity_summary",
            "fidelity_summary",
        ),
        (
            panel in {"Data", "Reports", "Campaign", "Analyze"} and config.fidelity is not None,
            "fidelity_coverage",
            "fidelity_coverage",
        ),
        (
            panel in report_panels and config.context is not None,
            "context_summary",
            "context_summary",
        ),
        (
            panel in report_panels and config.bo.acquisition == "qlog_nei",
            "qlog_nei_summary",
            "qlog_nei_summary",
        ),
    )
    return [(field, reader) for enabled, field, reader in rules if enabled]


@dataclass(frozen=True)
class ValidationResult:
    """Validation state for app display."""

    ok: bool
    label: str
    message: str = ""


@dataclass
class CampaignViewData:
    """Panel-specific read models collected lazily for the app."""

    summary: pd.DataFrame | None = None
    next_action: pd.DataFrame | None = None
    observed: pd.DataFrame | None = None
    pending: pd.DataFrame | None = None
    review_queue: pd.DataFrame | None = None
    observable: pd.DataFrame | None = None
    pareto_summary: pd.DataFrame | None = None
    pareto_front: pd.DataFrame | None = None
    cost_summary: pd.DataFrame | None = None
    replicate_summary: pd.DataFrame | None = None
    stage_summary: pd.DataFrame | None = None
    fidelity_summary: pd.DataFrame | None = None
    fidelity_coverage: pd.DataFrame | None = None
    context_summary: pd.DataFrame | None = None
    qlog_nei_summary: pd.DataFrame | None = None
    model_summary: pd.DataFrame | None = None
    model_profile_comparison: pd.DataFrame | None = None
    provenance: pd.DataFrame | None = None

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-like compatibility for existing app render helpers."""
        value = getattr(self, key, None)
        return default if value is None else value

    def __contains__(self, key: object) -> bool:
        """Return True when a panel table was collected."""
        return isinstance(key, str) and getattr(self, key, None) is not None

    def __getitem__(self, key: str) -> Any:
        """Return a collected panel table by name."""
        value = getattr(self, key, None)
        if value is None:
            raise KeyError(key)
        return value


@dataclass(frozen=True)
class StagedSuggestionResult:
    """Dry-run suggestions plus the existing staged-bundle shape."""

    suggestions: pd.DataFrame
    bundle: dict[str, object]
    quality: pd.DataFrame


@dataclass(frozen=True)
class MutationResult:
    """Result from a mutating app workflow operation."""

    service: CampaignAppService
    validation: ValidationResult


@dataclass(frozen=True)
class AppendResult:
    """Result from appending staged suggestions."""

    service: CampaignAppService
    validation: ValidationResult
    appended_fingerprint: str


@dataclass(frozen=True)
class PlotResult:
    """Result from rendering or exporting a plot."""

    figure: object
    written_path: Path | None = None


@dataclass
class CampaignAppService:
    """Internal non-HTTP service that delegates BO behavior to CampaignSession."""

    session: CampaignSession

    @classmethod
    def load(cls, config_path: str | Path, log_path: str | Path) -> CampaignAppService:
        """Load a campaign service from YAML config and CSV log paths."""
        return cls(CampaignSession.from_files(config_path=config_path, log_path=log_path))

    @classmethod
    def from_session(cls, session: CampaignSession) -> CampaignAppService:
        """Wrap an existing campaign session."""
        return cls(session)

    @property
    def config(self) -> CampaignConfig:
        """Return the active campaign config."""
        return self.session.config

    @property
    def df(self) -> pd.DataFrame:
        """Return the active campaign log DataFrame."""
        return self.session.df

    @property
    def config_path(self) -> Path:
        """Return the active config path."""
        return self.session.config_path

    @property
    def log_path(self) -> Path:
        """Return the active log path."""
        return self.session.log_path

    def validate(self) -> ValidationResult:
        """Validate the current campaign and return app display state."""
        try:
            self.session.validate()
        except BOForgeError as exc:
            return ValidationResult(False, "Validation issue", str(exc))
        return ValidationResult(True, "Valid", "")

    def collect_view_data(self, panel: str) -> CampaignViewData:
        """Collect only the read data needed by one active app panel."""
        data = CampaignViewData()
        readers = [
            *_PANEL_BASE_READERS.get(panel, ()),
            *_optional_panel_readers(self.config, panel),
        ]
        for field, reader in readers:
            value = (
                observable_rows(self.config, self.df)
                if reader is None
                else getattr(self.session, reader)()
            )
            setattr(data, field, value)
        if panel in {"Campaign", "Overview", "Data"} and self.session.is_provenance_managed:
            data.provenance = self.session.provenance_summary()
        return data

    @property
    def is_provenance_managed(self) -> bool:
        """Return whether the active campaign has a provenance manifest."""
        return self.session.is_provenance_managed

    def provenance_summary(self) -> pd.DataFrame:
        """Return campaign provenance fields without performing filesystem repair."""
        return self.session.provenance_summary()

    def suggest_dry_run(
        self,
        batch_size: int,
        stage: str | None = None,
        context_values: dict[str, object] | None = None,
    ) -> StagedSuggestionResult:
        """Generate non-mutating suggestions and return staged app state."""
        config_fingerprint, log_fingerprint = self._verified_source_fingerprints()
        suggestions = self.session.suggest_next(
            batch_size=batch_size,
            stage=stage,
            context_values=context_values,
        )
        quality = self.session.suggestion_quality(suggestions)
        if (
            file_fingerprint(self.config_path) != config_fingerprint
            or file_fingerprint(self.log_path) != log_fingerprint
        ):
            raise LogConflictError(
                "Campaign config or log changed while suggestions were being generated. "
                "Reload the campaign and generate a new batch."
            )
        bundle = make_staged_suggestion_bundle(
            suggestions,
            self.config_path,
            self.log_path,
            stage=stage,
            context_values=context_values,
            config_fingerprint=config_fingerprint,
            log_fingerprint=log_fingerprint,
        )
        return StagedSuggestionResult(suggestions, bundle, quality)

    def append_staged(
        self,
        bundle: dict[str, object],
        last_appended_fingerprint: str | None = None,
        stage: str | None = None,
        context_values: dict[str, object] | None = None,
    ) -> AppendResult:
        """Append a valid staged bundle, reload, and return refreshed service state."""
        reason = staged_bundle_invalidation_reason(
            bundle=bundle,
            config_path=self.config_path,
            log_path=self.log_path,
            last_appended_fingerprint=last_appended_fingerprint,
            stage=stage,
            context_values=context_values,
        )
        if reason is not None:
            raise ValueError(reason)
        if (
            self.session.config_fingerprint is not None
            and self.session.config_fingerprint != bundle.get("config_fingerprint")
        ) or (
            self.session.log_fingerprint is not None
            and self.session.log_fingerprint != bundle.get("log_fingerprint")
        ):
            raise LogConflictError(
                "The loaded campaign state does not match the staged suggestions. "
                "Reload the campaign and generate a new batch."
            )
        suggestions = staged_suggestions_from_bundle(bundle)
        self.session.append_suggestions(
            suggestions,
            expected_log_fingerprint=str(bundle.get("log_fingerprint", "")),
        )
        appended_fingerprint = str(bundle.get("suggestions_fingerprint", ""))
        return AppendResult(self, self.validate(), appended_fingerprint)

    def _verified_source_fingerprints(self) -> tuple[str, str]:
        """Return current source fingerprints only when they match loaded state."""
        config_fingerprint = file_fingerprint(self.config_path)
        log_fingerprint = file_fingerprint(self.log_path)
        if (
            self.session.config_fingerprint is not None
            and config_fingerprint != self.session.config_fingerprint
        ) or (
            self.session.log_fingerprint is not None
            and log_fingerprint != self.session.log_fingerprint
        ):
            raise LogConflictError(
                "Campaign config or log changed after it was loaded. Reload the campaign "
                "before generating suggestions."
            )
        return config_fingerprint, log_fingerprint

    def export_staged_suggestions(self, bundle: dict[str, object], path: str | Path) -> Path:
        """Export staged suggestions without mutating campaign state."""
        return export_staged_suggestions_csv(staged_suggestions_from_bundle(bundle), path)

    def review(
        self,
        row_id: str,
        decision: str,
        note: str = "",
        *,
        expected_log_fingerprint: str | None = None,
    ) -> MutationResult:
        """Apply a review decision and return refreshed service state."""
        self.session.review_suggestion(
            row_id,
            decision,
            note,
            expected_log_fingerprint=expected_log_fingerprint,
        )
        return MutationResult(self, self.validate())

    def mark_observed(
        self,
        row_id: str,
        objective_value: float | None = None,
        objective_values: dict[str, float] | None = None,
        actual_cost: float | None = None,
        *,
        expected_log_fingerprint: str | None = None,
    ) -> MutationResult:
        """Mark one suggestion observed and return refreshed service state."""
        self.session.mark_observed(
            row_id,
            objective_value=objective_value,
            objective_values=objective_values,
            actual_cost=actual_cost,
            expected_log_fingerprint=expected_log_fingerprint,
        )
        return MutationResult(self, self.validate())

    def report_text(self) -> str:
        """Return deterministic report text without writing files."""
        return _format_campaign_report(self.session.report())

    def export_report(self, path: str | Path) -> Path:
        """Export the campaign report and return the written path."""
        return self.session.export_report(path)

    def available_plot_kinds(self) -> list[str]:
        """Return plot kinds supported by this campaign for app routing."""
        return available_plot_kinds(self.config)

    def plot(self, kind: str, save_path: str | Path | None = None) -> PlotResult:
        """Render or export one supported plot kind."""
        route = _PLOT_ROUTES.get(kind)
        if route is None:
            raise ValueError(f"Unsupported plot kind: {kind}")
        kwargs = {"save_path": save_path} if save_path is not None else {}
        result = getattr(self.session, route.session_method)(**kwargs)
        return PlotResult(
            figure=extract_matplotlib_figure(result),
            written_path=Path(save_path) if save_path is not None else None,
        )

    def __getattr__(self, name: str) -> Any:
        """Delegate read-only CampaignSession helpers used by existing render code."""
        if name not in _SESSION_READ_HELPERS:
            raise AttributeError(name)
        return getattr(self.session, name)
