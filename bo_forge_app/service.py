"""Internal app service layer for Streamlit-facing campaign workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from bo_forge.config import CampaignConfig
from bo_forge.errors import BOForgeError
from bo_forge.session import CampaignSession, _format_campaign_report
from bo_forge_app.streamlit_helpers import (
    available_plot_kinds,
    export_staged_suggestions_csv,
    extract_matplotlib_figure,
    make_staged_suggestion_bundle,
    observable_rows,
    staged_bundle_invalidation_reason,
    staged_suggestions_from_bundle,
)

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
    "context_summary",
}


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
    context_summary: pd.DataFrame | None = None

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
        if panel in {"Overview", "Data", "Reports"}:
            data.summary = self.session.summary()
            data.next_action = self.session.next_action()
        if panel in {"Overview", "Data"}:
            data.observed = self.session.observed_data()
            data.pending = self.session.pending_suggestions()
        if panel == "Resolve":
            data.pending = self.session.pending_suggestions()
            data.observable = observable_rows(self.config, self.df)
            if self.config.review.enabled:
                data.review_queue = self.session.review_queue()
        if panel in {"Overview", "Data"} and self.config.is_multi_objective:
            data.pareto_summary = self.session.pareto_summary()
            if panel == "Data":
                data.pareto_front = self.session.pareto_front()
        if panel in {"Overview", "Data"} and self.config.cost is not None:
            data.cost_summary = self.session.cost_summary()
        if panel in {"Overview", "Data"} and self.config.replicates.enabled:
            data.replicate_summary = self.session.replicate_summary()
        if panel in {"Overview", "Data", "Reports"} and self.config.is_structured_campaign:
            data.stage_summary = self.session.stage_summary()
        if panel in {"Overview", "Data", "Reports"} and self.config.fidelity is not None:
            data.fidelity_summary = self.session.fidelity_summary()
        if panel in {"Overview", "Data", "Reports"} and self.config.context is not None:
            data.context_summary = self.session.context_summary()
        return data

    def suggest_dry_run(
        self,
        batch_size: int,
        stage: str | None = None,
        context_values: dict[str, object] | None = None,
    ) -> StagedSuggestionResult:
        """Generate non-mutating suggestions and return staged app state."""
        suggestions = self.session.suggest_next(
            batch_size=batch_size,
            stage=stage,
            context_values=context_values,
        )
        bundle = make_staged_suggestion_bundle(
            suggestions,
            self.config_path,
            self.log_path,
            stage=stage,
            context_values=context_values,
        )
        quality = self.session.suggestion_quality(suggestions)
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
        suggestions = staged_suggestions_from_bundle(bundle)
        self.session.append_suggestions(suggestions)
        appended_fingerprint = str(bundle.get("suggestions_fingerprint", ""))
        return AppendResult(self, self.validate(), appended_fingerprint)

    def export_staged_suggestions(self, bundle: dict[str, object], path: str | Path) -> Path:
        """Export staged suggestions without mutating campaign state."""
        return export_staged_suggestions_csv(staged_suggestions_from_bundle(bundle), path)

    def review(self, row_id: str, decision: str, note: str = "") -> MutationResult:
        """Apply a review decision and return refreshed service state."""
        self.session.review_suggestion(row_id, decision, note)
        return MutationResult(self, self.validate())

    def mark_observed(
        self,
        row_id: str,
        objective_value: float | None = None,
        objective_values: dict[str, float] | None = None,
        actual_cost: float | None = None,
    ) -> MutationResult:
        """Mark one suggestion observed and return refreshed service state."""
        self.session.mark_observed(
            row_id,
            objective_value=objective_value,
            objective_values=objective_values,
            actual_cost=actual_cost,
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
        plotters = {
            "progress": self.session.plot_progress,
            "diagnostics": self.session.plot_diagnostics,
            "pareto": self.session.plot_pareto,
            "pareto_parallel": self.session.plot_pareto_parallel,
            "hypervolume": self.session.plot_hypervolume,
            "cost_progress": self.session.plot_cost_progress,
            "replicates": self.session.plot_replicates,
            "stage_diagnostics": self.session.plot_stage_diagnostics,
            "fidelity_diagnostics": self.session.plot_fidelity_diagnostics,
            "context_diagnostics": self.session.plot_context_diagnostics,
        }
        if kind not in plotters:
            raise ValueError(f"Unsupported plot kind: {kind}")
        kwargs = {"save_path": save_path} if save_path is not None else {}
        result = plotters[kind](**kwargs)
        return PlotResult(
            figure=extract_matplotlib_figure(result),
            written_path=Path(save_path) if save_path is not None else None,
        )

    def __getattr__(self, name: str) -> Any:
        """Delegate read-only CampaignSession helpers used by existing render code."""
        if name not in _SESSION_READ_HELPERS:
            raise AttributeError(name)
        return getattr(self.session, name)
