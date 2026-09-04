"""Notebook-oriented campaign session wrapper."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from bo_forge._campaign.reports import (
    _base_report_tables,
    _bo_suggestion_reason,
    _format_campaign_report,
    _optional_report_readers,
    _pending_next_action,
    _suggest_and_append_call,
)
from bo_forge.config import CampaignConfig
from bo_forge.costs import (
    accepted_pending_estimated_cost,
    budget_remaining,
    observed_effective_cost,
)
from bo_forge.errors import LogConflictError
from bo_forge.logs import (
    _load_campaign_log_snapshot,
    _log_file_fingerprint,
    _session_log_fingerprint,
)
from bo_forge.logs import (
    append_suggestions as _append_suggestions,
)
from bo_forge.logs import (
    mark_observed as _mark_observed,
)
from bo_forge.logs import (
    review_suggestion as _review_suggestion,
)
from bo_forge.replicates import (
    best_replicate_group as _best_replicate_group,
)
from bo_forge.replicates import (
    modeling_observed_data,
)
from bo_forge.replicates import (
    replicate_summary as _replicate_summary,
)
from bo_forge.validation import (
    canonical_columns,
    get_observed_data,
    has_blocking_qlog_nehvi_review_suggestions,
    has_blocking_qlog_nei_review_suggestions,
    next_iteration,
    qlog_nehvi_active_pending_suggestions,
    qlog_nei_active_pending_suggestions,
    validate_campaign_data,
)


@dataclass
class CampaignSession:
    """Stateful notebook helper around the explicit BO Forge backend functions."""

    config_path: Path
    log_path: Path
    config: CampaignConfig
    df: pd.DataFrame
    log_fingerprint: str | None = None
    config_fingerprint: str | None = None
    _provenance_managed: bool | None = field(default=None, init=False, repr=False)

    @classmethod
    def initialize(cls, config_path: str | Path, log_path: str | Path) -> CampaignSession:
        """Create an empty provenance-managed campaign and return its session."""
        from bo_forge._campaign.provenance import initialize_campaign_session
        return initialize_campaign_session(cls, config_path, log_path)

    @classmethod
    def from_files(cls, config_path: str | Path, log_path: str | Path) -> CampaignSession:
        """Create a campaign session from a YAML config and CSV campaign log."""
        parsed_config_path = Path(config_path)
        parsed_log_path = Path(log_path)
        config_fingerprint = _log_file_fingerprint(parsed_config_path)
        config = CampaignConfig.from_yaml(parsed_config_path)
        if _log_file_fingerprint(parsed_config_path) != config_fingerprint:
            raise LogConflictError(
                "Campaign config changed while it was being loaded. Reload the campaign."
            )
        df, fingerprint = _load_campaign_log_snapshot(parsed_log_path, config)
        if _log_file_fingerprint(parsed_config_path) != config_fingerprint:
            raise LogConflictError(
                "Campaign config changed while it was being loaded. Reload the campaign."
            )
        from bo_forge._campaign.provenance import validate_manifest_for_load
        manifest = validate_manifest_for_load(
            parsed_config_path,
            parsed_log_path,
            config=config,
            log_row_count=len(df),
        )
        if manifest is not None and _log_file_fingerprint(parsed_log_path) != fingerprint:
            raise LogConflictError(
                "Campaign log changed while it was being loaded. Reload the campaign."
            )
        session = cls(
            config_path=parsed_config_path,
            log_path=parsed_log_path,
            config=config,
            df=df,
            log_fingerprint=fingerprint,
            config_fingerprint=config_fingerprint,
        )
        session._provenance_managed = manifest is not None
        return session

    def reload(self) -> pd.DataFrame:
        """Reload the campaign log from disk into the session."""
        from bo_forge._campaign.provenance import validate_manifest_for_load
        df, fingerprint = _load_campaign_log_snapshot(self.log_path, self.config)
        manifest = validate_manifest_for_load(
            self.config_path,
            self.log_path,
            config=self.config,
            log_row_count=len(df),
        )
        managed = manifest is not None
        if self._provenance_managed is not None and managed != self._provenance_managed:
            raise LogConflictError(
                "Campaign provenance state changed after it was loaded. Reload from files."
            )
        if managed and _log_file_fingerprint(self.log_path) != fingerprint:
            raise LogConflictError(
                "Campaign log changed while it was being reloaded. Reload the campaign."
            )
        self.df, self.log_fingerprint = df, fingerprint
        return self.df

    def validate(self) -> None:
        """Validate the current session DataFrame."""
        validate_campaign_data(self.config, self.df)

    def summary(self) -> pd.DataFrame:
        """Return a notebook-friendly two-column summary of campaign state."""
        self.validate()
        if self.config.is_multi_objective:
            return self._multi_objective_summary()
        observed = self.observed_data()
        pending = self.pending_suggestions()
        observed_count = len(observed)
        training_observed_count = len(modeling_observed_data(self.config, observed))
        pending_count = len(pending)
        initial_design_remaining = self._initial_design_remaining(training_observed_count)
        best = self.best_observation()
        if best.empty:
            best_row_id = None
            best_objective_value = None
        else:
            best_row_id = str(best["row_id"].iloc[0])
            best_objective_value = float(best[self.config.objective.name].iloc[0])

        rows = [
            ("campaign_name", self.config.campaign_name),
            ("campaign_status", self.campaign_status()),
            ("objective", self.config.objective.name),
            ("direction", self.config.objective.direction),
            ("total_rows", len(self.df)),
            ("observed_rows", observed_count),
            ("pending_suggestions", pending_count),
            ("initial_design_remaining", initial_design_remaining),
            ("next_iteration", next_iteration(self.df)),
            ("best_row_id", best_row_id),
            ("best_objective_value", best_objective_value),
            ("model_profile", self.config.model.profile),
        ]
        self._extend_structured_summary_rows(rows)
        self._extend_fidelity_summary_rows(rows)
        self._extend_context_summary_rows(rows)
        self._extend_qlog_nei_summary_rows(rows)
        if self.config.review.enabled:
            review_counts = self._review_status_counts()
            rows.extend(
                [
                    ("pending_review", review_counts["pending"]),
                    ("accepted_pending", review_counts["accepted"]),
                    ("rejected", review_counts["rejected"]),
                    ("deferred", review_counts["deferred"]),
                ]
            )
        if self.config.cost is not None:
            rows.extend(
                [
                    ("budget", self.config.cost.budget),
                    ("observed_effective_cost", observed_effective_cost(self.config, self.df)),
                    (
                        "accepted_pending_estimated_cost",
                        accepted_pending_estimated_cost(self.config, self.df),
                    ),
                    ("budget_remaining", budget_remaining(self.config, self.df)),
                ]
            )
        if self.config.replicates.enabled:
            replicate_summary = self.replicate_summary()
            best_group = self.best_replicate_group()
            if best_group.empty:
                best_replicate_group = None
                best_replicate_mean = None
            else:
                best_replicate_group = str(best_group["replicate_group"].iloc[0])
                best_replicate_mean = float(best_group["objective_mean"].iloc[0])
            rows.extend(
                [
                    ("replicate_groups", len(replicate_summary)),
                    (
                        "replicated_groups",
                        int((replicate_summary["n_replicates"] > 1).sum())
                        if not replicate_summary.empty
                        else 0,
                    ),
                    (
                        "max_replicates_per_group",
                        int(replicate_summary["n_replicates"].max())
                        if not replicate_summary.empty
                        else 0,
                    ),
                    ("best_replicate_group", best_replicate_group),
                    ("best_replicate_mean", best_replicate_mean),
                ]
            )
        return pd.DataFrame(rows, columns=["field", "value"])

    def _initial_design_remaining(self, training_observed_count: int) -> int:
        remaining = self.config.bo.initial_design_size - training_observed_count
        if self.config.bo.acquisition in {"qlog_nei", "qlog_nehvi"}:
            active_pending = (
                qlog_nei_active_pending_suggestions(self.df, self.config)
                if self.config.bo.acquisition == "qlog_nei"
                else qlog_nehvi_active_pending_suggestions(self.df, self.config)
            )
            if not active_pending.empty:
                pending_initial_count = int(
                    active_pending["source"].isin({"sobol", "random"}).sum()
                )
                remaining -= pending_initial_count
        return max(remaining, 0)

    def _multi_objective_summary(self) -> pd.DataFrame:
        observed = self.observed_data()
        pending = self.pending_suggestions()
        observed_count = len(observed)
        training_observed_count = len(modeling_observed_data(self.config, observed))
        pending_count = len(pending)
        initial_design_remaining = self._initial_design_remaining(training_observed_count)
        rows: list[tuple[str, object]] = [
            ("campaign_name", self.config.campaign_name),
            ("campaign_status", self.campaign_status()),
            ("objectives", ", ".join(self.config.objective_names)),
            (
                "directions",
                ", ".join(objective.direction for objective in self.config.objectives),
            ),
            (
                "reference_points",
                ", ".join(
                    f"{objective.name}={objective.reference_point:g}"
                    for objective in self.config.objectives
                ),
            ),
            ("total_rows", len(self.df)),
            ("observed_rows", observed_count),
            ("pending_suggestions", pending_count),
            ("initial_design_remaining", initial_design_remaining),
            ("next_iteration", next_iteration(self.df)),
            ("model_profile", self.config.model.profile),
        ]
        self._extend_structured_summary_rows(rows)
        self._extend_fidelity_summary_rows(rows)
        self._extend_context_summary_rows(rows)
        self._extend_qlog_nei_summary_rows(rows)
        if self.config.review.enabled:
            review_counts = self._review_status_counts()
            rows.extend(
                [
                    ("pending_review", review_counts["pending"]),
                    ("accepted_pending", review_counts["accepted"]),
                    ("rejected", review_counts["rejected"]),
                    ("deferred", review_counts["deferred"]),
                ]
            )
        if self.config.replicates.enabled:
            replicate_summary = self.replicate_summary()
            rows.extend(
                [
                    ("replicate_groups", len(replicate_summary)),
                    (
                        "replicated_groups",
                        int((replicate_summary["n_replicates"] > 1).sum())
                        if not replicate_summary.empty
                        else 0,
                    ),
                    (
                        "max_replicates_per_group",
                        int(replicate_summary["n_replicates"].max())
                        if not replicate_summary.empty
                        else 0,
                    ),
                ]
            )
        pareto_summary = self.pareto_summary()
        rows.extend(
            (str(row["field"]), row["value"]) for _, row in pareto_summary.iterrows()
        )
        return pd.DataFrame(rows, columns=["field", "value"])

    def _extend_structured_summary_rows(self, rows: list[tuple[str, object]]) -> None:
        if not self.config.is_structured_campaign:
            return
        rows.extend(
            [
                ("structured_campaign", True),
                ("stage_count", len(self.config.stages)),
                ("stages", ", ".join(self.config.stage_names)),
                (
                    "stage_active_variables",
                    "; ".join(
                        f"{stage.name}: {', '.join(stage.variables)}"
                        for stage in self.config.stages
                    ),
                ),
            ]
        )

    def _extend_fidelity_summary_rows(self, rows: list[tuple[str, object]]) -> None:
        if self.config.fidelity is None:
            return
        rows.extend(
            [
                ("multi_fidelity_campaign", True),
                ("fidelity_variable", self.config.fidelity.variable),
                ("target_fidelity", self.config.fidelity.target),
                ("fidelity_fixed_cost", self.config.fidelity.fixed_cost),
                ("fidelity_cost_weight", self.config.fidelity.fidelity_cost_weight),
                ("qmfkg_num_fantasies", self.config.fidelity.num_fantasies),
            ]
        )

    def _extend_context_summary_rows(self, rows: list[tuple[str, object]]) -> None:
        if self.config.context is None:
            return
        rows.extend(
            [
                ("contextual_campaign", True),
                ("context_variables", ", ".join(self.config.context_variable_names)),
                ("decision_variables", ", ".join(self.config.decision_variable_names)),
            ]
        )

    def _extend_qlog_nei_summary_rows(self, rows: list[tuple[str, object]]) -> None:
        if self.config.bo.acquisition != "qlog_nei":
            return
        values = {
            str(row["field"]): row["value"]
            for _, row in self.qlog_nei_summary().iterrows()
        }
        rows.extend(
            [
                ("qlog_nei_active_pending_rows", values["active_pending_rows"]),
                (
                    "qlog_nei_blocking_review_pending_rows",
                    values["blocking_review_pending_rows"],
                ),
                ("qlog_nei_ready", values["ready_for_qlog_nei"]),
                ("qlog_nei_x_pending_used", values["x_pending_used"]),
            ]
        )

    def _review_status_counts(self) -> dict[str, int]:
        suggested = self.df["status"] == "suggested"
        return {
            status: int((suggested & (self.df["review_status"] == status)).sum())
            for status in ["pending", "accepted", "rejected", "deferred"]
        }

    def observed_data(self) -> pd.DataFrame:
        """Return observed rows from the current session DataFrame."""
        return get_observed_data(self.config, self.df)

    def pending_suggestions(self) -> pd.DataFrame:
        """Return unresolved suggestions from the current session DataFrame."""
        self.validate()
        return self.df.loc[self.df["status"] == "suggested"].copy()

    def review_queue(self) -> pd.DataFrame:
        """Return suggested rows that are still pending review."""
        self.validate()
        if not self.config.review.enabled:
            return pd.DataFrame(columns=self.df.columns)
        return self.df.loc[
            (self.df["status"] == "suggested")
            & (self.df["review_status"] == "pending")
        ].copy()

    def cost_summary(self) -> pd.DataFrame:
        """Return cost and budget summary fields for the current campaign."""
        self.validate()
        if self.config.cost is None:
            return pd.DataFrame(
                columns=["field", "value"],
            )
        if self.config.is_multi_objective:
            pareto = self.pareto_summary()
            pareto_values = dict(zip(pareto["field"], pareto["value"], strict=True))
            rows = [
                ("total_observed_cost", observed_effective_cost(self.config, self.df)),
                ("accepted_pending_cost", accepted_pending_estimated_cost(self.config, self.df)),
                ("budget", self.config.cost.budget),
                ("budget_remaining", budget_remaining(self.config, self.df)),
                ("current_hypervolume", pareto_values.get("hypervolume")),
                ("pareto_count", pareto_values.get("pareto_count")),
            ]
            return pd.DataFrame(rows, columns=["field", "value"])
        best = self.best_observation()
        best_value = None if best.empty else float(best[self.config.objective.name].iloc[0])
        rows = [
            ("total_observed_cost", observed_effective_cost(self.config, self.df)),
            ("accepted_pending_cost", accepted_pending_estimated_cost(self.config, self.df)),
            ("budget", self.config.cost.budget),
            ("budget_remaining", budget_remaining(self.config, self.df)),
            ("best_observed_objective", best_value),
        ]
        return pd.DataFrame(rows, columns=["field", "value"])

    def campaign_status(self) -> str:
        """Return the current campaign status without mutating session state."""
        self.validate()
        pending_aware = self.config.bo.acquisition in {"qlog_nei", "qlog_nehvi"}
        if pending_aware:
            has_blocking_review = (
                has_blocking_qlog_nei_review_suggestions(self.df, self.config)
                if self.config.bo.acquisition == "qlog_nei"
                else has_blocking_qlog_nehvi_review_suggestions(self.df, self.config)
            )
            if has_blocking_review:
                return "has_pending_suggestions"
            active_pending = (
                qlog_nei_active_pending_suggestions(self.df, self.config)
                if self.config.bo.acquisition == "qlog_nei"
                else qlog_nehvi_active_pending_suggestions(self.df, self.config)
            )
            active_pending_initial = (
                active_pending["source"].isin({"sobol", "random"})
                if not active_pending.empty
                else pd.Series(dtype=bool)
            )
            pending_count = int(active_pending_initial.sum())
        elif self.config.review.enabled:
            pending_count = int(
                (
                    (self.df["status"] == "suggested")
                    & self.df["review_status"].isin(["pending", "accepted"])
                ).sum()
            )
        else:
            pending_count = int((self.df["status"] == "suggested").sum())
        observed = get_observed_data(self.config, self.df)
        observed_count = len(modeling_observed_data(self.config, observed))
        if observed_count < self.config.bo.initial_design_size:
            if (
                pending_aware
                and observed_count + pending_count < self.config.bo.initial_design_size
            ):
                return "ready_for_initial_design"
            if pending_count > 0:
                return "has_pending_suggestions"
            return "ready_for_initial_design"
        if not pending_aware and pending_count > 0:
            return "has_pending_suggestions"
        return "ready_for_bo"

    def next_action(self) -> pd.DataFrame:
        """Return the recommended next notebook action without mutating state."""
        campaign_status = self.campaign_status()
        if campaign_status == "has_pending_suggestions":
            action, reason, suggested_call = _pending_next_action(self)
        elif campaign_status == "ready_for_initial_design":
            action = "suggest_initial_design"
            reason = "Observed rows are below initial_design_size; request Sobol suggestions."
            suggested_call = _suggest_and_append_call(self.config, include_batch_size=False)
        else:
            action = "suggest_bo"
            reason = _bo_suggestion_reason(self)
            suggested_call = _suggest_and_append_call(self.config, include_batch_size=True)

        return pd.DataFrame(
            [
                {
                    "campaign_status": campaign_status,
                    "action": action,
                    "reason": reason,
                    "suggested_call": suggested_call,
                }
            ],
            columns=["campaign_status", "action", "reason", "suggested_call"],
        )

    def report(self) -> dict[str, pd.DataFrame]:
        """Return read-only campaign report tables for notebook display."""
        tables = _base_report_tables(self)
        for name, reader in _optional_report_readers(self):
            tables[name] = reader()
        if self.is_provenance_managed:
            tables["provenance"] = self.provenance_summary()
        return tables

    @property
    def is_provenance_managed(self) -> bool:
        """Return whether this campaign has a valid provenance manifest."""
        return bool(self._provenance_managed)

    def provenance_summary(self) -> pd.DataFrame:
        """Return ordered provenance fields for managed or legacy campaigns."""
        from bo_forge.provenance import provenance_summary
        return provenance_summary(self.config_path, self.log_path)

    def export_report(self, path: str | Path) -> Path:
        """Write a deterministic plain-text campaign report and return its path."""
        report_path = Path(path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        text = _format_campaign_report(self.report())
        report_path.write_text(text + "\n", encoding="utf-8")
        return report_path

    def best_observation(self) -> pd.DataFrame:
        """Return the best observed row as a canonical-order copy."""
        if self.config.is_multi_objective:
            raise ValueError(
                "best_observation() is only defined for single-objective campaigns; "
                "use pareto_front() for multi-objective campaigns."
            )
        observed = self.observed_data()
        columns = canonical_columns(self.config)
        if observed.empty:
            return pd.DataFrame(columns=columns)

        objective = self.config.objective.name
        values = pd.to_numeric(observed[objective])
        if self.config.objective.direction == "maximize":
            best_index = values.idxmax()
        else:
            best_index = values.idxmin()
        return observed.loc[[best_index], columns].copy()

    def pareto_front(self) -> pd.DataFrame:
        """Return nondominated observed rows for a multi-objective campaign."""
        from bo_forge.multi_objective import pareto_front

        return pareto_front(self.config, self.df)

    def pareto_summary(self) -> pd.DataFrame:
        """Return Pareto-front and hypervolume summary fields."""
        from bo_forge.multi_objective import pareto_summary

        return pareto_summary(self.config, self.df)

    def stage_summary(self) -> pd.DataFrame:
        """Return structured-campaign stage summary rows."""
        from bo_forge.structured import stage_summary

        return stage_summary(self.config, self.df)

    def fidelity_summary(self) -> pd.DataFrame:
        """Return multi-fidelity campaign summary fields."""
        from bo_forge.multifidelity import fidelity_summary

        return fidelity_summary(self.config, self.df)

    def fidelity_coverage(self) -> pd.DataFrame:
        """Return observed and active-suggestion coverage by fidelity value."""
        from bo_forge.multifidelity import fidelity_coverage

        return fidelity_coverage(self.config, self.df)

    def context_summary(self) -> pd.DataFrame:
        """Return contextual-campaign summary rows by context combination."""
        from bo_forge.contextual import context_summary

        return context_summary(self.config, self.df)

    def qlog_nei_summary(self) -> pd.DataFrame:
        """Return qLogNEI pending-state summary fields."""
        from bo_forge.noisy import qlog_nei_summary

        return qlog_nei_summary(self.config, self.df)

    def model_summary(self) -> pd.DataFrame:
        """Return model-profile and fitting-input summary fields."""
        from bo_forge.models import model_summary

        return model_summary(self.config, self.df)

    def model_profile_comparison(
        self, profiles: list[str] | tuple[str, ...] | None = None
    ) -> pd.DataFrame:
        """Return read-only model-profile comparison diagnostics."""
        from bo_forge.models import model_profile_comparison

        return model_profile_comparison(self.config, self.df, profiles=profiles)

    def replicate_summary(self) -> pd.DataFrame:
        """Return observed replicate-group summary statistics."""
        return _replicate_summary(self.config, self.df)

    def best_replicate_group(self) -> pd.DataFrame:
        """Return the best replicate group by mean objective."""
        return _best_replicate_group(self.config, self.df)

    def suggest_next(
        self,
        batch_size: int | None = None,
        stage: str | None = None,
        context_values: dict[str, object] | None = None,
    ) -> pd.DataFrame:
        """Return suggested candidates without mutating session state or writing to disk."""
        from bo_forge.suggestions import suggest_next

        return suggest_next(
            self.config,
            self.df.copy(deep=True),
            batch_size=batch_size,
            stage=stage,
            context_values=context_values,
        )

    def suggestion_quality(self, suggestions: pd.DataFrame) -> pd.DataFrame:
        """Return read-only quality diagnostics for suggested rows."""
        from bo_forge.suggestions import suggestion_quality_summary

        return suggestion_quality_summary(
            self.config,
            self.df.copy(deep=True),
            suggestions.copy(deep=True),
        )

    def append_suggestions(
        self,
        suggestions: pd.DataFrame,
        *,
        expected_log_fingerprint: str | None = None,
    ) -> pd.DataFrame:
        """Append suggestions to disk, reload the session, and return the refreshed log."""
        _append_suggestions(
            self.log_path,
            suggestions,
            config=self.config,
            expected_log_fingerprint=self._mutation_fingerprint(expected_log_fingerprint),
        )
        return self.reload()

    def mark_observed(
        self,
        row_id: str,
        objective_value: float | None = None,
        objective_values: dict[str, float] | None = None,
        actual_cost: float | None = None,
        *,
        expected_log_fingerprint: str | None = None,
    ) -> pd.DataFrame:
        """Mark one pending suggestion observed, reload, and return the refreshed log."""
        _mark_observed(
            self.log_path,
            row_id,
            objective_value=objective_value,
            objective_values=objective_values,
            actual_cost=actual_cost,
            config=self.config,
            expected_log_fingerprint=self._mutation_fingerprint(expected_log_fingerprint),
        )
        return self.reload()

    def review_suggestion(
        self,
        row_id: str,
        decision: str,
        note: str = "",
        *,
        expected_log_fingerprint: str | None = None,
    ) -> pd.DataFrame:
        """Review one pending suggestion, reload, and return the refreshed log."""
        _review_suggestion(
            self.log_path,
            row_id,
            decision,
            note,
            config=self.config,
            expected_log_fingerprint=self._mutation_fingerprint(expected_log_fingerprint),
        )
        return self.reload()

    def _mutation_fingerprint(self, expected: str | None) -> str | None:
        fingerprint = self.log_fingerprint if expected is None else expected
        if self._provenance_managed is None:
            return fingerprint
        return _session_log_fingerprint(fingerprint, managed=self._provenance_managed)

    def plot_progress(self, **kwargs: Any) -> Any:
        """Plot campaign progress and return figure/axes objects."""
        from bo_forge.diagnostics import plot_progress as _plot_progress

        return _plot_progress(self.config, self.df, **kwargs)

    def plot_diagnostics(self, **kwargs: Any) -> Any:
        """Plot campaign diagnostics and return figure/axes objects."""
        from bo_forge.diagnostics import plot_diagnostics as _plot_diagnostics

        return _plot_diagnostics(self.config, self.df, **kwargs)

    def plot_cost_progress(self, **kwargs: Any) -> Any:
        """Plot best observed objective against cumulative effective cost."""
        from bo_forge.diagnostics import plot_cost_progress as _plot_cost_progress

        return _plot_cost_progress(self.config, self.df, **kwargs)

    def plot_replicates(self, **kwargs: Any) -> Any:
        """Plot replicate-group objective summaries and return figure/axes objects."""
        from bo_forge.diagnostics import plot_replicates as _plot_replicates

        return _plot_replicates(self.config, self.df, **kwargs)

    def plot_pareto(self, **kwargs: Any) -> Any:
        """Plot observed Pareto diagnostics for a multi-objective campaign."""
        from bo_forge.diagnostics import plot_pareto as _plot_pareto

        return _plot_pareto(self.config, self.df, **kwargs)

    def plot_pareto_parallel(self, **kwargs: Any) -> Any:
        """Plot Pareto-front rows with normalized parallel coordinates."""
        from bo_forge.diagnostics import plot_pareto_parallel as _plot_pareto_parallel

        return _plot_pareto_parallel(self.config, self.df, **kwargs)

    def plot_hypervolume(self, **kwargs: Any) -> Any:
        """Plot hypervolume progress for a multi-objective campaign."""
        from bo_forge.diagnostics import plot_hypervolume as _plot_hypervolume

        return _plot_hypervolume(self.config, self.df, **kwargs)

    def plot_stage_diagnostics(self, **kwargs: Any) -> Any:
        """Plot structured-campaign stage diagnostics."""
        from bo_forge.diagnostics import plot_stage_diagnostics as _plot_stage_diagnostics

        return _plot_stage_diagnostics(self.config, self.df, **kwargs)

    def plot_fidelity_diagnostics(self, **kwargs: Any) -> Any:
        """Plot observed multi-fidelity diagnostics."""
        from bo_forge.diagnostics import (
            plot_fidelity_diagnostics as _plot_fidelity_diagnostics,
        )

        return _plot_fidelity_diagnostics(self.config, self.df, **kwargs)

    def plot_fidelity_progress(self, **kwargs: Any) -> Any:
        """Plot fidelity use and target-fidelity objective progress."""
        from bo_forge.diagnostics import plot_fidelity_progress as _plot_fidelity_progress

        return _plot_fidelity_progress(self.config, self.df, **kwargs)

    def plot_context_diagnostics(self, **kwargs: Any) -> Any:
        """Plot observed contextual diagnostics."""
        from bo_forge.diagnostics import (
            plot_context_diagnostics as _plot_context_diagnostics,
        )

        return _plot_context_diagnostics(self.config, self.df, **kwargs)

    def plot_model_diagnostics(self, **kwargs: Any) -> Any:
        """Plot model posterior-vs-observed diagnostics."""
        from bo_forge.diagnostics import (
            plot_model_diagnostics as _plot_model_diagnostics,
        )

        return _plot_model_diagnostics(self.config, self.df, **kwargs)

    def plot_model_comparison(self, **kwargs: Any) -> Any:
        """Plot model-profile comparison diagnostics."""
        from bo_forge.diagnostics import (
            plot_model_comparison as _plot_model_comparison,
        )

        return _plot_model_comparison(self.config, self.df, **kwargs)

    def plot_qlog_nei_diagnostics(self, **kwargs: Any) -> Any:
        """Plot qLogNEI pending-state diagnostics."""
        from bo_forge.diagnostics import (
            plot_qlog_nei_diagnostics as _plot_qlog_nei_diagnostics,
        )

        return _plot_qlog_nei_diagnostics(self.config, self.df, **kwargs)
