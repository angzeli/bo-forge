"""Notebook-oriented campaign session wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from bo_forge.config import CampaignConfig
from bo_forge.costs import (
    accepted_pending_estimated_cost,
    budget_remaining,
    observed_effective_cost,
)
from bo_forge.logs import (
    append_suggestions as _append_suggestions,
)
from bo_forge.logs import (
    load_campaign_log,
)
from bo_forge.logs import (
    mark_observed as _mark_observed,
)
from bo_forge.logs import (
    review_suggestion as _review_suggestion,
)
from bo_forge.multi_objective import (
    pareto_front as _pareto_front,
)
from bo_forge.multi_objective import (
    pareto_summary as _pareto_summary,
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
from bo_forge.structured import stage_summary as _stage_summary
from bo_forge.suggestions import (
    suggest_next as _suggest_next,
)
from bo_forge.suggestions import (
    suggestion_quality_summary,
)
from bo_forge.validation import (
    canonical_columns,
    get_observed_data,
    next_iteration,
    validate_campaign_data,
)


@dataclass
class CampaignSession:
    """Stateful notebook helper around the explicit BO Forge backend functions."""

    config_path: Path
    log_path: Path
    config: CampaignConfig
    df: pd.DataFrame

    @classmethod
    def from_files(cls, config_path: str | Path, log_path: str | Path) -> CampaignSession:
        """Create a campaign session from a YAML config and CSV campaign log."""
        parsed_config_path = Path(config_path)
        parsed_log_path = Path(log_path)
        config = CampaignConfig.from_yaml(parsed_config_path)
        df = load_campaign_log(parsed_log_path, config)
        return cls(
            config_path=parsed_config_path,
            log_path=parsed_log_path,
            config=config,
            df=df,
        )

    def reload(self) -> pd.DataFrame:
        """Reload the campaign log from disk into the session."""
        self.df = load_campaign_log(self.log_path, self.config)
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
        initial_design_remaining = max(
            self.config.bo.initial_design_size - training_observed_count,
            0,
        )
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
        ]
        self._extend_structured_summary_rows(rows)
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

    def _multi_objective_summary(self) -> pd.DataFrame:
        observed = self.observed_data()
        pending = self.pending_suggestions()
        observed_count = len(observed)
        training_observed_count = len(modeling_observed_data(self.config, observed))
        pending_count = len(pending)
        initial_design_remaining = max(
            self.config.bo.initial_design_size - training_observed_count,
            0,
        )
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
        ]
        self._extend_structured_summary_rows(rows)
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
        if self.config.review.enabled:
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
        if pending_count > 0:
            return "has_pending_suggestions"
        if observed_count < self.config.bo.initial_design_size:
            return "ready_for_initial_design"
        return "ready_for_bo"

    def next_action(self) -> pd.DataFrame:
        """Return the recommended next notebook action without mutating state."""
        campaign_status = self.campaign_status()
        structured_stage_arg = ""
        if self.config.is_structured_campaign:
            if len(self.config.stage_names) == 1:
                structured_stage_arg = f"stage={self.config.stage_names[0]!r}"
            else:
                structured_stage_arg = "stage='STAGE_NAME'"
        if campaign_status == "has_pending_suggestions":
            if self.config.review.enabled and not self.review_queue().empty:
                action = "review_pending_suggestions"
                reason = (
                    "There are suggestions awaiting review; accept, reject, or defer "
                    "them before requesting more."
                )
                suggested_call = (
                    "campaign.review_queue(); "
                    "campaign.review_suggestion(row_id, decision, note='')"
                )
            elif self.config.review.enabled:
                action = "run_accepted_suggestions"
                reason = "There are accepted suggestions awaiting experimental results."
                if self.config.is_multi_objective:
                    observed_call = (
                        "campaign.mark_observed(row_id, objective_values={...}, actual_cost=...)"
                        if self.config.cost is not None
                        else "campaign.mark_observed(row_id, objective_values={...})"
                    )
                    suggested_call = (
                        "campaign.pending_suggestions(); "
                        f"{observed_call}"
                    )
                else:
                    suggested_call = (
                        "campaign.pending_suggestions(); "
                        "campaign.mark_observed(row_id, objective_value, actual_cost=...)"
                    )
            else:
                action = "resolve_pending_suggestions"
                reason = (
                    "There are unresolved suggested rows; record results before "
                    "requesting more."
                )
                if self.config.is_multi_objective:
                    observed_call = (
                        "campaign.mark_observed(row_id, objective_values={...}, actual_cost=...)"
                        if self.config.cost is not None
                        else "campaign.mark_observed(row_id, objective_values={...})"
                    )
                    suggested_call = (
                        "campaign.pending_suggestions(); "
                        f"{observed_call}"
                    )
                else:
                    suggested_call = (
                        "campaign.pending_suggestions(); "
                        "campaign.mark_observed(row_id, objective_value)"
                    )
        elif campaign_status == "ready_for_initial_design":
            action = "suggest_initial_design"
            reason = "Observed rows are below initial_design_size; request Sobol suggestions."
            if structured_stage_arg:
                suggested_call = (
                    f"suggestions = campaign.suggest_next({structured_stage_arg}); "
                    "campaign.append_suggestions(suggestions)"
                )
            else:
                suggested_call = (
                    "suggestions = campaign.suggest_next(); "
                    "campaign.append_suggestions(suggestions)"
                )
        else:
            action = "suggest_bo"
            reason = "Initial design is complete and no pending suggestions remain."
            if structured_stage_arg:
                suggested_call = (
                    "suggestions = campaign.suggest_next("
                    f"batch_size=..., {structured_stage_arg}); "
                    "campaign.append_suggestions(suggestions)"
                )
            else:
                suggested_call = (
                    "suggestions = campaign.suggest_next(batch_size=...); "
                    "campaign.append_suggestions(suggestions)"
                )

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
        if self.config.is_multi_objective:
            tables = {
                "summary": self.summary(),
                "next_action": self.next_action(),
                "pareto_summary": self.pareto_summary(),
                "pareto_front": self.pareto_front(),
                "pending_suggestions": self.pending_suggestions(),
            }
            if self.config.review.enabled:
                tables["review_queue"] = self.review_queue()
            if self.config.replicates.enabled:
                tables["replicate_summary"] = self.replicate_summary()
            if self.config.cost is not None:
                tables["cost_summary"] = self.cost_summary()
            if self.config.is_structured_campaign:
                tables["stage_summary"] = self.stage_summary()
            return tables
        tables = {
            "summary": self.summary(),
            "next_action": self.next_action(),
            "best_observation": self.best_observation(),
            "best_replicate_group": self.best_replicate_group(),
            "replicate_summary": self.replicate_summary(),
            "pending_suggestions": self.pending_suggestions(),
            "review_queue": self.review_queue(),
            "cost_summary": self.cost_summary(),
        }
        if self.config.is_structured_campaign:
            tables["stage_summary"] = self.stage_summary()
        return tables

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
        return _pareto_front(self.config, self.df)

    def pareto_summary(self) -> pd.DataFrame:
        """Return Pareto-front and hypervolume summary fields."""
        return _pareto_summary(self.config, self.df)

    def stage_summary(self) -> pd.DataFrame:
        """Return structured-campaign stage summary rows."""
        return _stage_summary(self.config, self.df)

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
    ) -> pd.DataFrame:
        """Return suggested candidates without mutating session state or writing to disk."""
        return _suggest_next(
            self.config,
            self.df.copy(deep=True),
            batch_size=batch_size,
            stage=stage,
        )

    def suggestion_quality(self, suggestions: pd.DataFrame) -> pd.DataFrame:
        """Return read-only quality diagnostics for suggested rows."""
        return suggestion_quality_summary(
            self.config,
            self.df.copy(deep=True),
            suggestions.copy(deep=True),
        )

    def append_suggestions(self, suggestions: pd.DataFrame) -> pd.DataFrame:
        """Append suggestions to disk, reload the session, and return the refreshed log."""
        _append_suggestions(self.log_path, suggestions, config=self.config)
        return self.reload()

    def mark_observed(
        self,
        row_id: str,
        objective_value: float | None = None,
        objective_values: dict[str, float] | None = None,
        actual_cost: float | None = None,
    ) -> pd.DataFrame:
        """Mark one pending suggestion observed, reload, and return the refreshed log."""
        _mark_observed(
            self.log_path,
            row_id,
            objective_value=objective_value,
            objective_values=objective_values,
            actual_cost=actual_cost,
            config=self.config,
        )
        return self.reload()

    def review_suggestion(
        self,
        row_id: str,
        decision: str,
        note: str = "",
    ) -> pd.DataFrame:
        """Review one pending suggestion, reload, and return the refreshed log."""
        _review_suggestion(self.log_path, row_id, decision, note, config=self.config)
        return self.reload()

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


def _format_report_table(df: pd.DataFrame, empty_message: str) -> str:
    if df.empty:
        return empty_message
    return df.to_string(index=False)


def _format_campaign_report(tables: dict[str, pd.DataFrame]) -> str:
    if "pareto_front" in tables:
        sections = [
            "BO Forge Campaign Report\n========================",
            "Summary\n-------\n\n" + tables["summary"].to_string(index=False),
            "Next Action\n-----------\n\n" + _format_next_action(tables["next_action"]),
            "Pareto Summary\n--------------\n\n"
            + _format_report_table(tables["pareto_summary"], "No Pareto summary available."),
            "Pareto Front\n------------\n\n"
            + _format_report_table(tables["pareto_front"], "No Pareto observations yet."),
        ]
        if "replicate_summary" in tables:
            sections.append(
                "Replicate Summary\n-----------------\n\n"
                + _format_report_table(
                    tables["replicate_summary"],
                    "No replicate groups observed.",
                )
            )
        if "cost_summary" in tables:
            sections.append(
                "Cost Summary\n------------\n\n"
                + _format_report_table(tables["cost_summary"], "No cost model configured.")
            )
        if "stage_summary" in tables:
            sections.append(
                "Stage Summary\n-------------\n\n"
                + _format_report_table(tables["stage_summary"], "No structured stages configured.")
            )
        sections.append(
            "Pending Suggestions\n-------------------\n\n"
            + _format_report_table(tables["pending_suggestions"], "No pending suggestions.")
        )
        if "review_queue" in tables:
            sections.append(
                "Review Queue\n------------\n\n"
                + _format_report_table(
                    tables["review_queue"],
                    "No suggestions awaiting review.",
                )
            )
        return "\n\n".join(sections)
    return "\n\n".join(
        [
            "BO Forge Campaign Report\n========================",
            "Summary\n-------\n\n" + tables["summary"].to_string(index=False),
            "Next Action\n-----------\n\n" + _format_next_action(tables["next_action"]),
            "Best Raw Observation\n--------------------\n\n"
            + _format_best_observation(tables["best_observation"]),
            "Best Replicate Group By Mean Objective\n--------------------------------------\n\n"
            + _format_best_observation(
                tables["best_replicate_group"],
                empty_message="No replicate groups observed.",
            ),
            "Replicate Summary\n-----------------\n\n"
            + _format_report_table(tables["replicate_summary"], "No replicate groups observed."),
            "Pending Suggestions\n-------------------\n\n"
            + _format_report_table(tables["pending_suggestions"], "No pending suggestions."),
            "Review Queue\n------------\n\n"
            + _format_report_table(tables["review_queue"], "No suggestions awaiting review."),
            "Cost Summary\n------------\n\n"
            + _format_report_table(tables["cost_summary"], "No cost model configured."),
            *(
                [
                    "Stage Summary\n-------------\n\n"
                    + _format_report_table(
                        tables["stage_summary"],
                        "No structured stages configured.",
                    )
                ]
                if "stage_summary" in tables
                else []
            ),
        ]
    )


def _format_next_action(df: pd.DataFrame) -> str:
    if df.empty:
        return "No next action available."

    row = df.iloc[0]
    suggested_calls = [
        call.strip() for call in str(row["suggested_call"]).split(";") if call.strip()
    ]
    lines = [
        f"Campaign status: {_format_report_value(row['campaign_status'])}",
        f"Action: {_format_report_value(row['action'])}",
        "Reason:",
        f"  {_format_report_value(row['reason'])}",
        "Suggested call:",
    ]
    lines.extend(f"  {call}" for call in suggested_calls)
    return "\n".join(lines)


def _format_best_observation(
    df: pd.DataFrame,
    empty_message: str = "No best observation yet.",
) -> str:
    if df.empty:
        return empty_message

    row = df.iloc[0]
    return "\n".join(f"{column}: {_format_report_value(row[column])}" for column in df.columns)


def _format_report_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value)
