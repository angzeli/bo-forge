"""Notebook-oriented campaign session wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from bo_forge.config import CampaignConfig
from bo_forge.diagnostics import plot_diagnostics as _plot_diagnostics
from bo_forge.diagnostics import plot_progress as _plot_progress
from bo_forge.logs import (
    append_suggestions as _append_suggestions,
)
from bo_forge.logs import (
    load_campaign_log,
)
from bo_forge.logs import (
    mark_observed as _mark_observed,
)
from bo_forge.suggestions import suggest_next as _suggest_next
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
        observed = self.observed_data()
        pending = self.pending_suggestions()
        observed_count = len(observed)
        pending_count = len(pending)
        initial_design_remaining = max(self.config.bo.initial_design_size - observed_count, 0)
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
        return pd.DataFrame(rows, columns=["field", "value"])

    def observed_data(self) -> pd.DataFrame:
        """Return observed rows from the current session DataFrame."""
        return get_observed_data(self.config, self.df)

    def pending_suggestions(self) -> pd.DataFrame:
        """Return unresolved suggestions from the current session DataFrame."""
        self.validate()
        return self.df.loc[self.df["status"] == "suggested"].copy()

    def campaign_status(self) -> str:
        """Return the current campaign status without mutating session state."""
        self.validate()
        pending_count = int((self.df["status"] == "suggested").sum())
        observed_count = int((self.df["status"] == "observed").sum())
        if pending_count > 0:
            return "has_pending_suggestions"
        if observed_count < self.config.bo.initial_design_size:
            return "ready_for_initial_design"
        return "ready_for_bo"

    def next_action(self) -> pd.DataFrame:
        """Return the recommended next notebook action without mutating state."""
        campaign_status = self.campaign_status()
        if campaign_status == "has_pending_suggestions":
            action = "resolve_pending_suggestions"
            reason = "There are unresolved suggested rows; record results before requesting more."
            suggested_call = (
                "campaign.pending_suggestions(); "
                "campaign.mark_observed(row_id, objective_value)"
            )
        elif campaign_status == "ready_for_initial_design":
            action = "suggest_initial_design"
            reason = "Observed rows are below initial_design_size; request Sobol suggestions."
            suggested_call = (
                "suggestions = campaign.suggest_next(); "
                "campaign.append_suggestions(suggestions)"
            )
        else:
            action = "suggest_bo"
            reason = "Initial design is complete and no pending suggestions remain."
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
        return {
            "summary": self.summary(),
            "next_action": self.next_action(),
            "best_observation": self.best_observation(),
            "pending_suggestions": self.pending_suggestions(),
        }

    def export_report(self, path: str | Path) -> Path:
        """Write a deterministic plain-text campaign report and return its path."""
        report_path = Path(path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        tables = self.report()
        text = "\n\n".join(
            [
                "BO Forge Campaign Report\n========================",
                "Summary\n-------\n\n" + tables["summary"].to_string(index=False),
                "Next Action\n-----------\n\n" + _format_next_action(tables["next_action"]),
                "Best Observation\n----------------\n\n" + _format_best_observation(
                    tables["best_observation"]
                ),
                "Pending Suggestions\n-------------------\n\n"
                + _format_report_table(tables["pending_suggestions"], "No pending suggestions."),
            ]
        )
        report_path.write_text(text + "\n", encoding="utf-8")
        return report_path

    def best_observation(self) -> pd.DataFrame:
        """Return the best observed row as a canonical-order copy."""
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

    def suggest_next(self, batch_size: int | None = None) -> pd.DataFrame:
        """Return suggested candidates without mutating session state or writing to disk."""
        return _suggest_next(self.config, self.df.copy(deep=True), batch_size=batch_size)

    def append_suggestions(self, suggestions: pd.DataFrame) -> pd.DataFrame:
        """Append suggestions to disk, reload the session, and return the refreshed log."""
        _append_suggestions(self.log_path, suggestions)
        return self.reload()

    def mark_observed(self, row_id: str, objective_value: float) -> pd.DataFrame:
        """Mark one pending suggestion observed, reload, and return the refreshed log."""
        _mark_observed(self.log_path, row_id, objective_value)
        return self.reload()

    def plot_progress(self, **kwargs: Any) -> Any:
        """Plot campaign progress and return figure/axes objects."""
        return _plot_progress(self.config, self.df, **kwargs)

    def plot_diagnostics(self, **kwargs: Any) -> Any:
        """Plot campaign diagnostics and return figure/axes objects."""
        return _plot_diagnostics(self.config, self.df, **kwargs)


def _format_report_table(df: pd.DataFrame, empty_message: str) -> str:
    if df.empty:
        return empty_message
    return df.to_string(index=False)


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


def _format_best_observation(df: pd.DataFrame) -> str:
    if df.empty:
        return "No best observation yet."

    row = df.iloc[0]
    return "\n".join(f"{column}: {_format_report_value(row[column])}" for column in df.columns)


def _format_report_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value)
