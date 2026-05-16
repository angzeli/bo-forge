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
from bo_forge.validation import get_observed_data, next_iteration, validate_campaign_data


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
        best_row_id, best_objective_value = self._best_observation(observed)

        rows = [
            ("campaign_name", self.config.campaign_name),
            ("campaign_status", self._campaign_status(observed_count, pending_count)),
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

    def _campaign_status(self, observed_count: int, pending_count: int) -> str:
        if pending_count > 0:
            return "has_pending_suggestions"
        if observed_count < self.config.bo.initial_design_size:
            return "ready_for_initial_design"
        return "ready_for_bo"

    def _best_observation(self, observed: pd.DataFrame) -> tuple[str | None, float | None]:
        if observed.empty:
            return None, None

        objective = self.config.objective.name
        values = pd.to_numeric(observed[objective])
        if self.config.objective.direction == "maximize":
            best_index = values.idxmax()
        else:
            best_index = values.idxmin()
        return str(observed.loc[best_index, "row_id"]), float(values.loc[best_index])
