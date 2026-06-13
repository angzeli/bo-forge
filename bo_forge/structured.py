"""Read-only helpers for structured campaign inspection."""

from __future__ import annotations

import pandas as pd

from bo_forge.config import CampaignConfig
from bo_forge.multi_objective import pareto_front
from bo_forge.replicates import aggregate_observed_replicates
from bo_forge.validation import validate_campaign_data

STAGE_SUMMARY_COLUMNS = [
    "stage",
    "active_variables",
    "inactive_variables",
    "total_rows",
    "observed_rows",
    "suggested_rows",
    "pending_rows",
    "best_row_id",
    "best_objective_value",
    "pareto_count",
    "warning",
    "transition_readiness",
]


def stage_summary(config: CampaignConfig, df: pd.DataFrame) -> pd.DataFrame:
    """Return deterministic read-only structured-campaign stage summary rows."""
    if not config.is_structured_campaign:
        return pd.DataFrame(columns=STAGE_SUMMARY_COLUMNS)
    validate_campaign_data(config, df)

    rows: list[dict[str, object]] = []
    for stage in config.stages:
        stage_rows = df.loc[df["stage"] == stage.name]
        observed = stage_rows.loc[stage_rows["status"] == "observed"]
        suggested = stage_rows.loc[stage_rows["status"] == "suggested"]
        pending_rows = _pending_rows(config, suggested)
        best_row_id, best_value = _single_objective_best(config, observed)
        pareto_count = None
        if config.is_multi_objective:
            pareto_count = len(pareto_front(config, stage_rows)) if not observed.empty else 0
            best_row_id = None
            best_value = None

        warning = "" if not observed.empty else "No observed rows for stage."
        readiness = _transition_readiness(len(observed), pending_rows)
        inactive = [
            variable_name
            for variable_name in config.variable_names
            if variable_name not in set(stage.variables)
        ]
        rows.append(
            {
                "stage": stage.name,
                "active_variables": ", ".join(stage.variables),
                "inactive_variables": ", ".join(inactive),
                "total_rows": int(len(stage_rows)),
                "observed_rows": int(len(observed)),
                "suggested_rows": int(len(suggested)),
                "pending_rows": int(pending_rows),
                "best_row_id": best_row_id,
                "best_objective_value": best_value,
                "pareto_count": pareto_count,
                "warning": warning,
                "transition_readiness": readiness,
            }
        )
    return pd.DataFrame(rows, columns=STAGE_SUMMARY_COLUMNS)


def _pending_rows(config: CampaignConfig, suggested: pd.DataFrame) -> int:
    if suggested.empty:
        return 0
    if config.review.enabled:
        return int(suggested["review_status"].isin(["pending", "accepted"]).sum())
    return int(len(suggested))


def _single_objective_best(
    config: CampaignConfig,
    observed: pd.DataFrame,
) -> tuple[str | None, float | None]:
    if config.is_multi_objective or observed.empty:
        return None, None
    if config.replicates.enabled:
        return _single_objective_replicate_best(config, observed)
    objective = config.objective.name
    values = pd.to_numeric(observed[objective])
    index = values.idxmax() if config.objective.direction == "maximize" else values.idxmin()
    return str(observed.loc[index, "row_id"]), float(values.loc[index])


def _single_objective_replicate_best(
    config: CampaignConfig,
    observed: pd.DataFrame,
) -> tuple[str | None, float | None]:
    summary = aggregate_observed_replicates(config, observed)
    if summary.empty:
        return None, None
    values = pd.to_numeric(summary["objective_mean"])
    index = values.idxmax() if config.objective.direction == "maximize" else values.idxmin()
    return str(summary.loc[index, "replicate_group"]), float(values.loc[index])


def _transition_readiness(observed_rows: int, pending_rows: int) -> str:
    if pending_rows > 0:
        return "resolve_pending"
    if observed_rows == 0:
        return "needs_observations"
    return "ready_for_suggestions"
