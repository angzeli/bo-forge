"""Read-only helpers for noisy and pending-aware acquisitions."""

from __future__ import annotations

import pandas as pd

from bo_forge.config import CampaignConfig
from bo_forge.models import dataframe_to_training_tensors
from bo_forge.validation import (
    get_observed_data,
    has_blocking_qlog_nei_review_suggestions,
    qlog_nei_active_pending_suggestions,
    validate_campaign_data,
)


def qlog_nei_summary(config: CampaignConfig, df: pd.DataFrame) -> pd.DataFrame:
    """Return read-only pending-state summary fields for qLogNEI campaigns."""
    if config.bo.acquisition != "qlog_nei":
        raise ValueError("qlog_nei_summary() requires bo.acquisition: qlog_nei.")
    validate_campaign_data(config, df)

    observed = get_observed_data(config, df)
    if observed.empty:
        observed_baseline_rows = 0
        train_yvar_available = False
    else:
        training = dataframe_to_training_tensors(config, observed)
        observed_baseline_rows = int(training.train_x.shape[0])
        train_yvar_available = bool(training.train_yvar is not None)
    active_pending = qlog_nei_active_pending_suggestions(df, config)
    active_pending_initial_rows = (
        int(active_pending["source"].isin({"sobol", "random"}).sum())
        if not active_pending.empty
        else 0
    )
    blocking_review_pending_rows = _blocking_review_pending_count(config, df)
    rejected_or_deferred_pending_rows = _rejected_or_deferred_pending_count(config, df)
    initial_design_remaining = max(
        config.bo.initial_design_size
        - observed_baseline_rows
        - active_pending_initial_rows,
        0,
    )
    ready_for_qlog_nei = (
        observed_baseline_rows >= config.bo.initial_design_size
        and blocking_review_pending_rows == 0
    )

    rows = [
        ("campaign_name", config.campaign_name),
        ("observed_baseline_rows", observed_baseline_rows),
        ("active_pending_rows", int(len(active_pending))),
        ("active_pending_initial_rows", active_pending_initial_rows),
        ("blocking_review_pending_rows", blocking_review_pending_rows),
        ("rejected_or_deferred_pending_rows", rejected_or_deferred_pending_rows),
        ("initial_design_size", int(config.bo.initial_design_size)),
        ("initial_design_remaining", initial_design_remaining),
        ("ready_for_qlog_nei", bool(ready_for_qlog_nei)),
        ("x_pending_used", bool(ready_for_qlog_nei and not active_pending.empty)),
        ("train_yvar_available", train_yvar_available),
        ("model_profile", config.model.profile),
    ]
    return pd.DataFrame(rows, columns=["field", "value"])


def _blocking_review_pending_count(config: CampaignConfig, df: pd.DataFrame) -> int:
    if not config.review.enabled or not has_blocking_qlog_nei_review_suggestions(df, config):
        return 0
    return int(
        ((df["status"] == "suggested") & (df["review_status"] == "pending")).sum()
    )


def _rejected_or_deferred_pending_count(config: CampaignConfig, df: pd.DataFrame) -> int:
    if not config.review.enabled or df.empty:
        return 0
    return int(
        (
            (df["status"] == "suggested")
            & df["review_status"].isin({"rejected", "deferred"})
        ).sum()
    )
