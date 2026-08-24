"""Read-only campaign report assembly and next-action formatting."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from bo_forge.config import CampaignConfig

if TYPE_CHECKING:
    from bo_forge.session import CampaignSession

def _base_report_tables(session: CampaignSession) -> dict[str, pd.DataFrame]:
    if session.config.is_multi_objective:
        return {
            "summary": session.summary(),
            "next_action": session.next_action(),
            "model_summary": session.model_summary(),
            "pareto_summary": session.pareto_summary(),
            "pareto_front": session.pareto_front(),
            "pending_suggestions": session.pending_suggestions(),
        }
    return {
        "summary": session.summary(),
        "next_action": session.next_action(),
        "model_summary": session.model_summary(),
        "best_observation": session.best_observation(),
        "best_replicate_group": session.best_replicate_group(),
        "replicate_summary": session.replicate_summary(),
        "pending_suggestions": session.pending_suggestions(),
        "review_queue": session.review_queue(),
        "cost_summary": session.cost_summary(),
    }


def _optional_report_readers(session: CampaignSession) -> list[tuple[str, Any]]:
    config = session.config
    if config.is_multi_objective:
        options = [
            (config.review.enabled, "review_queue", session.review_queue),
            (config.replicates.enabled, "replicate_summary", session.replicate_summary),
            (config.cost is not None, "cost_summary", session.cost_summary),
            (config.is_structured_campaign, "stage_summary", session.stage_summary),
            (config.context is not None, "context_summary", session.context_summary),
            (
                config.bo.acquisition == "qlog_nei",
                "qlog_nei_summary",
                session.qlog_nei_summary,
            ),
        ]
    else:
        options = [
            (config.is_structured_campaign, "stage_summary", session.stage_summary),
            (config.fidelity is not None, "fidelity_summary", session.fidelity_summary),
            (config.fidelity is not None, "fidelity_coverage", session.fidelity_coverage),
            (config.context is not None, "context_summary", session.context_summary),
            (
                config.bo.acquisition == "qlog_nei",
                "qlog_nei_summary",
                session.qlog_nei_summary,
            ),
        ]
    return [(name, reader) for enabled, name, reader in options if enabled]


def _pending_next_action(session: CampaignSession) -> tuple[str, str, str]:
    if session.config.review.enabled and not session.review_queue().empty:
        return (
            "review_pending_suggestions",
            "There are suggestions awaiting review; accept, reject, or defer them before "
            "requesting more.",
            "campaign.review_queue(); "
            "campaign.review_suggestion(row_id, decision, note='')",
        )
    if session.config.review.enabled:
        return (
            "run_accepted_suggestions",
            "There are accepted suggestions awaiting experimental results.",
            _pending_observation_call(session.config, reviewed=True),
        )
    return (
        "resolve_pending_suggestions",
        "There are unresolved suggested rows; record results before requesting more.",
        _pending_observation_call(session.config, reviewed=False),
    )


def _pending_observation_call(config: CampaignConfig, *, reviewed: bool) -> str:
    if config.is_multi_objective:
        observed_call = (
            "campaign.mark_observed(row_id, objective_values={...}, actual_cost=...)"
            if config.cost is not None
            else "campaign.mark_observed(row_id, objective_values={...})"
        )
    elif reviewed:
        observed_call = "campaign.mark_observed(row_id, objective_value, actual_cost=...)"
    else:
        observed_call = "campaign.mark_observed(row_id, objective_value)"
    return f"campaign.pending_suggestions(); {observed_call}"


def _suggest_and_append_call(
    config: CampaignConfig,
    *,
    include_batch_size: bool,
) -> str:
    args: list[str] = []
    if include_batch_size:
        args.append("batch_size=...")
    if config.is_structured_campaign:
        stage_arg = (
            f"stage={config.stage_names[0]!r}"
            if len(config.stage_names) == 1
            else "stage='STAGE_NAME'"
        )
        args.append(stage_arg)
    if config.context is not None:
        args.append("context_values={...}")
    call_args = ", ".join(args)
    suggest_call = (
        f"suggestions = campaign.suggest_next({call_args})"
        if call_args
        else "suggestions = campaign.suggest_next()"
    )
    return f"{suggest_call}; campaign.append_suggestions(suggestions)"


def _bo_suggestion_reason(session: CampaignSession) -> str:
    pending_aware_label = {
        "qlog_nei": "qLogNEI",
        "qlog_nehvi": "qLogNEHVI",
    }.get(session.config.bo.acquisition)
    if pending_aware_label is not None and not session.pending_suggestions().empty:
        return (
            f"Initial design is complete; {pending_aware_label} can account "
            "for accepted pending suggestions as X_pending."
        )
    return "Initial design is complete and no pending suggestions remain."


def _format_report_table(df: pd.DataFrame, empty_message: str) -> str:
    if df.empty:
        return empty_message
    return df.to_string(index=False)


def _format_campaign_report(tables: dict[str, pd.DataFrame]) -> str:
    if "pareto_front" in tables:
        return _format_multi_objective_report(tables)
    return _format_single_objective_report(tables)


def _format_multi_objective_report(tables: dict[str, pd.DataFrame]) -> str:
    sections = [
        "BO Forge Campaign Report\n========================",
        "Summary\n-------\n\n" + tables["summary"].to_string(index=False),
        "Next Action\n-----------\n\n" + _format_next_action(tables["next_action"]),
        "Model Summary\n-------------\n\n"
        + _format_report_table(tables["model_summary"], "No model summary available."),
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
    if "context_summary" in tables:
        sections.append(
            "Context Summary\n---------------\n\n"
            + _format_report_table(
                tables["context_summary"],
                "No contextual observations or pending suggestions yet.",
            )
        )
    if "qlog_nei_summary" in tables:
        sections.append(
            "qLogNEI Summary\n---------------\n\n"
            + _format_report_table(
                tables["qlog_nei_summary"],
                "No qLogNEI summary available.",
            )
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


def _format_single_objective_report(tables: dict[str, pd.DataFrame]) -> str:
    return "\n\n".join(
        [
            "BO Forge Campaign Report\n========================",
            "Summary\n-------\n\n" + tables["summary"].to_string(index=False),
            "Next Action\n-----------\n\n" + _format_next_action(tables["next_action"]),
            "Model Summary\n-------------\n\n"
            + _format_report_table(tables["model_summary"], "No model summary available."),
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
                    "Fidelity Summary\n----------------\n\n"
                    + _format_report_table(
                        tables["fidelity_summary"],
                        "No fidelity section configured.",
                    )
                ]
                if "fidelity_summary" in tables
                else []
            ),
            *(
                [
                    "Fidelity Coverage\n-----------------\n\n"
                    + _format_report_table(
                        tables["fidelity_coverage"],
                        "No observed or active fidelity values.",
                    )
                ]
                if "fidelity_coverage" in tables
                else []
            ),
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
            *(
                [
                    "Context Summary\n---------------\n\n"
                    + _format_report_table(
                        tables["context_summary"],
                        "No contextual observations or pending suggestions yet.",
                    )
                ]
                if "context_summary" in tables
                else []
            ),
            *(
                [
                    "qLogNEI Summary\n---------------\n\n"
                    + _format_report_table(
                        tables["qlog_nei_summary"],
                        "No qLogNEI summary available.",
                    )
                ]
                if "qlog_nei_summary" in tables
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
