"""Registration and handlers for campaign inspection and report export."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from bo_forge._cli.common import _add_config_log_arguments, _load_session
from bo_forge._cli.errors import _CLIOutputError
from bo_forge._cli.output import emit
from bo_forge.errors import ConfigError


def _register_read_commands(subparsers: argparse._SubParsersAction) -> None:
    """Register read-only campaign inspection commands."""
    validate_parser = subparsers.add_parser("validate", help="Validate a campaign CSV log.")
    _add_config_log_arguments(validate_parser)
    validate_parser.set_defaults(handler=_cmd_validate)

    summary_parser = subparsers.add_parser("summary", help="Print campaign summary.")
    _add_config_log_arguments(summary_parser)
    summary_parser.set_defaults(handler=_cmd_summary)

    status_parser = subparsers.add_parser("status", help="Print campaign status.")
    _add_config_log_arguments(status_parser)
    status_parser.set_defaults(handler=_cmd_status)

    next_action_parser = subparsers.add_parser(
        "next-action",
        help="Print the recommended next campaign action.",
    )
    _add_config_log_arguments(next_action_parser)
    next_action_parser.set_defaults(handler=_cmd_next_action)

    cost_summary_parser = subparsers.add_parser(
        "cost-summary",
        help="Print campaign cost and budget summary.",
    )
    _add_config_log_arguments(cost_summary_parser)
    cost_summary_parser.set_defaults(handler=_cmd_cost_summary)

    replicate_summary_parser = subparsers.add_parser(
        "replicate-summary",
        help="Print observed replicate-group summary statistics.",
    )
    _add_config_log_arguments(replicate_summary_parser)
    replicate_summary_parser.set_defaults(handler=_cmd_replicate_summary)

    stage_summary_parser = subparsers.add_parser(
        "stage-summary",
        help="Print structured-campaign stage summary fields.",
    )
    _add_config_log_arguments(stage_summary_parser)
    stage_summary_parser.set_defaults(handler=_cmd_stage_summary)

    fidelity_summary_parser = subparsers.add_parser(
        "fidelity-summary",
        help="Print multi-fidelity campaign summary fields.",
    )
    _add_config_log_arguments(fidelity_summary_parser)
    fidelity_summary_parser.set_defaults(handler=_cmd_fidelity_summary)

    fidelity_coverage_parser = subparsers.add_parser(
        "fidelity-coverage",
        help="Print observed and active-suggestion coverage by fidelity value.",
    )
    _add_config_log_arguments(fidelity_coverage_parser)
    fidelity_coverage_parser.set_defaults(handler=_cmd_fidelity_coverage)

    context_summary_parser = subparsers.add_parser(
        "context-summary",
        help="Print contextual-campaign summary rows by context combination.",
    )
    _add_config_log_arguments(context_summary_parser)
    context_summary_parser.set_defaults(handler=_cmd_context_summary)

    qlog_nei_summary_parser = subparsers.add_parser(
        "qlog-nei-summary",
        help="Print qLogNEI pending-state summary fields.",
    )
    _add_config_log_arguments(qlog_nei_summary_parser)
    qlog_nei_summary_parser.set_defaults(handler=_cmd_qlog_nei_summary)

    model_summary_parser = subparsers.add_parser(
        "model-summary",
        help="Print model profile and fitting-input summary fields.",
    )
    _add_config_log_arguments(model_summary_parser)
    model_summary_parser.set_defaults(handler=_cmd_model_summary)

    model_compare_parser = subparsers.add_parser(
        "model-compare",
        help="Compare model profiles on current observed fitting rows.",
    )
    _add_config_log_arguments(model_compare_parser)
    model_compare_parser.add_argument(
        "--profile",
        action="append",
        choices=["default", "smooth", "rough", "robust"],
        help="Model profile to compare; repeat to compare a subset.",
    )
    model_compare_parser.set_defaults(handler=_cmd_model_compare)

    pareto_front_parser = subparsers.add_parser(
        "pareto-front",
        help="Print nondominated observed rows for a multi-objective campaign.",
    )
    _add_config_log_arguments(pareto_front_parser)
    pareto_front_parser.set_defaults(handler=_cmd_pareto_front)

    pareto_summary_parser = subparsers.add_parser(
        "pareto-summary",
        help="Print Pareto-front and hypervolume summary fields.",
    )
    _add_config_log_arguments(pareto_summary_parser)
    pareto_summary_parser.set_defaults(handler=_cmd_pareto_summary)

    report_parser = subparsers.add_parser("report", help="Print or export a campaign report.")
    _add_config_log_arguments(report_parser)
    report_parser.add_argument("--output", type=Path, help="Optional report output path.")
    report_parser.set_defaults(handler=_cmd_report)


def _cmd_validate(args: argparse.Namespace) -> int:
    campaign = _load_session(args)
    campaign.validate()
    emit(args, {"valid": True}, text="Campaign log is valid.")
    return 0


def _cmd_summary(args: argparse.Namespace) -> int:
    campaign = _load_session(args)
    emit(args, campaign.summary())
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    campaign = _load_session(args)
    status = campaign.campaign_status()
    emit(args, {"status": status}, text=status)
    return 0


def _cmd_next_action(args: argparse.Namespace) -> int:
    campaign = _load_session(args)
    emit(args, campaign.next_action())
    return 0


def _cmd_cost_summary(args: argparse.Namespace) -> int:
    campaign = _load_session(args)
    emit(args, campaign.cost_summary())
    return 0


def _cmd_replicate_summary(args: argparse.Namespace) -> int:
    campaign = _load_session(args)
    emit(args, campaign.replicate_summary())
    return 0


def _cmd_stage_summary(args: argparse.Namespace) -> int:
    campaign = _load_session(args)
    if not campaign.config.is_structured_campaign:
        raise ConfigError("stage-summary requires a structured campaign config.")
    emit(args, campaign.stage_summary())
    return 0


def _cmd_fidelity_summary(args: argparse.Namespace) -> int:
    campaign = _load_session(args)
    if campaign.config.fidelity is None:
        raise ConfigError("fidelity-summary requires a multi-fidelity config.")
    emit(args, campaign.fidelity_summary())
    return 0


def _cmd_fidelity_coverage(args: argparse.Namespace) -> int:
    campaign = _load_session(args)
    if campaign.config.fidelity is None:
        raise ConfigError("fidelity-coverage requires a multi-fidelity config.")
    coverage = campaign.fidelity_coverage()
    emit(args, coverage, text=coverage.where(pd.notna(coverage), "").to_string(index=False))
    return 0


def _cmd_context_summary(args: argparse.Namespace) -> int:
    campaign = _load_session(args)
    if campaign.config.context is None:
        raise ConfigError("context-summary requires a contextual config.")
    emit(args, campaign.context_summary())
    return 0


def _cmd_qlog_nei_summary(args: argparse.Namespace) -> int:
    campaign = _load_session(args)
    if campaign.config.bo.acquisition != "qlog_nei":
        raise ConfigError("qlog-nei-summary requires bo.acquisition: qlog_nei.")
    emit(args, campaign.qlog_nei_summary())
    return 0


def _cmd_model_summary(args: argparse.Namespace) -> int:
    campaign = _load_session(args)
    emit(args, campaign.model_summary())
    return 0


def _cmd_model_compare(args: argparse.Namespace) -> int:
    campaign = _load_session(args)
    emit(args, campaign.model_profile_comparison(profiles=args.profile))
    return 0


def _cmd_pareto_front(args: argparse.Namespace) -> int:
    campaign = _load_session(args)
    if not campaign.config.is_multi_objective:
        raise ConfigError("pareto-front requires a multi-objective config.")
    emit(args, campaign.pareto_front())
    return 0


def _cmd_pareto_summary(args: argparse.Namespace) -> int:
    campaign = _load_session(args)
    if not campaign.config.is_multi_objective:
        raise ConfigError("pareto-summary requires a multi-objective config.")
    emit(args, campaign.pareto_summary())
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    campaign = _load_session(args)
    if args.output is None:
        from bo_forge.session import _format_campaign_report

        print(_format_campaign_report(campaign.report()))
    else:
        try:
            report_path = campaign.export_report(args.output)
        except OSError as exc:
            raise _CLIOutputError(
                f"Could not write campaign report '{args.output}': {exc}"
            ) from exc
        print(f"Wrote campaign report: {report_path}")
    return 0
