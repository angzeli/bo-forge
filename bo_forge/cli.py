"""Command-line interface for BO Forge campaign workflows."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from bo_forge import __version__
from bo_forge.errors import BOForgeError
from bo_forge.session import CampaignSession, _format_campaign_report


def build_parser() -> argparse.ArgumentParser:
    """Build the BO Forge CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="bo-forge",
        description="Run BO Forge campaign workflows from the terminal.",
    )
    parser.add_argument("--version", action="version", version=f"bo-forge {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

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

    report_parser = subparsers.add_parser("report", help="Print or export a campaign report.")
    _add_config_log_arguments(report_parser)
    report_parser.add_argument("--output", type=Path, help="Optional report output path.")
    report_parser.set_defaults(handler=_cmd_report)

    suggest_parser = subparsers.add_parser("suggest", help="Generate campaign suggestions.")
    _add_config_log_arguments(suggest_parser)
    suggest_parser.add_argument("--batch-size", type=int, help="Override configured batch size.")
    suggest_parser.add_argument("--output", type=Path, help="Optional suggestions CSV output path.")
    suggest_parser.add_argument(
        "--append",
        action="store_true",
        help="Append generated suggestions to the canonical campaign log.",
    )
    suggest_parser.set_defaults(handler=_cmd_suggest)

    mark_parser = subparsers.add_parser(
        "mark-observed",
        help="Mark one pending suggestion as observed.",
    )
    _add_config_log_arguments(mark_parser)
    mark_parser.add_argument("--row-id", required=True, help="Suggested row_id to mark observed.")
    mark_parser.add_argument(
        "--objective-value",
        required=True,
        type=float,
        help="Observed objective value in user-facing units.",
    )
    mark_parser.set_defaults(handler=_cmd_mark_observed)

    plot_parser = subparsers.add_parser("plot", help="Export one campaign plot.")
    _add_config_log_arguments(plot_parser)
    plot_parser.add_argument(
        "--kind",
        choices=["progress", "diagnostics"],
        required=True,
        help="Plot type to export.",
    )
    plot_parser.add_argument("--output", type=Path, required=True, help="Figure output path.")
    plot_parser.set_defaults(handler=_cmd_plot)

    return parser


def run(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments, dispatch a command, and return an exit code."""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        if isinstance(exc.code, int):
            return exc.code
        return 1

    try:
        return int(args.handler(args))
    except BOForgeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def main(argv: Sequence[str] | None = None) -> None:
    """Console-script entrypoint."""
    raise SystemExit(run(argv))


def _add_config_log_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, type=Path, help="Campaign YAML config path.")
    parser.add_argument("--log", required=True, type=Path, help="Campaign CSV log path.")


def _load_session(args: argparse.Namespace) -> CampaignSession:
    return CampaignSession.from_files(args.config, args.log)


def _cmd_validate(args: argparse.Namespace) -> int:
    campaign = _load_session(args)
    campaign.validate()
    print("Campaign log is valid.")
    return 0


def _cmd_summary(args: argparse.Namespace) -> int:
    campaign = _load_session(args)
    _print_table(campaign.summary())
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    campaign = _load_session(args)
    print(campaign.campaign_status())
    return 0


def _cmd_next_action(args: argparse.Namespace) -> int:
    campaign = _load_session(args)
    _print_table(campaign.next_action())
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    campaign = _load_session(args)
    if args.output is None:
        print(_format_campaign_report(campaign.report()))
    else:
        report_path = campaign.export_report(args.output)
        print(f"Wrote campaign report: {report_path}")
    return 0


def _cmd_suggest(args: argparse.Namespace) -> int:
    campaign = _load_session(args)
    suggestions = campaign.suggest_next(batch_size=args.batch_size)

    print(f"Generated {len(suggestions)} suggestion(s).")
    if args.output is None:
        _print_table(suggestions)
    else:
        output_path = _write_csv(suggestions, args.output)
        print(f"Wrote suggestions CSV: {output_path}")

    if args.append:
        campaign.append_suggestions(suggestions)
        print(f"Appended suggestions to campaign log: {args.log}")
    return 0


def _cmd_mark_observed(args: argparse.Namespace) -> int:
    campaign = _load_session(args)
    campaign.mark_observed(args.row_id, args.objective_value)
    print(f"Marked row {args.row_id} as observed in campaign log: {args.log}")
    return 0


def _cmd_plot(args: argparse.Namespace) -> int:
    campaign = _load_session(args)
    if args.kind == "progress":
        campaign.plot_progress(save_path=args.output)
    else:
        campaign.plot_diagnostics(save_path=args.output)
    print(f"Wrote {args.kind} plot: {args.output}")
    return 0


def _print_table(df: pd.DataFrame) -> None:
    print(df.to_string(index=False))


def _write_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    main()
