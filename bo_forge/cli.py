"""Command-line interface for BO Forge campaign workflows."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.util
import io
import os
import platform
import sys
from collections.abc import Sequence
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd

from bo_forge import __version__
from bo_forge.config import CampaignConfig
from bo_forge.errors import (
    BOForgeError,
    ConfigError,
    LogValidationError,
    LogWriteError,
    SuggestionError,
)
from bo_forge.io import empty_campaign_log
from bo_forge.logs import load_campaign_log
from bo_forge.session import CampaignSession, _format_campaign_report


class _CLIOutputError(BOForgeError):
    """Raised when a CLI-owned output file cannot be written."""


class _CLIDoctorError(BOForgeError):
    """Raised when an expected doctor check fails."""


def build_parser() -> argparse.ArgumentParser:
    """Build the BO Forge CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="bo-forge",
        description="Run BO Forge campaign workflows from the terminal.",
    )
    parser.add_argument("--version", action="version", version=f"bo-forge {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor_parser = subparsers.add_parser("doctor", help="Check the active BO Forge environment.")
    doctor_parser.set_defaults(handler=_cmd_doctor)

    init_log_parser = subparsers.add_parser(
        "init-log",
        help="Create an empty canonical campaign CSV log from a config.",
    )
    _add_config_log_arguments(init_log_parser)
    init_log_parser.set_defaults(handler=_cmd_init_log)

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
        hint = _hint_for_error(exc)
        if hint is not None:
            print(hint, file=sys.stderr)
        return 1


def main(argv: Sequence[str] | None = None) -> None:
    """Console-script entrypoint."""
    raise SystemExit(run(argv))


def _add_config_log_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, type=Path, help="Campaign YAML config path.")
    parser.add_argument("--log", required=True, type=Path, help="Campaign CSV log path.")


def _load_session(args: argparse.Namespace) -> CampaignSession:
    return CampaignSession.from_files(args.config, args.log)


def _cmd_doctor(args: argparse.Namespace) -> int:
    lines = [
        "BO Forge doctor",
        f"BO Forge version: {__version__}",
        f"Python executable: {sys.executable}",
        f"Python version: {platform.python_version()}",
    ]
    for module_name in [
        "torch",
        "botorch",
        "gpytorch",
        "pandas",
        "yaml",
        "matplotlib",
        "bo_forge",
    ]:
        _doctor_import(module_name)
        lines.append(f"{module_name}: OK")

    if importlib.util.find_spec("bo_forge.__main__") is None:
        raise _CLIDoctorError("Module entrypoint 'bo_forge.__main__' is not available.")
    lines.append("module entrypoint: OK")
    lines.append("Status: OK")
    print("\n".join(lines))
    return 0


def _cmd_init_log(args: argparse.Namespace) -> int:
    config = CampaignConfig.from_yaml(args.config)
    log = empty_campaign_log(config)
    log_path = _write_empty_log(log, args.log)
    load_campaign_log(log_path, config)
    print(f"Created empty campaign log: {log_path}")
    return 0


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
        try:
            report_path = campaign.export_report(args.output)
        except OSError as exc:
            raise _CLIOutputError(
                f"Could not write campaign report '{args.output}': {exc}"
            ) from exc
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
    try:
        if args.kind == "progress":
            campaign.plot_progress(save_path=args.output)
        else:
            campaign.plot_diagnostics(save_path=args.output)
    except OSError as exc:
        raise _CLIOutputError(
            f"Could not write {args.kind} plot '{args.output}': {exc}"
        ) from exc
    print(f"Wrote {args.kind} plot: {args.output}")
    return 0


def _print_table(df: pd.DataFrame) -> None:
    print(df.to_string(index=False))


def _write_csv(df: pd.DataFrame, path: Path) -> Path:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    except OSError as exc:
        raise _CLIOutputError(f"Could not write suggestions CSV '{path}': {exc}") from exc
    return path


def _write_empty_log(df: pd.DataFrame, path: Path) -> Path:
    temp_path: Path | None = None
    try:
        if path.exists():
            raise _CLIOutputError(
                f"Cannot create empty campaign log '{path}' because file already exists."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            df.to_csv(temp_file, index=False)
        os.link(temp_path, path)
    except _CLIOutputError:
        raise
    except OSError as exc:
        raise _CLIOutputError(f"Could not write empty campaign log '{path}': {exc}") from exc
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    return path


def _doctor_import(module_name: str) -> None:
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            importlib.import_module(module_name)
    except ImportError as exc:
        raise _CLIDoctorError(
            f"Doctor check failed while importing '{module_name}': {exc}"
        ) from exc


def _hint_for_error(exc: BOForgeError) -> str | None:
    if isinstance(exc, _CLIOutputError):
        return None
    if isinstance(exc, ConfigError):
        return "Hint: Check the YAML config path and campaign settings."
    if isinstance(exc, LogValidationError):
        return "Hint: Check the CSV schema, statuses, objective values, and variable bounds."
    if isinstance(exc, SuggestionError):
        return (
            "Hint: Resolve pending suggestions or review the campaign state before "
            "requesting new suggestions."
        )
    if isinstance(exc, LogWriteError):
        return "Hint: Check the row_id, pending status, campaign log path, and file permissions."
    return None


if __name__ == "__main__":
    main()
