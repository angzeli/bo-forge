"""Campaign creation and mutation commands, including non-writing suggestion previews."""

from __future__ import annotations

import argparse
import math
import stat
import sys
from collections.abc import Sequence
from os.path import expanduser
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import pandas as pd

from bo_forge._campaign.exports import _validate_export_destination
from bo_forge._cli.common import _add_config_log_arguments, _load_session
from bo_forge._cli.errors import _CLIOutputError
from bo_forge._cli.output import _print_table, emit
from bo_forge.errors import LogWriteError, SuggestionError

if TYPE_CHECKING:
    from bo_forge.session import CampaignSession


def _register_init_log_command(subparsers: argparse._SubParsersAction) -> None:
    """Register initialization separately to retain its position after doctor."""
    init_log_parser = subparsers.add_parser(
        "init-log",
        help="Create an empty canonical campaign CSV log and provenance manifest.",
    )
    _add_config_log_arguments(init_log_parser, include_provenance_policy=False)
    init_log_parser.set_defaults(handler=_cmd_init_log)


def _register_mutation_commands(subparsers: argparse._SubParsersAction) -> None:
    """Register explicit campaign suggestion and mutation commands."""
    suggest_parser = subparsers.add_parser("suggest", help="Generate campaign suggestions.")
    _add_config_log_arguments(suggest_parser)
    suggest_parser.add_argument("--batch-size", type=int, help="Override configured batch size.")
    suggest_parser.add_argument(
        "--stage",
        help="Structured campaign stage name for stage-aware suggestions.",
    )
    suggest_parser.add_argument(
        "--context",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Context variable value for contextual campaigns; repeat as needed.",
    )
    suggest_parser.add_argument("--output", type=Path, help="Optional suggestions CSV output path.")
    suggest_parser.add_argument(
        "--append",
        action="store_true",
        help="Append generated suggestions to the canonical campaign log.",
    )
    suggest_parser.set_defaults(handler=_cmd_suggest)

    review_parser = subparsers.add_parser(
        "review",
        help="Accept, reject, or defer one suggested row.",
    )
    _add_config_log_arguments(review_parser)
    review_parser.add_argument("--row-id", required=True, help="Suggested row_id to review.")
    review_parser.add_argument(
        "--decision",
        choices=["accept", "reject", "defer"],
        required=True,
        help="Review decision.",
    )
    review_parser.add_argument("--note", default="", help="Optional one-line review note.")
    review_parser.set_defaults(handler=_cmd_review)

    mark_parser = subparsers.add_parser(
        "mark-observed",
        help="Mark one pending suggestion as observed.",
    )
    _add_config_log_arguments(mark_parser)
    mark_parser.add_argument("--row-id", required=True, help="Suggested row_id to mark observed.")
    mark_parser.add_argument(
        "--objective-value",
        type=float,
        help="Observed objective value in user-facing units.",
    )
    mark_parser.add_argument(
        "--objective",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Observed multi-objective value; repeat once per objective.",
    )
    mark_parser.add_argument(
        "--actual-cost",
        type=float,
        help="Optional observed experiment cost for cost-aware campaigns.",
    )
    mark_parser.set_defaults(handler=_cmd_mark_observed)


def _cmd_init_log(args: argparse.Namespace) -> int:
    from bo_forge._campaign.provenance import manifest_path_for_log
    from bo_forge.session import CampaignSession

    campaign = CampaignSession.initialize(args.config, args.log)
    print(f"Created empty campaign log: {campaign.log_path}")
    print(f"Created provenance manifest: {manifest_path_for_log(campaign.log_path)}")
    return 0


def _cmd_suggest(args: argparse.Namespace) -> int:
    campaign = _load_session(args)
    if args.output is not None:
        try:
            _validate_export_destination(
                expanduser(args.output), campaign.config_path, campaign.log_path,
            )
        except OSError as exc:
            raise _CLIOutputError(str(exc)) from exc
    context_values = _parse_context_values(args.context)
    suggestions = campaign.suggest_next(
        batch_size=args.batch_size,
        stage=args.stage,
        context_values=context_values,
    )

    _print_contextual_replicate_fallback_note(campaign, suggestions)

    if getattr(args, "format", "text") == "json":
        emit(args, suggestions)
        return 0

    print(f"Generated {len(suggestions)} suggestion(s).")
    if args.output is None:
        _print_table(suggestions)
    else:
        output_path = _write_csv(suggestions, args.output, campaign)
        print(f"Wrote suggestions CSV: {output_path}")

    if args.append:
        campaign.append_suggestions(suggestions)
        print(f"Appended suggestions to campaign log: {args.log}")
    return 0


def _print_contextual_replicate_fallback_note(
    campaign: CampaignSession,
    suggestions: pd.DataFrame,
) -> None:
    config = campaign.config
    if (
        config.context is None
        or not config.replicates.enabled
        or config.replicates.suggestion_policy != "uncertain_best"
        or suggestions.empty
        or suggestions["source"].isin({"sobol", "random"}).any()
    ):
        return
    existing_groups = set(campaign.df["replicate_group"].astype(str))
    selected_repeat = suggestions["replicate_group"].astype(str).isin(existing_groups).any()
    if not selected_repeat:
        print(
            "Note: No active repeat was selected for the requested context; "
            "BO Forge returned context-fixed exploration suggestions.",
            file=sys.stderr,
        )


def _parse_context_values(items: Sequence[str]) -> dict[str, str] | None:
    if not items:
        return None
    parsed: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SuggestionError(
                f"Malformed --context value '{item}'. Expected NAME=VALUE."
            )
        name, raw_value = item.split("=", 1)
        name = name.strip()
        if not name:
            raise SuggestionError(
                f"Malformed --context value '{item}'. Context variable name is blank."
            )
        if name in parsed:
            raise SuggestionError(f"Duplicate --context value for variable '{name}'.")
        parsed[name] = raw_value
    return parsed


def _cmd_review(args: argparse.Namespace) -> int:
    campaign = _load_session(args)
    campaign.review_suggestion(args.row_id, args.decision, args.note)
    print(
        f"Reviewed row {args.row_id} as {args.decision} in campaign log: {args.log}"
    )
    return 0


def _cmd_mark_observed(args: argparse.Namespace) -> int:
    campaign = _load_session(args)
    objective_values = _parse_cli_objective_values(args.objective)
    if args.objective_value is not None and objective_values:
        raise LogWriteError("Pass either --objective-value or --objective, not both.")
    if campaign.config.is_multi_objective:
        if args.objective_value is not None:
            raise LogWriteError(
                "--objective-value is not valid for multi-objective campaigns; "
                "use repeated --objective name=value arguments."
            )
        if args.actual_cost is not None and campaign.config.cost is None:
            raise LogWriteError("--actual-cost requires a config with a cost section.")
        campaign.mark_observed(
            args.row_id,
            objective_values=objective_values,
            actual_cost=args.actual_cost,
        )
    else:
        if objective_values:
            raise LogWriteError(
                "--objective is not valid for single-objective campaigns; use --objective-value."
            )
        campaign.mark_observed(
            args.row_id,
            objective_value=args.objective_value,
            actual_cost=args.actual_cost,
        )
    print(f"Marked row {args.row_id} as observed in campaign log: {args.log}")
    return 0


def _write_csv(df: pd.DataFrame, path: Path, campaign: CampaignSession) -> Path:
    try:
        output_path = Path(expanduser(path))
        _validate_export_destination(output_path, campaign.config_path, campaign.log_path)
        destination = output_path.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(
            dir=destination.parent, prefix=".suggestions-", ignore_cleanup_errors=True,
        ) as temporary:
            # Retain the requested basename for compression inference and ZIP members.
            prepared = Path(temporary) / output_path.name
            df.to_csv(prepared, index=False)
            _validate_export_destination(destination, campaign.config_path, campaign.log_path)
            if destination.exists():
                prepared.chmod(stat.S_IMODE(destination.stat().st_mode))
            prepared.replace(destination)
    except OSError as exc:
        raise _CLIOutputError(f"Could not write suggestions CSV '{path}': {exc}") from exc
    return path


def _parse_cli_objective_values(values: list[str]) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for item in values:
        if "=" not in item:
            raise LogWriteError(
                f"Malformed --objective value '{item}'. Expected NAME=VALUE."
            )
        name, raw_value = item.split("=", 1)
        name = name.strip()
        if not name:
            raise LogWriteError(
                f"Malformed --objective value '{item}'. Objective name is blank."
            )
        if name in parsed:
            raise LogWriteError(f"Duplicate --objective value for objective '{name}'.")
        try:
            parsed_value = float(raw_value)
        except ValueError as exc:
            raise LogWriteError(
                f"Objective value for '{name}' must be numeric: value={raw_value!r}."
            ) from exc
        if not math.isfinite(parsed_value):
            raise LogWriteError(
                f"Objective value for '{name}' must be finite: value={raw_value!r}."
            )
        parsed[name] = parsed_value
    return parsed
