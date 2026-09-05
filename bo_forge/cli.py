"""Command-line interface for BO Forge campaign workflows."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.util
import io
import math
import platform
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from bo_forge import __version__
from bo_forge.config import CampaignConfig
from bo_forge.errors import (
    BOForgeError,
    ConfigError,
    LogBusyError,
    LogConflictError,
    LogValidationError,
    LogWriteError,
    ProvenanceError,
    ProvenanceRecoveryRequired,
    SuggestionError,
)
from bo_forge.plot_registry import _PLOT_ROUTES, _canonical_plot_kind

if TYPE_CHECKING:
    from bo_forge.session import CampaignSession


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

    _register_environment_commands(subparsers)
    _register_read_commands(subparsers)
    from bo_forge._cli.provenance import register_provenance_commands

    register_provenance_commands(subparsers, _add_config_log_arguments)
    _register_mutation_commands(subparsers)
    _register_plot_command(subparsers)
    return parser


def _register_environment_commands(subparsers: argparse._SubParsersAction) -> None:
    """Register environment and log-initialization commands."""

    doctor_parser = subparsers.add_parser("doctor", help="Check the active BO Forge environment.")
    doctor_parser.set_defaults(handler=_cmd_doctor)

    init_log_parser = subparsers.add_parser(
        "init-log",
        help="Create an empty canonical campaign CSV log and provenance manifest.",
    )
    _add_config_log_arguments(init_log_parser, include_provenance_policy=False)
    init_log_parser.set_defaults(handler=_cmd_init_log)


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


def _register_plot_command(subparsers: argparse._SubParsersAction) -> None:
    """Register the single-output plot export command."""
    plot_parser = subparsers.add_parser("plot", help="Export one campaign plot.")
    _add_config_log_arguments(plot_parser)
    plot_parser.add_argument(
        "--kind",
        choices=[
            "progress",
            "diagnostics",
            "cost-progress",
            "replicates",
            "pareto",
            "pareto-parallel",
            "hypervolume",
            "stage-diagnostics",
            "fidelity-diagnostics",
            "fidelity-progress",
            "context-diagnostics",
            "qlog-nei-diagnostics",
            "model-diagnostics",
            "model-comparison",
        ],
        required=True,
        help="Plot type to export.",
    )
    plot_parser.add_argument("--output", type=Path, required=True, help="Figure output path.")
    plot_parser.set_defaults(handler=_cmd_plot)


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


def _add_config_log_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_provenance_policy: bool = True,
) -> None:
    parser.add_argument("--config", required=True, type=Path, help="Campaign YAML config path.")
    parser.add_argument("--log", required=True, type=Path, help="Campaign CSV log path.")
    if include_provenance_policy:
        parser.add_argument(
            "--require-provenance",
            action="store_true",
            help="Reject legacy campaigns that do not have a provenance manifest.",
        )


def _load_session(args: argparse.Namespace) -> CampaignSession:
    from bo_forge.session import CampaignSession

    policy = "required" if args.require_provenance else "compatible"
    return CampaignSession.from_files(
        args.config,
        args.log,
        provenance_policy=policy,
    )


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
    from bo_forge._campaign.provenance import manifest_path_for_log
    from bo_forge.session import CampaignSession

    campaign = CampaignSession.initialize(args.config, args.log)
    print(f"Created empty campaign log: {campaign.log_path}")
    print(f"Created provenance manifest: {manifest_path_for_log(campaign.log_path)}")
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


def _cmd_cost_summary(args: argparse.Namespace) -> int:
    campaign = _load_session(args)
    _print_table(campaign.cost_summary())
    return 0


def _cmd_replicate_summary(args: argparse.Namespace) -> int:
    campaign = _load_session(args)
    _print_table(campaign.replicate_summary())
    return 0


def _cmd_stage_summary(args: argparse.Namespace) -> int:
    campaign = _load_session(args)
    if not campaign.config.is_structured_campaign:
        raise ConfigError("stage-summary requires a structured campaign config.")
    _print_table(campaign.stage_summary())
    return 0


def _cmd_fidelity_summary(args: argparse.Namespace) -> int:
    campaign = _load_session(args)
    if campaign.config.fidelity is None:
        raise ConfigError("fidelity-summary requires a multi-fidelity config.")
    _print_table(campaign.fidelity_summary())
    return 0


def _cmd_fidelity_coverage(args: argparse.Namespace) -> int:
    campaign = _load_session(args)
    if campaign.config.fidelity is None:
        raise ConfigError("fidelity-coverage requires a multi-fidelity config.")
    coverage = campaign.fidelity_coverage()
    _print_table(coverage.where(pd.notna(coverage), ""))
    return 0


def _cmd_context_summary(args: argparse.Namespace) -> int:
    campaign = _load_session(args)
    if campaign.config.context is None:
        raise ConfigError("context-summary requires a contextual config.")
    _print_table(campaign.context_summary())
    return 0


def _cmd_qlog_nei_summary(args: argparse.Namespace) -> int:
    campaign = _load_session(args)
    if campaign.config.bo.acquisition != "qlog_nei":
        raise ConfigError("qlog-nei-summary requires bo.acquisition: qlog_nei.")
    _print_table(campaign.qlog_nei_summary())
    return 0


def _cmd_model_summary(args: argparse.Namespace) -> int:
    campaign = _load_session(args)
    _print_table(campaign.model_summary())
    return 0


def _cmd_model_compare(args: argparse.Namespace) -> int:
    campaign = _load_session(args)
    _print_table(campaign.model_profile_comparison(profiles=args.profile))
    return 0


def _cmd_pareto_front(args: argparse.Namespace) -> int:
    campaign = _load_session(args)
    if not campaign.config.is_multi_objective:
        raise ConfigError("pareto-front requires a multi-objective config.")
    _print_table(campaign.pareto_front())
    return 0


def _cmd_pareto_summary(args: argparse.Namespace) -> int:
    campaign = _load_session(args)
    if not campaign.config.is_multi_objective:
        raise ConfigError("pareto-summary requires a multi-objective config.")
    _print_table(campaign.pareto_summary())
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


def _cmd_suggest(args: argparse.Namespace) -> int:
    campaign = _load_session(args)
    context_values = _parse_context_values(args.context)
    suggestions = campaign.suggest_next(
        batch_size=args.batch_size,
        stage=args.stage,
        context_values=context_values,
    )

    _print_contextual_replicate_fallback_note(campaign, suggestions)

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


def _cmd_plot(args: argparse.Namespace) -> int:
    campaign = _load_session(args)
    try:
        _validate_cli_plot_request(campaign.config, args.kind)
        route = _PLOT_ROUTES[_canonical_plot_kind(args.kind)]
        getattr(campaign, route.session_method)(save_path=args.output)
    except OSError as exc:
        raise _CLIOutputError(
            f"Could not write {args.kind} plot '{args.output}': {exc}"
        ) from exc
    print(f"Wrote {args.kind} plot: {args.output}")
    return 0


def _validate_cli_plot_request(config: CampaignConfig, kind: str) -> None:
    if kind == "cost-progress" and config.cost is None:
        raise ConfigError("plot --kind cost-progress requires a config with a cost section.")
    if kind == "replicates" and not config.replicates.enabled:
        raise ConfigError(
            "plot --kind replicates requires a config with replicates.enabled: true."
        )
    if kind in {"pareto", "pareto-parallel", "hypervolume"}:
        _validate_multi_objective_plot_request(config, kind)
    if kind == "stage-diagnostics" and not config.is_structured_campaign:
        raise ConfigError("plot --kind stage-diagnostics requires a structured config.")
    if kind in {"fidelity-diagnostics", "fidelity-progress"} and config.fidelity is None:
        raise ConfigError(
            f"plot --kind {kind} requires a multi-fidelity config."
        )
    if kind == "context-diagnostics" and config.context is None:
        raise ConfigError("plot --kind context-diagnostics requires a contextual config.")
    if kind == "qlog-nei-diagnostics" and config.bo.acquisition != "qlog_nei":
        raise ConfigError(
            "plot --kind qlog-nei-diagnostics requires bo.acquisition: qlog_nei."
        )
    if kind in {"model-diagnostics", "model-comparison"}:
        _validate_model_plot_request(config, kind)


def _validate_multi_objective_plot_request(config: CampaignConfig, kind: str) -> None:
    if not config.is_multi_objective:
        raise ConfigError(f"plot --kind {kind} requires a multi-objective config.")
    if kind == "pareto-parallel" and len(config.objectives) < 3:
        raise ConfigError("plot --kind pareto-parallel requires at least three objectives.")


def _validate_model_plot_request(config: CampaignConfig, kind: str) -> None:
    if config.is_multi_objective:
        raise ConfigError(f"plot --kind {kind} requires a single-objective config.")
    if config.fidelity is not None:
        raise ConfigError(f"plot --kind {kind} does not support multi-fidelity configs.")
    if config.is_structured_campaign:
        raise ConfigError(f"plot --kind {kind} does not support structured configs.")


def _print_table(df: pd.DataFrame) -> None:
    print(df.to_string(index=False))


def _write_csv(df: pd.DataFrame, path: Path) -> Path:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
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
    if isinstance(exc, ProvenanceRecoveryRequired):
        return f"Hint: {exc.recovery_action}"
    if isinstance(exc, LogConflictError):
        action = getattr(exc, "recovery_action", None)
        if action:
            return f"Hint: {action}"
        return "Hint: Reload the campaign, inspect the latest log, and retry the mutation."
    if isinstance(exc, LogBusyError):
        return "Hint: Another local writer is active; wait briefly and retry."
    if isinstance(exc, SuggestionError):
        return _suggestion_error_hint(str(exc))
    if isinstance(exc, LogWriteError):
        return "Hint: Check the row_id, pending status, campaign log path, and file permissions."
    if isinstance(exc, ProvenanceError):
        if exc.recovery_action:
            return f"Hint: {exc.recovery_action}"
        return (
            "Hint: Check that the CSV and its .manifest.json sidecar exist together, "
            "then inspect their paths and JSON content."
        )
    return None


def _suggestion_error_hint(message: str) -> str:
    if "review_status='pending'" in message and "qLogNEI" in message:
        return (
            "Hint: qLogNEI can use accepted suggestions as X_pending, but rows "
            "still awaiting review must be accepted, rejected, or deferred first."
        )
    if "observe accepted pending initial suggestions" in message:
        return (
            "Hint: qLogNEI requires observed initial-design rows before "
            "model-based suggestions; mark accepted initial suggestions observed first."
        )
    if "Context" in message or "context" in message:
        return (
            "Hint: Use --context NAME=VALUE for each configured context "
            "variable, or add context.default_values in the YAML config."
        )
    if "qMFKG supports batch_size from 1 through 4" in message:
        return "Hint: Use --batch-size 1, 2, 3, or 4 for qMFKG suggestions."
    if "qMFKG acquisition optimization timed out" in message:
        return (
            "Hint: Increase or remove fidelity.optimizer_timeout_seconds, "
            "reduce qMFKG runtime settings, or relax restrictive constraints."
        )
    structured_markers = (
        "Structured campaign suggestions require an explicit stage",
        "Invalid structured campaign stage",
        "Unknown structured campaign stage",
        "has no active variables",
        "--stage is only valid",
    )
    if any(marker in message for marker in structured_markers):
        return (
            "Hint: Use --stage with one configured structured stage name, "
            "or omit --stage for non-structured campaigns."
        )
    return (
        "Hint: Resolve pending suggestions or review the campaign state before "
        "requesting new suggestions."
    )


if __name__ == "__main__":
    main()
