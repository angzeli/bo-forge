"""Explicit predictive evaluation command, separate from in-sample comparison."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from bo_forge.errors import ConfigError


def register_model_evaluate_command(
    subparsers: argparse._SubParsersAction,
    add_config_log_arguments: Callable[..., None],
) -> None:
    """Register a read-only evaluation with optional explicit artifact export."""
    parser = subparsers.add_parser(
        "model-evaluate", help="Evaluate model profiles using held-out predictions."
    )
    add_config_log_arguments(parser)
    parser.add_argument(
        "--profile", action="append", choices=["default", "smooth", "rough", "robust"],
        help="Model profile; repeat in the desired evaluation order.",
    )
    parser.add_argument("--folds", type=int, default=5, help="Cross-validation folds (default: 5).")
    parser.add_argument("--seed", type=int, default=0, help="Split seed (default: 0).")
    parser.add_argument("--output-dir", type=Path, help="Optional evaluation artifact directory.")
    parser.set_defaults(handler=_cmd_model_evaluate)


def _cmd_model_evaluate(args: argparse.Namespace) -> int:
    from bo_forge.cli import _CLIOutputError, _load_session, _print_table

    campaign = _load_session(args)
    try:
        existing_output = args.output_dir is not None and (
            args.output_dir.exists() or args.output_dir.is_symlink()
        )
    except OSError as exc:
        raise _CLIOutputError(
            f"Could not inspect predictive evaluation output '{args.output_dir}': {exc}"
        ) from exc
    if existing_output:
        raise _CLIOutputError(
            f"Predictive evaluation output directory must not already exist: '{args.output_dir}'."
        )
    try:
        result = campaign.model_predictive_evaluation(
            profiles=args.profile, folds=args.folds, seed=args.seed,
        )
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    if args.output_dir is not None:
        try:
            result.export(args.output_dir)
        except (OSError, ValueError, TypeError) as exc:
            raise _CLIOutputError(
                f"Could not export predictive evaluation to '{args.output_dir}': {exc}"
            ) from exc
    _print_table(result.summary)
    print("Fold outcomes:")
    _print_table(result.fold_outcomes)
    incomplete = bool(result.summary["fit_status"].ne("complete").any())
    if args.output_dir is not None:
        label = "incomplete predictive evaluation" if incomplete else "predictive evaluation"
        print(f"Wrote {label}: {args.output_dir}")
    if incomplete:
        print(
            "Predictive evaluation is incomplete; aggregate metrics are withheld for "
            "incomplete profiles. Inspect the summary and fold outcomes above.", file=sys.stderr,
        )
    return 1 if incomplete else 0
