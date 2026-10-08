"""Public CLI parser composition and entrypoints; command behavior lives in _cli."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from bo_forge import __version__
from bo_forge._cli.campaign import _register_init_log_command, _register_mutation_commands
from bo_forge._cli.common import _add_config_log_arguments
from bo_forge._cli.dispatch import run_cli
from bo_forge._cli.environment import _register_doctor_command, _register_plot_command
from bo_forge._cli.errors import _hint_for_error
from bo_forge._cli.inspection import _register_read_commands
from bo_forge._cli.output import InspectionParser, register_formats


def build_parser() -> argparse.ArgumentParser:
    """Build the BO Forge CLI argument parser."""
    parser = InspectionParser(
        prog="bo-forge",
        description="Run BO Forge campaign workflows from the terminal.",
    )
    parser.add_argument("--version", action="version", version=f"bo-forge {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Registration order is part of the terminal help contract.
    _register_doctor_command(subparsers)
    _register_init_log_command(subparsers)
    _register_read_commands(subparsers)
    from bo_forge._cli.model_evaluate import register_model_evaluate_command
    from bo_forge._cli.provenance import register_provenance_commands

    register_model_evaluate_command(subparsers, _add_config_log_arguments)
    register_provenance_commands(subparsers, _add_config_log_arguments)
    _register_mutation_commands(subparsers)
    _register_plot_command(subparsers)
    register_formats(subparsers)
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments, dispatch a command, and return an exit code."""
    return run_cli(build_parser(), argv, _hint_for_error)


def main(argv: Sequence[str] | None = None) -> None:
    """Console-script entrypoint."""
    raise SystemExit(run(argv))


if __name__ == "__main__":
    main()
