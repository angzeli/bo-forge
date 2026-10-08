"""Environment checks and explicit plot export; plot dependencies are loaded on demand."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.util
import io
import platform
import sys
from pathlib import Path

from bo_forge import __version__
from bo_forge._campaign.exports import _validate_export_destination
from bo_forge._cli.common import _add_config_log_arguments, _load_session
from bo_forge._cli.errors import _CLIDoctorError, _CLIOutputError
from bo_forge.config import CampaignConfig
from bo_forge.errors import ConfigError
from bo_forge.plot_registry import _PLOT_ROUTES, _canonical_plot_kind


def _register_doctor_command(subparsers: argparse._SubParsersAction) -> None:
    """Register the environment check before campaign commands."""
    doctor_parser = subparsers.add_parser("doctor", help="Check the active BO Forge environment.")
    doctor_parser.set_defaults(handler=_cmd_doctor)


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


def _cmd_plot(args: argparse.Namespace) -> int:
    campaign = _load_session(args)
    try:
        _validate_cli_plot_request(campaign.config, args.kind)
        _validate_export_destination(args.output, campaign.config_path, campaign.log_path)
        _validate_plot_format(args.output)
        route = _PLOT_ROUTES[_canonical_plot_kind(args.kind)]
        getattr(campaign, route.session_method)(save_path=args.output)
    except OSError as exc:
        raise _CLIOutputError(
            f"Could not write {args.kind} plot '{args.output}': {exc}"
        ) from exc
    print(f"Wrote {args.kind} plot: {args.output}")
    return 0


def _validate_plot_format(path: Path) -> None:
    from matplotlib import rcParams
    from matplotlib.backend_bases import FigureCanvasBase

    requested = (path.suffix[1:] or rcParams["savefig.format"]).lower()
    supported = FigureCanvasBase.get_supported_filetypes()
    if requested not in supported:
        raise OSError(
            f"Format '{requested}' is not supported (supported formats: "
            f"{', '.join(sorted(supported))})."
        )


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


def _doctor_import(module_name: str) -> None:
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            importlib.import_module(module_name)
    except ImportError as exc:
        raise _CLIDoctorError(
            f"Doctor check failed while importing '{module_name}': {exc}"
        ) from exc
