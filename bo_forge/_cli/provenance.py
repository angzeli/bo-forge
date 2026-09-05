"""Provenance CLI command registration and handlers."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import Any


def register_provenance_commands(
    subparsers: argparse._SubParsersAction,
    add_config_log_arguments: Callable[..., None],
) -> None:
    """Register provenance inspection and explicit recovery commands."""
    provenance_parser = subparsers.add_parser(
        "provenance",
        help="Print campaign provenance and integrity fields.",
    )
    add_config_log_arguments(provenance_parser)
    provenance_parser.set_defaults(handler=_cmd_provenance)

    recover_parser = subparsers.add_parser(
        "provenance-recover",
        help="Explicitly resolve an interrupted managed-campaign transaction.",
    )
    add_config_log_arguments(recover_parser, include_provenance_policy=False)
    recover_parser.add_argument(
        "--expected-log-fingerprint",
        help="Optional current log fingerprint required before recovery.",
    )
    recover_parser.set_defaults(handler=_cmd_provenance_recover)


def _cmd_provenance(args: argparse.Namespace) -> int:
    from bo_forge._campaign.provenance_resume import inspect_provenance

    inspection = inspect_provenance(
        args.config,
        args.log,
        provenance_policy="required" if args.require_provenance else "compatible",
    )
    if inspection.provenance_status == "legacy" and not inspection.log_file.exists():
        from bo_forge.provenance import provenance_summary

        summary = provenance_summary(args.config, args.log)
    else:
        summary = inspection.to_frame()
    _print_table(summary)
    values = dict(summary.itertuples(index=False, name=None))
    if values.get("provenance_status") == "managed" and values.get(
        "integrity_status"
    ) != "valid":
        import sys

        reason = values.get("reason_code") or "manifest_invalid"
        print(
            "Error: Managed campaign provenance is not in a finalized valid state. "
            f"Reason: {reason}.",
            file=sys.stderr,
        )
        if values.get("recovery_action"):
            print(f"Hint: {values['recovery_action']}", file=sys.stderr)
        return 1
    return 0


def _cmd_provenance_recover(args: argparse.Namespace) -> int:
    from bo_forge.provenance import recover_provenance

    summary = recover_provenance(
        args.config,
        args.log,
        expected_log_fingerprint=args.expected_log_fingerprint,
    )
    _print_table(summary)
    print("Provenance state verified; reload the campaign before continuing.")
    return 0


def _print_table(frame: Any) -> None:
    print(frame.to_string(index=False))
