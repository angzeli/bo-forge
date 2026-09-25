"""CLI dispatch with atomic JSON response assembly and legacy text diagnostics."""

from __future__ import annotations

import contextlib
import sys

from bo_forge._cli.output import (
    ArgumentFailure,
    SerializationError,
    error_payload,
    render,
    requested_json_command,
)
from bo_forge.errors import BOForgeError


def run_cli(parser, argv, hint_for_error):
    arguments = list(sys.argv[1:] if argv is None else argv)
    command = requested_json_command(parser, arguments)
    try:
        args = parser.parse_args(arguments)
    except ArgumentFailure as exc:
        exc.parser.print_usage(sys.stderr)
        print(f"{exc.parser.prog}: error: {exc.message}", file=sys.stderr)
        if command:
            print(render(command, error={
                "code": "argument_error", "message": exc.message, "hint": None, "details": {},
            }))
        return 2
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1
    if getattr(args, "format", "text") == "json":
        return _run_json(args, hint_for_error)
    try:
        return int(args.handler(args))
    except BOForgeError as exc:
        _diagnose(exc, hint_for_error(exc))
        return 1


def _diagnose(exc, hint):
    print(f"Error: {exc}", file=sys.stderr)
    if hint is not None:
        print(hint, file=sys.stderr)


def _run_json(args, hint_for_error):
    error = None
    code = 0
    with contextlib.redirect_stdout(sys.stderr):
        try:
            code = int(args.handler(args))
        except BOForgeError as exc:
            hint = hint_for_error(exc)
            _diagnose(exc, hint)
            error, code = error_payload(exc, hint), 1
    try:
        response = render(args.command, getattr(args, "_json_data", None), error)
    except SerializationError as exc:
        _diagnose(exc, None)
        response = render(args.command, error=error_payload(exc, None))
        code = 1
    print(response)
    return code
