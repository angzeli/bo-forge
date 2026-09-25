"""Version-one inspection output, independent of optional application packages."""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
import pandas as pd

from bo_forge import __version__
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

INSPECTION_COMMANDS = (
    "validate", "summary", "status", "next-action", "cost-summary",
    "replicate-summary", "stage-summary", "context-summary", "fidelity-summary",
    "fidelity-coverage", "qlog-nei-summary", "model-summary", "model-compare",
    "pareto-front", "pareto-summary", "provenance",
)


class SerializationError(BOForgeError):
    """An inspection result contains values outside the JSON contract."""


class ArgumentFailure(Exception):
    """Retain the failing parser's usage and message without writing stdout."""

    def __init__(self, parser, message):
        self.parser = parser
        self.message = message


class InspectionParser(argparse.ArgumentParser):
    """Let the entrypoint render parser errors in the requested format."""

    def error(self, message):
        raise ArgumentFailure(self, message)


def register_formats(subparsers):
    for name in INSPECTION_COMMANDS:
        subparsers.choices[name].add_argument(
            "--format", choices=("text", "json"), default="text",
            help="Inspection output format (default: text).",
        )


def requested_json_command(parser, argv):
    """Recognize explicit JSON intent even when parsing the request fails."""
    if not argv or argv[0] not in INSPECTION_COMMANDS:
        return None
    subparsers = next(action for action in parser._actions
                      if isinstance(action, argparse._SubParsersAction))
    command_parser = subparsers.choices[argv[0]]
    # Last valid format wins; a malformed repeat must not erase JSON intent.
    selected = None
    for index, token in enumerate(argv[1:], 1):
        if token == "--":
            break
        value = _format_value(command_parser, token, argv[index + 1:index + 2])
        if value in ("text", "json"):
            selected = value
    return argv[0] if selected == "json" else None


def _format_value(parser, token, following):
    # Inspect declared options, not _parse_optional's patch-version-dependent tuples.
    option, separator, explicit = token.partition("=")
    registered = parser._option_string_actions
    if option not in registered and parser.allow_abbrev and option.startswith("--"):
        matches = [name for name in registered if name.startswith(option)]
        option = matches[0] if len(matches) == 1 else None
    if option != "--format":
        return None
    return explicit if separator else next(iter(following), None)


def json_value(value):
    """Preserve scalar types; missing and nonfinite numbers are JSON null."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (str, bool)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, np.floating) and value.dtype.itemsize > 8:
        raise SerializationError("Extended-precision floating values are unsupported.")
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise SerializationError("JSON object keys must be strings.")
        return {key: json_value(item) for key, item in value.items()}
    raise SerializationError(f"Unsupported JSON value type: {type(value).__name__}.")


def table_payload(frame):
    columns = list(frame.columns)
    if not all(isinstance(name, str) for name in columns) or len(set(columns)) != len(columns):
        raise SerializationError("JSON table columns must be unique strings.")
    return {
        "columns": columns,
        "records": [
            {name: json_value(value) for name, value in zip(columns, row, strict=True)}
            for row in frame.itertuples(index=False, name=None)
        ],
    }


def emit(args, value, *, text=None):
    if getattr(args, "format", "text") == "json":
        args._json_data = table_payload(value) if isinstance(value, pd.DataFrame) else value
    else:
        print(text if text is not None else value.to_string(index=False))


_ERROR_CODES = (
    (SerializationError, "serialization_error"),
    (ProvenanceRecoveryRequired, "provenance_recovery_required"),
    (ProvenanceError, "provenance_error"),
    (LogBusyError, "log_busy_error"),
    (LogConflictError, "log_conflict_error"),
    (LogValidationError, "log_validation_error"),
    (LogWriteError, "log_write_error"),
    (ConfigError, "config_error"),
    (SuggestionError, "suggestion_error"),
    (BOForgeError, "bo_forge_error"),
)


def error_payload(exc, hint):
    code = next(code for cls, code in _ERROR_CODES if isinstance(exc, cls))
    details = {
        key: getattr(exc, key)
        for key in ("reason_code", "recovery_action") if hasattr(exc, key)
    }
    return {"code": code, "message": str(exc), "hint": hint, "details": details}


def render(command, data=None, error=None):
    return json.dumps(
        {"schema_version": 1, "bo_forge_version": __version__, "command": command,
         "ok": error is None, "data": json_value(data), "error": json_value(error)},
        ensure_ascii=False, allow_nan=False,
    )
