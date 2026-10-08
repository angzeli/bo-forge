"""CLI-owned failures and actionable hints; backend exception contracts stay unchanged."""

from __future__ import annotations

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


class _CLIOutputError(BOForgeError):
    """Raised when a CLI-owned output file cannot be written."""


class _CLIDoctorError(BOForgeError):
    """Raised when an expected doctor check fails."""


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


class SerializationError(BOForgeError):
    """An inspection result contains values outside the JSON contract."""


class ArgumentFailure(Exception):
    """Retain the failing parser's usage and message without writing stdout."""

    def __init__(self, parser, message):
        self.parser = parser
        self.message = message
