"""Custom exceptions for BO Forge."""


class BOForgeError(Exception):
    """Base exception for BO Forge errors."""


class ConfigError(BOForgeError):
    """Raised when campaign configuration is invalid."""


class LogValidationError(BOForgeError):
    """Raised when a campaign log fails validation."""


class SuggestionError(BOForgeError):
    """Raised when candidate suggestions cannot be generated."""


class LogWriteError(BOForgeError):
    """Raised when writing or post-write validation of a campaign log fails."""


class ProvenanceError(BOForgeError):
    """Raised when campaign provenance metadata is unreadable or unsupported."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "manifest_invalid",
        recovery_action: str | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.recovery_action = recovery_action


class LogConflictError(LogWriteError):
    """Raised when a campaign log changed after a caller captured its state."""


class ProvenanceRecoveryRequired(LogConflictError):
    """Raised when an interrupted managed mutation requires explicit recovery."""

    def __init__(self, message: str, *, reason_code: str, recovery_action: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.recovery_action = recovery_action


class LogBusyError(LogWriteError):
    """Raised when a campaign log mutation cannot acquire its process lock."""
