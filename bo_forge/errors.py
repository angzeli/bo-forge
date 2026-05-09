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

