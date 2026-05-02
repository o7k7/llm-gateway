from __future__ import annotations


class BackendError(Exception):
    """Base class for all backend errors."""

    def __init__(self, message: str, *, backend: str | None = None) -> None:
        super().__init__(message)
        self.backend = backend


class BackendUnavailableError(BackendError):
    """Backend is unreachable or returned 5xx."""


class BackendTimeoutError(BackendError):
    """Backend exceeded the configured timeout."""


class BackendRateLimitError(BackendError):
    """Backend signaled rate limiting (429 upstream)."""


class BackendAuthError(BackendError):
    """Backend returned 401/403 — usually a config problem, not a client one."""
