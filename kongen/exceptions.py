"""Exception hierarchy for the Kongen SDK."""

from __future__ import annotations


class KongenError(Exception):
    """Base exception for all Kongen SDK errors."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class AuthenticationError(KongenError):
    """Raised when the API key is invalid, expired, or missing.

    Check that your API key is correctly set and has not been revoked
    at https://kongenlabs.life/dashboard.
    """

    def __init__(self, message: str = "Invalid or missing API key.") -> None:
        super().__init__(message)


class RateLimitError(KongenError):
    """Raised when the API returns HTTP 429 (Too Many Requests).

    The retry_after attribute indicates how many seconds to wait
    before retrying. The SDK handles automatic retries with exponential
    backoff, so this exception is only raised after all retries are exhausted.
    """

    def __init__(
        self,
        message: str = "Rate limit exceeded.",
        retry_after: float | None = None,
    ) -> None:
        self.retry_after = retry_after
        if retry_after is not None:
            message = f"{message} Retry after {retry_after:.1f}s."
        super().__init__(message)


class TokensExhaustedError(KongenError):
    """Raised when the account has no Kongen Tokens remaining.

    Purchase additional tokens or upgrade your plan at
    https://kongenlabs.life/dashboard.
    """

    def __init__(
        self, message: str = "No Kongen Tokens remaining. Upgrade your plan."
    ) -> None:
        super().__init__(message)


class APIError(KongenError):
    """Raised for generic API server errors (5xx, unexpected responses).

    Attributes:
        status_code: The HTTP status code returned by the server.
        detail: Additional error detail from the response body, if available.
    """

    def __init__(
        self,
        message: str = "API request failed.",
        status_code: int | None = None,
        detail: str | None = None,
    ) -> None:
        self.status_code = status_code
        self.detail = detail
        parts = [message]
        if status_code is not None:
            parts.append(f"Status: {status_code}.")
        if detail:
            parts.append(f"Detail: {detail}")
        super().__init__(" ".join(parts))
