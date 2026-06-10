"""HTTP client for the Kongen Labs SCI Pattern Intelligence API."""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

from kongen.exceptions import (
    APIError,
    AuthenticationError,
    KongenError,
    RateLimitError,
    TokensExhaustedError,
)
from kongen.types import TokenUsage

_DEFAULT_BASE_URL = "https://api.kongenlabs.life"
_DEFAULT_TIMEOUT = 30.0
_MAX_RETRIES = 3
_INITIAL_BACKOFF = 1.0  # seconds
_USER_AGENT = "kongen-python/1.0.0"


class KongenClient:
    """Client for the Kongen Labs SCI Pattern Intelligence API.

    Handles authentication, automatic retry with exponential backoff on
    rate-limit responses, and token usage tracking.

    Args:
        api_key: Your Kongen API key (``kl_live_...`` or ``kl_test_...``).
            Falls back to the ``KONGEN_API_KEY`` environment variable.
        base_url: API base URL. Override for testing or on-prem deployments.
        timeout: Request timeout in seconds.

    Usage::

        from kongen import KongenClient

        client = KongenClient(api_key="kl_live_...")
        result = client.logic.score("Prove sqrt(2) is irrational")
        print(result.regime)
        client.close()

    Or as a context manager::

        with KongenClient(api_key="kl_live_...") as client:
            result = client.logic.score("Prove sqrt(2) is irrational")
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = _DEFAULT_BASE_URL,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        resolved_key = api_key or os.environ.get("KONGEN_API_KEY")
        if not resolved_key:
            raise AuthenticationError(
                "No API key provided. Pass api_key= or set KONGEN_API_KEY."
            )

        self._api_key = resolved_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._last_token_usage: TokenUsage | None = None

        self._http = httpx.Client(
            base_url=self._base_url,
            timeout=self._timeout,
            headers={
                "X-API-Key": self._api_key,
                "User-Agent": _USER_AGENT,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )

        # Lazy-initialized sub-clients
        self._logic: LogicAnalyzer | None = None
        self._transfer: TransferScorer | None = None
        self._organisms: OrganismManager | None = None

    # -- Sub-client accessors --------------------------------------------------

    @property
    def logic(self) -> LogicAnalyzer:
        """Access the Logic LLM reasoning regime analyzer."""
        if self._logic is None:
            from kongen.logic import LogicAnalyzer

            self._logic = LogicAnalyzer(self)
        return self._logic

    @property
    def transfer(self) -> TransferScorer:
        """Access the pattern transfer scorer."""
        if self._transfer is None:
            from kongen.transfer import TransferScorer

            self._transfer = TransferScorer(self)
        return self._transfer

    @property
    def organisms(self) -> OrganismManager:
        """Access the organism creation and management interface."""
        if self._organisms is None:
            from kongen.organism import OrganismManager

            self._organisms = OrganismManager(self)
        return self._organisms

    # -- Token usage -----------------------------------------------------------

    @property
    def token_usage(self) -> TokenUsage | None:
        """Most recent token usage snapshot, updated after each API call.

        Returns ``None`` if no API calls have been made yet.
        """
        return self._last_token_usage

    # -- Core request machinery ------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Send an authenticated request with retry and error handling.

        Args:
            method: HTTP method (GET, POST, etc.).
            path: API path (e.g., ``/v1/logic/score``).
            **kwargs: Passed through to ``httpx.Client.request``.

        Returns:
            Parsed JSON response body as a dict.

        Raises:
            AuthenticationError: On 401 or 403.
            RateLimitError: On 429 after retries are exhausted.
            TokensExhaustedError: On 402 (payment required).
            APIError: On any other non-2xx response.
            KongenError: On network / transport failures.
        """
        last_exception: Exception | None = None
        backoff = _INITIAL_BACKOFF

        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = self._http.request(method, path, **kwargs)
            except httpx.TransportError as exc:
                raise KongenError(f"Network error: {exc}") from exc

            # Update token usage from response headers if present
            self._update_token_usage(response)

            if response.status_code == 200:
                return response.json()  # type: ignore[no-any-return]

            # -- Error mapping -------------------------------------------------

            if response.status_code in (401, 403):
                detail = self._extract_detail(response)
                raise AuthenticationError(detail or "Invalid or missing API key.")

            if response.status_code == 402:
                raise TokensExhaustedError()

            if response.status_code == 429:
                retry_after = self._parse_retry_after(response)
                if attempt < _MAX_RETRIES:
                    wait = retry_after if retry_after is not None else backoff
                    time.sleep(wait)
                    backoff *= 2
                    last_exception = RateLimitError(retry_after=retry_after)
                    continue
                raise RateLimitError(retry_after=retry_after)

            # Any other error
            detail = self._extract_detail(response)
            raise APIError(
                message="API request failed.",
                status_code=response.status_code,
                detail=detail,
            )

        # Should not reach here, but satisfy type checker
        if last_exception is not None:
            raise last_exception
        raise KongenError("Request failed after retries.")  # pragma: no cover

    # -- Helpers ---------------------------------------------------------------

    def _update_token_usage(self, response: httpx.Response) -> None:
        """Extract token usage from response headers if present."""
        used = response.headers.get("X-Kongen-Tokens-Used")
        remaining = response.headers.get("X-Kongen-Tokens-Remaining")
        plan = response.headers.get("X-Kongen-Plan")
        if used is not None and remaining is not None:
            self._last_token_usage = TokenUsage(
                used=int(used),
                remaining=int(remaining),
                plan=plan or "unknown",
            )

    @staticmethod
    def _parse_retry_after(response: httpx.Response) -> float | None:
        """Parse the Retry-After header value in seconds."""
        value = response.headers.get("Retry-After")
        if value is not None:
            try:
                return float(value)
            except ValueError:
                return None
        return None

    @staticmethod
    def _extract_detail(response: httpx.Response) -> str | None:
        """Extract error detail from a JSON response body."""
        try:
            body = response.json()
            if isinstance(body, dict):
                return body.get("detail") or body.get("error") or body.get("message")
        except Exception:
            pass
        return None

    # -- Lifecycle -------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._http.close()

    def __enter__(self) -> KongenClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"KongenClient(base_url={self._base_url!r})"


# Avoid circular imports at module level — these are imported lazily in properties.
# The TYPE_CHECKING guard lets type checkers see the imports without runtime cost.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kongen.logic import LogicAnalyzer
    from kongen.organism import OrganismManager
    from kongen.transfer import TransferScorer
