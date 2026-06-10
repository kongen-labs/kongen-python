"""Comprehensive tests for the Kongen Labs SCI Pattern Intelligence SDK.

Tests cover:
    - Client initialization (key sources, missing key)
    - HTTP request lifecycle (200, 401, 402, 429, 500)
    - Token usage tracking from response headers
    - LogicAnalyzer.score() API path
    - LogicAnalyzer.analyze() local heuristic logic
    - TransferScorer.score_signal() and score_batch()
    - StructuralSignature model validation
    - Exception hierarchy and formatting
    - Context manager lifecycle
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import httpx
import pytest
from pydantic import ValidationError

from kongen.logic import LogicAnalyzer
from kongen.client import KongenClient
from kongen.exceptions import (
    APIError,
    AuthenticationError,
    KongenError,
    RateLimitError,
    TokensExhaustedError,
)
from kongen.transfer import TransferScorer
from kongen.types import (
    BatchTransferResult,
    LogicResult,
    StructuralSignature,
    TokenUsage,
    TransferResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_response(
    status_code: int = 200,
    json_body: dict | None = None,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    """Create a fake httpx.Response for mocking."""
    body = json.dumps(json_body or {}).encode()
    all_headers = dict(headers or {})
    all_headers.setdefault("content-type", "application/json")
    return httpx.Response(
        status_code=status_code,
        content=body,
        headers=all_headers,
    )


# ===========================================================================
# 1. Client Initialization
# ===========================================================================

class TestClientInitialization:
    """Verify KongenClient resolves API keys and validates arguments."""

    def test_init_with_explicit_key(self):
        client = KongenClient(api_key="kl_live_xyz")
        assert client._api_key == "kl_live_xyz"
        client.close()

    def test_init_from_env_variable(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("KONGEN_API_KEY", "kl_test_envkey")
        client = KongenClient()
        assert client._api_key == "kl_test_envkey"
        client.close()

    def test_explicit_key_takes_precedence_over_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("KONGEN_API_KEY", "kl_test_env")
        client = KongenClient(api_key="kl_live_explicit")
        assert client._api_key == "kl_live_explicit"
        client.close()

    def test_missing_key_raises_authentication_error(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("KONGEN_API_KEY", raising=False)
        with pytest.raises(AuthenticationError, match="No API key provided"):
            KongenClient()

    def test_custom_base_url(self):
        client = KongenClient(api_key="kl_test_x", base_url="https://custom.api.io/")
        assert client._base_url == "https://custom.api.io"  # trailing slash stripped
        client.close()

    def test_custom_timeout(self):
        client = KongenClient(api_key="kl_test_x", timeout=60.0)
        assert client._timeout == 60.0
        client.close()

    def test_repr(self, client: KongenClient):
        assert "KongenClient" in repr(client)
        assert "test.kongenlabs.life" in repr(client)

    def test_initial_token_usage_is_none(self, client: KongenClient):
        assert client.token_usage is None


# ===========================================================================
# 2. Client HTTP Request Handling
# ===========================================================================

class TestClientRequest:
    """Verify _request() handles status codes, retries, and errors."""

    def test_200_returns_json(self, client: KongenClient):
        response = _make_response(200, {"regime": "analytical"})
        with patch.object(client._http, "request", return_value=response):
            result = client._request("POST", "/v1/logic/score")
        assert result == {"regime": "analytical"}

    def test_401_raises_authentication_error(self, client: KongenClient):
        response = _make_response(401, {"detail": "API key expired"})
        with patch.object(client._http, "request", return_value=response):
            with pytest.raises(AuthenticationError, match="API key expired"):
                client._request("GET", "/v1/status")

    def test_403_raises_authentication_error(self, client: KongenClient):
        response = _make_response(403, {"error": "Forbidden"})
        with patch.object(client._http, "request", return_value=response):
            with pytest.raises(AuthenticationError, match="Forbidden"):
                client._request("GET", "/v1/status")

    def test_402_raises_tokens_exhausted(self, client: KongenClient):
        response = _make_response(402, {"detail": "No tokens"})
        with patch.object(client._http, "request", return_value=response):
            with pytest.raises(TokensExhaustedError, match="No Kongen Tokens remaining"):
                client._request("POST", "/v1/logic/score")

    def test_429_retries_and_succeeds(self, client: KongenClient):
        rate_limit_resp = _make_response(
            429, {"detail": "Rate limited"}, {"Retry-After": "0.01"}
        )
        ok_resp = _make_response(200, {"result": "ok"})

        call_count = 0

        def mock_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return rate_limit_resp
            return ok_resp

        with patch.object(client._http, "request", side_effect=mock_request):
            result = client._request("POST", "/v1/logic/score")

        assert result == {"result": "ok"}
        assert call_count == 3  # 2 retries + 1 success

    def test_429_exhausts_retries_raises_rate_limit_error(self, client: KongenClient):
        rate_limit_resp = _make_response(
            429, {"detail": "Too many requests"}, {"Retry-After": "0.01"}
        )
        with patch.object(client._http, "request", return_value=rate_limit_resp):
            with pytest.raises(RateLimitError, match="Rate limit exceeded"):
                client._request("POST", "/v1/logic/score")

    def test_429_uses_exponential_backoff_when_no_retry_after(self, client: KongenClient):
        """Without Retry-After header, backoff doubles each attempt."""
        rate_limit_resp = _make_response(429, {"detail": "Slow down"})
        ok_resp = _make_response(200, {"result": "ok"})

        attempts = []

        def mock_request(*args, **kwargs):
            attempts.append(1)
            if len(attempts) < 3:
                return rate_limit_resp
            return ok_resp

        with patch.object(client._http, "request", side_effect=mock_request):
            with patch("kongen.client.time.sleep") as mock_sleep:
                result = client._request("POST", "/v1/test")

        assert result == {"result": "ok"}
        # First backoff = 1.0s, second = 2.0s (doubled)
        assert mock_sleep.call_count == 2
        assert mock_sleep.call_args_list[0][0][0] == 1.0
        assert mock_sleep.call_args_list[1][0][0] == 2.0

    def test_500_raises_api_error(self, client: KongenClient):
        response = _make_response(500, {"detail": "Internal server error"})
        with patch.object(client._http, "request", return_value=response):
            with pytest.raises(APIError) as exc_info:
                client._request("POST", "/v1/logic/score")
        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "Internal server error"

    def test_503_raises_api_error_with_status(self, client: KongenClient):
        response = _make_response(503, {"message": "Service unavailable"})
        with patch.object(client._http, "request", return_value=response):
            with pytest.raises(APIError) as exc_info:
                client._request("GET", "/v1/health")
        assert exc_info.value.status_code == 503

    def test_transport_error_raises_kongen_error(self, client: KongenClient):
        with patch.object(
            client._http, "request", side_effect=httpx.ConnectError("Connection refused")
        ):
            with pytest.raises(KongenError, match="Network error"):
                client._request("GET", "/v1/health")

    def test_extract_detail_falls_back_to_error_key(self, client: KongenClient):
        response = _make_response(400, {"error": "Bad request detail"})
        with patch.object(client._http, "request", return_value=response):
            with pytest.raises(APIError) as exc_info:
                client._request("POST", "/v1/test")
        assert exc_info.value.detail == "Bad request detail"

    def test_extract_detail_falls_back_to_message_key(self, client: KongenClient):
        response = _make_response(400, {"message": "Msg detail"})
        with patch.object(client._http, "request", return_value=response):
            with pytest.raises(APIError) as exc_info:
                client._request("POST", "/v1/test")
        assert exc_info.value.detail == "Msg detail"

    def test_extract_detail_non_json_body_returns_none(self, client: KongenClient):
        resp = httpx.Response(
            status_code=502,
            content=b"<html>Bad Gateway</html>",
            headers={"content-type": "text/html"},
        )
        with patch.object(client._http, "request", return_value=resp):
            with pytest.raises(APIError) as exc_info:
                client._request("GET", "/v1/health")
        assert exc_info.value.detail is None

    def test_auth_error_no_detail_uses_default(self, client: KongenClient):
        resp = httpx.Response(
            status_code=401,
            content=b"Unauthorized",
            headers={"content-type": "text/plain"},
        )
        with patch.object(client._http, "request", return_value=resp):
            with pytest.raises(AuthenticationError, match="Invalid or missing API key"):
                client._request("GET", "/v1/test")


# ===========================================================================
# 3. Token Usage Tracking
# ===========================================================================

class TestTokenUsageTracking:
    """Verify token usage is parsed from response headers."""

    def test_token_usage_parsed_from_headers(self, client: KongenClient):
        response = _make_response(
            200,
            {"regime": "analytical"},
            {
                "X-Kongen-Tokens-Used": "42",
                "X-Kongen-Tokens-Remaining": "9958",
                "X-Kongen-Plan": "pro",
            },
        )
        with patch.object(client._http, "request", return_value=response):
            client._request("POST", "/v1/logic/score")

        usage = client.token_usage
        assert usage is not None
        assert usage.used == 42
        assert usage.remaining == 9958
        assert usage.plan == "pro"

    def test_token_usage_defaults_plan_to_unknown(self, client: KongenClient):
        response = _make_response(
            200,
            {"result": "ok"},
            {
                "X-Kongen-Tokens-Used": "1",
                "X-Kongen-Tokens-Remaining": "99",
            },
        )
        with patch.object(client._http, "request", return_value=response):
            client._request("GET", "/v1/test")

        assert client.token_usage is not None
        assert client.token_usage.plan == "unknown"

    def test_token_usage_not_updated_when_headers_missing(self, client: KongenClient):
        response = _make_response(200, {"result": "ok"})
        with patch.object(client._http, "request", return_value=response):
            client._request("GET", "/v1/test")

        assert client.token_usage is None

    def test_token_usage_updated_even_on_error_responses(self, client: KongenClient):
        response = _make_response(
            500,
            {"detail": "Server error"},
            {
                "X-Kongen-Tokens-Used": "100",
                "X-Kongen-Tokens-Remaining": "0",
                "X-Kongen-Plan": "free",
            },
        )
        with patch.object(client._http, "request", return_value=response):
            with pytest.raises(APIError):
                client._request("POST", "/v1/logic/score")

        usage = client.token_usage
        assert usage is not None
        assert usage.used == 100
        assert usage.remaining == 0

    def test_token_usage_updates_across_requests(self, client: KongenClient):
        resp1 = _make_response(
            200, {"ok": True},
            {"X-Kongen-Tokens-Used": "1", "X-Kongen-Tokens-Remaining": "99"},
        )
        resp2 = _make_response(
            200, {"ok": True},
            {"X-Kongen-Tokens-Used": "6", "X-Kongen-Tokens-Remaining": "94"},
        )
        with patch.object(client._http, "request", side_effect=[resp1, resp2]):
            client._request("GET", "/v1/test")
            assert client.token_usage.used == 1
            client._request("GET", "/v1/test")
            assert client.token_usage.used == 6
            assert client.token_usage.remaining == 94


# ===========================================================================
# 4. LogicAnalyzer.score() (API call)
# ===========================================================================

class TestLogicScore:
    """Verify LogicAnalyzer.score() serializes request and parses response."""

    def test_score_returns_logic_result(self, client: KongenClient):
        api_response = _make_response(
            200,
            {
                "regime": "analytical",
                "confidence_adj": 0.15,
                "tokens_used": 1,
                "tokens_remaining": 999,
            },
        )
        with patch.object(client._http, "request", return_value=api_response) as mock_req:
            result = client.logic.score("Prove sqrt(2) is irrational")

        assert isinstance(result, LogicResult)
        assert result.regime == "analytical"
        assert result.confidence_adj == 0.15
        assert result.tokens_used == 1
        assert result.tokens_remaining == 999

        # Verify correct endpoint was called
        call_args = mock_req.call_args
        assert call_args[0] == ("POST", "/v1/logic/score")
        payload = call_args[1]["json"]
        assert payload["text"] == "Prove sqrt(2) is irrational"
        assert "model_hint" not in payload

    def test_score_with_model_hint(self, client: KongenClient):
        api_response = _make_response(
            200,
            {
                "regime": "metacognitive",
                "confidence_adj": 0.28,
                "tokens_used": 1,
                "tokens_remaining": 500,
            },
        )
        with patch.object(client._http, "request", return_value=api_response) as mock_req:
            result = client.logic.score(
                "Think about how you think about this problem",
                model_hint="claude-3-haiku",
            )

        assert result.regime == "metacognitive"
        payload = mock_req.call_args[1]["json"]
        assert payload["model_hint"] == "claude-3-haiku"

    def test_logic_analyzer_lazy_initialized(self, client: KongenClient):
        assert client._logic is None
        _ = client.logic
        assert client._logic is not None
        assert isinstance(client._logic, LogicAnalyzer)

    def test_logic_repr(self, client: KongenClient):
        assert repr(client.logic) == "LogicAnalyzer()"


# ===========================================================================
# 5. LogicAnalyzer.analyze() (local-only, no API call)
# ===========================================================================

class TestLogicAnalyze:
    """Verify the local analyze() heuristic classifies regimes correctly."""

    def test_short_text_is_reflexive(self, client: KongenClient):
        result = client.logic.analyze("What is 2 + 2?")
        assert result["estimated_regime"] == "reflexive"
        assert result["token_count"] < 10

    def test_short_sentences_is_procedural(self, client: KongenClient):
        text = "Step one. Mix flour. Add eggs. Stir well. Bake it. Serve warm. Cool down. Repeat process. Done now. Eat food. Very tasty."
        result = client.logic.analyze(text)
        assert result["estimated_regime"] == "procedural"
        assert result["avg_sentence_length"] < 12

    def test_medium_sentences_is_analytical(self, client: KongenClient):
        text = (
            "The Pythagorean theorem establishes a fundamental relationship stating that "
            "in any right triangle the square of the hypotenuse equals the sum of the "
            "squares of the other two sides. This elegant result underpins much of Euclidean "
            "geometry and has countless practical applications in engineering."
        )
        result = client.logic.analyze(text)
        assert result["estimated_regime"] == "analytical"
        assert 12 <= result["avg_sentence_length"] < 25

    def test_many_long_sentences_is_synthesis(self, client: KongenClient):
        sentences = [
            "First we must carefully examine the broad economic implications of this complex and multifaceted policy decision across multiple interconnected global markets and extended time horizons",
            "Then we methodically connect these preliminary findings to the extensive and growing body of sociological research on community resilience and long-term adaptation strategies in rapidly changing urban environments",
            "The biological analogy of ecosystem balance and homeostatic regulation provides further critical insight into the underlying dynamics of complex system recovery after significant disruption events",
            "Historical precedents from medieval and early modern trade routes across Europe and Asia show remarkably similar and consistent patterns of economic consolidation and subsequent expansion",
            "Synthesizing all of these diverse perspectives from economics sociology biology and history reveals a compelling unified theoretical framework for understanding systemic transformation across disparate domains",
            "Finally the rigorous mathematical formalization of these observed cross-domain structural patterns strongly confirms and validates the original theoretical predictions we so carefully derived in our earlier analysis",
        ]
        text = ". ".join(sentences) + "."
        result = client.logic.analyze(text)
        assert result["estimated_regime"] == "synthesis"
        assert result["sentence_count"] > 5

    def test_long_avg_few_sentences_is_metacognitive(self, client: KongenClient):
        text = (
            "I need to think about how I approach this problem and whether my current "
            "reasoning strategy is actually the most effective way to decompose the "
            "underlying structure of the question rather than just applying a rote procedure "
            "from memory which might lead me astray in subtle ways that I would not notice "
            "until much later in the derivation process."
        )
        result = client.logic.analyze(text)
        assert result["estimated_regime"] == "metacognitive"
        assert result["avg_sentence_length"] >= 25
        assert result["sentence_count"] <= 5

    def test_analyze_returns_all_keys(self, client: KongenClient):
        result = client.logic.analyze("Hello world")
        assert set(result.keys()) == {
            "token_count",
            "sentence_count",
            "avg_sentence_length",
            "estimated_regime",
        }

    def test_analyze_does_not_make_api_call(self, client: KongenClient):
        with patch.object(client._http, "request") as mock_req:
            client.logic.analyze("Some text here")
        mock_req.assert_not_called()

    def test_analyze_handles_empty_string(self, client: KongenClient):
        result = client.logic.analyze("")
        assert result["token_count"] == 0
        assert result["sentence_count"] == 0
        assert result["estimated_regime"] == "reflexive"

    def test_analyze_counts_question_marks_as_sentence_endings(self, client: KongenClient):
        result = client.logic.analyze("What? Why? How?")
        assert result["sentence_count"] == 3

    def test_analyze_counts_exclamation_marks_as_sentence_endings(self, client: KongenClient):
        result = client.logic.analyze("Stop! Go! Now!")
        assert result["sentence_count"] == 3

    def test_analyze_rounds_avg_sentence_length(self, client: KongenClient):
        result = client.logic.analyze("One two three. Four five.")
        avg = result["avg_sentence_length"]
        # Should be rounded to 2 decimal places
        assert avg == round(avg, 2)


# ===========================================================================
# 6. TransferScorer.score_signal()
# ===========================================================================

class TestTransferScoreSignal:
    """Verify TransferScorer.score_signal() sends correct payload and parses result."""

    def test_score_signal_with_dict(self, client: KongenClient, sample_signature_dict):
        api_response = _make_response(
            200,
            {
                "classification": "BOUNDARY_FORMATION",
                "confidence": 0.94,
                "confidence_adj": 0.25,
                "evidence": [
                    {"domain": "genetics", "pattern_type": "BOUNDARY_FORMATION", "similarity": 0.97}
                ],
                "tokens_used": 5,
                "tokens_remaining": 995,
            },
        )
        with patch.object(client._http, "request", return_value=api_response) as mock_req:
            result = client.transfer.score_signal(sample_signature_dict)

        assert isinstance(result, TransferResult)
        assert result.classification == "BOUNDARY_FORMATION"
        assert result.confidence == 0.94
        assert result.confidence_adj == 0.25
        assert len(result.evidence) == 1
        assert result.tokens_used == 5

        payload = mock_req.call_args[1]["json"]
        assert payload["signature"]["complexity"] == 0.7

    def test_score_signal_with_structural_signature(self, client: KongenClient, sample_signature_dict):
        sig = StructuralSignature(**sample_signature_dict)
        api_response = _make_response(
            200,
            {
                "classification": "ACTIVATOR_DOMINANCE",
                "confidence": 0.85,
                "confidence_adj": 0.10,
                "evidence": [],
                "tokens_used": 5,
                "tokens_remaining": 990,
            },
        )
        with patch.object(client._http, "request", return_value=api_response) as mock_req:
            result = client.transfer.score_signal(sig, source_domain="capital")

        assert result.classification == "ACTIVATOR_DOMINANCE"
        payload = mock_req.call_args[1]["json"]
        assert payload["source_domain"] == "capital"
        assert payload["signature"]["constraint"] == 0.3

    def test_score_signal_without_source_domain(self, client: KongenClient, sample_signature_dict):
        api_response = _make_response(
            200,
            {
                "classification": "MIXED_FLEXIBILITY",
                "confidence": 0.70,
                "confidence_adj": 0.05,
                "tokens_used": 5,
                "tokens_remaining": 800,
            },
        )
        with patch.object(client._http, "request", return_value=api_response) as mock_req:
            client.transfer.score_signal(sample_signature_dict)

        payload = mock_req.call_args[1]["json"]
        assert "source_domain" not in payload

    def test_score_signal_invalid_dict_raises_validation_error(self, client: KongenClient):
        with pytest.raises(ValidationError):
            client.transfer.score_signal({"complexity": 0.5})  # missing fields

    def test_transfer_scorer_lazy_initialized(self, client: KongenClient):
        assert client._transfer is None
        _ = client.transfer
        assert client._transfer is not None
        assert isinstance(client._transfer, TransferScorer)

    def test_transfer_repr(self, client: KongenClient):
        assert repr(client.transfer) == "TransferScorer()"


# ===========================================================================
# 7. TransferScorer.score_batch()
# ===========================================================================

class TestTransferScoreBatch:
    """Verify score_batch() sends list of signatures and parses batch result."""

    def test_score_batch_returns_batch_result(self, client: KongenClient, sample_signature_dict):
        sig2 = dict(sample_signature_dict)
        sig2["complexity"] = 0.2
        sig2["constraint"] = 0.8
        sig2["balance"] = 0.25

        api_response = _make_response(
            200,
            {
                "results": [
                    {
                        "classification": "BOUNDARY_FORMATION",
                        "confidence": 0.94,
                        "confidence_adj": 0.25,
                        "evidence": [],
                        "tokens_used": 4,
                        "tokens_remaining": 992,
                    },
                    {
                        "classification": "INHIBITOR_CONSOLIDATION",
                        "confidence": 0.88,
                        "confidence_adj": 0.18,
                        "evidence": [],
                        "tokens_used": 4,
                        "tokens_remaining": 988,
                    },
                ],
                "total_tokens_used": 8,
                "tokens_remaining": 988,
            },
        )
        with patch.object(client._http, "request", return_value=api_response) as mock_req:
            batch = client.transfer.score_batch(
                [sample_signature_dict, sig2],
                source_domain="genetics",
            )

        assert isinstance(batch, BatchTransferResult)
        assert len(batch.results) == 2
        assert batch.results[0].classification == "BOUNDARY_FORMATION"
        assert batch.results[1].classification == "INHIBITOR_CONSOLIDATION"
        assert batch.total_tokens_used == 8
        assert batch.tokens_remaining == 988

        # Verify correct endpoint
        call_args = mock_req.call_args
        assert call_args[0] == ("POST", "/v1/transfer/score_batch")
        payload = call_args[1]["json"]
        assert len(payload["signatures"]) == 2
        assert payload["source_domain"] == "genetics"

    def test_score_batch_without_source_domain(self, client: KongenClient, sample_signature_dict):
        api_response = _make_response(
            200,
            {
                "results": [
                    {
                        "classification": "SCALE_COHERENCE",
                        "confidence": 0.75,
                        "confidence_adj": 0.10,
                        "evidence": [],
                        "tokens_used": 4,
                        "tokens_remaining": 100,
                    },
                ],
                "total_tokens_used": 4,
                "tokens_remaining": 100,
            },
        )
        with patch.object(client._http, "request", return_value=api_response) as mock_req:
            client.transfer.score_batch([sample_signature_dict])

        payload = mock_req.call_args[1]["json"]
        assert "source_domain" not in payload

    def test_score_batch_with_structural_signature_objects(
        self, client: KongenClient, sample_signature_dict
    ):
        sig = StructuralSignature(**sample_signature_dict)
        api_response = _make_response(
            200,
            {
                "results": [
                    {
                        "classification": "COMPETITIVE_SELECTION",
                        "confidence": 0.60,
                        "confidence_adj": 0.02,
                        "evidence": [],
                        "tokens_used": 4,
                        "tokens_remaining": 50,
                    },
                ],
                "total_tokens_used": 4,
                "tokens_remaining": 50,
            },
        )
        with patch.object(client._http, "request", return_value=api_response):
            batch = client.transfer.score_batch([sig])

        assert batch.results[0].classification == "COMPETITIVE_SELECTION"


# ===========================================================================
# 8. StructuralSignature Model
# ===========================================================================

class TestStructuralSignature:
    """Verify StructuralSignature validation and serialization."""

    def test_valid_signature(self, sample_signature_dict):
        sig = StructuralSignature(**sample_signature_dict)
        assert sig.complexity == 0.7
        assert sig.constraint == 0.3
        assert sig.boundary == 0.8
        assert sig.coherence == 0.6
        assert sig.magnitude == 1.5
        assert sig.balance == 2.33
        assert sig.gradient == 0.5

    def test_model_dump_roundtrip(self, sample_signature_dict):
        sig = StructuralSignature(**sample_signature_dict)
        dumped = sig.model_dump()
        reconstructed = StructuralSignature(**dumped)
        assert reconstructed == sig

    def test_model_validate_from_dict(self, sample_signature_dict):
        sig = StructuralSignature.model_validate(sample_signature_dict)
        assert sig.complexity == 0.7

    def test_missing_field_raises_validation_error(self):
        with pytest.raises(ValidationError) as exc_info:
            StructuralSignature(
                complexity=0.5,
                constraint=0.5,
                # missing boundary, coherence, etc.
            )
        errors = exc_info.value.errors()
        missing_fields = {e["loc"][0] for e in errors if e["type"] == "missing"}
        assert "boundary" in missing_fields
        assert "coherence" in missing_fields
        assert "magnitude" in missing_fields
        assert "balance" in missing_fields
        assert "gradient" in missing_fields

    def test_all_fields_are_required(self):
        with pytest.raises(ValidationError):
            StructuralSignature()

    def test_negative_values_are_accepted(self):
        """StructuralSignature does not constrain value ranges (allows negatives)."""
        sig = StructuralSignature(
            complexity=-0.1,
            constraint=-0.2,
            boundary=-0.3,
            coherence=-0.4,
            magnitude=-0.5,
            balance=-0.6,
            gradient=-0.7,
        )
        assert sig.complexity == -0.1

    def test_model_dump_keys(self, sample_signature_dict):
        sig = StructuralSignature(**sample_signature_dict)
        keys = set(sig.model_dump().keys())
        assert keys == {
            "complexity",
            "constraint",
            "boundary",
            "coherence",
            "magnitude",
            "balance",
            "gradient",
        }


# ===========================================================================
# 9. Exception Hierarchy
# ===========================================================================

class TestExceptionHierarchy:
    """Verify all exceptions inherit from KongenError and format correctly."""

    def test_all_exceptions_inherit_from_kongen_error(self):
        assert issubclass(AuthenticationError, KongenError)
        assert issubclass(RateLimitError, KongenError)
        assert issubclass(TokensExhaustedError, KongenError)
        assert issubclass(APIError, KongenError)

    def test_kongen_error_inherits_from_exception(self):
        assert issubclass(KongenError, Exception)

    def test_kongen_error_message(self):
        err = KongenError("Something went wrong")
        assert err.message == "Something went wrong"
        assert str(err) == "Something went wrong"

    def test_authentication_error_default_message(self):
        err = AuthenticationError()
        assert "Invalid or missing API key" in err.message

    def test_authentication_error_custom_message(self):
        err = AuthenticationError("Key expired at midnight")
        assert err.message == "Key expired at midnight"

    def test_rate_limit_error_without_retry_after(self):
        err = RateLimitError()
        assert err.retry_after is None
        assert err.message == "Rate limit exceeded."

    def test_rate_limit_error_with_retry_after(self):
        err = RateLimitError(retry_after=2.5)
        assert err.retry_after == 2.5
        assert "2.5s" in err.message

    def test_rate_limit_error_retry_after_formatting(self):
        err = RateLimitError(retry_after=10.0)
        assert "10.0s" in str(err)

    def test_tokens_exhausted_default_message(self):
        err = TokensExhaustedError()
        assert "No Kongen Tokens remaining" in err.message
        assert "Upgrade" in err.message

    def test_tokens_exhausted_custom_message(self):
        err = TokensExhaustedError("Buy more tokens please")
        assert err.message == "Buy more tokens please"

    def test_api_error_full_message(self):
        err = APIError(
            message="API request failed.",
            status_code=503,
            detail="Service unavailable",
        )
        assert err.status_code == 503
        assert err.detail == "Service unavailable"
        assert "503" in str(err)
        assert "Service unavailable" in str(err)

    def test_api_error_without_detail(self):
        err = APIError(message="Failed.", status_code=500)
        assert err.detail is None
        assert "500" in str(err)

    def test_api_error_without_status_code(self):
        err = APIError(message="Unknown error.")
        assert err.status_code is None
        assert err.detail is None

    def test_exceptions_can_be_caught_as_kongen_error(self):
        """All SDK exceptions can be caught with a single except KongenError."""
        for exc_class in (AuthenticationError, RateLimitError, TokensExhaustedError, APIError):
            try:
                if exc_class == APIError:
                    raise exc_class(message="test", status_code=500)
                else:
                    raise exc_class()
            except KongenError:
                pass  # Should be caught here


# ===========================================================================
# 10. Context Manager
# ===========================================================================

class TestContextManager:
    """Verify KongenClient works as a context manager."""

    def test_enter_returns_client(self, api_key: str):
        client = KongenClient(api_key=api_key)
        returned = client.__enter__()
        assert returned is client
        client.close()

    def test_exit_closes_http_client(self, api_key: str):
        client = KongenClient(api_key=api_key)
        with patch.object(client._http, "close") as mock_close:
            client.__exit__(None, None, None)
        mock_close.assert_called_once()

    def test_with_statement(self, api_key: str):
        with KongenClient(api_key=api_key) as client:
            assert isinstance(client, KongenClient)
            response = _make_response(200, {"status": "ok"})
            with patch.object(client._http, "request", return_value=response):
                result = client._request("GET", "/v1/health")
                assert result == {"status": "ok"}

    def test_with_statement_closes_on_exception(self, api_key: str):
        close_called = False
        original_close = KongenClient.close

        def tracking_close(self_):
            nonlocal close_called
            close_called = True
            original_close(self_)

        with patch.object(KongenClient, "close", tracking_close):
            try:
                with KongenClient(api_key=api_key) as client:
                    raise ValueError("Intentional error")
            except ValueError:
                pass

        assert close_called


# ===========================================================================
# 11. Additional Edge Cases
# ===========================================================================

class TestEdgeCases:
    """Additional edge-case and integration-style tests."""

    def test_parse_retry_after_valid(self, client: KongenClient):
        resp = _make_response(429, {}, {"Retry-After": "3.5"})
        assert client._parse_retry_after(resp) == 3.5

    def test_parse_retry_after_invalid(self, client: KongenClient):
        resp = _make_response(429, {}, {"Retry-After": "not-a-number"})
        assert client._parse_retry_after(resp) is None

    def test_parse_retry_after_missing(self, client: KongenClient):
        resp = _make_response(429, {})
        assert client._parse_retry_after(resp) is None

    def test_logic_result_model(self):
        result = LogicResult(
            regime="synthesis",
            confidence_adj=0.22,
            tokens_used=1,
            tokens_remaining=499,
        )
        assert result.regime == "synthesis"
        assert result.confidence_adj == 0.22

    def test_transfer_result_confidence_bounds(self):
        """Confidence field has ge=0.0, le=1.0 constraint."""
        with pytest.raises(ValidationError):
            TransferResult(
                classification="BOUNDARY_FORMATION",
                confidence=1.5,  # out of bounds
                confidence_adj=0.1,
                tokens_used=5,
                tokens_remaining=100,
            )

        with pytest.raises(ValidationError):
            TransferResult(
                classification="BOUNDARY_FORMATION",
                confidence=-0.1,  # out of bounds
                confidence_adj=0.1,
                tokens_used=5,
                tokens_remaining=100,
            )

    def test_transfer_result_default_evidence(self):
        result = TransferResult(
            classification="SCALE_COHERENCE",
            confidence=0.8,
            confidence_adj=0.1,
            tokens_used=5,
            tokens_remaining=100,
        )
        assert result.evidence == []

    def test_batch_transfer_result_empty_results(self):
        batch = BatchTransferResult(
            results=[],
            total_tokens_used=0,
            tokens_remaining=1000,
        )
        assert len(batch.results) == 0

    def test_token_usage_model(self):
        usage = TokenUsage(used=50, remaining=950, plan="enterprise")
        assert usage.used == 50
        assert usage.remaining == 950
        assert usage.plan == "enterprise"

    def test_normalize_signature_validates_dict_keys(self, client: KongenClient):
        """TransferScorer._normalize_signature validates even plain dicts."""
        with pytest.raises(ValidationError):
            TransferScorer._normalize_signature({"bad_key": 1.0})

    def test_sub_clients_are_cached(self, client: KongenClient):
        """Accessing .logic and .transfer twice returns the same instance."""
        logic1 = client.logic
        logic2 = client.logic
        assert logic1 is logic2

        transfer1 = client.transfer
        transfer2 = client.transfer
        assert transfer1 is transfer2

    def test_client_headers_include_auth_and_user_agent(self, api_key: str):
        client = KongenClient(api_key=api_key)
        headers = client._http.headers
        assert headers["x-api-key"] == api_key
        assert "kongen-python" in headers["user-agent"]
        assert headers["content-type"] == "application/json"
        client.close()
