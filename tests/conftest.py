"""Shared fixtures for Kongen SDK tests."""

from __future__ import annotations

import pytest

from kongen.client import KongenClient


@pytest.fixture()
def api_key() -> str:
    """A dummy API key for testing (never hits real API)."""
    return "kl_test_abc123"


@pytest.fixture()
def client(api_key: str) -> KongenClient:
    """A KongenClient with a dummy key. Close after each test."""
    c = KongenClient(api_key=api_key, base_url="https://test.kongenlabs.life")
    yield c
    c.close()


@pytest.fixture()
def sample_signature_dict() -> dict[str, float]:
    """A valid structural signature as a plain dict."""
    return {
        "complexity": 0.7,
        "constraint": 0.3,
        "boundary": 0.8,
        "coherence": 0.6,
        "magnitude": 1.5,
        "balance": 2.33,
        "gradient": 0.5,
    }
