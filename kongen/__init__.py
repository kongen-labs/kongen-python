"""Kongen Labs SCI Pattern Intelligence SDK.

Pattern transfer scoring and LLM reasoning regime detection.

Usage::

    from kongen import KongenClient

    client = KongenClient(api_key="kl_live_...")
    result = client.logic.score("Prove sqrt(2) is irrational")
    print(result.regime, result.confidence_adj)
"""

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
from kongen.organism import Organism, OrganismManager
from kongen.organism_types import (
    FieldRole,
    FieldSpec,
    NormalizationHint,
    OrganismConfig,
    ObservationResult,
    OutcomeResult,
    AnalogyNotification,
    OrganismRegistration,
    OrganismHealthReport,
)
from kongen.types import (
    BatchTransferResult,
    LogicResult,
    StructuralSignature,
    TokenUsage,
    TransferResult,
)

__version__ = "1.1.0"
__all__ = [
    # Core client
    "KongenClient",
    # Sub-clients
    "LogicAnalyzer",
    "TransferScorer",
    "Organism",
    "OrganismManager",
    # Organism types
    "FieldRole",
    "FieldSpec",
    "NormalizationHint",
    "OrganismConfig",
    "ObservationResult",
    "OutcomeResult",
    "AnalogyNotification",
    "OrganismRegistration",
    "OrganismHealthReport",
    # Types
    "LogicResult",
    "TransferResult",
    "BatchTransferResult",
    "StructuralSignature",
    "TokenUsage",
    # Exceptions
    "KongenError",
    "AuthenticationError",
    "RateLimitError",
    "TokensExhaustedError",
    "APIError",
]
