"""Pydantic models for Kongen API request and response types."""

from __future__ import annotations

from pydantic import BaseModel, Field


class StructuralSignature(BaseModel):
    """7-dimensional signal vector for pattern scoring."""

    complexity: float = Field(
        ..., description="Primary complexity signal."
    )
    constraint: float = Field(
        ..., description="Constraint signal."
    )
    boundary: float = Field(
        ..., description="Boundary signal."
    )
    coherence: float = Field(
        ..., description="Multi-scale coherence."
    )
    magnitude: float = Field(
        ..., description="Signal magnitude."
    )
    balance: float = Field(
        ..., description="Complexity-constraint balance."
    )
    gradient: float = Field(
        ..., description="Gradient signal."
    )


class LogicResult(BaseModel):
    """Result from Logic LLM reasoning regime detection."""

    regime: str = Field(
        ...,
        description=(
            "Detected reasoning regime: "
            "'trivial', 'fast', 'moderate', 'deep', or 'exhaustive'."
        ),
    )
    confidence_adj: float = Field(
        ..., description="Confidence adjustment from pattern analysis."
    )
    recommended_tokens: int = Field(
        default=0, description="Recommended max_tokens for this prompt's complexity."
    )
    tokens_used: int = Field(..., description="Kongen Tokens consumed by this call.")
    tokens_remaining: int = Field(
        ..., description="Kongen Tokens remaining in current billing period."
    )
    request_id: str = Field(
        default="", description="Unique request identifier."
    )
    watermark: str = Field(
        default="", description="Response watermark for provenance tracking."
    )


class TransferResult(BaseModel):
    """Result from pattern transfer scoring."""

    classification: str = Field(
        ..., description="Pattern classification.",
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Classification confidence."
    )
    confidence_adj: float = Field(
        ..., description="Confidence adjustment from pattern analysis."
    )
    evidence: list[dict[str, object]] = Field(
        default_factory=list,
        description="Supporting evidence from pattern domains.",
    )
    tokens_used: int = Field(..., description="Kongen Tokens consumed by this call.")
    tokens_remaining: int = Field(
        ..., description="Kongen Tokens remaining in current billing period."
    )
    request_id: str = Field(
        default="", description="Unique request identifier."
    )
    watermark: str = Field(
        default="", description="Response watermark for provenance tracking."
    )


class BatchTransferResult(BaseModel):
    """Result from batch pattern transfer scoring."""

    results: list[TransferResult] = Field(
        ..., description="Individual transfer results for each input signature."
    )
    total_tokens_used: int = Field(
        ..., description="Total Kongen Tokens consumed by the batch."
    )
    tokens_remaining: int = Field(
        ..., description="Kongen Tokens remaining in current billing period."
    )


class TokenUsage(BaseModel):
    """Current token usage and plan information."""

    used: int = Field(..., description="Kongen Tokens used in current billing period.")
    remaining: int = Field(
        ..., description="Kongen Tokens remaining in current billing period."
    )
    plan: str = Field(..., description="Active billing plan name.")
