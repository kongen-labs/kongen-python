"""Types for organism creation and lifecycle management.

These types define the public vocabulary for organism creators:
growth, constraint, boundary, stability, coherence, diffusion, intent.
The semantic role system uses domain-agnostic language; internal field
mechanics are never exposed in the public schema.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Semantic roles -- the public-facing vocabulary
# ---------------------------------------------------------------------------


class FieldRole(str, Enum):
    """Semantic roles a creator assigns to their domain fields.

    These map internally to UAF fields, but the creator only sees
    generic descriptions. The mapping logic is server-side IP.
    """

    GROWTH = "growth"
    """Forces that expand, activate, or increase signal strength.
    Examples: gene expression level, assertion count, buy pressure,
    temperature rise, population growth."""

    CONSTRAINT = "constraint"
    """Forces that limit, inhibit, or resist growth.
    Examples: methylation level, hedging density, sell pressure,
    friction coefficient, regulatory burden."""

    BOUNDARY = "boundary"
    """Indicators of sharp transitions, edges, or regime boundaries.
    Examples: tissue specificity, topic shift score, support/resistance,
    phase transition temperature, territorial border."""

    STABILITY = "stability"
    """How persistent or consistent a pattern is over time.
    Examples: expression variance inverse, argument consistency,
    price volatility inverse, structural integrity."""

    COHERENCE = "coherence"
    """Multi-scale consistency -- does the pattern hold at different
    zoom levels or timescales.
    Examples: lexical diversity, fractal dimension, cross-timeframe
    agreement, hierarchical consistency."""

    DIFFUSION = "diffusion"
    """How fast patterns spread through the system.
    Examples: information propagation rate, contagion speed,
    heat transfer rate, viral coefficient."""

    INTENT = "intent"
    """Purposeful or directed energy in the system.
    Examples: assertive + consistent communication, directed capital
    flow, targeted gene regulation, goal-directed locomotion."""


class NormalizationHint(str, Enum):
    """Hints about how a field's values should be normalized.

    Helps the server-side mapper scale raw domain values into
    the [0, 1] or ratio space without the creator knowing the
    target range.
    """

    ZERO_TO_ONE = "zero_to_one"
    """Values are already in [0, 1] range."""

    POSITIVE_UNBOUNDED = "positive_unbounded"
    """Values are positive but unbounded (e.g., price, expression level).
    Will be log-normalized."""

    SYMMETRIC = "symmetric"
    """Values center around zero (e.g., returns, sentiment score).
    Will be sigmoid-normalized."""

    RATIO = "ratio"
    """Values represent a ratio (e.g., win_rate, percentage).
    Will be clipped to [0, 1]."""

    COUNT = "count"
    """Integer counts (e.g., observation_count, sample_size).
    Will be log-normalized."""

    AUTO = "auto"
    """Let the server auto-detect normalization from data distribution."""


# ---------------------------------------------------------------------------
# Field schema definition
# ---------------------------------------------------------------------------


class FieldSpec(BaseModel):
    """Specification for a single field in the creator's data schema.

    The creator describes what this field represents and its semantic
    role. The server uses this to auto-map to internal representations.
    """

    role: FieldRole = Field(
        ...,
        description="Semantic role of this field in the domain.",
    )
    description: Optional[str] = Field(
        None,
        max_length=200,
        description="Optional human-readable description of the field.",
    )
    normalization: NormalizationHint = Field(
        NormalizationHint.AUTO,
        description="Hint about the value distribution for normalization.",
    )
    weight: float = Field(
        1.0,
        ge=0.0,
        le=10.0,
        description=(
            "Relative importance of this field within its role group. "
            "If multiple fields share a role, their weighted average "
            "determines the composite score. Default 1.0."
        ),
    )


# ---------------------------------------------------------------------------
# Organism configuration
# ---------------------------------------------------------------------------


class OrganismConfig(BaseModel):
    """Full configuration for creating an organism.

    Level 0: Just name + domain (observe with raw arrays).
    Level 1: Add schema with FieldSpec mappings (observe with dicts).
    Level 2: Add outcomes + webhook for full collective participation.
    """

    name: str = Field(
        ...,
        min_length=1,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_-]*$",
        description="Organism name (lowercase, alphanumeric + hyphens/underscores).",
    )
    domain: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Domain this organism operates in (e.g., 'meteorology', 'logistics').",
    )
    description: Optional[str] = Field(
        None,
        max_length=500,
        description="Optional description of what this organism does.",
    )
    version: str = Field(
        "0.1.0",
        description="Organism version (semver recommended).",
    )

    # Level 1: Schema
    schema_fields: Optional[dict[str, FieldSpec]] = Field(
        None,
        description=(
            "Named field specifications mapping field names to semantic roles. "
            "If None, the organism operates in Level 0 (raw array) mode."
        ),
    )

    # Level 2: Outcomes + feedback
    outcomes_enabled: bool = Field(
        False,
        description="Whether this organism will report pattern outcomes.",
    )
    analogies_webhook: Optional[str] = Field(
        None,
        description=(
            "Webhook URL for receiving cross-domain analogy notifications. "
            "When a structurally similar pattern in another domain has an "
            "outcome, this webhook is called with the analogy details."
        ),
    )


# ---------------------------------------------------------------------------
# Observation and response types
# ---------------------------------------------------------------------------


class ObservationResult(BaseModel):
    """Result from observing a pattern through the organism."""

    pattern_id: str = Field(
        ..., description="Unique pattern identifier assigned by the API."
    )
    classification: str = Field(
        ..., description="Universal pattern classification (opaque label)."
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Classification confidence."
    )
    confidence_adj: float = Field(
        ..., description="Confidence adjustment from cross-domain evidence."
    )
    evidence: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Supporting evidence from other domains.",
    )
    tokens_used: int = Field(..., description="Kongen Tokens consumed.")
    tokens_remaining: int = Field(..., description="Kongen Tokens remaining.")
    request_id: str = Field(default="", description="Request identifier.")


class OutcomeResult(BaseModel):
    """Result from reporting a pattern outcome."""

    pattern_id: str = Field(..., description="Pattern this outcome applies to.")
    accepted: bool = Field(..., description="Whether the outcome was accepted.")
    transfer_impact: Optional[float] = Field(
        None,
        description=(
            "How much this outcome influenced cross-domain scoring. "
            "None if the pattern has not been matched cross-domain."
        ),
    )
    tokens_used: int = Field(..., description="Kongen Tokens consumed.")
    tokens_remaining: int = Field(..., description="Kongen Tokens remaining.")


class AnalogyNotification(BaseModel):
    """Cross-domain analogy notification delivered via webhook.

    The creator receives this when a pattern in another domain
    that structurally matches one of their patterns has an outcome.
    Domain identifiers are opaque -- creators never learn which
    specific domain the analogy came from.
    """

    analogy_id: str = Field(..., description="Unique analogy identifier.")
    your_pattern_id: str = Field(
        ..., description="Your pattern that was matched."
    )
    source_domain_id: str = Field(
        ..., description="Opaque identifier for the source domain (e.g., 'd3')."
    )
    similarity: float = Field(
        ..., ge=0.0, le=1.0, description="Structural similarity score."
    )
    source_outcome: str = Field(
        ..., description="Outcome in the source domain: 'positive' or 'negative'."
    )
    source_magnitude: float = Field(
        ..., description="Magnitude of the source outcome (normalized 0-1)."
    )
    suggested_adj: float = Field(
        ...,
        description="Suggested confidence adjustment for your pattern [-0.2, +0.3].",
    )
    timestamp: str = Field(..., description="ISO 8601 timestamp.")


class OrganismRegistration(BaseModel):
    """Response from organism registration."""

    organism_id: str = Field(..., description="Unique organism identifier.")
    name: str = Field(..., description="Organism name.")
    domain: str = Field(..., description="Domain.")
    level: int = Field(
        ...,
        description="Operating level: 0 (raw), 1 (schema), 2 (full collective).",
    )
    vector_size: int = Field(
        ...,
        description=(
            "Expected observation vector size. For Level 0 this is 7. "
            "For Level 1+ this equals the number of schema fields."
        ),
    )
    api_key_scoped: bool = Field(
        ...,
        description="Whether this organism is scoped to the creating API key.",
    )
    tokens_used: int = Field(..., description="Kongen Tokens consumed.")
    tokens_remaining: int = Field(..., description="Kongen Tokens remaining.")


class OrganismHealthReport(BaseModel):
    """Health report for a registered organism."""

    organism_id: str = Field(..., description="Organism identifier.")
    status: str = Field(
        ...,
        description="Status: 'active', 'idle', 'degraded'.",
    )
    total_observations: int = Field(..., description="Total observations submitted.")
    total_outcomes: int = Field(..., description="Total outcomes reported.")
    classification_distribution: dict[str, int] = Field(
        default_factory=dict,
        description="Count of each pattern classification observed.",
    )
    cross_domain_matches: int = Field(
        0,
        description="Number of cross-domain analogy matches found.",
    )
    last_observation_at: Optional[str] = Field(
        None, description="ISO 8601 timestamp of last observation."
    )
