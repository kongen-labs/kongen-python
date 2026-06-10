# Changelog

All notable changes to the Kongen SDK will be documented in this file.

## [1.1.0] - 2026-06-10

### Added

- **Organism creation** — build domain-specific organisms that participate in
  cross-domain pattern intelligence via `client.organisms`
  - `OrganismManager` — `create()`, `get()`, `list()`
  - `Organism` — `observe()`, `observe_batch()`, `report_outcome()`,
    `health()`, `get_analogies()`, `update()`, `delete()`
  - Three operating levels: L0 raw 7-element float vectors, L1 named fields
    with semantic roles (auto-normalized), L2 full collective with outcomes
    and analogy webhooks
  - 11 new public types, including `OrganismConfig`, `FieldSpec`, `FieldRole`,
    `ObservationResult`, `OutcomeResult`, `AnalogyNotification`,
    `OrganismHealthReport`

### Changed

- `ChiryuAnalyzer` renamed to `LogicAnalyzer` (`client.logic`) across the SDK,
  examples, and docs
- Signal vector field names use neutral semantic terminology
  (e.g. `complexity`, `constraint`) in the public API surface

### Infrastructure

- Releases now published from GitLab CI via PyPI trusted publishing (OIDC) —
  no long-lived tokens

### Fixed

- Docs: removed `client.mcp` examples — the MCP surface lives in the separate
  `kongen-mcp` server package and was never part of this SDK

## [1.0.0] - 2026-03-14

### Added

- `KongenClient` — main client with API key auth and token tracking
- `LogicAnalyzer` — LLM reasoning regime detection via `logic.score()`
  - 5 regimes: reflexive, procedural, analytical, synthesis, metacognitive
  - Local heuristic pre-scoring + remote API scoring
  - Cross-domain boost factor from UAF pattern corpus
- `TransferScorer` — cross-domain structural pattern scoring
  - `transfer.score_signal()` — single signature scoring (50 KT)
  - `transfer.score_batch()` — batch scoring at 40 KT/signal
  - Structural classification into 7 universal pattern types
- Token usage tracking via `client.token_usage`
- Error hierarchy: `KongenError`, `APIError`, `AuthenticationError`, `TokensExhaustedError`, `RateLimitError`
- Type-safe Pydantic response models
- Environment-based config: `KONGEN_API_KEY`, `KONGEN_API_BASE_URL`
