"""Organism creation and lifecycle management.

Create domain-specific organisms that participate in cross-domain
pattern intelligence via the Kongen API. All pattern classification,
transfer scoring, and collective coordination happens server-side.

Three operating levels:

    Level 0 (Raw): Push 7-element float arrays. Minimal setup.
    Level 1 (Schema): Define named fields with semantic roles.
                      Push observations as dicts. Auto-normalized.
    Level 2 (Full): Schema + outcome reporting + analogy webhook.
                    Full participation in the pattern collective.

Usage::

    from kongen import KongenClient

    client = KongenClient(api_key="kl_live_...")

    # Level 0: Raw arrays
    org = client.organisms.create(name="weather-ryu", domain="meteorology")
    result = org.observe([0.7, 0.3, 0.8, 0.6, 1.5, 2.33, 0.5])

    # Level 1: Named fields with semantic roles
    org = client.organisms.create(
        name="weather-ryu",
        domain="meteorology",
        schema={
            "temperature": {"role": "growth"},
            "pressure":    {"role": "constraint"},
            "humidity":    {"role": "boundary"},
            "wind_var":    {"role": "stability"},
        },
    )
    result = org.observe({
        "temperature": 28.5,
        "pressure": 1013.25,
        "humidity": 0.65,
        "wind_var": 0.8,
    })
    print(result.classification, result.confidence)

    # Level 2: Full collective participation
    org = client.organisms.create(
        name="weather-ryu",
        domain="meteorology",
        schema={...},
        outcomes_enabled=True,
        analogies_webhook="https://my-api.com/kongen-callback",
    )
    result = org.observe({...})
    org.report_outcome(
        pattern_id=result.pattern_id,
        success=True,
        magnitude=0.85,
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Union

from kongen.organism_types import (
    AnalogyNotification,
    FieldRole,
    FieldSpec,
    NormalizationHint,
    ObservationResult,
    OrganismConfig,
    OrganismHealthReport,
    OrganismRegistration,
    OutcomeResult,
)

if TYPE_CHECKING:
    from kongen.client import KongenClient


# ---------------------------------------------------------------------------
# Organism instance -- returned by OrganismManager.create / .get
# ---------------------------------------------------------------------------


class Organism:
    """A registered organism that can observe patterns and report outcomes.

    Do not instantiate directly. Use :meth:`OrganismManager.create` or
    :meth:`OrganismManager.get`.
    """

    def __init__(
        self,
        client: KongenClient,
        registration: OrganismRegistration,
        config: OrganismConfig,
    ) -> None:
        self._client = client
        self._registration = registration
        self._config = config

    # -- Identity --------------------------------------------------------------

    @property
    def organism_id(self) -> str:
        """Server-assigned organism identifier."""
        return self._registration.organism_id

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def domain(self) -> str:
        return self._config.domain

    @property
    def level(self) -> int:
        """Operating level: 0 (raw), 1 (schema), 2 (full)."""
        return self._registration.level

    @property
    def vector_size(self) -> int:
        """Expected observation vector size."""
        return self._registration.vector_size

    # -- Observe ---------------------------------------------------------------

    def observe(
        self,
        data: Union[list[float], dict[str, float]],
        metadata: Optional[dict[str, Any]] = None,
    ) -> ObservationResult:
        """Submit an observation for pattern classification.

        Args:
            data: Observation data. For Level 0: a list of 7 floats.
                For Level 1+: a dict mapping field names to values,
                matching the schema defined at creation time.
            metadata: Optional metadata attached to this observation
                (not used for classification, stored for your reference).

        Returns:
            An :class:`~kongen.organism_types.ObservationResult` with
            pattern classification, confidence, and cross-domain evidence.

        Cost:
            50 Kongen Tokens (KT) per observation.

        Raises:
            ValueError: If the data shape does not match the organism's schema.
            APIError: On server errors.
            TokensExhaustedError: When out of tokens.

        Example::

            # Level 0
            result = org.observe([0.7, 0.3, 0.8, 0.6, 1.5, 2.33, 0.5])

            # Level 1
            result = org.observe({
                "temperature": 28.5,
                "pressure": 1013.25,
                "humidity": 0.65,
            })
        """
        payload = self._build_observe_payload(data, metadata)
        resp = self._client._request(
            "POST",
            f"/v1/organisms/{self.organism_id}/observe",
            json=payload,
        )
        return ObservationResult.model_validate(resp)

    def observe_batch(
        self,
        observations: list[Union[list[float], dict[str, float]]],
        metadata: Optional[list[dict[str, Any]]] = None,
    ) -> list[ObservationResult]:
        """Submit multiple observations in a single request.

        Args:
            observations: List of observation data (arrays or dicts).
            metadata: Optional per-observation metadata list.

        Returns:
            List of :class:`~kongen.organism_types.ObservationResult`.

        Cost:
            40 Kongen Tokens (KT) per observation (batch discount).
        """
        items = []
        for i, obs in enumerate(observations):
            meta = metadata[i] if metadata and i < len(metadata) else None
            items.append(self._build_observe_payload(obs, meta))

        resp = self._client._request(
            "POST",
            f"/v1/organisms/{self.organism_id}/observe_batch",
            json={"observations": items},
        )
        return [ObservationResult.model_validate(r) for r in resp["results"]]

    # -- Outcomes (Level 2) ----------------------------------------------------

    def report_outcome(
        self,
        pattern_id: str,
        success: bool,
        magnitude: float = 0.0,
        exit_reason: str = "manual",
        metadata: Optional[dict[str, Any]] = None,
    ) -> OutcomeResult:
        """Report the outcome of acting on a pattern.

        Only available for Level 2 organisms (outcomes_enabled=True).
        Outcomes feed back into the collective, improving cross-domain
        transfer accuracy for all participants.

        Args:
            pattern_id: The pattern_id from an ObservationResult.
            success: Whether the outcome was favorable.
            magnitude: Size of the outcome (0-1 normalized).
                0.0 = neutral, 1.0 = maximum positive/negative.
            exit_reason: Why the outcome ended. Free-form string for
                your own tracking. Not used in classification.
            metadata: Optional metadata for your reference.

        Returns:
            An :class:`~kongen.organism_types.OutcomeResult` confirming
            the outcome was recorded and its cross-domain impact.

        Cost:
            5 Kongen Tokens (KT) per outcome report.

        Raises:
            ValueError: If outcomes are not enabled for this organism.
        """
        if not self._config.outcomes_enabled:
            raise ValueError(
                f"Outcomes not enabled for organism '{self.name}'. "
                "Create with outcomes_enabled=True to report outcomes."
            )

        payload: dict[str, Any] = {
            "pattern_id": pattern_id,
            "success": success,
            "magnitude": magnitude,
            "exit_reason": exit_reason,
        }
        if metadata:
            payload["metadata"] = metadata

        resp = self._client._request(
            "POST",
            f"/v1/organisms/{self.organism_id}/outcomes",
            json=payload,
        )
        return OutcomeResult.model_validate(resp)

    # -- Health & Stats --------------------------------------------------------

    def health(self) -> OrganismHealthReport:
        """Get organism health and statistics.

        Returns:
            An :class:`~kongen.organism_types.OrganismHealthReport` with
            observation counts, classification distribution, and status.

        Cost:
            1 Kongen Token (KT).
        """
        resp = self._client._request(
            "GET",
            f"/v1/organisms/{self.organism_id}/health",
        )
        return OrganismHealthReport.model_validate(resp)

    # -- Analogies (Level 2, pull-based alternative to webhook) ----------------

    def get_analogies(
        self,
        since: Optional[str] = None,
        limit: int = 20,
    ) -> list[AnalogyNotification]:
        """Poll for cross-domain analogy notifications.

        Alternative to the webhook for creators who prefer polling.
        Only returns analogies for Level 2 organisms.

        Args:
            since: ISO 8601 timestamp. Only return analogies after this time.
            limit: Maximum number of analogies to return (1-100).

        Returns:
            List of :class:`~kongen.organism_types.AnalogyNotification`.

        Cost:
            1 Kongen Token (KT) per poll.
        """
        params: dict[str, Any] = {"limit": min(max(limit, 1), 100)}
        if since:
            params["since"] = since

        resp = self._client._request(
            "GET",
            f"/v1/organisms/{self.organism_id}/analogies",
            params=params,
        )
        return [AnalogyNotification.model_validate(a) for a in resp["analogies"]]

    # -- Lifecycle -------------------------------------------------------------

    def update(
        self,
        description: Optional[str] = None,
        version: Optional[str] = None,
        analogies_webhook: Optional[str] = None,
    ) -> None:
        """Update organism metadata.

        Schema and name cannot be changed after creation.
        Create a new organism version instead.

        Args:
            description: Updated description.
            version: Updated version string.
            analogies_webhook: Updated webhook URL (or None to remove).
        """
        payload: dict[str, Any] = {}
        if description is not None:
            payload["description"] = description
        if version is not None:
            payload["version"] = version
        if analogies_webhook is not None:
            payload["analogies_webhook"] = analogies_webhook

        if payload:
            self._client._request(
                "PATCH",
                f"/v1/organisms/{self.organism_id}",
                json=payload,
            )
            # Update local config
            if description is not None:
                self._config.description = description
            if version is not None:
                self._config.version = version
            if analogies_webhook is not None:
                self._config.analogies_webhook = analogies_webhook

    def delete(self) -> None:
        """Delete this organism and all its stored patterns.

        This action is irreversible. All observation history, outcomes,
        and analogy data are permanently removed.
        """
        self._client._request(
            "DELETE",
            f"/v1/organisms/{self.organism_id}",
        )

    # -- Internal --------------------------------------------------------------

    def _build_observe_payload(
        self,
        data: Union[list[float], dict[str, float]],
        metadata: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        """Validate and serialize observation data."""
        payload: dict[str, Any] = {}

        if isinstance(data, list):
            # Level 0: raw float array
            if self._config.schema_fields and self.level > 0:
                raise ValueError(
                    f"Organism '{self.name}' has a schema (Level {self.level}). "
                    "Pass a dict with field names, not a list."
                )
            if len(data) != self.vector_size:
                raise ValueError(
                    f"Expected {self.vector_size} values, got {len(data)}."
                )
            payload["values"] = data

        elif isinstance(data, dict):
            # Level 1+: named fields
            if not self._config.schema_fields:
                raise ValueError(
                    f"Organism '{self.name}' has no schema (Level 0). "
                    "Pass a list of {self.vector_size} floats, not a dict."
                )
            # Validate all schema fields are present
            missing = set(self._config.schema_fields.keys()) - set(data.keys())
            if missing:
                raise ValueError(
                    f"Missing fields: {', '.join(sorted(missing))}. "
                    f"Schema requires: {', '.join(sorted(self._config.schema_fields.keys()))}."
                )
            extra = set(data.keys()) - set(self._config.schema_fields.keys())
            if extra:
                raise ValueError(
                    f"Unknown fields: {', '.join(sorted(extra))}. "
                    f"Schema accepts: {', '.join(sorted(self._config.schema_fields.keys()))}."
                )
            payload["fields"] = data

        else:
            raise TypeError(
                f"Expected list[float] or dict[str, float], got {type(data).__name__}."
            )

        if metadata:
            payload["metadata"] = metadata

        return payload

    def __repr__(self) -> str:
        return (
            f"Organism(name={self.name!r}, domain={self.domain!r}, "
            f"level={self.level}, id={self.organism_id!r})"
        )


# ---------------------------------------------------------------------------
# OrganismManager -- accessed via client.organisms
# ---------------------------------------------------------------------------


class OrganismManager:
    """Manage organisms via the Kongen API.

    Access this via :attr:`KongenClient.organisms`.

    Example::

        org = client.organisms.create(
            name="weather-ryu",
            domain="meteorology",
            schema={"temp": {"role": "growth"}, "pressure": {"role": "constraint"}},
        )

        # Later, reconnect to the same organism
        org = client.organisms.get("weather-ryu")
    """

    def __init__(self, client: KongenClient) -> None:
        self._client = client

    def create(
        self,
        name: str,
        domain: str,
        description: Optional[str] = None,
        version: str = "0.1.0",
        schema: Optional[dict[str, Union[dict[str, Any], FieldSpec]]] = None,
        outcomes_enabled: bool = False,
        analogies_webhook: Optional[str] = None,
    ) -> Organism:
        """Register a new organism with the Kongen API.

        Args:
            name: Organism name (lowercase, alphanumeric + hyphens).
            domain: Domain this organism operates in.
            description: Optional description.
            version: Version string (semver recommended).
            schema: Optional field schema mapping field names to
                :class:`~kongen.organism_types.FieldSpec` or plain dicts
                with at minimum a ``role`` key. Enables Level 1+ mode.
            outcomes_enabled: Enable outcome reporting (Level 2).
            analogies_webhook: Webhook URL for cross-domain analogies (Level 2).

        Returns:
            An :class:`Organism` instance ready for observations.

        Cost:
            10 Kongen Tokens (KT) for registration.

        Example::

            # Level 0 -- minimal
            org = client.organisms.create(name="my-org", domain="finance")

            # Level 1 -- with schema
            org = client.organisms.create(
                name="my-org",
                domain="finance",
                schema={
                    "revenue":  {"role": "growth", "normalization": "positive_unbounded"},
                    "debt":     {"role": "constraint", "normalization": "positive_unbounded"},
                    "margin":   {"role": "boundary", "normalization": "ratio"},
                    "earnings_var": {"role": "stability", "normalization": "positive_unbounded"},
                },
            )

            # Level 2 -- full collective
            org = client.organisms.create(
                name="my-org",
                domain="finance",
                schema={...},
                outcomes_enabled=True,
                analogies_webhook="https://my-api.com/kongen-cb",
            )
        """
        # Normalize schema dicts to FieldSpec
        schema_fields: Optional[dict[str, FieldSpec]] = None
        if schema is not None:
            schema_fields = {}
            for field_name, spec in schema.items():
                if isinstance(spec, FieldSpec):
                    schema_fields[field_name] = spec
                elif isinstance(spec, dict):
                    schema_fields[field_name] = FieldSpec.model_validate(spec)
                else:
                    raise TypeError(
                        f"Schema field '{field_name}': expected dict or FieldSpec, "
                        f"got {type(spec).__name__}."
                    )

        config = OrganismConfig(
            name=name,
            domain=domain,
            description=description,
            version=version,
            schema_fields=schema_fields,
            outcomes_enabled=outcomes_enabled,
            analogies_webhook=analogies_webhook,
        )

        # Register with the API
        payload = config.model_dump(exclude_none=True)
        # Convert FieldSpec to serializable dicts
        if payload.get("schema_fields"):
            payload["schema_fields"] = {
                k: v.model_dump() if isinstance(v, FieldSpec) else v
                for k, v in payload["schema_fields"].items()
            }

        resp = self._client._request("POST", "/v1/organisms", json=payload)
        registration = OrganismRegistration.model_validate(resp)

        return Organism(self._client, registration, config)

    def get(self, name_or_id: str) -> Organism:
        """Retrieve an existing organism by name or ID.

        Args:
            name_or_id: Organism name or server-assigned ID.

        Returns:
            An :class:`Organism` instance.

        Cost:
            1 Kongen Token (KT).
        """
        resp = self._client._request("GET", f"/v1/organisms/{name_or_id}")

        registration = OrganismRegistration.model_validate(resp["registration"])

        # Reconstruct config from server response
        config_data = resp["config"]
        if config_data.get("schema_fields"):
            config_data["schema_fields"] = {
                k: FieldSpec.model_validate(v)
                for k, v in config_data["schema_fields"].items()
            }
        config = OrganismConfig.model_validate(config_data)

        return Organism(self._client, registration, config)

    def list(self) -> list[dict[str, Any]]:
        """List all organisms registered under the current API key.

        Returns:
            List of organism summaries (name, domain, level, status).

        Cost:
            1 Kongen Token (KT).
        """
        resp = self._client._request("GET", "/v1/organisms")
        return resp["organisms"]

    def __repr__(self) -> str:
        return "OrganismManager()"
