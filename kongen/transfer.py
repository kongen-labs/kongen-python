"""Pattern transfer scoring."""

from __future__ import annotations

from typing import TYPE_CHECKING

from kongen.types import BatchTransferResult, StructuralSignature, TransferResult

if TYPE_CHECKING:
    from kongen.client import KongenClient


class TransferScorer:
    """Score structural signatures against the pattern library.

    The transfer scorer compares a signal vector to reference
    patterns and returns a classification with supporting
    evidence and a confidence adjustment.

    This class is not instantiated directly. Access it via
    :attr:`KongenClient.transfer`.
    """

    def __init__(self, client: KongenClient) -> None:
        self._client = client

    def score_signal(
        self,
        signature: StructuralSignature | dict[str, float],
        source_domain: str | None = None,
    ) -> TransferResult:
        """Score a single structural signature against reference patterns.

        Args:
            signature: A :class:`~kongen.types.StructuralSignature` or a plain
                dict with the same keys (``complexity``,
                ``constraint``, etc.).
            source_domain: Optional hint for the originating domain.
                Excludes same-domain patterns from evidence to avoid
                self-reinforcement.

        Returns:
            A :class:`~kongen.types.TransferResult` with classification,
            confidence, confidence adjustment, and evidence.

        Cost:
            50 Kongen Tokens (KT) per call.

        Example::

            result = client.transfer.score_signal({
                "complexity": 0.7,
                "constraint": 0.3,
                "boundary": 0.8,
                "coherence": 0.6,
                "magnitude": 1.5,
                "balance": 2.33,
                "gradient": 0.5,
            })
            print(result.classification)  # "BOUNDARY_FORMATION"
            print(result.confidence)      # 0.94
        """
        sig_dict = self._normalize_signature(signature)

        payload: dict[str, object] = {"signature": sig_dict}
        if source_domain is not None:
            payload["source_domain"] = source_domain

        data = self._client._request("POST", "/v1/transfer/score", json=payload)
        return TransferResult.model_validate(data)

    def score_batch(
        self,
        signatures: list[StructuralSignature | dict[str, float]],
        source_domain: str | None = None,
    ) -> BatchTransferResult:
        """Score multiple structural signatures in a single request.

        More efficient than calling :meth:`score_signal` in a loop --
        batch scoring costs 40 KT per signal instead of 50 KT.

        Args:
            signatures: A list of :class:`~kongen.types.StructuralSignature`
                or plain dicts.
            source_domain: Optional hint for the originating domain.

        Returns:
            A :class:`~kongen.types.BatchTransferResult` with per-signal
            results and aggregate token accounting.

        Cost:
            40 Kongen Tokens (KT) per signal in the batch.

        Example::

            batch = client.transfer.score_batch([
                {"complexity": 0.7, "constraint": 0.3, ...},
                {"complexity": 0.2, "constraint": 0.8, ...},
            ])
            for r in batch.results:
                print(r.classification, r.confidence)
        """
        sig_dicts = [
            {"signature": self._normalize_signature(s)} for s in signatures
        ]

        payload: dict[str, object] = {"signatures": sig_dicts}
        if source_domain is not None:
            payload["source_domain"] = source_domain

        data = self._client._request("POST", "/v1/transfer/score_batch", json=payload)
        return BatchTransferResult.model_validate(data)

    @staticmethod
    def _normalize_signature(
        signature: StructuralSignature | dict[str, float],
    ) -> dict[str, float]:
        """Convert a signature to a plain dict for serialization."""
        if isinstance(signature, StructuralSignature):
            return signature.model_dump()
        # Validate dict keys by constructing and dumping
        return StructuralSignature.model_validate(signature).model_dump()

    def __repr__(self) -> str:
        return "TransferScorer()"
