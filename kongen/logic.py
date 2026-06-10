"""Logic — LLM reasoning regime detection."""

from __future__ import annotations

from typing import TYPE_CHECKING

from kongen.types import LogicResult

if TYPE_CHECKING:
    from kongen.client import KongenClient


class LogicAnalyzer:
    """Detect reasoning regimes in LLM-generated text.

    The Logic analyzer classifies prompt complexity and returns a
    confidence adjustment for downstream scoring.

    This class is not instantiated directly. Access it via
    :attr:`KongenClient.logic`.
    """

    def __init__(self, client: KongenClient) -> None:
        self._client = client

    def score(
        self,
        text: str,
        model_hint: str | None = None,
    ) -> LogicResult:
        """Score text for reasoning regime via the Kongen API.

        Sends the text to the Logic endpoint which classifies it into
        a reasoning regime and returns a confidence adjustment.

        Args:
            text: The prompt or LLM response to analyze.
            model_hint: Optional model identifier (e.g., ``"claude-3-haiku"``)
                to improve regime calibration.

        Returns:
            A :class:`~kongen.types.LogicResult` with regime classification,
            confidence adjustment, and token accounting.

        Cost:
            1 Kongen Token (KT) per call.

        Example::

            result = client.logic.score("Prove that sqrt(2) is irrational")
            print(result.regime)       # "analytical"
            print(result.confidence_adj) # 0.15
        """
        payload: dict[str, object] = {"text": text}
        if model_hint is not None:
            payload["model_hint"] = model_hint

        data = self._client._request("POST", "/v1/logic/score", json=payload)
        return LogicResult.model_validate(data)

    def analyze(self, text: str) -> dict[str, object]:
        """Run local-only text analysis (no API call, 0 KT).

        Performs basic structural analysis of the input text without
        contacting the Kongen API. Useful for pre-filtering or offline
        exploration. A full Cython-accelerated analyzer will ship in v1.1.

        Args:
            text: The text to analyze locally.

        Returns:
            A dictionary with basic structural metrics:
            ``token_count``, ``sentence_count``, ``avg_sentence_length``,
            and ``estimated_regime``.

        Example::

            info = client.logic.analyze("What is 2 + 2?")
            print(info["estimated_regime"])  # "reflexive"
        """
        sentences = [s.strip() for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]
        token_count = len(text.split())
        sentence_count = len(sentences)
        avg_sentence_length = token_count / max(sentence_count, 1)

        # Rough heuristic -- will be replaced by Cython analyzer in v1.1
        if token_count < 10:
            estimated_regime = "reflexive"
        elif avg_sentence_length < 12:
            estimated_regime = "procedural"
        elif avg_sentence_length < 25:
            estimated_regime = "analytical"
        elif sentence_count > 5:
            estimated_regime = "synthesis"
        else:
            estimated_regime = "metacognitive"

        return {
            "token_count": token_count,
            "sentence_count": sentence_count,
            "avg_sentence_length": round(avg_sentence_length, 2),
            "estimated_regime": estimated_regime,
        }

    def __repr__(self) -> str:
        return "LogicAnalyzer()"
