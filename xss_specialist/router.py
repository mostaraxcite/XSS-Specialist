"""Two-stage triage router for the compact XSS classifier.

The router may use a conservative static heuristic to resolve low-confidence
cases, but only the browser execution layer can confirm an XSS finding.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class FastDecision:
    verdict: str
    confidence: float


@dataclass(frozen=True)
class RoutedDecision:
    verdict: str
    confidence: float
    route: str
    requires_oracle: bool
    specialist_output: str | None = None
    escalation_reason: str | None = None


class SecurityRouter:
    """Route a snippet through ML triage and, when useful, a static heuristic.

    requires_oracle means the result must pass through the real browser oracle
    before it can become a CONFIRMED finding. The static heuristic is never
    execution evidence.
    """

    def __init__(
        self,
        fast_predict: Callable[[str], FastDecision],
        triage_predict: Callable[[str], str],
        direct_threshold: float = 0.90,
        xss_threshold: float = 0.85,
    ):
        self.fast_predict = fast_predict
        self.triage_predict = triage_predict
        self.direct_threshold = direct_threshold
        self.xss_threshold = xss_threshold

    def predict(self, code: str) -> RoutedDecision:
        fast = self.fast_predict(code)
        verdict, confidence = fast.verdict, fast.confidence

        if verdict == "SAFE" and confidence >= self.direct_threshold:
            return RoutedDecision(
                verdict="SAFE",
                confidence=confidence,
                route="encoder",
                requires_oracle=False,
            )

        if verdict == "XSS" and confidence >= self.xss_threshold:
            return RoutedDecision(
                verdict="XSS",
                confidence=confidence,
                route="browser_required",
                requires_oracle=True,
                escalation_reason="xss_needs_browser_confirmation",
            )

        analysis = self.triage_predict(code)
        parsed = self._parse_verdict(analysis)
        if parsed == "XSS":
            route = "browser_required"
            requires_oracle = True
        elif parsed == "SAFE":
            route = "heuristic"
            requires_oracle = False
        else:
            route = "abstain"
            requires_oracle = True

        return RoutedDecision(
            verdict=parsed,
            confidence=confidence,
            route=route,
            requires_oracle=requires_oracle,
            specialist_output=analysis,
            escalation_reason="low_confidence",
        )

    @staticmethod
    def _parse_verdict(text: str) -> str:
        normalized = text.upper()
        for label in ("POSSIBLE_XSS", "XSS", "SAFE"):
            if f"CLASSIFICATION: {label}" in normalized or f"VERDICT: {label}" in normalized:
                return label
        return "POSSIBLE_XSS"
