"""Bridge triage decisions to the authoritative browser oracle.

The ML/static layers may request confirmation, but only observed browser
execution can upgrade a routed decision to confirmed XSS. Failure to execute,
unsupported syntax, or an unavailable browser is an abstention rather than
proof of safety.
"""
from __future__ import annotations

from dataclasses import dataclass

from verification.browser_oracle import verify_dom_case
from xss_specialist.router import RoutedDecision


@dataclass(frozen=True)
class FinalDecision:
    verdict: str
    confirmed: bool
    confirmation_status: str
    browser_evidence: dict | None


def finalize_with_browser(code: str, routed: RoutedDecision) -> FinalDecision:
    if not routed.requires_oracle:
        return FinalDecision(
            verdict=routed.verdict,
            confirmed=False,
            confirmation_status="NOT_REQUIRED",
            browser_evidence=None,
        )

    evidence = verify_dom_case({
        "code": code,
        "language": "javascript",
        "vulnerable": True,
    })
    compact = {
        key: evidence.get(key)
        for key in ("status", "executed", "scope", "js_error")
        if key in evidence
    }
    if evidence.get("executed") is True:
        return FinalDecision(
            verdict="XSS",
            confirmed=True,
            confirmation_status="CONFIRMED_EXECUTION",
            browser_evidence=compact,
        )

    return FinalDecision(
        verdict="POSSIBLE_XSS",
        confirmed=False,
        confirmation_status="UNCONFIRMED",
        browser_evidence=compact,
    )
