"""Conservative static XSS triage heuristic.

This module is intentionally *not* an execution oracle. It recognizes a small
set of obvious source-to-sink shapes so low-confidence ML predictions can be
triaged cheaply. A positive result is still UNCONFIRMED and must go through the
real browser oracle in :mod:`verification.browser_oracle` before a finding can
be called confirmed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


TAINTED_SOURCE = re.compile(
    r"(location\.(?:search|hash|href)|document\.(?:URL|documentURI|cookie|referrer)|"
    r"window\.name|req\.(?:body|query|params)|URLSearchParams|formData|\.value\b)",
    re.IGNORECASE,
)

HTML_SINK = re.compile(
    r"\b(innerHTML|outerHTML|insertAdjacentHTML|document\.write|document\.writeln)\b"
)
JS_SINK = re.compile(r"\b(eval|Function)\s*\(")
EVENT_SETTER = re.compile(
    r"setAttribute\s*\(\s*['\"]on[a-z]+['\"]\s*,",
    re.IGNORECASE,
)
DANGEROUS_URL_LITERAL = re.compile(
    r"setAttribute\s*\(\s*['\"](?:src|href|action|data)['\"]\s*,\s*"
    r"['\"]\s*(?:javascript:|data:text/html)",
    re.IGNORECASE,
)

SAFE_ONLY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*\w+\.textContent\s*=\s*[^;\n]+;?\s*$", re.IGNORECASE),
    re.compile(
        r"^\s*\w+\.setAttribute\s*\(\s*['\"](?:class|id|aria-[a-z-]+)['\"]\s*,"
        r"\s*['\"][^'\"]*['\"]\s*\)\s*;?\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*\w+\.setAttribute\s*\(\s*['\"](?:src|href|action|data)['\"]\s*,"
        r"\s*['\"](?!\s*(?:javascript:|data:text/html))[^'\"]*['\"]\s*\)\s*;?\s*$",
        re.IGNORECASE,
    ),
)


@dataclass(frozen=True)
class OracleVerdict:
    """Historical name kept for API compatibility; this is a triage verdict."""

    label: str
    rule: str | None
    confidence: float


def analyze(code: str) -> OracleVerdict:
    if not code or not code.strip():
        return OracleVerdict("SAFE", "empty", 0.6)

    for pattern in SAFE_ONLY_PATTERNS:
        if pattern.search(code):
            return OracleVerdict("SAFE", "safe_static_shape", 0.9)

    tainted = bool(TAINTED_SOURCE.search(code))
    if tainted and HTML_SINK.search(code):
        return OracleVerdict("XSS", "tainted_html_sink", 0.95)
    if tainted and JS_SINK.search(code):
        return OracleVerdict("XSS", "tainted_js_sink", 0.95)
    if tainted and EVENT_SETTER.search(code):
        return OracleVerdict("XSS", "tainted_event_handler", 0.95)
    if DANGEROUS_URL_LITERAL.search(code):
        return OracleVerdict("XSS", "dangerous_url_literal", 0.95)

    return OracleVerdict("POSSIBLE_XSS", None, 0.0)


def explain(code: str) -> list[str]:
    verdict = analyze(code)
    if verdict.rule:
        return [f"static triage rule: {verdict.rule}"]
    return ["no hard static rule fired; browser verification or human review required"]
