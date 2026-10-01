"""Wire the adapter registry into the XSS triage router."""
from __future__ import annotations

from typing import Callable

from xss_specialist.adapter_registry import AdapterSpec, default_registry, load_classifier
from xss_specialist.oracle import analyze as static_analyze
from xss_specialist.router import FastDecision, SecurityRouter


def _make_fast_predict(
    adapter_name: str | None = None,
) -> tuple[Callable[[str], FastDecision], AdapterSpec]:
    classify, spec = load_classifier(adapter_name)

    def fast_predict(code: str) -> FastDecision:
        scores = classify(code, truncation=True)[0]
        best = max(scores, key=lambda item: item["score"])
        return FastDecision(str(best["label"]), float(best["score"]))

    return fast_predict, spec


def _static_triage_predict(code: str) -> str:
    verdict = static_analyze(code)
    return f"Classification: {verdict.label}"


def build_router(
    adapter_name: str | None = None,
    *,
    direct_threshold: float = 0.90,
    xss_threshold: float = 0.85,
    specialist: Callable[[str], str] | None = None,
) -> tuple[SecurityRouter, AdapterSpec]:
    fast_predict, spec = _make_fast_predict(adapter_name)
    triage = specialist or _static_triage_predict
    router = SecurityRouter(
        fast_predict=fast_predict,
        triage_predict=triage,
        direct_threshold=direct_threshold,
        xss_threshold=xss_threshold,
    )
    return router, spec


def active_adapter_name() -> str:
    return default_registry().active
