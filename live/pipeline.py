"""Live-assessment orchestrator.

assess_candidate: marker probe -> (if reflected) context classification from where the marker landed
-> context-tailored execution probes -> sanitizer identity -> correlated finding. The oracle is
authoritative for CONFIRMED. The specialist model may be attached for advisory context analysis, but
it never sets the finding state.

assess_target: run assess_candidate over a candidate list (from the crawler/mapper or a seed list),
under one Enforcer, collecting all evidence for the report.
"""
from __future__ import annotations

import re

from live.executor import execute
from live.findings import correlate
from live.probes import plan
from live.sanitizer_id import verify_identity

_URL_ATTRS = {"href", "src", "action", "formaction", "poster", "data"}


def classify_context(html_src: str, marker: str) -> str:
    """Classify the marker's reflection context from the surrounding source."""
    idx = html_src.find(marker)
    if idx < 0:
        return "unknown"

    before = html_src[max(0, idx - 160):idx]

    # Inside <script> ... marker ... </script>
    last_script_open = before.lower().rfind("<script")
    last_script_close = before.lower().rfind("</script>")
    if last_script_open > last_script_close:
        return "js_string"

    # Inside an HTML tag / attribute value.
    tag_open = before.rfind("<")
    tag_close = before.rfind(">")
    if tag_open > tag_close:
        seg = before[tag_open:]

        quoted = re.search(r'([:\w-]+)\s*=\s*"[^"]*$', seg)
        if quoted is None:
            quoted = re.search(r"([:\w-]+)\s*=\s*'[^']*$", seg)
        if quoted:
            attr = quoted.group(1).lower()
            return "html_attr_url" if attr in _URL_ATTRS else "html_attr"

        unquoted = re.search(r"([:\w-]+)\s*=\s*[^\s>]*$", seg)
        if unquoted:
            attr = unquoted.group(1).lower()
            return "html_attr_url" if attr in _URL_ATTRS else "html_attr_unquoted"

    if before.rstrip().endswith("<!--") or (
        "<!--" in before and "-->" not in before[before.rfind("<!--"):]
    ):
        return "html_comment"
    return "html_text"


def assess_candidate(candidate: dict, enforcer, fid: str, timeout_ms: int = 12000,
                     caps: dict | None = None) -> dict:
    """Run the bounded marker -> context -> execution -> correlation pipeline."""
    caps = caps or {}
    cap_interactions = caps.get("interactions", True)
    cap_jscode = caps.get("js_code_probe", True)
    cap_quoted_split = caps.get("quoted_split", True)

    # 1) harmless marker
    mp = plan(candidate, "marker")[0]
    marker_ev = execute(candidate, mp, enforcer, timeout_ms)
    reflected = bool(marker_ev.get("reflected_html") or marker_ev.get("reflected_dom"))

    # 2) classify the actual reflection context
    ctx = candidate.get("context", "unknown")
    if reflected and ctx == "unknown":
        if marker_ev.get("reflected_dom") and not marker_ev.get("reflected_html"):
            ctx = "dom_html"
        else:
            win = marker_ev.get("context_window", "")
            ctx = classify_context(win, mp.marker) if win else "html_text"
            if not cap_quoted_split and ctx == "html_attr_unquoted":
                ctx = "html_attr"
        candidate = {**candidate, "context": ctx}

    # 3) bounded interactions derived from the observed context
    if "interactions" not in candidate:
        win = marker_ev.get("context_window", "") or ""
        acts = []
        m = re.search(r"on(\w+)\s*=", win)
        if m:
            ev_name = m.group(1).lower()
            acts = {
                "mouseover": ["hover"],
                "focus": ["focus"],
                "click": ["click"],
                "mouseenter": ["hover"],
                "keyup": ["focus"],
                "keydown": ["focus"],
            }.get(ev_name, ["hover", "focus", "click"])
        elif ctx == "html_attr_url" or "href=" in win.lower() or "src=" in win.lower():
            acts = ["click"]
        elif ctx == "html_attr":
            acts = ["hover", "focus", "click"]
        elif ctx in ("dom_html", "dom_attr"):
            acts = ["hashnav"]
        candidate = {**candidate, "interactions": acts if cap_interactions else []}

    # 4) context-shaped execution probes
    exec_evs = []
    if reflected:
        for pr in plan(candidate, "exec"):
            ev = execute(candidate, pr, enforcer, timeout_ms)
            exec_evs.append(ev)
            if ev.get("executed"):
                break

    # Code sinks can consume input without reflecting it.
    if cap_jscode and not any(e.get("executed") for e in exec_evs):
        from live.probes import js_code_probe

        jp = js_code_probe(candidate.get("delivery", "query"))
        exec_evs.append(execute(candidate, jp, enforcer, timeout_ms))

    # 5) sanitizer identity and final correlation
    san_name = candidate.get("observed_sanitizer", "")
    sv = verify_identity(san_name) if san_name else None
    finding = correlate(candidate, marker_ev, exec_evs, sv, fid)
    return {"finding": finding, "marker_ev": marker_ev, "exec_evs": exec_evs}


def assess_target(candidates: list, enforcer, prefix: str = "F") -> dict:
    results = []
    for i, cand in enumerate(candidates):
        if enforcer.budget_left() <= 0:
            break
        results.append(assess_candidate(cand, enforcer, fid=f"{prefix}-{i:03d}"))
    return {
        "results": results,
        "requests": enforcer.requests_made,
        "blocked": enforcer.blocked_log,
    }
