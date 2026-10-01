"""Headless browser executor: delivers a planned probe to an in-scope URL and records browser
evidence via the authoritative oracle (`verification.browser_oracle.run_probe_on_url`).

Delivery embeds the probe payload into the candidate's parameter — as a query value, a URL fragment,
or (for POST forms) a urlencoded POST body to the form action. Each delivery maps to a test class
(query -> reflected, hash -> dom, form -> post) that the Scope must allow. Scope is enforced before
every navigation and again inside the browser on every request it issues; out-of-scope requests
and redirect targets are blocked before the browser follows them.
"""
from __future__ import annotations

from urllib.parse import urlencode, urlparse, urlunparse, parse_qsl

from live.probes import Probe
from live.scope import Enforcer
from verification.browser_oracle import run_probe_on_url


def _embed(url: str, param: str, value: str, delivery: str) -> str:
    p = urlparse(url)
    if delivery == "hash":
        return urlunparse(p._replace(fragment=value))
    q = dict(parse_qsl(p.query, keep_blank_values=True))
    q[param] = value
    return urlunparse(p._replace(query=urlencode(q)))


DELIVERY_TEST_CLASS = {"query": "reflected", "hash": "dom", "form": "post"}


def execute(candidate: dict, probe: Probe, enforcer: Enforcer, timeout_ms: int = 4000) -> dict:
    """Deliver one probe. Returns a browser-evidence record; the oracle's `executed` is authoritative."""
    param = candidate.get("param", "q")
    test_class = DELIVERY_TEST_CLASS.get(probe.delivery)
    if test_class is None or not enforcer.scope.allows(test_class):
        reason = f"test_class_not_authorized:{test_class or probe.delivery}"
        enforcer.blocked_log.append({"url": candidate["url"], "reason": reason, "kind": probe.kind})
        return {"blocked": True, "reason": reason, "marker": probe.marker, "url": candidate["url"]}
    post_data = None
    if probe.delivery == "form":
        url, post_data = candidate["url"], {param: probe.payload}
    else:
        url = _embed(candidate["url"], param, probe.payload, probe.delivery)
    ok, reason = enforcer.check(url, kind=probe.kind)
    if not ok:
        return {"blocked": True, "reason": reason, "marker": probe.marker, "url": url}
    ev = run_probe_on_url(url, probe.marker, delivery=probe.delivery,
                          param=param, timeout_ms=timeout_ms,
                          extra_headers=enforcer.scope.extra_headers or None,
                          cookies=enforcer.scope.cookies or None,
                          raw_signature=probe.raw_signature,
                          interactions=candidate.get("interactions"),
                          in_scope=enforcer.scope.in_scope, post_data=post_data)
    enforcer.note_request(url, final_url=ev.get("final_url") or url, status=ev.get("status"),
                          kind=probe.kind)
    enforcer.blocked_log.extend({**b, "from": url} for b in ev.get("scope_blocked", []))
    return {"blocked": False, "probe_kind": probe.kind, "probe_note": probe.note,
            "context": probe.context, "delivery": probe.delivery,
            "param": param, **ev}
