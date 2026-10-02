from types import SimpleNamespace

from live.crawler import map_inputs
from live.findings import INCONCLUSIVE, LIKELY, NOT_VULNERABLE, correlate
from verification.browser_oracle import _survival_evidence


def _candidate(context="html_attr"):
    return {
        "url": "http://example.test/page?q=1",
        "method": "GET",
        "param": "q",
        "input_type": "GET parameter",
        "context": context,
    }


def _marker(reflected=True):
    return {
        "reflected_html": reflected,
        "reflected_dom": False,
        "blocked": False,
        "requested_url": "http://example.test/page?q=marker",
        "final_url": "http://example.test/page?q=marker",
        "status": 200,
        "marker": "xzmarker",
        "probe_note": "harmless reflection marker",
        "console": [],
    }


def test_reflection_without_execution_is_not_declared_safe():
    f = correlate(_candidate(), _marker(True), [
        {"executed": False, "raw_reflected": False, "breakout_survived": False, "blocked": False}
    ], None, "F-1")
    assert f.status == INCONCLUSIVE


def test_marker_local_breakout_survival_is_likely_not_confirmed():
    f = correlate(_candidate(), _marker(True), [
        {
            "executed": False,
            "raw_reflected": False,
            "breakout_survived": True,
            "blocked": False,
            "requested_url": "http://example.test/page?q=probe",
            "final_url": "http://example.test/page?q=probe",
            "status": 200,
            "marker": "xzprobe",
            "probe_note": "quoted attr breakout",
            "console": [],
        }
    ], None, "F-2")
    assert f.status == LIKELY
    assert f.browser_result == "not-executed"


def test_verified_safe_defense_can_close_reflected_candidate():
    sanitizer = SimpleNamespace(
        status="VERIFIED_SAFE", observed_name="known-encoder", note="verified", evidence_id="E-1"
    )
    f = correlate(_candidate(), _marker(True), [
        {"executed": False, "raw_reflected": False, "breakout_survived": False, "blocked": False}
    ], sanitizer, "F-3")
    assert f.status == NOT_VULNERABLE


def test_hash_candidates_require_dom_source_evidence():
    crawl = {
        "routes": [
            {"url": "http://example.test/a", "dom_hash_source": False},
            {"url": "http://example.test/b", "dom_hash_source": True},
        ],
        "forms": [],
        "js_routes": [],
    }
    cands = map_inputs(crawl)
    hashes = [c for c in cands if c["delivery"] == "hash"]
    assert len(hashes) == 1
    assert hashes[0]["url"] == "http://example.test/b"


def test_marker_local_survival_detects_breakout_without_full_payload_match():
    marker = "xzprobe"
    payload = f'"><img src=x onerror="window.__X[\'{marker}\']=1">'
    raw = f'<input value="{payload}">'
    ev = _survival_evidence(payload, marker, raw, raw)
    assert ev["payload_marker_reflected"] is True
    assert ev["breakout_survived"] is True
    assert '"><' in ev["survival_tokens"]


def test_encoded_markup_does_not_count_as_live_breakout():
    marker = "xzsafe"
    payload = f'<span>{marker}</span><img src=x onerror="window.__X[\'{marker}\']=1">'
    encoded = (
        f'<div>&lt;span&gt;{marker}&lt;/span&gt;'
        f'&lt;img src=x onerror="window.__X[\'{marker}\']=1"&gt;</div>'
    )
    ev = _survival_evidence(payload, marker, encoded, encoded)
    assert ev["payload_marker_reflected"] is True
    assert ev["breakout_survived"] is False
    assert ev["survival_tokens"] == []


def test_json_escaped_js_breakout_is_not_live():
    marker = "xzjson"
    payload = f'";window.__X[\'{marker}\']=1;//'
    encoded_payload = payload.replace('"', '\\"')
    raw = f'<script>var s = "{encoded_payload}";</script>'
    ev = _survival_evidence(payload, marker, raw, raw)
    assert ev["payload_marker_reflected"] is True
    assert ev["breakout_survived"] is False
