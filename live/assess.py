"""Top-level authorized live-assessment orchestrator + CLI.

Flow: Scope -> crawl -> map inputs -> assess each candidate (marker -> context -> exec -> oracle ->
finding) -> save evidence + report. Refuses any EXTERNAL (non-loopback) target unless the local
acceptance evaluation has passed (a gate file written by `live.evaluate`). Never auto-expands scope.
Live findings are NOT fed to training — they go only to a quarantine queue.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from live.crawler import crawl, map_inputs
from live.executor import DELIVERY_TEST_CLASS
from live.pipeline import assess_candidate
from live.report import save_assessment
from live.scope import TEST_CLASSES, Enforcer, Scope, is_loopback

ROOT = Path(__file__).resolve().parents[1]
# Written by `python -m live.evaluate` on THIS machine. It lives under runs/ (git-ignored) so a
# fresh clone never inherits a passed gate from the committed evidence in reports/.
ACCEPT_GATE = ROOT / "runs" / "live_acceptance_gate.json"


def local_eval_passed() -> bool:
    if not ACCEPT_GATE.exists():
        return False
    try:
        return bool(json.loads(ACCEPT_GATE.read_text()).get("ACCEPTED"))
    except Exception:
        return False


def external_hosts(scope: Scope) -> list[str]:
    """Every non-loopback host the scope could reach (target, extra hosts, subdomain suffixes)."""
    hosts = [h for h in scope.allowed_hosts if not is_loopback(h)]
    return hosts + [f"*.{d}" for d in scope.allowed_subdomains]


def run_assessment(scope: Scope, name: str) -> dict:
    if external_hosts(scope):
        if not local_eval_passed():
            raise SystemExit("REFUSED: external target requires a PASSED local acceptance evaluation "
                             "(run `python -m live.evaluate`). Not touching an external target.")
        if not scope.authorized_external:
            raise SystemExit("REFUSED: external (non-loopback) target requires explicit per-target "
                             "authorization (Scope.authorized_external=True). Not touching it.")
    enf = Enforcer(scope)
    crawl_result = crawl(enf)
    candidates = map_inputs(crawl_result)
    allowed = []
    for cand in candidates:     # record what the authorized test classes leave untested
        cls = DELIVERY_TEST_CLASS.get(cand["delivery"])
        if scope.allows(cls):
            allowed.append(cand)
        else:
            enf.blocked_log.append({"url": cand["url"], "param": cand["param"], "kind": "candidate",
                                    "reason": f"test_class_not_authorized:{cls}"})
    candidates = allowed
    results = []
    for i, cand in enumerate(candidates):
        if enf.budget_left() <= 0:
            break
        results.append(assess_candidate(cand, enf, fid=f"F-{i:03d}"))
    assess_result = {"results": results, "requests": enf.requests_made, "blocked": enf.blocked_log}
    # quarantine note: live findings never auto-train
    Path("reports/live_assessments").mkdir(parents=True, exist_ok=True)
    saved = save_assessment(name, scope, enf, crawl_result, candidates, assess_result)
    (Path(saved["dir"]) / "QUARANTINE.txt").write_text(
        "Live findings are for review only. They MUST NOT be used to train the specialist "
        "automatically. Promotion of any knowledge requires the offline verified pipeline.\n")
    return saved


def main():
    ap = argparse.ArgumentParser(prog="xss-live")
    ap.add_argument("--target", required=True)
    ap.add_argument("--name", default="assessment")
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--max-requests", type=int, default=300)
    ap.add_argument("--rate", type=float, default=5.0)
    ap.add_argument("--allowed-prefix", action="append", default=[])
    ap.add_argument("--exclude", action="append", default=[])
    ap.add_argument("--authorized-external", action="store_true",
                    help="explicit per-target authorization to test a non-loopback host")
    ap.add_argument("--test-class", action="append", choices=[c for c in TEST_CLASSES if c != "stored"],
                    help="test classes to run (default: reflected, dom); 'post' sends POST forms")
    a = ap.parse_args()
    scope = Scope(base_url=a.target, allowed_prefixes=a.allowed_prefix or None,
                  excluded_paths=a.exclude, max_depth=a.max_depth,
                  max_requests=a.max_requests, rate_limit_rps=a.rate,
                  authorized_external=a.authorized_external,
                  allowed_test_classes=a.test_class or ["reflected", "dom"])
    saved = run_assessment(scope, a.name)
    print(json.dumps(saved["summary"], indent=2))
    print("artifacts ->", saved["dir"])


if __name__ == "__main__":
    main()
