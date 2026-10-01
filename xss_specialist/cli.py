"""Runtime CLI for the standalone XSS Specialist."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEMANTIC = ROOT / "models" / "semantic-v4e"
FLOW = ROOT / "models" / "flow-v2"


def _doctor() -> int:
    checks = {
        "semantic_model": (SEMANTIC / "dual_branch_config.json").exists(),
        "flow_model": (FLOW / "flow_branch_config.json").exists(),
    }
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            p.chromium.launch(headless=True).close()
        checks["chromium"] = True
    except Exception:
        checks["chromium"] = False
    ok = all(checks.values())
    print(json.dumps({"ok": ok, "checks": checks}, indent=2))
    return 0 if ok else 1


def _fields(args) -> int:
    from xss_specialist.field_advisor import StructuralFieldAdvisor

    advisor = StructuralFieldAdvisor(SEMANTIC, FLOW)
    relation = {
        "source_expression": args.source,
        "sink_expression": args.sink,
        "flow_excerpt": args.statement or f"{args.sink} = {args.source};",
    }
    result = advisor.review_fields(
        source_expression=args.source,
        sink_expression=args.sink,
        defense_expression=args.defense,
        relation=relation,
    )
    print(json.dumps(result, indent=2))
    return 0


def _accept() -> int:
    from live.evaluate import main as evaluate_main
    result = evaluate_main()
    return int(result or 0)


def _scan(args) -> int:
    from live.pilot import PilotAuthorization, run_pilot

    if not args.acknowledge_authorization:
        raise SystemExit(
            "REFUSED: external scanning requires --acknowledge-authorization."
        )

    auth = PilotAuthorization(
        target=args.target,
        operator_identity=args.operator,
        assessment_id=args.assessment_id,
        authorization_acknowledged=True,
        allowed_prefixes=args.allowed_prefix or ["/"],
        excluded_paths=args.exclude or [],
        request_budget=args.budget,
        rate_limit_rps=args.rate,
        allowed_test_classes=["reflected", "dom"],
        stored_xss_permitted=False,
    )

    saved = run_pilot(auth, authorized_external=args.authorized_external)
    print(json.dumps(saved["summary"], indent=2))
    print("evidence ->", saved["dir"])
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="xss")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="verify model files and Chromium")

    p = sub.add_parser("fields", help="run semantic + flow inference")
    p.add_argument("--source", required=True)
    p.add_argument("--sink", required=True)
    p.add_argument("--defense", default="")
    p.add_argument("--statement", default="")

    sub.add_parser("accept", help="run the local browser acceptance gate")

    p = sub.add_parser("scan", help="run a bounded authorized live scan")
    p.add_argument("target")
    p.add_argument("--operator", default="Mostafa")
    p.add_argument("--assessment-id", default="manual-canary")
    p.add_argument("--acknowledge-authorization", action="store_true")
    p.add_argument("--authorized-external", action="store_true")
    p.add_argument("--allowed-prefix", action="append", default=[])
    p.add_argument("--exclude", action="append", default=[])
    p.add_argument("--budget", type=int, default=80)
    p.add_argument("--rate", type=float, default=1.0)

    args = ap.parse_args(argv)

    if args.cmd == "doctor":
        return _doctor()
    if args.cmd == "fields":
        return _fields(args)
    if args.cmd == "accept":
        return _accept()
    if args.cmd == "scan":
        return _scan(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
