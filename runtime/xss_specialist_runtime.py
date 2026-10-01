"""Minimal runtime CLI for the frozen XSS Specialist models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEMANTIC = ROOT / "models" / "semantic-v4e"
FLOW = ROOT / "models" / "flow-v2"


def doctor() -> int:
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


def fields(args) -> int:
    from xss_specialist.field_advisor import StructuralFieldAdvisor
    advisor = StructuralFieldAdvisor(SEMANTIC, FLOW)
    relation = {
        "source_expression": args.source,
        "sink_expression": args.sink,
        "statement": args.statement or f"{args.sink} = {args.source};",
    }
    result = advisor.review_fields(
        source_expression=args.source,
        sink_expression=args.sink,
        defense_expression=args.defense,
        relation=relation,
    )
    print(json.dumps(result, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="xss-specialist-runtime")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor")
    p = sub.add_parser("fields")
    p.add_argument("--source", required=True)
    p.add_argument("--sink", required=True)
    p.add_argument("--defense", default="")
    p.add_argument("--statement", default="")
    a = ap.parse_args()
    if a.cmd == "doctor":
        return doctor()
    if a.cmd == "fields":
        return fields(a)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
