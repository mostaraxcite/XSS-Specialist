"""Explicit offline evaluation with group-disjoint provenance and strict abstention accounting."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from xss_specialist.taint import analyze

GROUPS = ("repository", "framework", "template", "generator")


def audit_splits(splits: dict[str, list[dict]]) -> dict:
    """Fail closed on absent identities, overlap or exact-code contamination.

    Metadata must be assigned by corpus owners, never inferred from labels.
    This check does not establish semantic independence or corpus quality.
    """
    identities: dict[str, dict[str, set[str]]] = {}
    for name, rows in splits.items():
        if not rows:
            raise ValueError(f"empty split: {name}")
        identities[name] = {key: set() for key in (*GROUPS, "code_sha256")}
        for row in rows:
            if row.get("label") not in ("SAFE", "XSS") or not isinstance(row.get("code"), str):
                raise ValueError(f"{name}: expected code and binary SAFE/XSS ground truth")
            provenance = row.get("provenance", {})
            for key in GROUPS:
                value = provenance.get(key)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{name}: missing provenance.{key}")
                identities[name][key].add(value.strip())
            identities[name]["code_sha256"].add(hashlib.sha256(row["code"].strip().encode()).hexdigest())
    names = list(identities)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            for key in identities[left]:
                if identities[left][key] & identities[right][key]:
                    raise ValueError(f"split overlap: {left}/{right} in {key}")
    return {"passed": True, "groups": list(GROUPS),
            "sizes": {name: len(rows) for name, rows in splits.items()}}


def score(rows: list[dict]) -> dict:
    counts = {"safe": 0, "xss": 0, "false_positive": 0, "false_negative": 0,
              "abstentions": 0, "safe_abstentions": 0, "xss_abstentions": 0}
    for row in rows:
        if row.get("label") not in ("SAFE", "XSS"):
            raise ValueError("binary SAFE/XSS ground truth required")
        verdict = analyze(row["code"])["verdict"]
        safe = row["label"] == "SAFE"
        counts["safe" if safe else "xss"] += 1
        if verdict == "INCONCLUSIVE":
            counts["abstentions"] += 1
            counts["safe_abstentions" if safe else "xss_abstentions"] += 1
        if safe and verdict == "CANDIDATE":
            counts["false_positive"] += 1
        # Abstaining on an XSS case is a missed detection for promotion purposes.
        if not safe and verdict != "CANDIDATE":
            counts["false_negative"] += 1
    ratio = lambda numerator, denominator: numerator / denominator if denominator else None
    fpr = ratio(counts["false_positive"], counts["safe"])
    fnr = ratio(counts["false_negative"], counts["xss"])
    coverage = ratio(len(rows) - counts["abstentions"], len(rows))
    # Safe abstentions cannot be used to evade the false-positive review burden.
    burden = ratio(counts["false_positive"] + counts["safe_abstentions"], counts["safe"])
    return {**counts, "external_safe_fpr": fpr, "strict_xss_fnr": fnr,
            "decision_coverage": coverage, "safe_review_rate": burden,
            "thresholds_met": bool(fpr is not None and fnr is not None and coverage is not None
                                   and fpr < .15 and fnr < .08 and coverage >= .90 and burden < .15),
            "promotion_allowed": False,
            "promotion_note": "Metrics alone cannot authorize promotion; independently curated locked evaluation required."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--dev", required=True, type=Path)
    parser.add_argument("--external", required=True, type=Path,
                        help="Explicit locked evaluation invocation; never loaded during development")
    args = parser.parse_args()
    splits = {k: [json.loads(line) for line in getattr(args, k).read_text().splitlines() if line.strip()]
              for k in ("train", "dev", "external")}
    audit = audit_splits(splits)
    print(json.dumps({"audit": audit, "external": score(splits["external"])}, indent=2))


if __name__ == "__main__":
    main()
