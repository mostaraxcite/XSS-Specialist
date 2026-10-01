"""Small advisory relation model. No execution and no final XSS verdicts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

LABELS = ("CONNECTED", "DISCONNECTED", "UNKNOWN")
GROUPS = ("repository", "framework", "template", "generator")
FIELDS = ("source_expression", "sink_expression", "flow_excerpt")
MAX_EXCERPT = 4096


def verify_repository_excerpt(row: dict) -> bool:
    """Verify native-code provenance before accepting generator/template N/A.

    Repository snapshots are inert data; no source code is imported or executed.
    Labels still need a separate review and are not inferred by this check.
    """
    provenance = row.get("provenance", {})
    if provenance.get("code_origin") != "repository":
        return False
    commit = provenance.get("commit", "")
    if not re.fullmatch(r"[a-f0-9]{40}", commit):
        raise ValueError("native source requires a pinned upstream commit")
    reference = row.get("verification", {}).get("reference", "")
    expected = f"https://github.com/{provenance.get('repository')}/blob/{commit}/{provenance.get('source_file')}#L"
    if not reference.startswith(expected):
        raise ValueError("native source reference must identify the pinned repository/file")
    root = Path(__file__).resolve().parents[1]
    snapshot = (root / provenance.get("source_snapshot", "")).resolve()
    approved_roots = [
        (root / "data/flow-relations-v1/sources").resolve(),
        (root / "data/flow-relations-v2/sources").resolve(),
    ]
    if not any(snapshot.is_relative_to(approved) for approved in approved_roots) or not snapshot.is_file():
        raise ValueError(
            "native source snapshot must be under an approved flow-relations source root"
        )
    content = snapshot.read_bytes()
    if hashlib.sha256(content).hexdigest() != provenance.get("source_sha256"):
        raise ValueError("native source snapshot hash mismatch")
    span = provenance.get("excerpt_span", {})
    start, end = span.get("start_byte"), span.get("end_byte")
    if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end <= len(content):
        raise ValueError("invalid native source byte span")
    if content[start:end].decode() != row.get("flow_excerpt"):
        raise ValueError("relation excerpt differs from pinned source bytes")
    location = row.get("query_location", {})
    for key in ("source", "sink"):
        query_span = location.get(key + "_span", {})
        query_start, query_end = query_span.get("start_byte"), query_span.get("end_byte")
        if not isinstance(query_start, int) or not isinstance(query_end, int) or not 0 <= query_start < query_end <= len(content):
            raise ValueError(f"invalid {key} expression byte span")
        if content[query_start:query_end].decode() != row.get(key + "_expression"):
            raise ValueError(f"{key} expression differs from pinned source bytes")
    return True


def relation_text(row: dict) -> str:
    if row.get("task") != "FLOW_RELATION":
        raise ValueError("expected task=FLOW_RELATION; whole-snippet classifier records are unsupported")
    parts = []
    for key in FIELDS:
        value = row.get(key)
        if not isinstance(value, str) or not value.strip() or len(value) > MAX_EXCERPT:
            raise ValueError(f"{key} must be a nonempty reviewed span of at most {MAX_EXCERPT} characters")
        parts.append(f"[{key}]\n{value}")
    return "\n".join(parts)



def canonical_relation_text(row: dict) -> str:
    """Canonical role-aware flow text that removes identifier memorization.

    The original reviewed spans stay untouched for provenance/audit. This view is
    only for the advisory flow model: exact source/sink expressions are replaced
    by role tokens inside the excerpt so the model learns connectivity shape.
    """
    if row.get("task") != "FLOW_RELATION":
        raise ValueError("expected task=FLOW_RELATION")
    source = row.get("source_expression")
    sink = row.get("sink_expression")
    excerpt = row.get("flow_excerpt")
    for name, value in (("source_expression", source), ("sink_expression", sink), ("flow_excerpt", excerpt)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a nonempty string")

    normalized = excerpt
    replacements = sorted(
        ((source, "__SOURCE__"), (sink, "__SINK__")),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    for value, token in replacements:
        normalized = normalized.replace(value, token)

    return (
        "[SOURCE]\n__SOURCE__\n"
        "[SINK]\n__SINK__\n"
        f"[FLOW]\n{normalized}"
    )



def deterministic_relation(row: dict) -> dict:
    """Resolve only obvious local value relations; otherwise abstain UNKNOWN."""
    source = str(row.get("source_expression", "")).strip()
    sink = str(row.get("sink_expression", "")).strip()
    excerpt = str(row.get("flow_excerpt", "")).strip()
    if not source or not sink or not excerpt:
        return {"relation": "UNKNOWN", "reason": "missing_field", "authoritative": False}

    compact = " ".join(excerpt.split())
    source_esc = re.escape(source)
    sink_esc = re.escape(sink)

    # Opaque call/new return: source is an argument to a call whose result is
    # written to the sink. The bounded deterministic engine deliberately abstains.
    opaque = re.search(
        rf"(?:const|let|var)?\s*{sink_esc}\s*=\s*(?:new\s+)?[A-Za-z_$][\w$\.]*\([^;]*{source_esc}[^;]*\)",
        compact,
    )
    if opaque:
        return {"relation": "UNKNOWN", "reason": "opaque_call_return", "authoritative": False}

    direct_assign = re.search(
        rf"{sink_esc}\s*=\s*{source_esc}(?:\s*;|\s*$)",
        compact,
    )
    direct_call = re.search(
        rf"{sink_esc}\s*\([^;]*\b{source_esc}\b[^;]*\)",
        compact,
    )
    if direct_assign or direct_call:
        return {"relation": "CONNECTED", "reason": "direct_local_value_use", "authoritative": True}

    literal_assign = re.search(
        rf"{sink_esc}\s*=\s*(?:['\"][^'\"]*['\"]|\d+(?:\.\d+)?|true|false|null|undefined)(?:\s*;|\s*$)",
        compact,
        re.I,
    )
    if literal_assign:
        return {"relation": "DISCONNECTED", "reason": "literal_overwrite", "authoritative": True}

    return {"relation": "UNKNOWN", "reason": "bounded_abstain", "authoritative": False}


def audit_relation_splits(splits: dict[str, list[dict]]) -> dict:
    identities = {}
    counts = {}
    for name, rows in splits.items():
        if not rows:
            raise ValueError(f"empty split: {name}")
        identities[name] = {key: set() for key in (*GROUPS, "content", "id")}
        counts[name] = {label: 0 for label in LABELS}
        for row in rows:
            content = relation_text(row)
            label = row.get("label")
            if label not in LABELS:
                raise ValueError("relation label must be CONNECTED, DISCONNECTED or UNKNOWN")
            if not isinstance(row.get("id"), str) or not row["id"].strip():
                raise ValueError("relation id is required")
            review = row.get("verification", {})
            if review.get("status") != "REVIEWED" or any(
                not isinstance(review.get(k), str) or not review[k].strip()
                for k in ("reviewer", "rationale", "reference")
            ):
                raise ValueError("independent reviewed relation labels and evidence reference are required")
            provenance = row.get("provenance", {})
            native = verify_repository_excerpt(row)
            for key in GROUPS:
                value = provenance.get(key)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"missing provenance.{key}")
                # Templates/generators describe generated CODE, not the extraction
                # tool or annotation author. Never invent split-specific identities.
                if value == "not-applicable":
                    if key not in ("template", "generator") or not native:
                        raise ValueError("not-applicable requires verified native code provenance")
                    continue
                identities[name][key].add(value.strip())
            digest = hashlib.sha256(content.encode()).hexdigest()
            if digest in identities[name]["content"] or row["id"] in identities[name]["id"]:
                raise ValueError(f"duplicate relation in {name}")
            identities[name]["content"].add(digest)
            identities[name]["id"].add(row["id"])
            counts[name][label] += 1
    names = list(splits)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            for key in identities[left]:
                if identities[left][key] & identities[right][key]:
                    raise ValueError(f"relation split overlap: {left}/{right} in {key}")
    return {"passed": True, "class_counts": counts, "groups": list(GROUPS),
            "generator_template_policy": "Disjoint for generated code; N/A only for byte-verified native repository excerpts",
            "note": "Metadata and review declarations are validated; authenticity requires corpus-owner review."}


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class FlowRelationAdvisor:
    """Read inert JSON weights; output priorities, never safety or confirmation."""
    def __init__(self, path: Path):
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer

        artifact = json.loads(path.read_text())
        if artifact.get("schema") != "xss-flow-relation-v1" or artifact.get("final_judge") is not False:
            raise ValueError("unsupported or authoritative relation artifact")
        self.classes = artifact["classes"]
        if sorted(self.classes) != sorted(LABELS):
            raise ValueError("unexpected relation classes")
        self.vectorizer = TfidfVectorizer(vocabulary=artifact["vocabulary"],
                                         token_pattern=r"(?u)\b\w+\b", lowercase=False,
                                         ngram_range=(1, 2), sublinear_tf=True)
        self.vectorizer.idf_ = np.asarray(artifact["idf"], dtype=float)
        self.weights = np.asarray(artifact["coefficients"], dtype=float)
        self.intercept = np.asarray(artifact["intercept"], dtype=float)
        if self.weights.shape != (3, len(artifact["vocabulary"])) or self.intercept.shape != (3,):
            raise ValueError("invalid weight dimensions")
        if not all(np.isfinite(a).all() for a in (self.weights, self.intercept, self.vectorizer.idf_)):
            raise ValueError("nonfinite model weights")

    def predict(self, row: dict) -> dict:
        import numpy as np

        features = self.vectorizer.transform([relation_text(row)])
        logits = np.asarray(features @ self.weights.T).reshape(-1) + self.intercept
        values = np.exp(logits - logits.max())
        values /= values.sum()
        distribution = {k: float(v) for k, v in zip(self.classes, values)}
        return {"task": "FLOW_RELATION", "probabilities": distribution,
                "relation": max(distribution, key=distribution.get),
                "advisory_only": True, "confirmed": False, "requires_review": True}


def advise_review(result: dict, advice: dict) -> dict:
    """Add relation evidence to priority only; leave every authoritative field intact."""
    reviewed = {**result, "relation_advice": advice}
    if advice.get("relation") == "CONNECTED":
        reviewed["review_priority"] = "high"
    return reviewed
