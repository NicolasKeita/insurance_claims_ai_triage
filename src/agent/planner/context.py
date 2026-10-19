"""Bounded evidence context shared by the planner and finalizer prompts."""

import json

from agent.config import MAX_FINALIZER_EVIDENCE_CHARS


def compact_data(value, depth=0):
    """Bound model context independently of source/service output size."""
    if depth > 5:
        return "[nested data omitted]"
    if isinstance(value, str):
        return value[:1600]
    if isinstance(value, dict):
        return {str(k)[:300]: compact_data(v, depth + 1) for k, v in list(value.items())[:40]}
    if isinstance(value, list):
        return [compact_data(v, depth + 1) for v in value[:20]]
    return value


def finalizer_evidence(registry):
    """
    Keep every provenance ID and bound the aggregate evidence context.

    JSON escaping can occupy more than one character; fit exactly.
    """
    evidence = [compact_data(e) for e in registry.values()]
    if len(json.dumps(evidence, ensure_ascii=False)) <= MAX_FINALIZER_EVIDENCE_CHARS:
        return evidence, False
    per_item = MAX_FINALIZER_EVIDENCE_CHARS // max(1, len(evidence)) - 4
    bounded = []
    for item in evidence:
        if len(json.dumps(item, ensure_ascii=False)) <= per_item:
            bounded.append(item)
            continue
        bounded.append(_shrink(item, per_item))
    return bounded, True


def _shrink(item, per_item):
    entry = {
        "evidence_id": item["evidence_id"],
        "source_type": item["source_type"],
        "summary": "",
        "context_truncated": True,
    }
    space = max(0, per_item - len(json.dumps(entry, ensure_ascii=False)))
    entry["summary"] = item["summary"][:space]
    while len(json.dumps(entry, ensure_ascii=False)) > per_item and entry["summary"]:
        entry["summary"] = entry["summary"][:-1]
    return entry
