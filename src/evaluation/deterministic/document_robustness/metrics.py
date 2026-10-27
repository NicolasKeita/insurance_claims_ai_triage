"""Whitespace-only OCR diagnostics with explicit edit counts and denominators."""

import re
from collections.abc import Sequence


def normalize_text(value: str) -> str:
    """Collapse whitespace only; preserve case, punctuation, digits and dates."""
    return re.sub(r"\s+", " ", value).strip()


def edit_distance(reference: Sequence, actual: Sequence) -> int:
    """Levenshtein distance using two rows and exact token equality."""
    if reference == actual:
        return 0
    start, end = 0, 0
    while start < min(len(reference), len(actual)) and reference[start] == actual[start]:
        start += 1
    while end < min(len(reference), len(actual)) - start and reference[-end - 1] == actual[-end - 1]:
        end += 1
    reference = reference[start : len(reference) - end if end else None]
    actual = actual[start : len(actual) - end if end else None]
    if len(reference) < len(actual):
        reference, actual = actual, reference
    previous = list(range(len(actual) + 1))
    for row, expected in enumerate(reference, start=1):
        current = [row]
        for column, observed in enumerate(actual, start=1):
            current.append(min(current[-1] + 1, previous[column] + 1, previous[column - 1] + (expected != observed)))
        previous = current
    return previous[-1]


def error_rate(errors: int, reference_count: int) -> float | None:
    """Empty/empty is zero; insertions against an empty reference are undefined."""
    return errors / reference_count if reference_count else (0.0 if errors == 0 else None)


def text_metrics(reference: str, actual: str) -> dict:
    expected, observed = normalize_text(reference), normalize_text(actual)
    words, observed_words = expected.split(), observed.split()
    characters = edit_distance(expected, observed)
    word_errors = edit_distance(words, observed_words)
    return {
        "character_errors": characters,
        "reference_characters": len(expected),
        "word_errors": word_errors,
        "reference_words": len(words),
        "cer": error_rate(characters, len(expected)),
        "wer": error_rate(word_errors, len(words)),
    }


def compare_baseline(baseline: dict, current: dict) -> list[dict]:
    """Compare a compatible future summary without inventing accuracy gates."""
    if baseline.get("benchmark_version") != current.get("benchmark_version"):
        raise ValueError("Document OCR benchmark version mismatch")
    if baseline.get("extraction_config") != current.get("extraction_config"):
        raise ValueError("Document OCR extraction configuration mismatch")
    previous = baseline.get("metrics", baseline)
    observed = current.get("metrics", current)
    regressions = []
    directions = {"critical_field_accuracy": -1, "parse_failure_count": 1, "consistency_preservation_rate": -1}
    for metric, direction in directions.items():
        before, after = previous.get(metric), observed.get(metric)
        if before is not None and after is not None and direction * (after - before) > 0:
            regressions.append({"metric": metric, "baseline": before, "current": after})
    passed = {(row["document_id"], row["variant"]) for row in baseline.get("documents", []) if row.get("parse_success")}
    for row in current.get("documents", []):
        if (row["document_id"], row["variant"]) in passed and not row.get("parse_success"):
            regressions.append(
                {"metric": "new_parse_failure", "document_id": row["document_id"], "variant": row["variant"]}
            )
    return regressions
