"""Existing source cases and prepared snapshots for CLI unit tests."""

import json
import sys

from claims.review.demo.provider import DEMO_FORMAT
from tests.claims.review_archive._support import IMAGE, PDF


def arguments(monkeypatch, *values):
    monkeypatch.setattr(sys, "argv", ["export_analysis_results.py", *map(str, values)])


def write_case(root, review):
    root.mkdir(parents=True)
    (root / "claim.json").write_text(review.claim.model_dump_json(indent=2), encoding="utf-8")
    (root / "documents").mkdir()
    (root / "documents" / "declaration finale.pdf").write_bytes(PDF)
    (root / "images").mkdir()
    (root / "images" / "photo originale.png").write_bytes(IMAGE)
    return root


def write_prepared(root, reviews):
    root.mkdir(parents=True)
    entries = []
    for index, review in enumerate(reviews):
        scenario = f"source-case-{index}"
        write_case(root / "fixtures" / scenario, review)
        snapshot = f"snapshot-{index}.json"
        (root / snapshot).write_text(review.model_dump_json(indent=2), encoding="utf-8")
        entries.append(
            {
                "claim_id": review.claim.claim_id,
                "scenario_id": scenario,
                "title": f"Recorded case {index}",
                "description": "Already computed controlled review",
                "snapshot": snapshot,
                "images": {"photo originale.png": f"fixtures/{scenario}/images/photo originale.png"},
            }
        )
    manifest = {"format": DEMO_FORMAT, "claims": entries}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root
