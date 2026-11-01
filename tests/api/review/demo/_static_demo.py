"""Tiny stored synthetic review fixtures; no pipeline runner or model calls."""

import hashlib
import json

import pymupdf

from claims.review.archive import export_review
from tests.claims.review_archive._support import IMAGE, recorded_review

BODY = (
    "DEMO SYNTHETIC CLAIM FORM\nClaim ID: CLAIM-0001\nIncident date: 20/01/2026\n"
    "Location: Lyon\nVehicle: Renault Clio\nVehicle year: 2021\nCollision type: Front collision\n"
    "Declared damage:\nFront bumper\nInjuries declared: No"
)


def make_archive(root, *, mode="demo", body=BODY, analysis_id="FIRST"):
    assets = root / "source-assets"
    assets.mkdir(exist_ok=True)
    document = assets / "declaration finale.pdf"
    image = assets / "photo originale.png"
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((45, 45), body, fontsize=10)
        pdf.save(document, no_new_id=True)
    image.write_bytes(IMAGE)
    review = recorded_review()
    review.mode = mode
    review.claim.customer.customer_id = "DEMO-E2E-CUSTOMER-visual_additional_damage"
    review.claim.policy.policy_id = "DEMO-E2E-POLICY-visual_additional_damage"
    return export_review(
        review,
        output_root=root / "archives",
        analysis_id=analysis_id,
        title="Stored synthetic claim",
        source={"kind": "prepared-review", "reference": review.claim.claim_id},
        document_paths={document.name: document},
        image_paths={image.name: image},
    )


def tree_bytes(root):
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def without_urls(value):
    if isinstance(value, list):
        return [without_urls(item) for item in value]
    if isinstance(value, dict):
        return {key: without_urls(item) for key, item in value.items() if key not in {"image_url", "document_url"}}
    return value
