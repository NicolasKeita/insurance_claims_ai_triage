"""Export saved demonstration results and sources for an autonomous static site."""

import tempfile
from pathlib import Path

from claims.review.archive.assets import plain_name
from claims.review.demo.provider import DemoReviewService
from evaluation.deterministic.document_generation import STYLE_VERSION
from evaluation.reporting.static_demo.assets import static_review
from evaluation.reporting.static_demo.storage import FORMAT, publish, safe_output, write_json


def export_static_demo(source_dir, output_dir, *, replace=False):
    """Publish saved demo reviews only; never invoke analysis, services or models."""
    service = DemoReviewService(safe_output(source_dir))
    if service.archive is None:
        raise ValueError("Static export requires an existing exported demo analysis archive")
    output = safe_output(output_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    claims = []
    with tempfile.TemporaryDirectory(prefix=".static-demo-", dir=output.parent) as temporary:
        staging = Path(temporary)
        for entry in service.archive.analyses.values():
            claim_id, analysis_id = entry["claim_id"], entry["analysis_id"]
            review = service.get_review(claim_id, analysis_id)
            if review.mode != "demo":
                raise ValueError("Only synthetic demonstration reviews may be exported as static demo data")
            filename = f"{plain_name(claim_id)}_{plain_name(analysis_id)}.json"
            relative = f"reviews/{filename}"
            data, files = static_review(service, review, entry, staging)
            write_json(staging / relative, data)
            claims.append(
                {
                    "claim_id": claim_id,
                    "analysis_id": analysis_id,
                    "review": relative,
                    "source_fingerprint": entry["fingerprint"],
                    "source_review_sha256": entry["review_sha256"],
                    "files": files,
                }
            )
        write_json(staging / "catalog.json", [item.model_dump() for item in service.list_demo_claims()])
        write_json(
            staging / "manifest.json",
            {
                "format": FORMAT,
                "template_version": STYLE_VERSION,
                "scope": "Presentation-only; recorded analysis results were not recomputed.",
                "claims": claims,
            },
        )
        return publish(staging, output, replace)
