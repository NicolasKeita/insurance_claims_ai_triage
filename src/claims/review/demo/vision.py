"""Validate saved visual observations without loading either vision model."""

from claims.review.demo.images import import_images
from claims.review.models import ReviewSection
from vision.models import ClaimMultimodalVisionAssessment

COMPUTED_FIELDS = {
    "observations",
    "fusion_policy_version",
    "aggregate_mapped_damage_types",
    "aggregate_vehicle_parts",
    "ambiguous_associations",
    "detection_count",
    "observation_summary",
}


def _without_computed(value):
    if isinstance(value, dict):
        return {key: _without_computed(item) for key, item in value.items() if key not in COMPUTED_FIELDS}
    if isinstance(value, list):
        return [_without_computed(item) for item in value]
    return value


def import_vision(root, scenario, observation, sections, timestamp):
    result = observation.get("components", {}).get("vision", {})
    if result.get("status") in {"SKIP", "FAIL"}:
        return _unavailable_vision(root, scenario, observation, sections, timestamp)
    value = observation.get("vision") or observation["facts"].get("vision")
    if value is None:
        return {}
    value = _without_computed(value)
    for image in value.get("images", []):
        image.pop("filename", None)
    checked = ClaimMultimodalVisionAssessment.model_validate(value)
    if checked.claim_id != scenario.inputs["claim_id"]:
        raise ValueError("Saved visual assessment belongs to a different claim")
    payload = checked.model_dump(mode="json")
    images = import_images(root, scenario, observation, payload)
    sections["vision"] = ReviewSection(
        status="AVAILABLE",
        data=payload,
        source=f"vision:{checked.claim_id}",
        updated_at=timestamp,
        reason="Previously computed model observations on authorized demo imagery; no inference on viewing.",
    )
    sections["visual_consistency"] = ReviewSection(
        status="AVAILABLE",
        data=payload["consistency"],
        source=f"visual_consistency:{checked.claim_id}",
        updated_at=timestamp,
    )
    return images


def _unavailable_vision(root, scenario, observation, sections, timestamp):
    failed = observation["components"]["vision"]["status"] == "FAIL"
    status = "FAILED" if failed else "UNAVAILABLE"
    claim = observation["facts"]["claim"]
    payload = {"claim_id": claim["claim_id"], "images": [{"filename": name} for name in claim["images"]]}
    images = import_images(root, scenario, observation, payload)
    sections["vision"] = ReviewSection(
        status=status,
        data=payload,
        source=f"vision:{claim['claim_id']}",
        updated_at=timestamp,
        reason="Vision assessment failed or was unavailable in the saved run; only original photographs are retained.",
    )
    sections["visual_consistency"] = ReviewSection(
        status=status, reason="Visual consistency is unavailable because the visual assessment did not complete."
    )
    return images
