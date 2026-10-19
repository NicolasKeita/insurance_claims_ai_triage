"""Explicit vision test boundary; the fusion service under test is real."""

from hashlib import sha256

from evaluation.models import ComponentResult, TraceEvent


def fake_vision_service(case, fixture):
    """Construct only explicit test observations; the fusion service is real."""
    from vision.detector import FakeDamageDetector
    from vision.models import NormalizedBoundingBox, NormalizedPolygon, VehiclePartDetection, VisualDamageDetection
    from vision.multimodal_service import ClaimMultimodalVisionService
    from vision.parts_detector import FakeVehiclePartDetector

    detections, parts = [], []
    observations = fixture.get("detections", [fixture] if fixture.get("damage_type") else [])
    for item in observations:
        damage = _morphology(item, VisualDamageDetection, NormalizedBoundingBox, NormalizedPolygon)
        detections.append(damage)
        parts.append(_part(item, VehiclePartDetection, NormalizedBoundingBox, NormalizedPolygon))
    name = case.claim.images[0] if case.claim.images else "synthetic_geometry.png"
    return ClaimMultimodalVisionService(FakeDamageDetector({name: detections}), FakeVehiclePartDetector({name: parts}))


def _geometry(values, bounding_box, polygon):
    x1, y1, x2, y2 = values
    return {
        "bounding_box": bounding_box(x_min=x1, y_min=y1, x_max=x2, y_max=y2),
        "polygon": polygon(points=[{"x": x1, "y": y1}, {"x": x2, "y": y1}, {"x": x2, "y": y2}, {"x": x1, "y": y2}]),
    }


def _morphology(item, detection, bounding_box, polygon):
    box = item.get("damage_box", [0.3, 0.3, 0.5, 0.5])
    return detection(
        damage_type=item.get("damage_type", "SCRATCH").lower(),
        confidence=item.get("damage_confidence", 0.95),
        model_identifier="fake-damage-detector",
        model_version="test-only",
        **_geometry(box, bounding_box, polygon),
    )


def _part(item, detection, bounding_box, polygon):
    part_box = item.get("part_box", [0.1, 0.1, 0.9, 0.9])
    return detection(
        part_type=item.get("vehicle_part", "HOOD").lower(),
        confidence=item.get("part_confidence", 0.96),
        **_geometry(part_box, bounding_box, polygon),
    )


def _skip_vision(run, reason):
    run.output.components["vision"] = ComponentResult(status="SKIP", reason=reason, metadata={"applicable": False})
    run.output.trace.append(TraceEvent(component="vision", status="SKIP", outputs={"reason": reason}))


def _image_provenance(run):
    return [
        {
            "filename": filename,
            "synthetic": True,
            "asset_sha256": sha256(run.case.image_path(filename).read_bytes()).hexdigest(),
            "description": "Generated geometry-only pixels; detector observations supplied explicitly by the scenario.",
        }
        for filename in run.case.claim.images
    ]


def assess_vision(run):
    """Record explicit vision observations, or an explicit non-applicable skip."""
    if not run.scenario.vision_required:
        _skip_vision(run, "Document-centric scenario; no vision fixture requested.")
        return
    from claims.consistency import map_quote_damage_types

    output = run.output
    run.visual = run.recorder.record(
        "vision",
        lambda: fake_vision_service(run.case, run.scenario.inputs.get("vision_fixture", {})).assess_claim(
            run.case, quoted_damage_types=map_quote_damage_types(run.quote)[0]
        ),
    )
    output.vision = {
        **run.visual.model_dump(mode="json"),
        "observations": [item.model_dump(mode="json") for item in run.visual.observations],
    }
    output.facts["vision"] = output.vision
    output.model_versions.update(
        damage=run.visual.damage_model_metadata.version or "unknown",
        parts=run.visual.part_model_metadata.version or "unknown",
    )
    output.policy_versions["fusion"] = run.visual.fusion_policy.version
    output.provenance["images"] = _image_provenance(run)
