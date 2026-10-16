"""Explicit vision test boundary; the fusion service under test is real."""


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