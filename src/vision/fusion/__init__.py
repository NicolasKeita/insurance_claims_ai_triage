"""Mask-first same-image geometry; containment, not IoU, decides association."""

from vision.fusion._association import (
    damage_part_candidates,
    select_primary_association,
)
from vision.fusion._fusion import fuse_image_assessments
from vision.fusion._overlap import (
    FusionGeometryError,
    GeometryOverlap,
    calculate_overlap,
)
from vision.fusion._snapshot import validate_image_association_snapshot

__all__ = [
    "FusionGeometryError",
    "GeometryOverlap",
    "calculate_overlap",
    "damage_part_candidates",
    "fuse_image_assessments",
    "select_primary_association",
    "validate_image_association_snapshot",
]
