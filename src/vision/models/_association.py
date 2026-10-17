"""Damage-to-part association records and their preserved geometry contract."""

import math
from enum import Enum

from pydantic import Field, model_validator

from vision.models._assessments import VehiclePartDetection
from vision.models._base import UnitFloat, VisionDomainModel


class AssociationMethod(str, Enum):
    MASK = "MASK"
    BBOX_FALLBACK = "BBOX_FALLBACK"


class AssociationStatus(str, Enum):
    ASSIGNED = "assigned"
    AMBIGUOUS = "ambiguous"
    UNASSIGNED = "unassigned"


class DamagePartAssociation(VisionDomainModel):
    damage_detection_index: int = Field(ge=0)
    part_detection_index: int = Field(ge=0)
    candidate_part: VehiclePartDetection
    damage_confidence: UnitFloat
    part_confidence: UnitFloat
    intersection_area: float = Field(ge=0, le=1, allow_inf_nan=False)
    damage_area: float = Field(gt=0, le=1, allow_inf_nan=False)
    part_area: float = Field(gt=0, le=1, allow_inf_nan=False)
    damage_containment_ratio: UnitFloat
    iou: UnitFloat
    association_method: AssociationMethod
    eligible: bool

    @model_validator(mode="after")
    def validate_overlap(self):
        if self.intersection_area > min(self.damage_area, self.part_area) + 1e-12:
            raise ValueError("Intersection area cannot exceed either source area")
        containment = self.intersection_area / self.damage_area
        union = self.damage_area + self.part_area - self.intersection_area
        if union <= 0:
            raise ValueError("Association union area must be positive")
        if not math.isclose(
            self.damage_containment_ratio, containment, rel_tol=1e-9, abs_tol=1e-12
        ) or not math.isclose(self.iou, self.intersection_area / union, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("Association containment and IoU must agree with the preserved areas")
        return self
