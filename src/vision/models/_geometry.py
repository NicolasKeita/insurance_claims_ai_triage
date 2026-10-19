"""Normalized, EXIF-oriented point, polygon and bounding-box geometry."""

from pydantic import Field, model_validator

from vision.models._base import UnitFloat, VisionDomainModel


class NormalizedBoundingBox(VisionDomainModel):
    x_min: UnitFloat
    y_min: UnitFloat
    x_max: UnitFloat
    y_max: UnitFloat

    @model_validator(mode="after")
    def validate_extent(self):
        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError("Bounding box must have positive width and height")
        return self


class NormalizedPoint(VisionDomainModel):
    x: UnitFloat
    y: UnitFloat


class NormalizedPolygon(VisionDomainModel):
    """Step29 exterior plus optional holes and disconnected mask components.

    Existing single-contour artifacts retain their loading contract. New mask
    fields preserve observed topology and are checked by the geometry layer.
    """

    points: list[NormalizedPoint] = Field(min_length=3)
    holes: list[list[NormalizedPoint]] = Field(default_factory=list)
    components: list["NormalizedPolygon"] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_area(self):
        if abs(_ring_area_twice(self.points)) <= 1e-12:
            raise ValueError("Polygon must enclose a nonzero area")
        for hole in self.holes:
            if len(hole) < 3:
                raise ValueError("Polygon hole must contain at least three points")
            if abs(_ring_area_twice(hole)) <= 1e-12:
                raise ValueError("Polygon hole must enclose a nonzero area")
        return self


def _ring_area_twice(ring: list[NormalizedPoint]) -> float:
    return sum(
        point.x * ring[(index + 1) % len(ring)].y - ring[(index + 1) % len(ring)].x * point.y
        for index, point in enumerate(ring)
    )
