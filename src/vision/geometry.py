"""Robust geometry validation for fusion, preserving step29 domain loading."""

from pydantic import ValidationError

from vision.models import NormalizedPolygon


class PolygonGeometryError(ValueError):
    pass


def validated_polygon_geometry(polygon: NormalizedPolygon):
    from shapely.geometry import MultiPolygon, Polygon
    from shapely.validation import explain_validity

    try:
        validated = NormalizedPolygon.model_validate(polygon.model_dump())
    except ValidationError as error:
        raise PolygonGeometryError("Invalid normalized polygon") from error
    def component_geometries(component):
        geometry = Polygon(
            [(point.x, point.y) for point in component.points],
            holes=[[(point.x, point.y) for point in hole] for hole in component.holes],
        )
        if not geometry.is_valid:
            raise PolygonGeometryError(f"Invalid polygon topology: {explain_validity(geometry)}")
        return [geometry, *(shape for child in component.components for shape in component_geometries(child))]

    components = component_geometries(validated)
    geometry = components[0] if len(components) == 1 else MultiPolygon(components)
    if not geometry.is_valid:
        raise PolygonGeometryError(f"Invalid polygon topology: {explain_validity(geometry)}")
    return geometry
