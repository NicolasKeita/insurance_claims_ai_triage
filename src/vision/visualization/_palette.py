"""Debug rendering palette and mask painting."""

from typing import Any

from vision.taxonomy import VisualDamageType

COLORS = dict(zip(VisualDamageType, ["#e65100", "#1565c0", "#6a1b9a", "#00838f", "#ad1457", "#2e7d32"]))
PART_COLOR = "#00bcd4"


def _paint_polygon(target: Any, polygon: Any, color: str, alpha: int) -> None:
    """Display the stored mask, including holes and disconnected components."""
    from PIL import Image, ImageColor, ImageDraw

    mask = Image.new("L", target.size)
    painter = ImageDraw.Draw(mask)

    def points(ring):
        return [(point.x * (target.width - 1), point.y * (target.height - 1)) for point in ring]

    def component(value):
        painter.polygon(points(value.points), fill=alpha)
        for hole in value.holes:
            painter.polygon(points(hole), fill=0)
        for child in value.components:
            component(child)

    component(polygon)
    tint = Image.new("RGBA", target.size, (*ImageColor.getrgb(color), 0))
    try:
        tint.putalpha(mask)
        target.alpha_composite(tint)
    finally:
        tint.close()
        mask.close()
