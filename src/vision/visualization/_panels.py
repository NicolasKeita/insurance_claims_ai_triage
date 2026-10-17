"""Shared multimodal panel rendering context and painters."""

from dataclasses import dataclass
from typing import Any

from vision.models import ImageDamageAssessment, ImagePartAssessment, ImageVisualAssessment
from vision.visualization._palette import COLORS, PART_COLOR, _paint_polygon


@dataclass(frozen=True)
class RenderContext:
    """Everything the panel painters need from one stored multimodal result."""

    image: Any
    damage: ImageDamageAssessment
    parts: ImagePartAssessment
    assessment: ImageVisualAssessment
    font: Any


def validated_observations(image: Any, assessment: ImageVisualAssessment) -> tuple[Any, Any]:
    damage, parts = assessment.damage_assessment, assessment.part_assessment
    if image.size != (damage.width, damage.height) or image.size != (parts.width, parts.height):
        raise ValueError("Assessment dimensions differ from the EXIF-oriented source image")
    if damage.filename != parts.filename or image.size[0] <= 0:
        raise ValueError("Multimodal panels must refer to the same image")
    return damage, parts


def panel_font() -> Any:
    from PIL import ImageFont

    try:
        return ImageFont.truetype("arial.ttf", 16)
    except OSError:
        return ImageFont.load_default(size=16)


def _draw_observations(panel: Any, detections: list[Any], kind: str, context: RenderContext) -> None:
    """Recreate the painter after compositing a polygon overlay."""
    from PIL import Image, ImageDraw

    image = context.image
    overlay = Image.new("RGBA", image.size)
    painter = ImageDraw.Draw(overlay)
    for detection in detections:
        color = PART_COLOR if kind == "part" else COLORS[detection.damage_type]
        if detection.polygon is not None:
            _paint_polygon(overlay, detection.polygon, color, 65)
            painter = ImageDraw.Draw(overlay)
        else:
            box = detection.bounding_box
            painter.rectangle(
                (
                    box.x_min * (image.width - 1),
                    box.y_min * (image.height - 1),
                    box.x_max * (image.width - 1),
                    box.y_max * (image.height - 1),
                ),
                outline=color,
                width=2,
            )
    panel.alpha_composite(overlay)
    overlay.close()
    _draw_labels(panel, detections, kind, context)


def _draw_labels(panel: Any, detections: list[Any], kind: str, context: RenderContext) -> None:
    """Keep labels visible even when the detection reaches the right edge."""
    from PIL import ImageDraw

    image, font = context.image, context.font
    painter = ImageDraw.Draw(panel)
    for index, detection in enumerate(detections):
        color = PART_COLOR if kind == "part" else COLORS[detection.damage_type]
        category = detection.part_type if kind == "part" else detection.damage_type
        label = f"{'P' if kind == 'part' else 'D'}{index}: {category.value} {detection.confidence:.2f}"
        position = (
            detection.bounding_box.x_min * (image.width - 1),
            max(0, detection.bounding_box.y_min * (image.height - 1) - 20),
        )
        bounds = painter.textbbox((0, 0), label, font=font)
        position = (max(0, min(position[0], image.width - (bounds[2] - bounds[0]))), position[1])
        painter.rectangle(painter.textbbox(position, label, font=font), fill="#101820")
        painter.text(position, label, font=font, fill=color)


def _draw_association_links(panel: Any, context: RenderContext) -> None:
    """Draw one line per primary association, or per eligible candidate when ambiguous."""
    from PIL import ImageDraw

    image = context.image
    draw = ImageDraw.Draw(panel)
    for item in context.assessment.evidence:
        candidates = (
            [item.primary_association]
            if item.primary_association is not None
            else [association for association in item.associations if association.eligible]
        )
        for association in candidates:
            first, second = item.damage_detection.bounding_box, association.candidate_part.bounding_box
            start = (
                (first.x_min + first.x_max) / 2 * (image.width - 1),
                (first.y_min + first.y_max) / 2 * (image.height - 1),
            )
            end = (
                (second.x_min + second.x_max) / 2 * (image.width - 1),
                (second.y_min + second.y_max) / 2 * (image.height - 1),
            )
            draw.line((start, end), fill="#ffeb3b", width=3)
            draw.ellipse((end[0] - 4, end[1] - 4, end[0] + 4, end[1] + 4), fill="#ffeb3b")
