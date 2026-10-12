"""Debug rendering from domain observations, never an application data source."""

from pathlib import Path

from vision.images import load_validated_image
from vision.models import ImageDamageAssessment
from vision.taxonomy import VisualDamageType

COLORS = dict(zip(VisualDamageType, ["#e65100", "#1565c0", "#6a1b9a", "#00838f", "#ad1457", "#2e7d32"]))


def save_annotated_image(source: Path, assessment: ImageDamageAssessment, destination: Path) -> Path:
    from PIL import Image, ImageColor, ImageDraw

    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("Annotated output must differ from the original image")
    image = load_validated_image(source)
    try:
        if image.size != (assessment.width, assessment.height):
            raise ValueError("Assessment dimensions differ from the EXIF-oriented source image")
        overlay = Image.new("RGBA", image.size)
        overlay_draw = ImageDraw.Draw(overlay)
        for detection in assessment.detections:
            if detection.polygon is not None:
                points = [(point.x * (image.width - 1), point.y * (image.height - 1)) for point in detection.polygon.points]
                overlay_draw.polygon(points, fill=(*ImageColor.getrgb(COLORS[detection.damage_type]), 75))
        rendered = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
        overlay.close()
        try:
            draw = ImageDraw.Draw(rendered)
            for detection in assessment.detections:
                box = detection.bounding_box
                rectangle = (box.x_min * (image.width - 1), box.y_min * (image.height - 1), box.x_max * (image.width - 1), box.y_max * (image.height - 1))
                color = COLORS[detection.damage_type]
                draw.rectangle(rectangle, outline=color, width=2)
                label = f"{detection.damage_type.value.replace('_', ' ')} {detection.confidence:.2f}"
                position = (rectangle[0], max(0, rectangle[1] - 14))
                draw.rectangle(draw.textbbox(position, label), fill=color)
                draw.text(position, label, fill="white")
            destination.parent.mkdir(parents=True, exist_ok=True)
            rendered.save(destination)
        finally:
            rendered.close()
    finally:
        image.close()
    return destination
