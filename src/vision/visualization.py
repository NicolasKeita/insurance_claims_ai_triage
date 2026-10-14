"""Debug rendering from domain observations, never an application data source."""

from pathlib import Path

from vision.images import load_validated_image
from vision.models import ImageDamageAssessment, ImageVisualAssessment
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


def save_multimodal_annotated_image(
    source: Path, assessment: ImageVisualAssessment, destination: Path,
) -> Path:
    """Render the two observations and their associations from one stored result.

    This function performs no inference and never supplies business evidence.
    Part contours are cyan; damage contours use the existing damage palette.
    """
    import textwrap
    from PIL import Image, ImageColor, ImageDraw, ImageFont

    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("Annotated output must differ from the original image")
    image = load_validated_image(source)
    panels = []
    canvas = None
    try:
        damage, parts = assessment.damage_assessment, assessment.part_assessment
        if image.size != (damage.width, damage.height) or image.size != (parts.width, parts.height):
            raise ValueError("Assessment dimensions differ from the EXIF-oriented source image")
        if damage.filename != parts.filename or image.size[0] <= 0:
            raise ValueError("Multimodal panels must refer to the same image")
        image.thumbnail((700, 700))
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except OSError:
            font = ImageFont.load_default(size=16)

        def draw_observations(panel, detections, *, kind):
            overlay = Image.new("RGBA", image.size)
            painter = ImageDraw.Draw(overlay)
            for index, detection in enumerate(detections):
                color = "#00bcd4" if kind == "part" else COLORS[detection.damage_type]
                if detection.polygon is not None:
                    points = [(p.x * (image.width - 1), p.y * (image.height - 1)) for p in detection.polygon.points]
                    painter.polygon(points, fill=(*ImageColor.getrgb(color), 40), outline=color, width=2)
                else:
                    box = detection.bounding_box
                    painter.rectangle((box.x_min * (image.width - 1), box.y_min * (image.height - 1),
                                       box.x_max * (image.width - 1), box.y_max * (image.height - 1)), outline=color, width=2)
            panel.alpha_composite(overlay)
            overlay.close()
            painter = ImageDraw.Draw(panel)
            for index, detection in enumerate(detections):
                color = "#00bcd4" if kind == "part" else COLORS[detection.damage_type]
                category = detection.part_type if kind == "part" else detection.damage_type
                label = f"{'P' if kind == 'part' else 'D'}{index}: {category.value} {detection.confidence:.2f}"
                position = (detection.bounding_box.x_min * (image.width - 1),
                            max(0, detection.bounding_box.y_min * (image.height - 1) - 20))
                # Keep labels visible even when the detection reaches the right edge.
                bounds = painter.textbbox((0, 0), label, font=font)
                position = (max(0, min(position[0], image.width - (bounds[2] - bounds[0]))), position[1])
                painter.rectangle(painter.textbbox(position, label, font=font), fill="#101820")
                painter.text(position, label, font=font, fill=color)

        for kind in ("damage", "part", "fusion"):
            panel = image.convert("RGBA")
            panels.append(panel)
            if kind in ("part", "fusion"):
                draw_observations(panel, parts.detections, kind="part")
            if kind in ("damage", "fusion"):
                draw_observations(panel, damage.detections, kind="damage")
            if kind == "fusion":
                draw = ImageDraw.Draw(panel)
                for item in assessment.evidence:
                    candidates = ([item.primary_association] if item.primary_association is not None
                                  else [association for association in item.associations if association.eligible])
                    for association in candidates:
                        first, second = item.damage_detection.bounding_box, association.candidate_part.bounding_box
                        start = ((first.x_min + first.x_max) / 2 * (image.width - 1),
                                 (first.y_min + first.y_max) / 2 * (image.height - 1))
                        end = ((second.x_min + second.x_max) / 2 * (image.width - 1),
                               (second.y_min + second.y_max) / 2 * (image.height - 1))
                        draw.line((start, end), fill="#ffeb3b", width=3)
                        draw.ellipse((end[0] - 4, end[1] - 4, end[0] + 4, end[1] + 4), fill="#ffeb3b")

        captions = []
        for item in assessment.evidence:
            part = item.vehicle_part.value.upper() if item.vehicle_part is not None else item.association_status.value.upper()
            text = f"D{item.damage_detection_index}: {item.visual_damage_type.value.upper()} -> {part}"
            if item.containment_ratio is not None:
                text += f" ({item.containment_ratio:.2f} containment)"
            if item.primary_association is not None:
                text += f" [{item.primary_association.association_method.value}; IoU {item.primary_association.iou:.3f}]"
            captions.extend(textwrap.wrap(text, width=max(30, image.width * 3 // 9)))
            if item.association_status.value == "ambiguous":
                choices = ", ".join(f"P{a.part_detection_index} {a.candidate_part.part_type.value}: {a.damage_containment_ratio:.2f}"
                                    for a in item.associations if a.eligible)
                captions.extend(textwrap.wrap(f"  Candidates: {choices}", width=max(30, image.width * 3 // 9)))
        if not captions:
            captions = ["No damage observations above the detector threshold. Non-detection is not evidence of absence."]
        margin, header = 12, 58
        canvas = Image.new("RGB", (image.width * 3 + margin * 4, header + image.height + 28 * len(captions) + 42), "#101820")
        draw = ImageDraw.Draw(canvas)
        for index, (panel, title) in enumerate(zip(panels, ("Damage masks", "Vehicle part masks", "Spatial associations"))):
            x = margin + index * (image.width + margin)
            rgb_panel = panel.convert("RGB")
            try:
                canvas.paste(rgb_panel, (x, header))
            finally:
                rgb_panel.close()
            draw.text((x, 12), title, fill="white", font=font)
        for index, text in enumerate(captions):
            draw.text((margin, header + image.height + 14 + index * 28), text, fill="white", font=font)
        destination.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(destination)
    finally:
        if canvas is not None:
            canvas.close()
        for panel in panels:
            panel.close()
        image.close()
    return destination
