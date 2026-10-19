"""Three-panel multimodal rendering with captions and association links."""

from pathlib import Path
from typing import Any

from vision.images import load_validated_image
from vision.models import ImageVisualAssessment
from vision.visualization._panels import (
    RenderContext,
    _draw_association_links,
    _draw_observations,
    panel_font,
    validated_observations,
)

PANEL_TITLES = ("Damage masks", "Vehicle part masks", "Spatial associations")
NO_OBSERVATION_CAPTION = (
    "No damage observations above the detector threshold. Non-detection is not evidence of absence."
)


def _render_panel(panel: Any, kind: str, context: RenderContext) -> None:
    if kind in ("part", "fusion"):
        _draw_observations(panel, context.parts.detections, "part", context)
    if kind in ("damage", "fusion"):
        _draw_observations(panel, context.damage.detections, "damage", context)
    if kind == "fusion":
        _draw_association_links(panel, context)


def _captions(context: RenderContext) -> list[str]:
    import textwrap

    width = max(30, context.image.width * 3 // 9)
    captions = []
    for item in context.assessment.evidence:
        part = (
            item.vehicle_part.value.upper() if item.vehicle_part is not None else item.association_status.value.upper()
        )
        text = f"D{item.damage_detection_index}: {item.visual_damage_type.value.upper()} -> {part}"
        if item.containment_ratio is not None:
            text += f" ({item.containment_ratio:.2f} containment)"
        if item.primary_association is not None:
            text += f" [{item.primary_association.association_method.value}; IoU {item.primary_association.iou:.3f}]"
        captions.extend(textwrap.wrap(text, width=width))
        if item.association_status.value == "ambiguous":
            choices = ", ".join(
                f"P{a.part_detection_index} {a.candidate_part.part_type.value}: {a.damage_containment_ratio:.2f}"
                for a in item.associations
                if a.eligible
            )
            captions.extend(textwrap.wrap(f"  Candidates: {choices}", width=width))
    return captions or [NO_OBSERVATION_CAPTION]


def _assemble_canvas(context: RenderContext, panels: list[Any], captions: list[str]) -> Any:
    from PIL import Image, ImageDraw

    image = context.image
    margin, header = 12, 58
    canvas = Image.new(
        "RGB",
        (image.width * 3 + margin * 4, header + image.height + 28 * len(captions) + 42),
        "#101820",
    )
    draw = ImageDraw.Draw(canvas)
    for index, (panel, title) in enumerate(zip(panels, PANEL_TITLES)):
        x = margin + index * (image.width + margin)
        rgb_panel = panel.convert("RGB")
        try:
            canvas.paste(rgb_panel, (x, header))
        finally:
            rgb_panel.close()
        draw.text((x, 12), title, fill="white", font=context.font)
    for index, text in enumerate(captions):
        draw.text((margin, header + image.height + 14 + index * 28), text, fill="white", font=context.font)
    return canvas


def save_multimodal_annotated_image(
    source: Path,
    assessment: ImageVisualAssessment,
    destination: Path,
) -> Path:
    """Render the two observations and their associations from one stored result.

    This function performs no inference and never supplies business evidence.
    Part contours are cyan; damage contours use the existing damage palette.
    """
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("Annotated output must differ from the original image")
    image = load_validated_image(source)
    panels: list[Any] = []
    canvas = None
    try:
        damage, parts = validated_observations(image, assessment)
        image.thumbnail((700, 700))
        context = RenderContext(
            image=image,
            damage=damage,
            parts=parts,
            assessment=assessment,
            font=panel_font(),
        )
        for kind in ("damage", "part", "fusion"):
            panel = image.convert("RGBA")
            panels.append(panel)
            _render_panel(panel, kind, context)
        canvas = _assemble_canvas(context, panels, _captions(context))
        destination.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(destination)
    finally:
        if canvas is not None:
            canvas.close()
        for panel in panels:
            panel.close()
        image.close()
    return destination
