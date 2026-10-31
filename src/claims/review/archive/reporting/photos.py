"""Original copied images and finite recorded vector annotations."""

import math

from .text import _data, _label, _relative_url, _text


def _coordinate(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) * 100 if math.isfinite(value) and 0 <= value <= 1 else None


def _polygon(value, depth=0) -> str:
    if not isinstance(value, dict) or depth > 8:
        return ""
    rings = [value.get("points", []), *value.get("holes", [])]
    paths = []
    for ring in rings:
        if not isinstance(ring, list) or not 3 <= len(ring) <= 10000:
            continue
        points = []
        for point in ring:
            if not isinstance(point, dict):
                break
            x, y = _coordinate(point.get("x")), _coordinate(point.get("y"))
            if x is None or y is None:
                break
            points.append(f"{x:g} {y:g}")
        if len(points) == len(ring):
            paths.append(f"M {' L '.join(points)} Z")
    for component in value.get("components", []):
        path = _polygon(component, depth + 1)
        if path:
            paths.append(path)
    return " ".join(paths)


def _overlay(image) -> str:
    assessment = image.get("damage_assessment")
    detections = assessment.get("detections", []) if isinstance(assessment, dict) else []
    shapes = []
    for item in detections:
        if not isinstance(item, dict):
            continue
        path = _polygon(item.get("polygon"))
        if path:
            shapes.append(f'<path d="{_text(path)}"/>')
        box = item.get("bounding_box")
        if not isinstance(box, dict):
            continue
        values = [_coordinate(box.get(key)) for key in ("x_min", "y_min", "x_max", "y_max")]
        if all(value is not None for value in values):
            x1, y1, x2, y2 = values
            if x2 > x1 and y2 > y1:
                shapes.append(f'<rect x="{x1:g}" y="{y1:g}" width="{x2 - x1:g}" height="{y2 - y1:g}"/>')
    if not shapes:
        return ""
    return (
        '<svg class="overlay" viewBox="0 0 100 100" preserveAspectRatio="none" role="img" '
        'aria-label="Contours et cadres enregistrés par le modèle">' + "".join(shapes) + "</svg>"
    )


def _photo(asset, review) -> str:
    url = _relative_url(asset.get("path")) if asset.get("status") == "COPIED" else None
    filename = asset.get("filename", "Photographie du dossier")
    if not url:
        return (
            '<figure class="card photo"><div class="photo-placeholder"><h3>Photo indisponible</h3>'
            f"<p>{_text(_label(asset.get('reason') or asset.get('status')))}</p>"
            f"<small>{_text(filename)}</small></div></figure>"
        )
    images = _data(review.vision).get("images", [])
    recorded = next((item for item in images if item.get("filename") == filename), {})
    overlay = _overlay(recorded) if review.vision.status in {"AVAILABLE", "NO_EVIDENCE"} else ""
    return (
        '<figure class="card photo"><div class="photo-stage">'
        f'<img src="{url}" alt="Photographie du sinistre : {_text(filename)}" loading="lazy">{overlay}</div>'
        f'<figcaption><a href="{url}">{_text(filename)}</a>'
        + (" · Annotations enregistrées" if overlay else " · Photographie originale")
        + "</figcaption></figure>"
    )
