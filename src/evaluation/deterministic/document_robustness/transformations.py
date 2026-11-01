"""Plausible deterministic scans; all numeric settings are recorded with each case."""

import hashlib
from io import BytesIO
from pathlib import Path

from evaluation.deterministic.document_generation import BODY_FONT_SIZE, STYLE_VERSION

VARIANTS = (
    "native_clean",
    "scan_clean",
    "scan_low_resolution",
    "scan_jpeg_compressed",
    "scan_rotated_small",
    "scan_blurred",
    "scan_noisy",
    "mixed_native_and_scan",
)


def variant_config(variant: str, document_id: str) -> dict:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown document variant: {variant}")
    seed = int.from_bytes(hashlib.sha256(document_id.encode()).digest()[:4], "big")
    return {
        "variant": variant,
        "source_dpi": 100 if variant == "scan_low_resolution" else 180,
        "jpeg_quality": 35 if variant == "scan_jpeg_compressed" else None,
        "rotation_degrees": 2.0 if variant == "scan_rotated_small" else 0.0,
        "blur_radius": 0.65 if variant == "scan_blurred" else 0.0,
        "noise_stddev": 8.0 if variant == "scan_noisy" else 0.0,
        "noise_mean": 0.0,
        "seed": seed,
        "font_size": BODY_FONT_SIZE,
        "language": "eng",
        "template_version": STYLE_VERSION,
    }


def _native_page(document, text, *, professional=False, **layout):
    if professional:
        from evaluation.deterministic.document_generation import render_page

        return render_page(document, text, **layout)
    page = document.new_page(width=595, height=842)
    page.insert_text((45, 55), text, fontsize=12, fontname="helv")
    return page


def _raster(text, config, **layout):
    import pymupdf
    from PIL import Image

    with pymupdf.open() as source:
        page = _native_page(source, text, **layout)
        pixmap = page.get_pixmap(dpi=config["source_dpi"], colorspace=pymupdf.csRGB, alpha=False)
        return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)


def _degrade(image, config):
    from PIL import Image, ImageFilter

    if config["rotation_degrees"]:
        image = image.rotate(config["rotation_degrees"], resample=Image.Resampling.BICUBIC, fillcolor="white")
    if config["blur_radius"]:
        image = image.filter(ImageFilter.GaussianBlur(config["blur_radius"]))
    if config["noise_stddev"]:
        import numpy as np

        values = np.asarray(image).astype(np.float32)
        noise = np.random.default_rng(config["seed"]).normal(0, config["noise_stddev"], values.shape[:2])
        image = Image.fromarray(np.clip(values + noise[:, :, None], 0, 255).astype(np.uint8))
    return image


def _scan_page(document, text, config, **layout):
    image = _degrade(_raster(text, config, **layout), config)
    encoded = BytesIO()
    image.save(
        encoded,
        format="JPEG" if config["jpeg_quality"] else "PNG",
        **({"quality": config["jpeg_quality"]} if config["jpeg_quality"] else {}),
    )
    page = document.new_page(width=595, height=842)
    page.insert_image(page.rect, stream=encoded.getvalue())


def materialize_variant(reference, variant: str, path: Path) -> dict:
    """Create a PDF without overwriting; mixed pages partition the same source text."""
    import pymupdf

    if path.exists():
        raise FileExistsError(f"Document variant already exists: {path}")
    config = variant_config(variant, reference.document_id)
    body_text = getattr(reference, "body_text", None)
    text = reference.text if body_text is None else body_text
    layout = {"kind": reference.document_type, "professional": body_text is not None}
    if body_text is None:
        config.update(font_size=12, template_version="plain_v1")
    path.parent.mkdir(parents=True, exist_ok=True)
    with pymupdf.open() as document:
        if variant == "native_clean":
            _native_page(document, text, **layout)
        elif variant == "mixed_native_and_scan":
            lines = text.splitlines()
            split = max(1, len(lines) // 2)
            _native_page(document, "\n".join(lines[:split]), footer=False, **layout)
            _scan_page(document, "\n".join(lines[split:]), config, header=False, **layout)
        else:
            _scan_page(document, text, config, **layout)
        document.save(path, no_new_id=True)
    return config
