"""Plausible deterministic scans; all numeric settings are recorded with each case."""

import hashlib
from io import BytesIO
from pathlib import Path

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
        "font_size": 12,
        "language": "eng",
    }


def _native_page(document, text):
    page = document.new_page(width=595, height=842)
    page.insert_text((45, 55), text, fontsize=12, fontname="helv")
    return page


def _raster(text, config):
    import pymupdf
    from PIL import Image

    with pymupdf.open() as source:
        page = _native_page(source, text)
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


def _scan_page(document, text, config):
    image = _degrade(_raster(text, config), config)
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with pymupdf.open() as document:
        if variant == "native_clean":
            _native_page(document, reference.text)
        elif variant == "mixed_native_and_scan":
            lines = reference.text.splitlines()
            split = max(1, len(lines) // 2)
            _native_page(document, "\n".join(lines[:split]))
            _scan_page(document, "\n".join(lines[split:]), config)
        else:
            _scan_page(document, reference.text, config)
        document.save(path, no_new_id=True)
    return config
