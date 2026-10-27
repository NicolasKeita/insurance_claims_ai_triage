"""Small deterministic raster fixtures require no Tesseract installation."""

import pymupdf
import pytest

from claims.assets import DocumentAsset
from claims.document_ingestion import extract_pdf_text, extract_pdf_text_auto
from claims.enums import DocumentType, TextExtractionMethod
from evaluation.deterministic.document_robustness.corpus import build_corpus
from evaluation.deterministic.document_robustness.transformations import VARIANTS, materialize_variant


class RecordedOcr:
    def __init__(self):
        self.calls = 0

    def extract_text(self, image):
        self.calls += 1
        assert image.width > 0 and image.height > 0
        return "Controlled raster text"


@pytest.mark.parametrize("variant", VARIANTS)
def test_page_detection_and_actual_method(tmp_path, variant):
    reference = build_corpus()[0].documents[0]
    path = tmp_path / f"{variant}.pdf"
    materialize_variant(reference, variant, path)
    asset = DocumentAsset(DocumentType.CLAIM_FORM, path.name, path)
    native = extract_pdf_text(asset)
    engine = RecordedOcr()
    actual = extract_pdf_text_auto(asset, engine, dpi=72)
    if variant == "native_clean":
        assert native.has_text and engine.calls == 0
        assert actual.method == TextExtractionMethod.NATIVE
    elif variant == "mixed_native_and_scan":
        assert native.page_count == 2 and native.pages[0] and not native.pages[1]
        assert actual.method == TextExtractionMethod.HYBRID and engine.calls == 1
    else:
        assert not native.has_text
        assert actual.method == TextExtractionMethod.OCR and engine.calls == 1


def test_noise_and_small_rotation_are_reproducible(tmp_path):
    reference = build_corpus()[0].documents[0]
    clean = tmp_path / "clean.pdf"
    materialize_variant(reference, "scan_clean", clean)
    with pymupdf.open(clean) as document:
        clean_pixels = document[0].get_pixmap().samples
    for variant in ("scan_noisy", "scan_rotated_small"):
        first, second = tmp_path / f"{variant}_a.pdf", tmp_path / f"{variant}_b.pdf"
        config = materialize_variant(reference, variant, first)
        assert materialize_variant(reference, variant, second) == config
        with pymupdf.open(first) as left, pymupdf.open(second) as right:
            assert left[0].get_pixmap().samples == right[0].get_pixmap().samples
            assert left[0].get_pixmap().samples != clean_pixels
        if variant == "scan_rotated_small":
            assert 0 < config["rotation_degrees"] < 5


def test_variants_record_settings_and_never_overwrite(tmp_path):
    reference = build_corpus()[0].documents[0]
    path = tmp_path / "scan.pdf"
    config = materialize_variant(reference, "scan_jpeg_compressed", path)
    assert config["jpeg_quality"] == 35
    assert config["source_dpi"] == 180
    assert config["noise_stddev"] == 0
    with pytest.raises(FileExistsError):
        materialize_variant(reference, "scan_clean", path)
    with pytest.raises(ValueError):
        materialize_variant(reference, "unknown", tmp_path / "invalid.pdf")
