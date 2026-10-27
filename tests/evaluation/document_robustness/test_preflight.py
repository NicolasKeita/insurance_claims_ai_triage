"""Preflight reports observed versions/languages and precise failures without installs."""

import sys
from types import SimpleNamespace

import pytest

from evaluation.deterministic.document_robustness.runtime import preflight


@pytest.fixture
def fake_tesseract(monkeypatch):
    fake = SimpleNamespace(
        pytesseract=SimpleNamespace(tesseract_cmd="tesseract"),
        get_tesseract_version=lambda: "5.4.1-test",
        get_languages=lambda **_: ["osd", "eng"],
    )
    monkeypatch.setitem(sys.modules, "pytesseract", fake)
    monkeypatch.setattr(preflight.importlib, "import_module", lambda _: object())
    monkeypatch.setattr(preflight.shutil, "which", lambda _: "C:/demo/tesseract.exe")
    monkeypatch.setattr(preflight, "_package_version", lambda _: "demo-version")
    return fake


def test_preflight_records_real_api_values_and_explicit_language(fake_tesseract):
    actual = preflight.ocr_preflight("eng", "C:/demo/tesseract.exe")
    assert actual["available"] is True and actual["errors"] == []
    assert actual["tesseract_version"] == "5.4.1-test"
    assert actual["available_languages"] == ["eng", "osd"]
    assert actual["language"] == "eng"
    assert actual["executable"] == "C:/demo/tesseract.exe"
    assert actual["packages"]["PyMuPDF"] == "demo-version"
    assert actual["native_available"] is True and actual["document_generation_available"] is True
    assert actual["missing_imports"] == []
    assert fake_tesseract.pytesseract.tesseract_cmd == "C:/demo/tesseract.exe"


def test_missing_requested_language_is_not_ready(fake_tesseract):
    actual = preflight.ocr_preflight("eng+fra")
    assert actual["available"] is False
    assert actual["status"] == "OCR benchmark: NOT EXECUTED"
    assert actual["errors"] == ["Missing Tesseract language packs: fra"]


@pytest.mark.parametrize("api", ("get_tesseract_version", "get_languages"))
def test_executable_and_language_api_errors_are_preserved(monkeypatch, fake_tesseract, api):
    def fail(**_):
        raise RuntimeError("controlled Tesseract failure")

    monkeypatch.setattr(fake_tesseract, api, fail)
    actual = preflight.ocr_preflight()
    assert actual["available"] is False
    assert actual["errors"] == ["Tesseract: RuntimeError: controlled Tesseract failure"]


@pytest.mark.parametrize("missing", ("pymupdf", "PIL.Image", "numpy", "pypdf", "pytesseract"))
def test_import_failure_records_module_and_cause(monkeypatch, fake_tesseract, missing):
    def import_module(name):
        if name == missing:
            raise ImportError("controlled missing dependency")
        return object()

    monkeypatch.setattr(preflight.importlib, "import_module", import_module)
    actual = preflight.ocr_preflight()
    assert actual["available"] is False and actual["tesseract_version"] is None
    assert actual["errors"] == [f"{missing}: ImportError: controlled missing dependency"]
    assert actual["missing_imports"] == [missing]
    assert actual["native_available"] is (missing == "pytesseract")
    assert actual["document_generation_available"] is (missing in ("pypdf", "pytesseract"))


def test_tesseract_system_exit_is_reported(monkeypatch, fake_tesseract):
    def fail():
        raise SystemExit(7)

    monkeypatch.setattr(fake_tesseract, "get_tesseract_version", fail)
    actual = preflight.ocr_preflight()
    assert actual["available"] is False
    assert actual["errors"] == ["Tesseract: SystemExit: 7"]


def test_missing_executable_is_explicit(monkeypatch, fake_tesseract):
    monkeypatch.setattr(preflight.shutil, "which", lambda _: None)
    actual = preflight.ocr_preflight()
    assert actual["available"] is False
    assert actual["errors"] == ["Tesseract executable is not accessible: tesseract"]
