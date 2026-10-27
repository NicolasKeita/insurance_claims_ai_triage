"""Inspect the actual local OCR installation without installing anything."""

import importlib
import platform
import shutil
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def _package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def ocr_preflight(language="eng", tesseract_cmd=None):
    result = {
        "python": platform.python_version(),
        "language": language,
        "packages": {name: _package_version(name) for name in ("PyMuPDF", "pytesseract", "Pillow", "numpy", "pypdf")},
        "executable": None,
        "tesseract_version": None,
        "available_languages": [],
        "errors": [],
        "missing_imports": [],
    }
    _inspect_imports(result)
    if not result["errors"]:
        _inspect_tesseract(result, tesseract_cmd)
    result["available"] = not result["errors"]
    result["status"] = "OCR benchmark: READY" if result["available"] else "OCR benchmark: NOT EXECUTED"
    return result


def _inspect_imports(result):
    for name in ("pymupdf", "PIL.Image", "numpy", "pypdf", "pytesseract"):
        try:
            importlib.import_module(name)
        except (Exception, SystemExit) as error:
            result["missing_imports"].append(name)
            result["errors"].append(f"{name}: {type(error).__name__}: {error}")
    unavailable = set(result["missing_imports"])
    result["document_generation_available"] = not unavailable.intersection({"pymupdf", "PIL.Image", "numpy"})
    result["native_available"] = result["document_generation_available"] and "pypdf" not in unavailable


def _inspect_tesseract(result, command):
    try:
        import pytesseract

        if command:
            pytesseract.pytesseract.tesseract_cmd = str(command)
        configured = pytesseract.pytesseract.tesseract_cmd
        result["executable"] = shutil.which(configured) or (
            str(Path(configured).resolve()) if Path(configured).is_file() else None
        )
        if not result["executable"]:
            result["errors"].append(f"Tesseract executable is not accessible: {configured}")
            return
        result["tesseract_version"] = str(pytesseract.get_tesseract_version())
        result["available_languages"] = sorted(pytesseract.get_languages(config=""))
        missing = set(result["language"].split("+")) - set(result["available_languages"])
        if missing:
            result["errors"].append(f"Missing Tesseract language packs: {', '.join(sorted(missing))}")
    except (Exception, SystemExit) as error:
        result["errors"].append(f"Tesseract: {type(error).__name__}: {error}")
