from typing import Protocol

from PIL import Image
import pytesseract

from pytesseract import (
    TesseractError,
    TesseractNotFoundError,
)


class OcrEngine(Protocol):
    def extract_text(
        self,
        image: Image.Image,
    ) -> str:
        ...

class OcrError(Exception):
    pass

class TesseractOcrEngine:
    def __init__(
        self,
        language: str = "eng",
        timeout: float = 30,
    ):
        self.language = language
        self.timeout = timeout

    def extract_text(
        self,
        image: Image.Image,
    ) -> str:
        try:
            return pytesseract.image_to_string(
                image,
                lang=self.language,
                timeout=self.timeout,
            ).strip()

        except TesseractNotFoundError as error:
            raise OcrError(
                "Tesseract executable was not found"
            ) from error

        except TesseractError as error:
            raise OcrError(
                f"Tesseract OCR failed: {error}"
            ) from error

        except RuntimeError as error:
            raise OcrError(
                "Tesseract OCR timed out"
            ) from error