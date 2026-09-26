from pathlib import Path

from claims.models import Claim


class ClaimFileValidationError(Exception):
    pass


def validate_claim_files(
    claim: Claim,
    claim_dir: Path,
) -> None:
    documents_dir = claim_dir / "documents"
    images_dir = claim_dir / "images"

    for document in claim.documents:
        if not document.available:
            continue

        if document.filename is None:
            raise ClaimFileValidationError(
                "Available document has no filename"
            )

        file_path = documents_dir / document.filename

        if not file_path.is_file():
            raise ClaimFileValidationError(
                f"Missing document file: {file_path}"
            )

    for image_filename in claim.images:
        file_path = images_dir / image_filename

        if not file_path.is_file():
            raise ClaimFileValidationError(
                f"Missing image file: {file_path}"
            )