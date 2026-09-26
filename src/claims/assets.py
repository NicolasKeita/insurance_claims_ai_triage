import mimetypes
from dataclasses import dataclass
from pathlib import Path

from claims.enums import DocumentType
from claims.case import ClaimCase


@dataclass(frozen=True)
class DocumentAsset:
    type: DocumentType
    filename: str
    path: Path

    @property
    def extension(self) -> str:
        return self.path.suffix.lower()

    @property
    def size_bytes(self) -> int:
        return self.path.stat().st_size

    @property
    def media_type(self) -> str | None:
        media_type, _ = mimetypes.guess_type(
            self.filename
        )

        return media_type


@dataclass(frozen=True)
class ImageAsset:
    filename: str
    path: Path

    @property
    def extension(self) -> str:
        return self.path.suffix.lower()

    @property
    def size_bytes(self) -> int:
        return self.path.stat().st_size

    @property
    def media_type(self) -> str | None:
        media_type, _ = mimetypes.guess_type(
            self.filename
        )

        return media_type

def resolve_document_assets(
    case: ClaimCase,
) -> list[DocumentAsset]:
    assets = []

    for document in case.claim.documents:
        if not document.available:
            continue

        if document.filename is None:
            continue

        asset = DocumentAsset(
            type=document.type,
            filename=document.filename,
            path=case.document_path(
                document.filename
            ),
        )

        assets.append(asset)

    return assets


def resolve_image_assets(
    case: ClaimCase,
) -> list[ImageAsset]:
    return [
        ImageAsset(
            filename=filename,
            path=case.image_path(filename),
        )
        for filename in case.claim.images
    ]