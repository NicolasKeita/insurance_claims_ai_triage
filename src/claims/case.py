from dataclasses import dataclass
from pathlib import Path

from claims.models import Claim


@dataclass(frozen=True)
class ClaimCase:
    claim: Claim
    root_dir: Path

    @property
    def documents_dir(self) -> Path:
        return self.root_dir / "documents"

    @property
    def images_dir(self) -> Path:
        return self.root_dir / "images"

    def document_path(self, filename: str) -> Path:
        return self.documents_dir / filename

    def image_path(self, filename: str) -> Path:
        return self.images_dir / filename