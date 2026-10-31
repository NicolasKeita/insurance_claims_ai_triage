"""Explicit options for portable review export."""

from dataclasses import dataclass, field
from pathlib import Path

from claims.review.archive.storage.catalogue import DEFAULT_OUTPUT_ROOT


@dataclass
class ExportOptions:
    output_root: Path = DEFAULT_OUTPUT_ROOT
    analysis_id: str | None = None
    title: str | None = None
    document_paths: dict = field(default_factory=dict)
    image_paths: dict = field(default_factory=dict)
    source: dict = field(default_factory=dict)
