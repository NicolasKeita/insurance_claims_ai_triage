"""Shared configuration and fixed dataset contracts."""

from vision.models import VisionModelMetadata
from vision.taxonomy import CARPARTS_CLASS_TO_PART

NAMES = dict(enumerate(CARPARTS_CLASS_TO_PART))

METADATA = VisionModelMetadata(identifier="synthetic-parts", framework="fake", version="test-only")
