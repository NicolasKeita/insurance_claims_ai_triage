"""Claim orchestration: one inference per model/photo and pre-mapped quote types."""

from vision.multimodal_service._consistency import assess_multimodal_consistency
from vision.multimodal_service._service import ClaimMultimodalVisionService

__all__ = ["ClaimMultimodalVisionService", "assess_multimodal_consistency"]
