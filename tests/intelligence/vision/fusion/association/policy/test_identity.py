"""Test identity."""

import pytest

from tests.intelligence.vision.fusion.association._constants import METADATA
from tests.intelligence.vision.fusion.association._support import damage, fused, part
from vision.fusion import FusionGeometryError, fuse_image_assessments
from vision.models import (
    ImageDamageAssessment,
    ImagePartAssessment,
)


@pytest.mark.parametrize("changes", [{"filename": "other.png"}, {"width": 99}, {"height": 99}])
def test_different_images_or_coordinate_spaces_cannot_be_fused(changes):
    with pytest.raises(FusionGeometryError, match="same image"):
        fused([damage()], [part()], **changes)


def test_same_filename_with_different_asset_paths_rejected():
    with pytest.raises(FusionGeometryError, match="different image asset"):
        fuse_image_assessments(
            ImageDamageAssessment(filename="photo.png", asset_reference="first/photo.png", width=10, height=10),
            ImagePartAssessment(
                filename="photo.png",
                asset_reference="second/photo.png",
                width=10,
                height=10,
                model_metadata=METADATA,
                confidence_threshold=0.25,
            ),
            damage_model_metadata=METADATA,
        )
