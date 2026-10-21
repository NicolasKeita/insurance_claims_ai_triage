"""Test identity."""

import pytest

from tests.vision.datasets.hf_inspection._support import fixture
from vision.cardd_hf import (
    normalize_fiftyone_dataset,
)
from vision.dataset import DatasetValidationError


def test_multiple_usable_fields_require_explicit_choice(tmp_path):
    raw, sample, label = fixture(tmp_path)
    sample["boxes"]["detections"][0] = label
    with pytest.raises(DatasetValidationError, match="Multiple usable annotation fields"):
        normalize_fiftyone_dataset(raw, "owner/subset", None)
    result = normalize_fiftyone_dataset(raw, "owner/subset", None, annotation_field="damage_regions")
    assert result.annotation_fields == ("damage_regions",)


@pytest.mark.parametrize(
    "what,match",
    [
        ("duplicate", "Duplicate stable sample ID"),
        ("dimension", "dimensions disagree"),
        ("missing", "Downloaded image is missing"),
    ],
)
def test_image_identity_and_actual_image_metadata_are_checked(tmp_path, what, match):
    raw, sample, _ = fixture(tmp_path)
    if what == "duplicate":
        raw.samples.append(sample)
    elif what == "dimension":
        sample["metadata"]["width"] = 10
    else:
        sample["filepath"] = str(tmp_path / "does_not_exist.png")
    with pytest.raises(DatasetValidationError, match=match):
        normalize_fiftyone_dataset(raw, "owner/subset", None)
