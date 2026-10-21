"""Test balance."""

from dataclasses import replace

from tests.vision.datasets.hf_split._support import (
    Instance,
    cooccurrence_patterns,
    fixture,
)
from vision.cardd_hf_split import (
    create_or_load_cardd_hf_split,
    validate_cardd_hf_split,
)


def test_common_cooccurring_class_stays_balanced_after_rare_label_assignment(tmp_path):
    """
    Many common-only images and rare class combinations can cause a naive rare-first greedy allocation to
    reserve most common images for train.
    """
    patterns = cooccurrence_patterns()
    labels = [pattern for pattern, count in patterns for _ in range(count)]
    dataset = fixture(tmp_path, count=len(labels))
    classes = tuple(sorted({label for pattern in labels for label in pattern}))
    samples = tuple(
        replace(sample, instances=tuple(Instance(label) for label in pattern))
        for sample, pattern in zip(dataset.samples, labels)
    )
    dataset = replace(dataset, samples=samples, classes=classes)
    manifest = create_or_load_cardd_hf_split(dataset, tmp_path / "split.json")
    for name in ("train", "val", "test"):
        assert len(manifest["sample_ids"][name]) == manifest["target_counts"][name]
        for label in classes:
            total = sum(label in pattern for pattern in labels)
            observed = manifest["splits"][name]["images_per_category"][label]
            assert abs(observed - total * manifest["ratios"][name]) <= 2
    assert manifest["warnings"] == []
    assert validate_cardd_hf_split(dataset, manifest) == manifest
