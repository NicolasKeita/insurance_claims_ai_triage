"""Synthetic multilabel/content fixtures test project splits, never CarDD quality."""

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from vision.cardd_hf_split import (
    ALGORITHM,
    DEFAULT_SPLIT_MANIFEST_PATH,
    SPLIT_VERSION,
    create_or_load_cardd_hf_split,
    validate_cardd_hf_split,
)
from vision.dataset import DatasetValidationError


@dataclass(frozen=True)
class Instance:
    label: str
    polygons: tuple = (((0.1, 0.1), (0.8, 0.1), (0.8, 0.8)),)
    annotation_field: str = "ground_truth"
    source_kind: str = "polyline"


@dataclass(frozen=True)
class Sample:
    sample_id: str
    filepath: Path
    width: int
    height: int
    instances: tuple
    image_sha256: str


@dataclass(frozen=True)
class Dataset:
    samples: tuple
    repository: str = "synthetic/CarDD-mirror"
    revision: str | None = "a" * 40
    field_schema: dict | None = None
    annotation_fields: tuple = ("ground_truth",)
    classes: tuple = ("dent", "scratch", "crack")
    source_format: str = "FiftyOneDataset"


def fixture(tmp_path, count=100):
    samples = []
    for index in range(count):
        path = tmp_path / f"image_{index:04d}.bin"
        content = f"synthetic pixels {index}".encode()
        path.write_bytes(content)
        labels = ["dent"]
        if index % 2 == 0:
            labels.append("scratch")
        if index % 5 == 0:
            labels.append("crack")
        samples.append(
            Sample(
                str(index),
                path,
                100,
                80,
                tuple(Instance(label) for label in labels),
                hashlib.sha256(content).hexdigest(),
            )
        )
    return Dataset(tuple(samples), field_schema={"ground_truth": "Polylines"})


def assigned(manifest, sample_id):
    return next(name for name, ids in manifest["sample_ids"].items() if sample_id in ids)


def test_custom_multilabel_split_has_exact_counts_and_balanced_classes(tmp_path):
    dataset = fixture(tmp_path)
    manifest = create_or_load_cardd_hf_split(dataset, tmp_path / "split.json")
    assert manifest["target_counts"] == {"train": 70, "val": 15, "test": 15}
    assert {name: len(ids) for name, ids in manifest["sample_ids"].items()} == manifest["target_counts"]
    for name in ("train", "val", "test"):
        for label, total in (("dent", 100), ("scratch", 50), ("crack", 20)):
            assert abs(manifest["splits"][name]["images_per_category"][label] - total * manifest["ratios"][name]) <= 1
    assert manifest["warnings"] == []
    assert manifest["official_splits_used"] is False
    assert "HF mirror" in manifest["source_name"]
    assert set(manifest["annotation_hashes"]) == {"train", "val", "test"}
    assert validate_cardd_hf_split(dataset, manifest) == manifest
    assert manifest["split_version"] == SPLIT_VERSION == "cardd_hf_split_v2"
    assert manifest["algorithm"] == ALGORITHM
    assert DEFAULT_SPLIT_MANIFEST_PATH.name == "cardd_hf_split_v2.json"


def test_split_independent_of_input_order_and_cache_root(tmp_path):
    first_dir, second_dir = tmp_path / "first", tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first = fixture(first_dir)
    second = fixture(second_dir)
    second = replace(second, samples=tuple(reversed(second.samples)))
    a = create_or_load_cardd_hf_split(first, tmp_path / "a.json")
    b = create_or_load_cardd_hf_split(second, tmp_path / "b.json")
    assert a == b
    assert str(tmp_path) not in json.dumps(a)
    assert validate_cardd_hf_split(second, a) == a


def test_frozen_manifest_is_reused_without_rewrite(tmp_path, monkeypatch):
    import vision.cardd_hf_split as module

    dataset = fixture(tmp_path)
    path = tmp_path / "split.json"
    first = create_or_load_cardd_hf_split(dataset, path)
    before = path.read_bytes(), path.stat().st_mtime_ns
    monkeypatch.setattr(module, "_assign", lambda *args: pytest.fail("A frozen split must not be regenerated"))
    assert create_or_load_cardd_hf_split(dataset, path) == first
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before


def test_existing_v1_manifest_stays_valid_and_immutable_after_v2_upgrade(tmp_path, monkeypatch):
    import vision.cardd_hf_split as module

    dataset = fixture(tmp_path)
    path = tmp_path / "cardd_hf_split_v1.json"
    legacy = create_or_load_cardd_hf_split(dataset, tmp_path / "initial.json")
    legacy["split_version"] = "cardd_hf_split_v1"
    legacy["algorithm"] = "deterministic_multilabel_greedy_grouped_v1"
    stable = {
        "source_fingerprint": legacy["source_fingerprint"],
        "sample_ids": legacy["sample_ids"],
        "split_version": legacy["split_version"],
        "algorithm": legacy["algorithm"],
        "seed": legacy["seed"],
        "ratios": legacy["ratios"],
    }
    legacy["dataset_fingerprint"] = hashlib.sha256(
        json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path.write_text(json.dumps(legacy), encoding="utf-8")
    before = path.read_bytes(), path.stat().st_mtime_ns
    monkeypatch.setattr(module, "_assign", lambda *args: pytest.fail("The legacy frozen split must be reused"))
    assert create_or_load_cardd_hf_split(dataset, path) == legacy
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before


def test_common_cooccurring_class_stays_balanced_after_rare_label_assignment(tmp_path):
    # Many common-only images and rare class combinations can cause a naive
    # rare-first greedy allocation to reserve most common images for train.
    patterns = [
        (("scratch",), 70),
        (("glass_shatter",), 40),
        (("dent", "scratch"), 36),
        (("dent",), 36),
        (("tire_flat",), 17),
        (("dent", "lamp_broken", "scratch"), 10),
        (("dent", "lamp_broken"), 9),
        (("crack", "dent", "scratch"), 8),
        (("crack", "scratch"), 7),
        (("crack", "dent", "lamp_broken", "scratch"), 7),
        (("lamp_broken",), 7),
        (("crack",), 6),
        (("lamp_broken", "scratch"), 6),
        (("dent", "glass_shatter"), 5),
        (("crack", "dent"), 4),
        (("crack", "dent", "lamp_broken"), 4),
        (("crack", "lamp_broken", "scratch"), 2),
        (("crack", "lamp_broken"), 2),
    ]
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


@pytest.mark.parametrize(
    "change",
    [
        lambda dataset: replace(dataset, revision="b" * 40),
        lambda dataset: replace(dataset, repository="other/mirror"),
        lambda dataset: replace(dataset, classes=tuple(reversed(dataset.classes))),
        lambda dataset: replace(dataset, samples=dataset.samples[:-1]),
        lambda dataset: replace(
            dataset,
            samples=(
                replace(
                    dataset.samples[0], instances=(Instance("dent", polygons=(((0.2, 0.2), (0.9, 0.2), (0.9, 0.9)),)),)
                ),
                *dataset.samples[1:],
            ),
        ),
        lambda dataset: replace(dataset, annotation_fields=("other_field",)),
        lambda dataset: replace(
            dataset, samples=(replace(dataset.samples[0], sample_id="other-id"), *dataset.samples[1:])
        ),
    ],
)
def test_frozen_manifest_rejects_changed_source_without_replacement(tmp_path, change):
    dataset = fixture(tmp_path)
    path = tmp_path / "split.json"
    create_or_load_cardd_hf_split(dataset, path)
    before = path.read_bytes()
    with pytest.raises(DatasetValidationError, match="inconsistent"):
        create_or_load_cardd_hf_split(change(dataset), path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("kwargs", [{"seed": 43}, {"ratios": {"train": 0.6, "val": 0.2, "test": 0.2}}])
def test_frozen_manifest_rejects_changed_split_parameters(tmp_path, kwargs):
    dataset = fixture(tmp_path)
    path = tmp_path / "split.json"
    create_or_load_cardd_hf_split(dataset, path)
    with pytest.raises(DatasetValidationError, match="inconsistent"):
        create_or_load_cardd_hf_split(dataset, path, **kwargs)


def test_changed_actual_image_bytes_fail_even_if_declared_hash_is_stale(tmp_path):
    dataset = fixture(tmp_path)
    path = tmp_path / "split.json"
    create_or_load_cardd_hf_split(dataset, path)
    dataset.samples[0].filepath.write_bytes(b"changed pixels")
    with pytest.raises(DatasetValidationError, match="content changed"):
        create_or_load_cardd_hf_split(dataset, path)


def test_changed_content_with_updated_hash_rejects_frozen_manifest(tmp_path):
    dataset = fixture(tmp_path)
    path = tmp_path / "split.json"
    create_or_load_cardd_hf_split(dataset, path)
    dataset.samples[0].filepath.write_bytes(b"changed pixels")
    dataset = replace(
        dataset,
        samples=(
            replace(dataset.samples[0], image_sha256=hashlib.sha256(b"changed pixels").hexdigest()),
            *dataset.samples[1:],
        ),
    )
    with pytest.raises(DatasetValidationError, match="inconsistent"):
        create_or_load_cardd_hf_split(dataset, path)


def test_byte_identical_images_are_kept_together_without_losing_samples(tmp_path):
    dataset = fixture(tmp_path)
    original = dataset.samples[0]
    duplicate = dataset.samples[1]
    duplicate.filepath.write_bytes(original.filepath.read_bytes())
    dataset = replace(
        dataset, samples=(original, replace(duplicate, image_sha256=original.image_sha256), *dataset.samples[2:])
    )
    manifest = create_or_load_cardd_hf_split(dataset, tmp_path / "split.json")
    assert assigned(manifest, original.sample_id) == assigned(manifest, duplicate.sample_id)
    assert manifest["duplicate_grouping"]["duplicate_image_sha256_groups"] == 1
    assert manifest["duplicate_grouping"]["samples_in_duplicate_groups"] == 2
    assert sum(len(ids) for ids in manifest["sample_ids"].values()) == 100
    assert validate_cardd_hf_split(dataset, manifest) == manifest


def test_same_filepath_different_ids_are_kept_together(tmp_path):
    dataset = fixture(tmp_path)
    original = dataset.samples[0]
    dataset = replace(
        dataset,
        samples=(
            original,
            replace(dataset.samples[1], filepath=original.filepath, image_sha256=original.image_sha256),
            *dataset.samples[2:],
        ),
    )
    manifest = create_or_load_cardd_hf_split(dataset, tmp_path / "split.json")
    assert assigned(manifest, "0") == assigned(manifest, "1")
    assert manifest["duplicate_grouping"]["duplicate_filepath_groups"] == 1


def test_frozen_split_detects_identical_bytes_crossing_splits_after_id_swap(tmp_path):
    dataset = fixture(tmp_path)
    original, duplicate = dataset.samples[:2]
    duplicate.filepath.write_bytes(original.filepath.read_bytes())
    dataset = replace(
        dataset, samples=(original, replace(duplicate, image_sha256=original.image_sha256), *dataset.samples[2:])
    )
    manifest = create_or_load_cardd_hf_split(dataset, tmp_path / "split.json")
    original_split = assigned(manifest, "0")
    other_split = next(name for name in ("train", "val", "test") if name != original_split)
    # IDs remain unique and split sizes unchanged; content still must not leak.
    donor_id = manifest["sample_ids"][other_split][0]
    manifest["sample_ids"][original_split].remove("1")
    manifest["sample_ids"][original_split].append(donor_id)
    manifest["sample_ids"][other_split].remove(donor_id)
    manifest["sample_ids"][other_split].append("1")
    with pytest.raises(DatasetValidationError, match="identical image content"):
        validate_cardd_hf_split(dataset, manifest)


def test_duplicate_groups_incompatible_with_exact_sizes_fail_cleanly(tmp_path):
    dataset = fixture(tmp_path, count=6)
    samples = list(dataset.samples)
    for first, second in ((0, 1), (2, 3), (4, 5)):
        samples[second].filepath.write_bytes(samples[first].filepath.read_bytes())
        samples[second] = replace(samples[second], image_sha256=samples[first].image_sha256)
    path = tmp_path / "split.json"
    with pytest.raises(DatasetValidationError, match="cannot fit"):
        create_or_load_cardd_hf_split(replace(dataset, samples=tuple(samples)), path)
    assert not path.exists()


def test_duplicate_sample_ids_are_rejected(tmp_path):
    dataset = fixture(tmp_path)
    dataset = replace(
        dataset, samples=(dataset.samples[0], replace(dataset.samples[1], sample_id="0"), *dataset.samples[2:])
    )
    with pytest.raises(DatasetValidationError, match="duplicate sample ID"):
        create_or_load_cardd_hf_split(dataset, tmp_path / "split.json")


def test_manifest_sample_overlap_and_distribution_tampering_rejected(tmp_path):
    dataset = fixture(tmp_path)
    manifest = create_or_load_cardd_hf_split(dataset, tmp_path / "split.json")
    manifest["sample_ids"]["val"][0] = manifest["sample_ids"]["train"][0]
    with pytest.raises(DatasetValidationError, match="leakage"):
        validate_cardd_hf_split(dataset, manifest)
    manifest = create_or_load_cardd_hf_split(dataset, tmp_path / "split.json")
    manifest["splits"]["test"]["images_per_category"]["dent"] -= 1
    with pytest.raises(DatasetValidationError, match="distribution"):
        validate_cardd_hf_split(dataset, manifest)


def test_three_rare_class_images_represented_in_each_split(tmp_path):
    dataset = fixture(tmp_path)
    samples = tuple(
        replace(sample, instances=(Instance("dent"),) + ((Instance("crack"),) if int(sample.sample_id) < 3 else ()))
        for sample in dataset.samples
    )
    dataset = replace(dataset, samples=samples, classes=("dent", "crack"))
    manifest = create_or_load_cardd_hf_split(dataset, tmp_path / "split.json")
    assert [manifest["splits"][name]["images_per_category"]["crack"] for name in ("train", "val", "test")] == [1, 1, 1]
    assert manifest["warnings"] == []


def test_impossible_rare_class_coverage_is_explicit(tmp_path):
    dataset = fixture(tmp_path)
    samples = tuple(
        replace(sample, instances=(Instance("dent"),) + ((Instance("crack"),) if sample.sample_id == "0" else ()))
        for sample in dataset.samples
    )
    manifest = create_or_load_cardd_hf_split(
        replace(dataset, samples=samples, classes=("dent", "crack")), tmp_path / "split.json"
    )
    assert any("'crack'" in warning and "impossible" in warning for warning in manifest["warnings"])


def test_corrupt_existing_manifest_is_never_silently_regenerated(tmp_path):
    dataset = fixture(tmp_path)
    path = tmp_path / "split.json"
    path.write_text("partial JSON")
    with pytest.raises(DatasetValidationError, match="will not be regenerated"):
        create_or_load_cardd_hf_split(dataset, path)
    assert path.read_text() == "partial JSON"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed": True},
        {"ratios": {"train": 0.7, "val": 0.3}},
        {"ratios": {"train": 0.5, "val": 0.1, "test": 0.1}},
        {"ratios": {"train": 0.7, "val": 0.3, "test": 0}},
        {"ratios": {"train": float("nan"), "val": 0.15, "test": 0.15}},
    ],
)
def test_invalid_split_parameters_rejected(tmp_path, kwargs):
    with pytest.raises(DatasetValidationError):
        create_or_load_cardd_hf_split(fixture(tmp_path), tmp_path / "split.json", **kwargs)
