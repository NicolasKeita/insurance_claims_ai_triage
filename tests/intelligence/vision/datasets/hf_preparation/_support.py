"""Support."""

from pathlib import Path

from PIL import Image

from vision.dataset import annotation_hash
from vision.dataset.cardd_hf import HfDamageDataset, HfDamageInstance, HfDamageSample
from vision.dataset.cardd_hf_preparation import prepare_cardd_hf_yolo


def mirror_fixture(tmp_path: Path) -> HfDamageDataset:
    source = tmp_path / "external_fiftyone_source"
    source.mkdir()
    samples = []
    for index in range(12):
        path = source / f"synthetic_{index}.png"
        Image.new("RGB", (20, 10), color=(index * 17, 40, 10)).save(path)
        polygons = (((0.1, 0.1), (0.6, 0.1), (0.6, 0.6), (0.1, 0.6)),)
        label = "dent" if index % 2 else "scratch"
        instances = (HfDamageInstance(label, polygons, "ground_truth", "polyline"),)
        if index % 3 == 0:
            other = "scratch" if label == "dent" else "dent"
            instances += (
                HfDamageInstance(
                    other, (((0.7, 0.1), (0.9, 0.1), (0.9, 0.4), (0.7, 0.4)),), "ground_truth", "polyline"
                ),
            )
        samples.append(HfDamageSample(f"stable-id-{index:02d}", path, 20, 10, instances, annotation_hash(path)))
    return HfDamageDataset(
        "synthetic/CarDD-mirror",
        "a" * 40,
        tuple(samples),
        {"ground_truth": "Polylines"},
        ("ground_truth",),
        ("dent", "scratch"),
    )


def prepare_fixture(tmp_path: Path, dataset=None):
    source = dataset or mirror_fixture(tmp_path)
    output = tmp_path / "prepared_hf"
    manifest = prepare_cardd_hf_yolo(
        source,
        output,
        split_manifest_path=tmp_path / "split_v1.json",
        manifest_path=tmp_path / "manifest.json",
        converter="native",
    )
    return source, output, manifest
