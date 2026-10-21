"""Support."""

import hashlib
from dataclasses import dataclass
from pathlib import Path


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


def cooccurrence_patterns():
    return [
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
