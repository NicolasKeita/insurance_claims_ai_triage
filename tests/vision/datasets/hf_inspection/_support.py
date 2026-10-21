"""Support."""

from pathlib import Path

import numpy as np
from PIL import Image


class FixtureDataset:
    def __init__(self, samples, schema=None):
        self.samples = samples
        self.schema = schema or {
            "filepath": "StringField",
            "coco_id": "IntField",
            "boxes": "EmbeddedDocumentField(Detections)",
            "damage_regions": "EmbeddedDocumentField(Detections)",
        }

    def get_field_schema(self):
        return self.schema

    def iter_samples(self):
        return iter(self.samples)


def fixture(tmp_path: Path):
    image = tmp_path / "actual_image.png"
    Image.new("RGB", (20, 10), color="red").save(image)
    label = {
        "_cls": "Detection",
        "label": "glass shatter",
        "bounding_box": [0.1, 0.2, 0.5, 0.6],
        "mask": np.ones((3, 5), dtype=bool),
    }
    sample = {
        "filepath": str(image),
        "_id": "unstable-local-database-id",
        "coco_id": 7,
        "metadata": {"width": 20, "height": 10},
        "boxes": {"_cls": "Detections", "detections": [dict(label, mask=None)]},
        "damage_regions": {"_cls": "Detections", "detections": [label]},
    }
    return FixtureDataset([sample]), sample, label
