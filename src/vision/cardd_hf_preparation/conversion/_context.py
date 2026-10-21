"""Context."""

from __future__ import annotations

from collections import Counter


class _HfConversion:
    """Inspected source identity and immutable mappings used for staged conversion."""

    def __init__(self, dataset, split_document, manifest):
        self.split_document = split_document
        self.manifest = manifest
        self.classes = sorted(dataset.classes)
        self.category_mapping = [
            {key: item[key] for key in ("category_id", "name", "yolo_class_id")}
            for item in manifest["category_mapping"]
        ]
        self.class_ids = {item["name"]: item["yolo_class_id"] for item in self.category_mapping}
        self.sample_lookup = {sample.sample_id: sample for sample in dataset.samples}
        self.image_ids = {sample_id: index for index, sample_id in enumerate(sorted(self.sample_lookup), start=1)}


class _HfSplitInventory:
    """Prepared COCO records, expected labels and source counts for one HF split."""

    def __init__(self):
        self.annotations, self.images, self.entries = [], [], []
        self.expected = {}
        self.annotation_counts, self.image_counts = Counter(), Counter()
        self.stitched_count = self.hole_count = 0
