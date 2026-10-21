"""References."""

from __future__ import annotations

from pathlib import PurePosixPath

from vision.dataset._files import _contained_file, _integer, _relative_filename
from vision.dataset._models import DatasetValidationError


def _image_identities(images, split, image_dir):
    split_ids, split_names, label_names = set(), set(), set()
    for image in images:
        if not isinstance(image, dict):
            raise DatasetValidationError(f"{split}: COCO image must be an object")
        image_id = _integer(image.get("id"), f"{split} image ID", minimum=0)
        filename = _relative_filename(image.get("file_name"))
        _integer(image.get("width"), f"{split} image width", minimum=1)
        _integer(image.get("height"), f"{split} image height", minimum=1)
        label_name = str(PurePosixPath(filename).with_suffix(".txt")).casefold()
        if image_id in split_ids or filename.casefold() in split_names or label_name in label_names:
            raise DatasetValidationError(f"{split}: duplicate image ID, filename or label filename")
        _contained_file(image_dir, filename)
        split_ids.add(image_id)
        split_names.add(filename.casefold())
        label_names.add(label_name)
    return split_ids, split_names


def _annotation_references(annotations, split, mapping, split_ids):
    annotation_ids = set()
    category_ids = {item["category_id"] for item in mapping}
    for annotation in annotations:
        if not isinstance(annotation, dict):
            raise DatasetValidationError(f"{split}: COCO annotation must be an object")
        annotation_id = _integer(annotation.get("id"), f"{split} annotation ID", minimum=0)
        if annotation_id in annotation_ids:
            raise DatasetValidationError(f"{split}: duplicate annotation ID")
        annotation_ids.add(annotation_id)
        if _integer(annotation.get("image_id"), "Annotation image ID") not in split_ids:
            raise DatasetValidationError(f"{split}: annotation references an unknown image ID")
        if _integer(annotation.get("category_id"), "Annotation category ID") not in category_ids:
            raise DatasetValidationError(f"{split}: annotation references an unknown category ID")
