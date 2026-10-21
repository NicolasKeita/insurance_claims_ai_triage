"""Files."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from vision.cardd_hf_preparation._constants import DEFAULT_HF_PREPARED_DIR
from vision.dataset import (
    DatasetValidationError,
)


def _write_hf_sample(stage, split, sample, filename, rows):
    image_relative, label_relative = (
        f"images/{split}/{filename}",
        f"labels/{split}/{Path(filename).stem}.txt",
    )
    image_path, label_path = stage / image_relative, stage / label_relative
    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sample.filepath, image_path)
    label_path.write_text(
        "".join(
            str(class_id) + " " + " ".join(format(value, ".12g") for value in polygon) + "\n"
            for class_id, polygon in rows
        ),
        encoding="utf-8",
    )
    return image_relative, label_relative


def _hf_output(dataset, output_dir, manifest_path, converter):
    if converter not in ("auto", "native", "ultralytics"):
        raise DatasetValidationError("Invalid segmentation converter")
    configured = (
        output_dir if output_dir is not None else os.environ.get("CARDD_HF_PREPARED_DIR", DEFAULT_HF_PREPARED_DIR)
    )
    output = Path(configured).expanduser().resolve()
    source_paths = [Path(sample.filepath).resolve() for sample in dataset.samples]
    if not source_paths:
        raise DatasetValidationError("Cannot prepare an empty HF mirror")
    if any(
        source == output or source.is_relative_to(output) or output.is_relative_to(source.parent)
        for source in source_paths
    ):
        raise DatasetValidationError("Prepared output must be outside and separate from the HF source dataset")
    if manifest_path is not None and any(
        Path(manifest_path).resolve().is_relative_to(source.parent) for source in source_paths
    ):
        raise DatasetValidationError("Generated manifest must be outside the HF source dataset")
    if output.exists():
        raise DatasetValidationError(
            f"Prepared output already exists: {output}. "
            "Choose a new versioned --output-dir; existing data is never overwritten."
        )
    return output
