"""Validate source/output separation before creating any staged files."""

from pathlib import Path

from vision.dataset._models import DatasetValidationError


def _preparation_output(dataset, output_dir, manifest_path, unsupported, converter):
    if unsupported not in ("error", "skip") or converter not in ("auto", "native", "ultralytics"):
        raise DatasetValidationError("Invalid unsupported annotation policy or converter")
    output = Path(output_dir).resolve()
    source_locations = [
        dataset.root,
        *(split.image_dir for split in dataset.splits.values()),
        *(split.annotation_path.parent for split in dataset.splits.values()),
    ]
    if any(
        output == source or output.is_relative_to(source) or source.is_relative_to(output)
        for source in source_locations
    ):
        raise DatasetValidationError("Prepared output must be outside and separate from the source dataset")
    if manifest_path is not None and any(
        Path(manifest_path).resolve().is_relative_to(source) for source in source_locations
    ):
        raise DatasetValidationError("Generated manifest must be outside the source dataset")
    if output.exists():
        raise DatasetValidationError(
            f"Prepared output already exists: {output}. "
            "Choose a new versioned --output-dir; existing data is never overwritten."
        )
    return output
