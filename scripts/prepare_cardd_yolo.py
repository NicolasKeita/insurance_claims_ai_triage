"""Prepare validated YOLO segmentation labels without changing CarDD sources."""

import argparse
import json
from pathlib import Path

from vision.dataset import DEFAULT_MANIFEST_PATH, DEFAULT_PREPARED_DIR, DatasetValidationError, discover_cardd_dataset, prepare_cardd_yolo


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, help="Override CARDD_DATASET_DIR")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PREPARED_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--unsupported", choices=("error", "skip"), default="error", help="Explicit policy for missing, RLE, crowd or invalid segmentations; multipart polygons are stitched, never replaced by bbox masks")
    parser.add_argument("--converter", choices=("auto", "native", "ultralytics"), default="auto", help="auto prefers validated official convert_coco and records any polygon-only fallback")
    args = parser.parse_args()
    try:
        manifest = prepare_cardd_yolo(discover_cardd_dataset(args.dataset_dir), args.output_dir, manifest_path=args.manifest, unsupported=args.unsupported, converter=args.converter)
    except (DatasetValidationError, OSError) as exc:
        parser.exit(2, f"Dataset preparation failed: {exc}\n")
    print(json.dumps({"prepared_dataset": str(args.output_dir.resolve() / "dataset.yaml"), "manifest": str(args.manifest.resolve()), "conversion": manifest["conversion"], "splits": manifest["splits"], "dataset_fingerprint": manifest["dataset_fingerprint"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
