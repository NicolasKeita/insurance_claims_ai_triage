"""Prepare real HF/FiftyOne instance polygons using a frozen project split."""

import argparse
import json
import os
from pathlib import Path

from vision.dataset import DEFAULT_MANIFEST_PATH, DEFAULT_PREPARED_DIR, DatasetValidationError, discover_cardd_dataset, prepare_cardd_yolo
from vision.cardd_hf_preparation import DEFAULT_HF_MANIFEST_PATH, DEFAULT_HF_PREPARED_DIR, prepare_cardd_hf_yolo
from inspect_cardd_dataset import add_hf_arguments, load_hf_arguments


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_hf_arguments(parser)
    parser.add_argument("--output-dir", type=Path, help="HF default: CARDD_HF_PREPARED_DIR or data/external_prepared/cardd_hf_v1; external directories are supported")
    parser.add_argument("--manifest", type=Path, help="HF default: artifacts/vision/cardd_hf_dataset_manifest.json")
    parser.add_argument("--split-manifest", type=Path, help="Frozen custom split; default artifacts/vision/cardd_hf_split_v2.json. Existing v1/v2 manifests are validated and reused.")
    parser.add_argument("--unsupported", choices=("error", "skip"), default="error", help="HF annotations always require usable segmentation; skip is only supported by the legacy COCO adapter")
    parser.add_argument("--converter", choices=("auto", "native", "ultralytics"), default="auto", help="auto prefers validated official convert_coco and records any polygon-only fallback")
    args = parser.parse_args()
    try:
        if args.source == "coco":
            output = args.output_dir or DEFAULT_PREPARED_DIR
            manifest_path = args.manifest or DEFAULT_MANIFEST_PATH
            manifest = prepare_cardd_yolo(discover_cardd_dataset(args.dataset_dir), output, manifest_path=manifest_path, unsupported=args.unsupported, converter=args.converter)
        else:
            if args.unsupported != "error":
                raise DatasetValidationError("The HF mirror preparation requires all instance segmentations to be usable; --unsupported skip is a legacy COCO option")
            output = args.output_dir or Path(os.environ.get("CARDD_HF_PREPARED_DIR", DEFAULT_HF_PREPARED_DIR))
            manifest_path = args.manifest or DEFAULT_HF_MANIFEST_PATH
            manifest = prepare_cardd_hf_yolo(load_hf_arguments(args), output, split_manifest_path=args.split_manifest, manifest_path=manifest_path, converter=args.converter)
    except (DatasetValidationError, OSError) as exc:
        parser.exit(2, f"Dataset preparation failed: {exc}\n")
    print(json.dumps({"prepared_dataset": str(output.resolve() / "dataset.yaml"), "manifest": str(manifest_path.resolve()), "source_name": manifest.get("source_name", "legacy local COCO adapter"), "source_repository": manifest.get("source_repository"), "source_revision": manifest.get("source_revision"), "actual_sample_count": manifest.get("actual_sample_count"), "split_version": manifest.get("split_version"), "conversion": manifest["conversion"], "splits": manifest["splits"], "dataset_fingerprint": manifest["dataset_fingerprint"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
