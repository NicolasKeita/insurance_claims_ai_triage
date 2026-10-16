"""Inspect actual fields and instance segmentation in the CarDD HF mirror."""

import argparse
import json
import os
from pathlib import Path

from vision.dataset import DatasetValidationError, discover_cardd_dataset, inspect_cardd_dataset


def add_hf_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", choices=("hf", "coco"), default="hf", help="hf uses the project's mirror and custom splits; coco is the legacy local COCO adapter")
    parser.add_argument("--repository", default=os.environ.get("CARDD_HF_DATASET", "harpreetsahota/CarDD"))
    parser.add_argument("--revision", help="Pin a Hub commit; a downloadable revision is resolved to its actual SHA")
    parser.add_argument("--cache-dir", type=Path, help="External storage root (or CARDD_HF_CACHE_DIR); never committed")
    parser.add_argument("--annotation-field", help="Use an explicitly chosen inspected FiftyOne field; otherwise detect instance polygon/mask fields")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--download", action="store_true", help="Explicitly load/download through FiftyOne's official load_from_hub integration")
    modes.add_argument("--dataset-name", help="Inspect an existing local persistent FiftyOne dataset without downloading")
    parser.add_argument("--dataset-dir", type=Path, help="Legacy --source coco only: override CARDD_DATASET_DIR")


def load_hf_arguments(args: argparse.Namespace):
    from vision.cardd_hf import CarddHfConfig, default_hf_cache_dir, load_cardd_hf, normalize_fiftyone_dataset

    if args.dataset_dir is not None:
        raise DatasetValidationError("--dataset-dir requires --source coco; the default source is the HF mirror")
    if args.download:
        return load_cardd_hf(
            CarddHfConfig(
                repository=args.repository,
                revision=args.revision,
                cache_dir=args.cache_dir,
                annotation_field=args.annotation_field,
            )
        )
    if args.dataset_name:
        cache_dir = args.cache_dir.expanduser().resolve() if args.cache_dir is not None else default_hf_cache_dir()
        os.environ.setdefault("FIFTYONE_DATABASE_DIR", str(cache_dir / "fiftyone_db"))
        try:
            import fiftyone as fo
            raw = fo.load_dataset(args.dataset_name)
        except ImportError as exc:
            raise DatasetValidationError("FiftyOne is unavailable. Install the project's cardd-hf extra with Python 3.11–3.13 before loading local datasets.") from exc
        except Exception as exc:
            raise DatasetValidationError(f"Cannot load local FiftyOne dataset {args.dataset_name}: {exc}") from exc
        info = getattr(raw, "info", {}) or {}
        stored_repository = info.get("cardd_hf_source_repository", info.get("source_repo"))
        stored_revision = info.get("cardd_hf_source_revision", info.get("source_revision"))
        if not stored_repository or stored_repository != args.repository:
            raise DatasetValidationError("Local FiftyOne dataset source repository is missing or disagrees with --repository. Load once with --download to record verified Hub provenance.")
        if not {"cardd_hf_source_revision", "source_revision"}.intersection(info) or (args.revision is not None and args.revision != stored_revision):
            raise DatasetValidationError("Local FiftyOne dataset source revision is missing or disagrees with --revision. Use its resolved commit SHA or load with --download.")
        return normalize_fiftyone_dataset(raw, repository=stored_repository, revision=stored_revision, annotation_field=args.annotation_field)
    raise DatasetValidationError("HF inspection requires --download (official FiftyOne Hub integration) or --dataset-name (an existing local FiftyOne dataset). Storage can be configured outside Git with --cache-dir or CARDD_HF_CACHE_DIR.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_hf_arguments(parser)
    args = parser.parse_args()
    try:
        if args.source == "coco":
            summary = inspect_cardd_dataset(discover_cardd_dataset(args.dataset_dir))
        else:
            from vision.cardd_hf import inspect_cardd_hf

            summary = inspect_cardd_hf(load_hf_arguments(args))
    except (DatasetValidationError, OSError) as exc:
        parser.exit(2, f"Dataset inspection failed: {exc}\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
