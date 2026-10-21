"""Explicitly download and inspect the documented official dataset."""

import argparse
from pathlib import Path

from vision.dataset.carparts_dataset import DEFAULT_DATASET_ROOT, DEFAULT_MANIFEST, prepare_carparts_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Official Carparts-Seg download; never invoked by imports or pytest.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    manifest = prepare_carparts_dataset(dataset_root=args.dataset_root, manifest_path=args.manifest)
    print({split: values["image_count"] for split, values in manifest["splits"].items()})
    print(f"Manifest: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
