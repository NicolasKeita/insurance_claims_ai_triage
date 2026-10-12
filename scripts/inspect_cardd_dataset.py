"""Inspect externally obtained CarDD splits without writing any files."""

import argparse
import json
from pathlib import Path

from vision.dataset import DatasetValidationError, discover_cardd_dataset, inspect_cardd_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, help="Override CARDD_DATASET_DIR (official CarDD_COCO directory)")
    args = parser.parse_args()
    try:
        summary = inspect_cardd_dataset(discover_cardd_dataset(args.dataset_dir))
    except DatasetValidationError as exc:
        parser.exit(2, f"Dataset inspection failed: {exc}\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
