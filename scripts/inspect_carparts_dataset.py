"""Read-only actual counts, distribution and validation; no download."""

import argparse
import json
from pathlib import Path

from vision.carparts_dataset import DEFAULT_DATASET_YAML, inspect_carparts_dataset
from vision.training import write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect every local Carparts-Seg image/label without downloading.")
    parser.add_argument("--dataset-yaml", type=Path, default=DEFAULT_DATASET_YAML)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = inspect_carparts_dataset(args.dataset_yaml)
    display = {key: value for key, value in report.items() if key != "prepared_files"}
    print(json.dumps(display, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, report)
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
