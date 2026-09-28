import argparse
import json
from pathlib import Path

from ml.synthetic_dataset import (
    generate_synthetic_triage_dataset,
    save_synthetic_triage_dataset,
)


parser = argparse.ArgumentParser()

parser.add_argument(
    "--samples",
    type=int,
    default=5000,
)

parser.add_argument(
    "--seed",
    type=int,
    default=42,
)

parser.add_argument(
    "--output",
    type=Path,
    default=Path(
        "data/ml/triage_synthetic_v1.csv"
    ),
)

args = parser.parse_args()


dataset = (
    generate_synthetic_triage_dataset(
        sample_count=args.samples,
        seed=args.seed,
    )
)


save_synthetic_triage_dataset(
    dataset,
    args.output,
)


distribution = (
    dataset["workflow"]
    .value_counts()
    .sort_index()
)


metadata = {
    "dataset_version": (
        "triage_synthetic_v1"
    ),
    "sample_count": len(dataset),
    "seed": args.seed,
    "class_distribution": (
        distribution.to_dict()
    ),
}


metadata_path = (
    args.output.with_suffix(
        ".metadata.json"
    )
)


metadata_path.write_text(
    json.dumps(
        metadata,
        indent=2,
    ),
    encoding="utf-8",
)


print(
    f"Dataset: {args.output}"
)

print(
    f"Metadata: {metadata_path}"
)

print()

print("Workflow distribution:")
print(distribution)