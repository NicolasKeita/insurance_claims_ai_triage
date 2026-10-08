"""Generate the reference and mixed evaluation datasets for novelty detection."""

import argparse
import json
from pathlib import Path

from ml.anomaly_dataset import (
    ANOMALY_KINDS,
    FEATURE_COLUMNS,
    generate_anomaly_datasets,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-samples", type=int, default=4000)
    parser.add_argument("--evaluation-samples", type=int, default=1000)
    parser.add_argument("--anomaly-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reference-output",
        type=Path,
        default=Path("data/ml/anomaly_reference_v1.csv"),
    )
    parser.add_argument(
        "--evaluation-output",
        type=Path,
        default=Path("data/ml/anomaly_evaluation_v1.csv"),
    )
    args = parser.parse_args()

    reference, evaluation = generate_anomaly_datasets(
        reference_count=args.reference_samples,
        evaluation_count=args.evaluation_samples,
        anomaly_fraction=args.anomaly_fraction,
        seed=args.seed,
    )

    distribution = {
        kind: int((evaluation["anomaly_kind"] == kind).sum())
        for kind in ANOMALY_KINDS
    }
    distribution["NONE"] = int(
        (~evaluation["is_synthetic_anomaly"]).sum()
    )

    outputs = (
        (
            args.reference_output,
            reference,
            {
                "dataset_version": "anomaly_reference_v1",
                "role": "reference",
                "sample_count": len(reference),
                "seed": args.seed,
                "feature_columns": FEATURE_COLUMNS,
                "source": "synthetic_triage_v1",
            },
        ),
        (
            args.evaluation_output,
            evaluation,
            {
                "dataset_version": "anomaly_evaluation_v1",
                "role": "evaluation",
                "sample_count": len(evaluation),
                "seed": args.seed,
                "requested_anomaly_fraction": args.anomaly_fraction,
                "actual_anomaly_fraction": (
                    float(evaluation["is_synthetic_anomaly"].mean())
                ),
                "anomaly_kind_distribution": distribution,
                "feature_columns": FEATURE_COLUMNS,
                "source": "synthetic_triage_v1",
                "injection_stage": "after_reference_evaluation_split",
            },
        ),
    )

    for path, dataset, metadata in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(path, index=False)
        metadata_path = path.with_suffix(".metadata.json")
        metadata_path.write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        print(f"Dataset: {path}")
        print(f"Metadata: {metadata_path}")

    print("Anomaly kind distribution:")
    for kind, count in distribution.items():
        print(f"  {kind}: {count}")


if __name__ == "__main__":
    main()
