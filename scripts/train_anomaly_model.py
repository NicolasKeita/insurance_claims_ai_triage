import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from ml.anomaly_evaluation import evaluate_anomaly_detector
from ml.anomaly_training import train_anomaly_detector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("data/ml/anomaly_reference_v1.csv"),
    )
    parser.add_argument(
        "--evaluation",
        type=Path,
        default=Path("data/ml/anomaly_evaluation_v1.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/anomaly_v1"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=0.98,
    )
    args = parser.parse_args()

    reference = pd.read_csv(args.reference)
    evaluation = pd.read_csv(args.evaluation)
    detector = train_anomaly_detector(
        reference,
        seed=args.seed,
        threshold_percentile=args.threshold_percentile,
    )
    result = evaluate_anomaly_detector(detector, evaluation)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "anomaly_detector.joblib"
    metrics_path = args.output_dir / "metrics.json"
    errors_path = args.output_dir / "errors.csv"

    joblib.dump(detector, model_path)
    metrics = result.metrics.model_dump(mode="json")
    metrics.update(
        {
            "reference_dataset": str(args.reference),
            "evaluation_dataset": str(args.evaluation),
            "seed": args.seed,
        }
    )
    metrics_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    result.errors.to_csv(errors_path, index=False)

    print(f"Reference:  {len(reference)} cases ({args.reference})")
    print(f"Evaluation: {len(evaluation)} cases ({args.evaluation})")
    print(
        f"Threshold:  {detector.threshold:.4f} "
        f"at reference percentile {args.threshold_percentile:.2%}"
    )
    for name in ("precision", "recall", "f1", "flagged_count", "flagged_rate"):
        print(f"{name}: {metrics[name]}")
    print(f"average_precision: {metrics['average_precision']}")
    print(f"roc_auc: {metrics['roc_auc']}")
    print(f"Confusion matrix: {metrics['confusion_matrix']}")
    print("By anomaly kind:")
    for kind, performance in metrics["by_anomaly_kind"].items():
        recall = performance["recall"]
        recall_text = "n/a" if recall is None else f"{recall:.3f}"
        print(
            f"  {kind:25} count={performance['sample_count']:3d} "
            f"flagged={performance['flagged_count']:3d} "
            f"recall={recall_text}"
        )
    print(f"Model:   {model_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Errors:  {errors_path}")


if __name__ == "__main__":
    main()
