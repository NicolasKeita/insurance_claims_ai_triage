import argparse
import json
from pathlib import Path

import joblib

from ml.training import (
    evaluate_model,
    load_triage_dataset,
    split_triage_dataset,
    train_models,
)

parser = argparse.ArgumentParser()


parser.add_argument(
    "--dataset",
    type=Path,
    default=Path(
        "data/ml/"
        "triage_synthetic_v1.csv"
    ),
)


parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path(
        "artifacts/triage_v1"
    ),
)


parser.add_argument(
    "--seed",
    type=int,
    default=42,
)


parser.add_argument(
    "--test-size",
    type=float,
    default=0.20,
)


args = parser.parse_args()

dataset = load_triage_dataset(
    args.dataset
)


(
    X_train,
    X_test,
    y_train,
    y_test,
) = split_triage_dataset(
    dataset,
    test_size=args.test_size,
    seed=args.seed,
)

print(
    f"Dataset: {len(dataset)} samples"
)

print(
    f"Train:   {len(X_train)} samples"
)

print(
    f"Test:    {len(X_test)} samples"
)

print()
print("Training models...")


models = train_models(
    X_train,
    y_train,
    seed=args.seed,
)

results = []


for name, model in models.items():
    metrics = evaluate_model(
        name=name,
        model=model,
        X_test=X_test,
        y_test=y_test,
    )

    results.append(metrics)

results.sort(
    key=lambda item: item.f1_macro,
    reverse=True,
)

print()
print(
    f"{'MODEL':24}"
    f"{'ACCURACY':>12}"
    f"{'BAL ACC':>12}"
    f"{'F1 MACRO':>12}"
    f"{'F1 WEIGHTED':>14}"
)

print("-" * 74)


for metrics in results:
    print(
        f"{metrics.model_name:24}"
        f"{metrics.accuracy:>12.3f}"
        f"{metrics.balanced_accuracy:>12.3f}"
        f"{metrics.f1_macro:>12.3f}"
        f"{metrics.f1_weighted:>14.3f}"
    )

args.output_dir.mkdir(
    parents=True,
    exist_ok=True,
)


for name, model in models.items():
    model_path = (
        args.output_dir
        / f"{name}.joblib"
    )

    joblib.dump(
        model,
        model_path,
    )

metrics_payload = {
    "dataset": str(
        args.dataset
    ),
    "dataset_size": len(
        dataset
    ),
    "train_size": len(
        X_train
    ),
    "test_size": len(
        X_test
    ),
    "seed": args.seed,

    "models": [
        metrics.model_dump()
        for metrics in results
    ],
}

metrics_path = (
    args.output_dir
    / "metrics.json"
)


metrics_path.write_text(
    json.dumps(
        metrics_payload,
        indent=2,
    ),
    encoding="utf-8",
)

print()
print(
    f"Artifacts: {args.output_dir}"
)

print(
    f"Metrics:   {metrics_path}"
)