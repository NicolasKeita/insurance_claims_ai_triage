"""Track a reproducible novelty detector and its full MLflow PyFunc model."""

import argparse
from importlib.metadata import version
from pathlib import Path

import mlflow
import mlflow.data
import mlflow.pyfunc
import pandas as pd
from mlflow.models import infer_signature

from ml.anomaly_evaluation import evaluate_anomaly_detector
from ml.anomaly_pyfunc import AnomalyPyfuncModel
from ml.anomaly_registry import ANOMALY_EXPERIMENT_NAME
from ml.anomaly_tracking import (
    mlflow_anomaly_metrics,
    read_dataset_metadata,
    verify_anomaly_model_round_trip,
)
from ml.anomaly_training import select_anomaly_features, train_anomaly_detector
from ml.tracking import (
    compute_file_sha256,
    configure_mlflow_tracking,
    get_git_commit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference", type=Path, default=Path("data/ml/anomaly_reference_v1.csv")
    )
    parser.add_argument(
        "--evaluation", type=Path, default=Path("data/ml/anomaly_evaluation_v1.csv")
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold-percentile", type=float, default=0.98)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument(
        "--previous-analysis", type=Path, default=Path("artifacts/anomaly_v1")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_mlflow_tracking()
    reference_metadata = read_dataset_metadata(args.reference, expected_role="reference")
    evaluation_metadata = read_dataset_metadata(args.evaluation, expected_role="evaluation")
    reference = pd.read_csv(args.reference)
    evaluation = pd.read_csv(args.evaluation)
    detector = train_anomaly_detector(
        reference,
        seed=args.seed,
        threshold_percentile=args.threshold_percentile,
        n_estimators=args.n_estimators,
    )
    evaluation_result = evaluate_anomaly_detector(detector, evaluation)
    git_commit = get_git_commit()
    reference_sha256 = compute_file_sha256(args.reference)
    evaluation_sha256 = compute_file_sha256(args.evaluation)

    ordinary_example = reference.head(2)
    unusual_example = evaluation.loc[evaluation["is_synthetic_anomaly"]].head(1)
    examples = select_anomaly_features(
        pd.concat([ordinary_example, unusual_example], ignore_index=True)
    )
    if len(examples) < 3:
        raise RuntimeError("At least one synthetic anomaly is needed for the smoke test")
    expected_output = AnomalyPyfuncModel(detector).predict(None, examples)
    signature = infer_signature(examples, expected_output)

    mlflow.set_experiment(ANOMALY_EXPERIMENT_NAME)
    with mlflow.start_run(run_name="anomaly-synthetic-v1") as run:
        mlflow.set_tags(
            {
                "task": "novelty_detection",
                "dataset.synthetic": "true",
                "model.algorithm": "IsolationForest",
            }
        )
        if git_commit:
            mlflow.set_tag("git.commit", git_commit)
        params = {
            "dataset.reference.version": reference_metadata["dataset_version"],
            "dataset.reference.sha256": reference_sha256,
            "dataset.evaluation.version": evaluation_metadata["dataset_version"],
            "dataset.evaluation.sha256": evaluation_sha256,
            "seed": args.seed,
            "threshold.percentile": args.threshold_percentile,
            "threshold.value": detector.threshold,
        }
        forest = detector.pipeline.named_steps["isolation_forest"]
        params.update(
            {
                f"isolation_forest.{name}": str(value)
                for name, value in forest.get_params(deep=False).items()
            }
        )
        mlflow.log_params(params)
        mlflow.log_metrics(mlflow_anomaly_metrics(evaluation_result.metrics))

        mlflow.log_input(
            mlflow.data.from_pandas(
                reference,
                source=str(args.reference),
                name=reference_metadata["dataset_version"],
            ),
            context="training",
        )
        mlflow.log_input(
            mlflow.data.from_pandas(
                evaluation,
                source=str(args.evaluation),
                name=evaluation_metadata["dataset_version"],
                targets="is_synthetic_anomaly",
            ),
            context="evaluation",
        )
        for role, dataset_path in (
            ("reference", args.reference),
            ("evaluation", args.evaluation),
        ):
            mlflow.log_artifact(str(dataset_path), artifact_path=f"datasets/{role}")
            mlflow.log_artifact(
                str(dataset_path.with_suffix(".metadata.json")),
                artifact_path=f"datasets/{role}",
            )
        mlflow.log_dict(
            evaluation_result.metrics.model_dump(mode="json"),
            "evaluation/metrics.json",
        )
        mlflow.log_text(
            evaluation_result.errors.to_csv(index=False),
            "evaluation/errors.csv",
        )
        mlflow.log_dict(
            {
                name: metrics.model_dump(mode="json")
                for name, metrics in evaluation_result.metrics.by_anomaly_kind.items()
            },
            "evaluation/by_anomaly_kind.json",
        )
        if args.previous_analysis.is_dir():
            for name in ("metrics.json", "errors.csv"):
                path = args.previous_analysis / name
                if path.is_file():
                    mlflow.log_artifact(str(path), artifact_path="previous_analysis")

        requirements = [
            f"{package}=={version(package)}"
            for package in ("mlflow", "numpy", "pandas", "scikit-learn", "pydantic")
        ]
        model_info = mlflow.pyfunc.log_model(
            name="model",
            python_model=AnomalyPyfuncModel(detector),
            code_paths=["src/ml", "src/claims"],
            signature=signature,
            input_example=examples,
            pip_requirements=requirements,
            metadata={
                "task": "novelty_detection",
                "threshold": detector.threshold,
                "threshold_percentile": detector.threshold_percentile,
                "score_convention": "higher_is_more_anomalous",
            },
        )
        loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
        verified = verify_anomaly_model_round_trip(detector, loaded_model, examples)
        mlflow.set_tag("candidate.model_uri", model_info.model_uri)
        mlflow.set_tag("candidate.model", "IsolationForest")
        mlflow.set_tag("model.round_trip", "passed")

    print(f"Experiment: {ANOMALY_EXPERIMENT_NAME}")
    print(f"Run: {run.info.run_id}")
    print(f"Model URI: {model_info.model_uri}")
    print(f"Threshold: {detector.threshold}")
    print(f"F1: {evaluation_result.metrics.f1}")
    print(f"Round trip: passed ({len(verified)} rows)")


if __name__ == "__main__":
    main()
