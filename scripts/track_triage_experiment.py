import os
from pathlib import Path

import mlflow
import mlflow.data
import mlflow.sklearn

from mlflow.models import (
    infer_signature,
)

from ml.analysis import (
    build_classification_report,
    build_error_analysis,
    cross_validate_model,
    select_candidate_model,
)
from ml.tracking import (
    compute_file_sha256,
    get_classifier_params,
    get_git_commit,
)
from ml.training import (
    TARGET_COLUMN,
    build_models,
    evaluate_model,
    load_triage_dataset,
    split_triage_dataset,
)

DATASET_PATH = Path(
    "data/ml/triage_synthetic_v1.csv"
)

DATASET_METADATA_PATH = Path(
    "data/ml/"
    "triage_synthetic_v1.metadata.json"
)

ANALYSIS_DIR = Path(
    "artifacts/triage_v1/analysis"
)

EXPERIMENT_NAME = (
    "insurance-claims-triage"
)

DATASET_VERSION = (
    "triage_synthetic_v1"
)

SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

tracking_uri = os.environ.get(
    "MLFLOW_TRACKING_URI"
)

if tracking_uri is None:
    raise RuntimeError(
        "MLFLOW_TRACKING_URI is required"
    )


mlflow.set_tracking_uri(
    tracking_uri
)


mlflow.set_experiment(
    EXPERIMENT_NAME
)

dataset = load_triage_dataset(
    DATASET_PATH
)


(
    X_train,
    X_test,
    y_train,
    y_test,
) = split_triage_dataset(
    dataset,
    test_size=TEST_SIZE,
    seed=SEED,
)

dataset_sha256 = (
    compute_file_sha256(
        DATASET_PATH
    )
)

git_commit = get_git_commit()

train_data = X_train.copy()

train_data[
    TARGET_COLUMN
] = y_train.to_numpy()


test_data = X_test.copy()

test_data[
    TARGET_COLUMN
] = y_test.to_numpy()

mlflow_train_dataset = (
    mlflow.data.from_pandas(
        train_data,
        source=str(
            DATASET_PATH
        ),
        name=(
            DATASET_VERSION
            + "_train"
        ),
        targets=TARGET_COLUMN,
    )
)


mlflow_test_dataset = (
    mlflow.data.from_pandas(
        test_data,
        source=str(
            DATASET_PATH
        ),
        name=(
            DATASET_VERSION
            + "_test"
        ),
        targets=TARGET_COLUMN,
    )
)

models = build_models(
    seed=SEED
)

cv_results = []
model_uris: dict[
    str,
    str,
] = {}

with mlflow.start_run(
    run_name=(
        "triage-synthetic-v1"
    )
) as parent_run:
    mlflow.set_tags(
        {
            "task": (
                "multiclass_triage"
            ),
            "dataset.synthetic": (
                "true"
            ),
            "dataset.version": (
                DATASET_VERSION
            ),
        }
    )
    if git_commit is not None:
        mlflow.set_tag(
            "git.commit",
            git_commit,
        )

    mlflow.log_params(
        {
            "dataset_version": (
                DATASET_VERSION
            ),
            "dataset_sha256": (
                dataset_sha256
            ),
            "dataset_size": (
                len(dataset)
            ),
            "train_size": (
                len(X_train)
            ),
            "test_size": (
                len(X_test)
            ),
            "split_test_ratio": (
                TEST_SIZE
            ),
            "seed": SEED,
            "cv_folds": CV_FOLDS,
        }
    )
    mlflow.log_input(
        mlflow_train_dataset,
        context="training",
    )

    mlflow.log_input(
        mlflow_test_dataset,
        context="testing",
    )
    mlflow.log_artifact(
        str(DATASET_PATH),
        artifact_path="dataset",
    )

    if (
        DATASET_METADATA_PATH
        .is_file()
    ):
        mlflow.log_artifact(
            str(
                DATASET_METADATA_PATH
            ),
            artifact_path="dataset",
        )
    if ANALYSIS_DIR.is_dir():
        mlflow.log_artifacts(
            str(ANALYSIS_DIR),
            artifact_path=(
                "previous_analysis"
            ),
        )
    for name, model in (
        models.items()
    ):
        with mlflow.start_run(
            run_name=name,
            nested=True,
        ):
            mlflow.set_tag(
                "model.name",
                name,
            )

            mlflow.set_tag(
                "model.role",
                (
                    "baseline"
                    if name == "dummy"
                    else "candidate"
                ),
            )
            mlflow.log_param(
                "dataset_sha256",
                dataset_sha256,
            )
            mlflow.log_params(
                get_classifier_params(
                    model
                )
            )
            mlflow.log_input(
                mlflow_train_dataset,
                context="training",
            )

            mlflow.log_input(
                mlflow_test_dataset,
                context="testing",
            )
            cv_metrics = (
                cross_validate_model(
                    name=name,
                    model=model,
                    X_train=X_train,
                    y_train=y_train,
                    folds=CV_FOLDS,
                    seed=SEED,
                )
            )

            cv_results.append(
                cv_metrics
            )
            mlflow.log_metrics(
                {
                    "cv_f1_macro_mean": (
                        cv_metrics
                        .f1_macro_mean
                    ),
                    "cv_f1_macro_std": (
                        cv_metrics
                        .f1_macro_std
                    ),
                    (
                        "cv_balanced_"
                        "accuracy_mean"
                    ): (
                        cv_metrics
                        .balanced_accuracy_mean
                    ),
                    (
                        "cv_balanced_"
                        "accuracy_std"
                    ): (
                        cv_metrics
                        .balanced_accuracy_std
                    ),
                }
            )
            model.fit(
                X_train,
                y_train,
            )
            test_metrics = (
                evaluate_model(
                    name=name,
                    model=model,
                    X_test=X_test,
                    y_test=y_test,
                )
            )
            mlflow.log_metrics(
                {
                    "test_accuracy": (
                        test_metrics
                        .accuracy
                    ),
                    (
                        "test_balanced_"
                        "accuracy"
                    ): (
                        test_metrics
                        .balanced_accuracy
                    ),
                    "test_f1_macro": (
                        test_metrics
                        .f1_macro
                    ),
                    "test_f1_weighted": (
                        test_metrics
                        .f1_weighted
                    ),
                }
            )
            mlflow.log_dict(
                {
                    "labels": (
                        test_metrics.labels
                    ),
                    "matrix": (
                        test_metrics
                        .confusion_matrix
                    ),
                },
                (
                    "evaluation/"
                    "confusion_matrix.json"
                ),
            )
            report = (
                build_classification_report(
                    model=model,
                    X_test=X_test,
                    y_test=y_test,
                )
            )
            mlflow.log_text(
                report.to_csv(),
                (
                    "evaluation/"
                    "classification_report.csv"
                ),
            )
            errors = (
                build_error_analysis(
                    model=model,
                    X_test=X_test,
                    y_test=y_test,
                )
            )
            mlflow.log_metric(
                "test_error_count",
                len(errors),
            )

            mlflow.log_text(
                errors.head(
                    100
                ).to_csv(
                    index=False
                ),
                (
                    "evaluation/"
                    "top_100_errors.csv"
                ),
            )
            input_example = (
                X_train.head(5).copy()
            )
            integer_columns = input_example.select_dtypes(
                include="integer"
            ).columns
            input_example[integer_columns] = input_example[
                integer_columns
            ].astype("float64")

            example_predictions = (
                model.predict(
                    input_example
                )
            )
            signature = (
                infer_signature(
                    input_example,
                    example_predictions,
                )
            )
            model_info = (
                mlflow.sklearn.log_model(
                    sk_model=model,
                    name="model",
                    signature=signature,
                    input_example=(
                        input_example
                    ),
                    skops_trusted_types=(
                        ["sklearn.tree._tree.Tree"]
                        if name in {
                            "random_forest",
                            "gradient_boosting",
                        }
                        else None
                    ),
                )
            )
            model_uris[
                name
            ] = model_info.model_uri
            print(
                f"{name:24} "
                f"CV F1="
                f"{cv_metrics.f1_macro_mean:.3f} "
                f"| test F1="
                f"{test_metrics.f1_macro:.3f}"
            )
    candidate_name = (
        select_candidate_model(
            cv_results
        )
    )
    candidate_uri = (
        model_uris[
            candidate_name
        ]
    )
    mlflow.set_tag(
        "candidate.model",
        candidate_name,
    )

    mlflow.set_tag(
        "candidate.model_uri",
        candidate_uri,
    )
    mlflow.log_dict(
        {
            "candidate_model": (
                candidate_name
            ),
            "candidate_model_uri": (
                candidate_uri
            ),
            "dataset_version": (
                DATASET_VERSION
            ),
            "dataset_sha256": (
                dataset_sha256
            ),
            "git_commit": (
                git_commit
            ),
        },
        "experiment_summary.json",
    )
    loaded_candidate = (
        mlflow.sklearn.load_model(
            candidate_uri
        )
    )
    smoke_predictions = (
        loaded_candidate.predict(
            X_test.head(3)
        )
    )
    if len(
        smoke_predictions
    ) != 3:
        raise RuntimeError(
            "Reloaded MLflow model "
            "failed smoke test"
        )
print()
print(
    f"Experiment: "
    f"{EXPERIMENT_NAME}"
)

print(
    f"Candidate:  "
    f"{candidate_name}"
)

print(
    f"Model URI:  "
    f"{candidate_uri}"
)

print(
    f"Parent run: "
    f"{parent_run.info.run_id}"
)
