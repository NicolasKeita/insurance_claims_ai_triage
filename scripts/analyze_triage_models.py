import json
from pathlib import Path

from ml.analysis import (
    build_classification_report,
    build_error_analysis,
    compute_permutation_importance,
    cross_validate_models,
    select_candidate_model,
)
from ml.training import (
    build_models,
    load_triage_dataset,
    split_triage_dataset,
)
from ml.explainability import (
    compute_tree_shap,
    save_class_beeswarm,
    save_local_waterfall,
)

DATASET = Path(
    "data/ml/triage_synthetic_v1.csv"
)

OUTPUT_DIR = Path(
    "artifacts/triage_v1/analysis"
)

SEED = 42

dataset = load_triage_dataset(
    DATASET
)

(
    X_train,
    X_test,
    y_train,
    y_test,
) = split_triage_dataset(
    dataset,
    test_size=0.20,
    seed=SEED,
)

models = build_models(
    seed=SEED
)

cv_results = cross_validate_models(
    models=models,
    X_train=X_train,
    y_train=y_train,
    folds=5,
    seed=SEED,
)

print()
print("CROSS VALIDATION")
print("=" * 70)

for result in cv_results:
    print(
        f"{result.model_name:24}"
        f" F1={result.f1_macro_mean:.3f}"
        f" ± {result.f1_macro_std:.3f}"
        f" | BAL={result.balanced_accuracy_mean:.3f}"
        f" ± {result.balanced_accuracy_std:.3f}"
    )

candidate_name = (
    select_candidate_model(
        cv_results
    )
)

candidate = models[
    candidate_name
]

print()
print(
    f"Candidate model: "
    f"{candidate_name}"
)

candidate.fit(
    X_train,
    y_train,
)

report = (
    build_classification_report(
        model=candidate,
        X_test=X_test,
        y_test=y_test,
    )
)

print()
print("CLASSIFICATION REPORT")
print("=" * 70)

print(report)

errors = build_error_analysis(
    model=candidate,
    X_test=X_test,
    y_test=y_test,
)

print()
print(
    f"Misclassified samples: "
    f"{len(errors)}"
)

print()

print(
    errors.head(20)
)

importance = (
    compute_permutation_importance(
        model=candidate,
        X_test=X_test,
        y_test=y_test,
        seed=SEED,
    )
)

print()
print("PERMUTATION IMPORTANCE")
print("=" * 70)

print(importance)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

report.to_csv(
    OUTPUT_DIR
    / "classification_report.csv"
)

errors.to_csv(
    OUTPUT_DIR
    / "errors.csv",
    index=False,
)

importance.to_csv(
    OUTPUT_DIR
    / "permutation_importance.csv",
    index=False,
)

(
    OUTPUT_DIR
    / "cross_validation.json"
).write_text(
    json.dumps(
        {
            "candidate_model": (
                candidate_name
            ),
            "models": [
                result.model_dump()
                for result in cv_results
            ],
        },
        indent=2,
    ),
    encoding="utf-8",
)

random_forest = models["random_forest"]
random_forest.fit(X_train, y_train)

shap_sample = X_test.iloc[:200].copy()
(
    shap_values,
    transformed_shap_sample,
    shap_classifier,
) = compute_tree_shap(
    model=random_forest,
    X=shap_sample,
)

save_class_beeswarm(
    explanation=shap_values,
    classifier=shap_classifier,
    target_class="INVESTIGATION",
    output_path=OUTPUT_DIR / "shap_investigation.png",
)

save_local_waterfall(
    explanation=shap_values,
    classifier=shap_classifier,
    sample_index=0,
    target_class="INVESTIGATION",
    output_path=OUTPUT_DIR / "shap_sample_0_investigation.png",
)
