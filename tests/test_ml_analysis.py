from ml.analysis import (
    build_error_analysis,
    compute_permutation_importance,
    cross_validate_model,
)
from ml.synthetic_dataset import generate_synthetic_triage_dataset
from ml.training import FEATURE_COLUMNS, build_models, split_triage_dataset


def test_logistic_cross_validation():
    dataset = generate_synthetic_triage_dataset(sample_count=800, seed=42)
    X_train, _, y_train, _ = split_triage_dataset(dataset, seed=42)
    model = build_models(seed=42)["logistic_regression"]

    result = cross_validate_model(
        name="logistic_regression",
        model=model,
        X_train=X_train,
        y_train=y_train,
        folds=3,
        seed=42,
    )

    assert 0 <= result.f1_macro_mean <= 1
    assert result.f1_macro_std >= 0


def test_error_analysis_contains_only_errors():
    dataset = generate_synthetic_triage_dataset(sample_count=500, seed=42)
    X_train, X_test, y_train, y_test = split_triage_dataset(dataset, seed=42)
    model = build_models(seed=42)["logistic_regression"]
    model.fit(X_train, y_train)

    errors = build_error_analysis(model=model, X_test=X_test, y_test=y_test)

    assert (errors["actual"] != errors["predicted"]).all()


def test_permutation_importance_has_all_features():
    dataset = generate_synthetic_triage_dataset(sample_count=500, seed=42)
    X_train, X_test, y_train, y_test = split_triage_dataset(dataset, seed=42)
    model = build_models(seed=42)["logistic_regression"]
    model.fit(X_train, y_train)

    importance = compute_permutation_importance(
        model=model,
        X_test=X_test,
        y_test=y_test,
        seed=42,
    )

    assert set(importance["feature"]) == set(FEATURE_COLUMNS)
