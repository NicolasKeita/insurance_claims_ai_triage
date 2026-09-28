from ml.synthetic_dataset import (
    generate_synthetic_triage_dataset,
)

from ml.training import (
    FEATURE_COLUMNS,
    build_models,
    split_triage_dataset,
)

def test_train_test_split():
    dataset = (
        generate_synthetic_triage_dataset(
            sample_count=500,
            seed=42,
        )
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = split_triage_dataset(
        dataset,
        test_size=0.20,
        seed=42,
    )

    assert len(X_train) == 400
    assert len(X_test) == 100

    assert len(y_train) == 400
    assert len(y_test) == 100

def test_split_contains_only_features():
    dataset = (
        generate_synthetic_triage_dataset(
            sample_count=100,
            seed=42,
        )
    )

    (
        X_train,
        _,
        _,
        _,
    ) = split_triage_dataset(
        dataset,
        seed=42,
    )

    assert list(
        X_train.columns
    ) == FEATURE_COLUMNS

    assert "workflow" not in (
        X_train.columns
    )

def test_expected_models_exist():
    models = build_models(
        seed=42
    )

    assert set(models) == {
        "dummy",
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
    }

def test_logistic_regression_can_train():
    dataset = (
        generate_synthetic_triage_dataset(
            sample_count=400,
            seed=42,
        )
    )

    (
        X_train,
        X_test,
        y_train,
        _,
    ) = split_triage_dataset(
        dataset,
        seed=42,
    )

    model = build_models(
        seed=42
    )[
        "logistic_regression"
    ]

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(
        X_test
    )

    assert len(predictions) == (
        len(X_test)
    )