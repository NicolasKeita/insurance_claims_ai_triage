"""Classification reports, error analysis and permutation importance."""

import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline


def build_classification_report(*, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    return pd.DataFrame(report).transpose()


def build_error_analysis(*, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    result = X_test.copy()
    result["actual"] = y_test.to_numpy()
    result["predicted"] = predictions
    result["prediction_confidence"] = probabilities.max(axis=1)
    errors = result[result["actual"] != result["predicted"]].copy()
    return errors.sort_values(by="prediction_confidence", ascending=False)


def compute_permutation_importance(
    *, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, seed: int = 42
) -> pd.DataFrame:
    result = permutation_importance(
        model, X_test, y_test, scoring="f1_macro", n_repeats=20, random_state=seed, n_jobs=-1
    )
    importance = pd.DataFrame(
        {
            "feature": (X_test.columns),
            "importance_mean": (result.importances_mean),
            "importance_std": (result.importances_std),
        }
    )
    return importance.sort_values(by="importance_mean", ascending=False)
