"""Dataset loading and stratified splitting for triage training."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.training.schema import FEATURE_COLUMNS, TARGET_COLUMN


def load_triage_dataset(path: Path) -> pd.DataFrame:
    dataset = pd.read_csv(path)
    required_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing_columns = required_columns - set(dataset.columns)
    if missing_columns:
        raise ValueError("Dataset is missing columns: " + ", ".join(sorted(missing_columns)))
    if dataset.empty:
        raise ValueError("Dataset is empty")
    if dataset[FEATURE_COLUMNS + [TARGET_COLUMN]].isnull().any().any():
        raise ValueError("Dataset contains missing values")
    return dataset


def split_triage_dataset(dataset: pd.DataFrame, *, test_size: float = 0.20, seed: int = 42):
    X = dataset[FEATURE_COLUMNS].copy()
    y = dataset[TARGET_COLUMN].copy()
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
