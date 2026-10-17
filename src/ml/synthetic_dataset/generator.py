"""Row generation and persistence for the synthetic triage dataset."""

from pathlib import Path

import numpy as np
import pandas as pd

from ml.dataset import TriageTrainingExample
from ml.synthetic_dataset.distribution import _generate_features
from ml.synthetic_dataset.workflow import _sample_workflow


def _generate_example(rng: np.random.Generator) -> TriageTrainingExample:
    features = _generate_features(rng)
    workflow = _sample_workflow(rng, features)
    return TriageTrainingExample(**features, workflow=workflow)


def generate_synthetic_triage_dataset(sample_count: int = 5000, seed: int = 42) -> pd.DataFrame:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    rng = np.random.default_rng(seed)
    examples = [_generate_example(rng) for _ in range(sample_count)]
    rows = [example.model_dump(mode="json") for example in examples]
    return pd.DataFrame(rows)


def save_synthetic_triage_dataset(dataset: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(path, index=False)
