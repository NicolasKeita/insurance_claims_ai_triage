"""Synthetic benchmark data for claim novelty detection."""

import numpy as np
import pandas as pd

from claims.enums import CollisionType
from ml.features import TriageFeatures
from ml.synthetic_dataset import generate_synthetic_triage_dataset


ANOMALY_KINDS = (
    "EXTREME_REPAIR_COST",
    "DAMAGE_COST_MISMATCH",
    "DOCUMENT_INCONSISTENCY",
    "MULTIVARIATE_ODDITY",
)

NORMAL_ANOMALY_KIND = "NONE"
FEATURE_COLUMNS = list(TriageFeatures.model_fields)


def _inject_anomaly(
    row: dict,
    kind: str,
    rng: np.random.Generator,
) -> dict:
    mutated = row.copy()

    if kind == "EXTREME_REPAIR_COST":
        mutated["collision_type"] = CollisionType.PARKING_DAMAGE
        mutated["repair_amount"] = round(
            float(rng.uniform(18000, 30000)), 2
        )
        mutated["declared_damage_count"] = int(
            rng.integers(2, 5)
        )
    elif kind == "DAMAGE_COST_MISMATCH":
        mutated["declared_damage_count"] = int(
            rng.integers(7, 9)
        )
        mutated["repair_amount"] = round(
            float(rng.uniform(180, 500)), 2
        )
        mutated["additional_damage_count"] = 0
    elif kind == "DOCUMENT_INCONSISTENCY":
        mutated["document_count"] = int(rng.integers(5, 8))
        mutated["missing_document_count"] = int(
            rng.integers(3, min(6, mutated["document_count"]))
        )
        mutated["document_completeness_ratio"] = (
            mutated["document_count"]
            - mutated["missing_document_count"]
        ) / mutated["document_count"]
        mutated["additional_damage_count"] = int(
            rng.integers(2, 5)
        )
        mutated["consistency_issue_count"] = int(
            rng.integers(5, 9)
        )
    elif kind == "MULTIVARIATE_ODDITY":
        # Each value is plausible in isolation; their combination is rare.
        mutated["collision_type"] = CollisionType.PARKING_DAMAGE
        mutated["vehicle_age"] = int(rng.integers(10, 15))
        mutated["repair_amount"] = round(
            float(rng.uniform(5000, 8000)), 2
        )
        mutated["declared_damage_count"] = int(
            rng.integers(3, 6)
        )
        mutated["additional_damage_count"] = 1
        mutated["injuries_declared"] = True
        mutated["consistency_issue_count"] = int(
            rng.integers(1, 3)
        )
    else:
        raise ValueError(f"Unknown anomaly kind: {kind}")

    return TriageFeatures.model_validate(mutated).model_dump(mode="json")


def generate_anomaly_datasets(
    reference_count: int = 4000,
    evaluation_count: int = 1000,
    anomaly_fraction: float = 0.20,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split normal dossiers, then inject anomalies only into evaluation rows."""
    if reference_count <= 0:
        raise ValueError("reference_count must be positive")
    if evaluation_count <= 0:
        raise ValueError("evaluation_count must be positive")
    if not 0 <= anomaly_fraction <= 1:
        raise ValueError("anomaly_fraction must be between 0 and 1")

    baseline = generate_synthetic_triage_dataset(
        sample_count=reference_count + evaluation_count,
        seed=seed,
    )
    reference = baseline.iloc[:reference_count][FEATURE_COLUMNS].copy()
    reference.reset_index(drop=True, inplace=True)

    evaluation_base = baseline.iloc[reference_count:][
        FEATURE_COLUMNS
    ].copy()
    rows = evaluation_base.to_dict(orient="records")
    rng = np.random.default_rng(seed)
    anomaly_count = int(round(evaluation_count * anomaly_fraction))
    anomaly_indices = rng.permutation(evaluation_count)[:anomaly_count]
    labels = [False] * evaluation_count
    kinds = [NORMAL_ANOMALY_KIND] * evaluation_count

    for position, row_index in enumerate(anomaly_indices):
        index = int(row_index)
        kind = ANOMALY_KINDS[position % len(ANOMALY_KINDS)]
        rows[index] = _inject_anomaly(rows[index], kind, rng)
        labels[index] = True
        kinds[index] = kind

    evaluation = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    evaluation["is_synthetic_anomaly"] = labels
    evaluation["anomaly_kind"] = kinds
    return reference, evaluation
