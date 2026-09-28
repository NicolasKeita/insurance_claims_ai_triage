from pathlib import Path

import numpy as np
import pandas as pd

from claims.enums import (
    ClaimType,
    CollisionType,
    TriageWorkflow,
)
from ml.dataset import TriageTrainingExample

COLLISION_TYPES = (
    CollisionType.FRONT_COLLISION,
    CollisionType.REAR_COLLISION,
    CollisionType.SIDE_COLLISION,
    CollisionType.PARKING_DAMAGE,
)


COLLISION_PROBABILITIES = np.array(
    [
        0.34,
        0.28,
        0.25,
        0.13,
    ]
)

REPAIR_COST_FACTOR = {
    CollisionType.FRONT_COLLISION: 1.15,
    CollisionType.REAR_COLLISION: 0.95,
    CollisionType.SIDE_COLLISION: 1.20,
    CollisionType.PARKING_DAMAGE: 0.50,
}

def _softmax(
    values: np.ndarray,
) -> np.ndarray:
    shifted = values - np.max(values)

    exp_values = np.exp(shifted)

    return exp_values / exp_values.sum()

def _sample_collision_type(
    rng: np.random.Generator,
) -> CollisionType:
    index = int(
        rng.choice(
            len(COLLISION_TYPES),
            p=COLLISION_PROBABILITIES,
        )
    )

    return COLLISION_TYPES[index]

def _generate_features(
    rng: np.random.Generator,
) -> dict:
    collision_type = (
        _sample_collision_type(rng)
    )

    vehicle_age = int(
        np.clip(
            rng.poisson(6),
            0,
            20,
        )
    )
    repair_amount = float(
        np.clip(
            rng.lognormal(
                mean=np.log(2200),
                sigma=0.72,
            )
            * REPAIR_COST_FACTOR[
                collision_type
            ],
            150,
            30000,
        )
    )

    repair_amount = round(
        repair_amount,
        2,
    )
    declared_damage_count = int(
        np.clip(
            rng.poisson(
                1.0
                + repair_amount / 5000
            )
            + 1,
            1,
            8,
        )
    )
    additional_damage_count = int(
        np.clip(
            rng.poisson(
                0.15
                + repair_amount / 18000
            ),
            0,
            4,
        )
    )
    document_count = int(
        rng.integers(
            3,
            7,
        )
    )
    missing_probability = (
        0.07
        + 0.03
        * min(
            additional_damage_count,
            2,
        )
        + rng.uniform(
            0,
            0.05,
        )
    )
    missing_document_count = int(
        rng.binomial(
            document_count,
            min(
                missing_probability,
                0.40,
            ),
        )
    )
    document_completeness_ratio = (
        document_count
        - missing_document_count
    ) / document_count
    injury_probability = 0.025
    if (
        collision_type
        == CollisionType.FRONT_COLLISION
    ):
        injury_probability += 0.045

    if (
        collision_type
        == CollisionType.SIDE_COLLISION
    ):
        injury_probability += 0.065
    injury_probability += min(
        repair_amount / 50000,
        0.18,
    )
    injuries_declared = bool(
        rng.random()
        < min(
            injury_probability,
            0.35,
        )
    )
    consistency_issue_count = int(
        np.clip(
            rng.poisson(
                0.15
                + 0.55
                * additional_damage_count
                + 0.35
                * missing_document_count
            ),
            0,
            6,
        )
    )
    return {
        "claim_type": ClaimType.AUTO_COLLISION,
        "collision_type": collision_type,
        "vehicle_age": vehicle_age,
        "repair_amount": repair_amount,
        "declared_damage_count": (
            declared_damage_count
        ),
        "additional_damage_count": (
            additional_damage_count
        ),
        "document_count": document_count,
        "missing_document_count": (
            missing_document_count
        ),
        "document_completeness_ratio": (
            document_completeness_ratio
        ),
        "injuries_declared": (
            injuries_declared
        ),
        "consistency_issue_count": (
            consistency_issue_count
        ),
    }

def _sample_workflow(
    rng: np.random.Generator,
    features: dict,
) -> TriageWorkflow:
    repair_amount = (
        features["repair_amount"]
    )

    severity = (
        np.log1p(repair_amount)
        / np.log(30001)
        * 2.0
    )
    severity += (
        1.4
        if features["injuries_declared"]
        else 0
    )

    severity += (
        0.12
        * features[
            "declared_damage_count"
        ]
    )
    collision_bonus = {
        CollisionType.FRONT_COLLISION: 0.12,
        CollisionType.REAR_COLLISION: 0.0,
        CollisionType.SIDE_COLLISION: 0.20,
        CollisionType.PARKING_DAMAGE: -0.15,
    }

    severity += collision_bonus[
        features["collision_type"]
    ]
    inconsistency = (
        0.55
        * features[
            "consistency_issue_count"
        ]
        + 0.45
        * features[
            "additional_damage_count"
        ]
        + 0.30
        * features[
            "missing_document_count"
        ]
        + 0.60
        * (
            1
            - features[
                "document_completeness_ratio"
            ]
        )
    )
    logits = np.array(
        [
            # FAST_TRACK
            (
                2.8
                - 1.45 * severity
                - 1.60 * inconsistency
            ),

            # STANDARD
            (
                1.1
                - 0.25 * severity
                - 0.25 * inconsistency
            ),

            # EXPERT_REVIEW
            (
                -1.0
                + 1.35 * severity
                + 0.15 * inconsistency
            ),

            # INVESTIGATION
            (
                -1.6
                + 0.15 * severity
                + 1.45 * inconsistency
            ),
        ],
        dtype=float,
    )
    logits += np.array(
        [
            1.1,
            0.8,
            -0.9,
            0.1,
        ]
    )
    logits += rng.normal(
        loc=0,
        scale=0.45,
        size=4,
    )
    probabilities = _softmax(
        logits
    )
    workflows = (
        TriageWorkflow.FAST_TRACK,
        TriageWorkflow.STANDARD,
        TriageWorkflow.EXPERT_REVIEW,
        TriageWorkflow.INVESTIGATION,
    )
    index = int(
        rng.choice(
            len(workflows),
            p=probabilities,
        )
    )

    return workflows[index]

def _generate_example(
    rng: np.random.Generator,
) -> TriageTrainingExample:
    features = _generate_features(
        rng
    )

    workflow = _sample_workflow(
        rng,
        features,
    )

    return TriageTrainingExample(
        **features,
        workflow=workflow,
    )

def generate_synthetic_triage_dataset(
    sample_count: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    if sample_count <= 0:
        raise ValueError(
            "sample_count must be positive"
        )

    rng = np.random.default_rng(
        seed
    )

    examples = [
        _generate_example(rng)
        for _ in range(sample_count)
    ]

    rows = [
        example.model_dump(
            mode="json"
        )
        for example in examples
    ]

    return pd.DataFrame(rows)

def save_synthetic_triage_dataset(
    dataset: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset.to_csv(
        path,
        index=False,
    )