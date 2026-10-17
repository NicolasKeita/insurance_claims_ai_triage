"""Workflow logits derived from synthetic claim severity and inconsistency."""

import numpy as np

from claims.enums import CollisionType, TriageWorkflow
from ml.synthetic_dataset.distribution import _softmax

WORKFLOWS = (
    TriageWorkflow.FAST_TRACK,
    TriageWorkflow.STANDARD,
    TriageWorkflow.EXPERT_REVIEW,
    TriageWorkflow.INVESTIGATION,
)


def _sample_workflow(rng: np.random.Generator, features: dict) -> TriageWorkflow:
    """
    FAST_TRACK

    STANDARD

    EXPERT_REVIEW

    INVESTIGATION
    """
    logits = _workflow_logits(features) + rng.normal(loc=0, scale=0.45, size=4)
    probabilities = _softmax(logits)
    index = int(rng.choice(len(WORKFLOWS), p=probabilities))
    return WORKFLOWS[index]


def _workflow_logits(features):
    severity = _severity(features)
    inconsistency = _inconsistency(features)
    logits = np.array(
        [
            (2.8 - 1.45 * severity - 1.60 * inconsistency),
            (1.1 - 0.25 * severity - 0.25 * inconsistency),
            (-1.0 + 1.35 * severity + 0.15 * inconsistency),
            (-1.6 + 0.15 * severity + 1.45 * inconsistency),
        ],
        dtype=float,
    )
    return logits + np.array([1.1, 0.8, -0.9, 0.1])


def _severity(features):
    severity = np.log1p(features["repair_amount"]) / np.log(30001) * 2.0
    severity += 1.4 if features["injuries_declared"] else 0
    severity += 0.12 * features["declared_damage_count"]
    collision_bonus = {
        CollisionType.FRONT_COLLISION: 0.12,
        CollisionType.REAR_COLLISION: 0.0,
        CollisionType.SIDE_COLLISION: 0.20,
        CollisionType.PARKING_DAMAGE: -0.15,
    }
    return severity + collision_bonus[features["collision_type"]]


def _inconsistency(features):
    return (
        0.55 * features["consistency_issue_count"]
        + 0.45 * features["additional_damage_count"]
        + 0.30 * features["missing_document_count"]
        + 0.60 * (1 - features["document_completeness_ratio"])
    )
