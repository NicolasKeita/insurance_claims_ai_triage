"""Collision and repair-cost distributions for synthetic claims."""

import numpy as np

from claims.enums import ClaimType, CollisionType

COLLISION_TYPES = (
    CollisionType.FRONT_COLLISION,
    CollisionType.REAR_COLLISION,
    CollisionType.SIDE_COLLISION,
    CollisionType.PARKING_DAMAGE,
)
COLLISION_PROBABILITIES = np.array([0.34, 0.28, 0.25, 0.13])
REPAIR_COST_FACTOR = {
    CollisionType.FRONT_COLLISION: 1.15,
    CollisionType.REAR_COLLISION: 0.95,
    CollisionType.SIDE_COLLISION: 1.20,
    CollisionType.PARKING_DAMAGE: 0.50,
}


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum()


def _sample_collision_type(rng: np.random.Generator) -> CollisionType:
    index = int(rng.choice(len(COLLISION_TYPES), p=COLLISION_PROBABILITIES))
    return COLLISION_TYPES[index]


def _generate_features(rng: np.random.Generator) -> dict:
    collision_type = _sample_collision_type(rng)
    vehicle_age = int(np.clip(rng.poisson(6), 0, 20))
    repair_amount = _sample_repair_amount(rng, collision_type)
    declared_damage_count = int(np.clip(rng.poisson(1.0 + repair_amount / 5000) + 1, 1, 8))
    additional_damage_count = int(np.clip(rng.poisson(0.15 + repair_amount / 18000), 0, 4))
    document_count = int(rng.integers(3, 7))
    missing_document_count = _sample_missing_documents(rng, document_count, additional_damage_count)
    document_completeness_ratio = (document_count - missing_document_count) / document_count
    injuries_declared = _sample_injuries(rng, collision_type, repair_amount)
    consistency_issue_count = int(
        np.clip(rng.poisson(0.15 + 0.55 * additional_damage_count + 0.35 * missing_document_count), 0, 6)
    )
    return {
        "claim_type": ClaimType.AUTO_COLLISION,
        "collision_type": collision_type,
        "vehicle_age": vehicle_age,
        "repair_amount": repair_amount,
        "declared_damage_count": (declared_damage_count),
        "additional_damage_count": (additional_damage_count),
        "document_count": document_count,
        "missing_document_count": (missing_document_count),
        "document_completeness_ratio": (document_completeness_ratio),
        "injuries_declared": (injuries_declared),
        "consistency_issue_count": (consistency_issue_count),
    }


def _sample_repair_amount(rng, collision_type):
    amount = float(
        np.clip(rng.lognormal(mean=np.log(2200), sigma=0.72) * REPAIR_COST_FACTOR[collision_type], 150, 30000)
    )
    return round(amount, 2)


def _sample_missing_documents(rng, document_count, additional_damage_count):
    probability = 0.07 + 0.03 * min(additional_damage_count, 2) + rng.uniform(0, 0.05)
    return int(rng.binomial(document_count, min(probability, 0.40)))


def _sample_injuries(rng, collision_type, repair_amount):
    probability = 0.025
    if collision_type == CollisionType.FRONT_COLLISION:
        probability += 0.045
    if collision_type == CollisionType.SIDE_COLLISION:
        probability += 0.065
    probability += min(repair_amount / 50000, 0.18)
    return bool(rng.random() < min(probability, 0.35))
