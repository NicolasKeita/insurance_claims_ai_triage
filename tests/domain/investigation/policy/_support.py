"""Support."""

from claims.consistency import ClaimConsistencyReport
from ml.anomaly_training import AnomalyAssessment
from ml.features import TriageFeatures


def make_consistency(**changes) -> ClaimConsistencyReport:
    values = dict(
        claim_id_match=True,
        incident_date_match=True,
        location_match=True,
        vehicle_match=True,
        collision_type_match=True,
        injuries_match=True,
        declared_damage_match=True,
        additional_quote_damage=(),
        unmapped_quote_items=(),
    )
    values.update(changes)
    return ClaimConsistencyReport(**values)


def make_anomaly(**changes) -> AnomalyAssessment:
    values = dict(
        is_anomalous=False,
        anomaly_score=0.42,
        threshold=0.62,
        reference_percentile=0.75,
        statistical_signals=[],
    )
    values.update(changes)
    return AnomalyAssessment(**values)


def make_features(**changes) -> TriageFeatures:
    values = dict(
        claim_type="AUTO_COLLISION",
        collision_type="FRONT_COLLISION",
        vehicle_age=5,
        repair_amount=3160,
        declared_damage_count=2,
        additional_damage_count=0,
        document_count=4,
        missing_document_count=0,
        document_completeness_ratio=1.0,
        injuries_declared=False,
        consistency_issue_count=0,
    )
    values.update(changes)
    return TriageFeatures(**values)
