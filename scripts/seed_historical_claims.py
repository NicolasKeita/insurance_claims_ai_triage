"""Idempotent STEP25 demo fixtures; never overwrite unexpected business IDs."""

from datetime import timedelta
from decimal import Decimal

from claims.models import Claim
from ml.anomaly_inference import RegisteredAnomalyAssessment
from ml.anomaly_training import AnomalyAssessment
from persistence.database import DatabaseConfig, create_database_engine, create_session_factory
from persistence.repositories import AssessmentRepository, ClaimRepository


REFERENCE_ID = "CLAIM-2026-00001"
DEMO_MODEL = "demo-step25-anomaly-fixture"


def demo_anomaly(flag: bool, version: str) -> RegisteredAnomalyAssessment:
    return RegisteredAnomalyAssessment(
        registered_model=DEMO_MODEL, model_version=version, model_alias="demo-fixture",
        assessment=AnomalyAssessment(
            is_anomalous=flag, anomaly_score=0.72 if flag else 0.42,
            threshold=0.62, reference_percentile=0.95 if flag else 0.50,
            statistical_signals=[],
        ),
    )


def demo_claims(reference: Claim) -> list[tuple[Claim, tuple[bool, ...]]]:
    if reference.customer.customer_id != "CUSTOMER-00001" or reference.repair_estimate.currency != "EUR":
        raise ValueError("Demo requires the reference customer CUSTOMER-00001 and EUR")
    specs = [
        ("P01", 2, "100", (True, False)),
        ("P02", 15, "200", (False, True)),
        ("P03", 30, "300", (True,)),
        ("P04", 90, "400", ()),
        ("P05", 365, "500", ()),
        ("P06", 400, "600", ()),
        ("SAME-DAY", 0, "9000", (True,)),
        ("LATER", -1, "9000", (True,)),
        ("OTHER-CUSTOMER", 1, "9000", (True,)),
    ]
    result = []
    for suffix, days, amount, flags in specs:
        payload = reference.model_dump()
        payload.update(
            claim_id=f"DEMO-STEP25-HISTORY-{suffix}", documents=[], images=[],
            policy={"policy_id": "DEMO-STEP25-HISTORY-POLICY", "product": "DEMO_HISTORY_FIXTURE"},
            repair_estimate={"amount": Decimal(amount), "currency": "EUR"},
        )
        payload["incident"].update(
            date=reference.incident.date - timedelta(days=days),
            location="DEMO STEP25 HISTORY v1 (synthetic fixture)",
        )
        if suffix == "OTHER-CUSTOMER":
            payload["customer"] = {"customer_id": "DEMO-STEP25-HISTORY-CUSTOMER-CONTROL"}
        result.append((Claim.model_validate(payload), flags))
    return result


def seed(session) -> list[str]:
    claims = ClaimRepository(session)
    reference = claims.get_by_claim_id(REFERENCE_ID)
    if reference is None:
        raise ValueError("Import CLAIM-2026-00001 before seeding historical fixtures")
    # Serialize concurrent runs of this explicit demo script within the DB.
    from sqlalchemy import text
    session.execute(text("SELECT pg_advisory_xact_lock(250025)"))
    assessments = AssessmentRepository(session)
    scenario = demo_claims(reference)
    present = set()
    # Check the whole scenario before adding anything. Transaction rollback also
    # protects against errors encountered during insertion.
    for claim, flags in scenario:
        existing = claims.get_by_claim_id(claim.claim_id)
        if existing is None:
            continue
        expected = [demo_anomaly(flag, str(i)) for i, flag in enumerate(flags, 1)]
        actual = [record.result for record in assessments.list_anomaly_history(claim.claim_id)]
        if existing != claim or actual != expected:
            raise ValueError(f"Unexpected data at reserved fixture ID {claim.claim_id}; preserved")
        present.add(claim.claim_id)
    messages = []
    for claim, flags in scenario:
        if claim.claim_id in present:
            messages.append(f"Existing fixture verified: {claim.claim_id}")
            continue
        claims.add(claim)
        for i, flag in enumerate(flags, 1):
            assessments.add_anomaly(claim.claim_id, demo_anomaly(flag, str(i)))
        messages.append(f"Created demo fixture: {claim.claim_id}; incident={claim.incident.date}; "
                        f"amount={claim.repair_estimate.amount} EUR; anomaly states={flags}")
    return messages


def main() -> None:
    engine = create_database_engine(DatabaseConfig.from_env())
    try:
        with create_session_factory(engine).begin() as session:
            messages = seed(session)
        print("Synthetic/demo data only; anomaly states are illustrative model-output fixtures.")
        print("\n".join(messages))
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
