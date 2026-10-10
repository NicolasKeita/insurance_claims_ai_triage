"""Small additional retrieval demo; separate customer preserves STEP25 history."""

from datetime import timedelta

from claims.models import Claim
from persistence.database import DatabaseConfig, create_database_engine, create_session_factory
from persistence.repositories import ClaimRepository


def demo_claims(reference: Claim) -> list[Claim]:
    specs = [
        ("FRONT-CLIO", 3, "FRONT_COLLISION", "Renault", "Clio", 2020, "2950", ["FRONT_BUMPER", "LEFT_HEADLIGHT"]),
        ("FRONT-PEUGEOT", 12, "FRONT_COLLISION", "Peugeot", "208", 2022, "3400", ["FRONT_BUMPER", "LEFT_HEADLIGHT"]),
        ("FRONT-HOOD", 28, "FRONT_COLLISION", "Renault", "Clio", 2019, "6400", ["FRONT_BUMPER", "HOOD"]),
        ("PARKING", 9, "PARKING_DAMAGE", "Toyota", "Yaris", 2023, "450", []),
        ("SIDE", 21, "SIDE_COLLISION", "Volkswagen", "Golf", 2018, "4200", []),
        ("REAR", 42, "REAR_COLLISION", "Ford", "Focus", 2017, "1800", []),
    ]
    result = []
    for suffix, days, collision, make, model, year, amount, damage in specs:
        payload = reference.model_dump()
        payload.update(
            claim_id=f"DEMO-STEP26-SIMILAR-{suffix}", documents=[], images=[],
            customer={"customer_id": "DEMO-STEP26-SIMILAR-CUSTOMER"},
            policy={"policy_id": "DEMO-STEP26-SIMILAR-POLICY", "product": "DEMO_SIMILAR_FIXTURE"},
            vehicle={"make": make, "model": model, "year": year}, declared_damage=damage,
            repair_estimate={"amount": amount, "currency": "EUR"},
        )
        payload["incident"].update(date=reference.incident.date - timedelta(days=days),
                                   location="Bordeaux", collision_type=collision)
        result.append(Claim.model_validate(payload))
    return result


def seed(session):
    from sqlalchemy import text

    session.execute(text("SELECT pg_advisory_xact_lock(260026)"))
    repository = ClaimRepository(session)
    reference = repository.get_by_claim_id("CLAIM-2026-00001")
    if reference is None:
        raise ValueError("Import CLAIM-2026-00001 before seeding retrieval fixtures")
    scenario = demo_claims(reference)
    missing = []
    for claim in scenario:
        existing = repository.get_by_claim_id(claim.claim_id)
        if existing is None:
            missing.append(claim)
        elif existing != claim:
            raise ValueError(f"Unexpected data at reserved fixture ID {claim.claim_id}; preserved")
    for claim in missing:
        repository.add(claim)
    return [c.claim_id for c in scenario], len(missing)


def main():
    engine = create_database_engine(DatabaseConfig.from_env())
    try:
        with create_session_factory(engine).begin() as session:
            ids, created = seed(session)
        print(f"Synthetic retrieval demo: created={created}; verified={len(ids)}")
        print("\n".join(ids))
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
