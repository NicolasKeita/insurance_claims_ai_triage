"""Atomically import the filesystem reference case; reject duplicates."""

from pathlib import Path

from claims.loader import load_claim_case
from persistence.database import (
    DatabaseConfig, create_database_engine, create_session_factory,
)
from persistence.repositories import ClaimAlreadyExistsError
from persistence.service import PersistenceService


def main() -> int:
    case = load_claim_case(Path(__file__).resolve().parents[1] / "data" / "CLAIM-2026-00001")
    engine = create_database_engine(DatabaseConfig.from_env())
    try:
        service = PersistenceService(create_session_factory(engine))
        try:
            result = service.import_claim(case)
        except ClaimAlreadyExistsError as error:
            print(f"Import rejected: {error}. Existing data was preserved.")
            return 1
        print(f"Customer {case.claim.customer.customer_id}: "
              f"{'created' if result.customer_created else 'reused'}")
        print(f"Policy {case.claim.policy.policy_id}: "
              f"{'created' if result.policy_created else 'reused'}")
        print(f"Claim inserted: {result.claim_id}")
        print(f"Documents: {result.document_count}; images: {result.image_count}")
        return 0
    finally:
        engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
