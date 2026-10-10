"""Read PostgreSQL historical facts and show policy v2 signals, without writes."""

import argparse
import json

from investigation.policy import INVESTIGATION_POLICY_V2
from investigation.signals import signals_from_history
from persistence.database import DatabaseConfig, create_database_engine, create_session_factory
from persistence.history import HistoricalClaimRepository
from persistence.repositories import ClaimNotFoundError, ClaimRepository


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claim-id", default="CLAIM-2026-00001")
    args = parser.parse_args()
    engine = create_database_engine(DatabaseConfig.from_env())
    try:
        with create_session_factory(engine)() as session:
            claim = ClaimRepository(session).get_by_claim_id(args.claim_id)
            if claim is None:
                raise ClaimNotFoundError(f"Claim {args.claim_id} does not exist")
            profile = HistoricalClaimRepository(session).get_profile(args.claim_id)
        print(f"Current claim: {claim.claim_id}; incident_date={claim.incident.date}")
        print(f"Customer: {claim.customer.customer_id}")
        print("Historical profile:")
        print(profile.model_dump_json(indent=2))
        print(f"Historical investigation signals ({INVESTIGATION_POLICY_V2.version}):")
        print(json.dumps([signal.model_dump(mode="json") for signal in
                          signals_from_history(profile, INVESTIGATION_POLICY_V2)], indent=2))
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
