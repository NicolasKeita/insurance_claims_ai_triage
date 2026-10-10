"""Read persisted data or append a JSON response from /v1/claims/assess."""

import argparse
import json
from pathlib import Path

from investigation.models import InvestigationAssessment
from ml.anomaly_inference import RegisteredAnomalyAssessment
from ml.inference import TriagePrediction
from persistence.database import DatabaseConfig, create_database_engine, create_session_factory
from persistence.repositories import ClaimNotFoundError
from persistence.service import PersistenceService


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claim-id", default="CLAIM-2026-00001")
    parser.add_argument("--assessment-json", type=Path,
                        help="Append an existing combined API response (all three results).")
    args = parser.parse_args()
    engine = create_database_engine(DatabaseConfig.from_env())
    try:
        service = PersistenceService(create_session_factory(engine))
        try:
            if args.assessment_json:
                payload = json.loads(args.assessment_json.read_text(encoding="utf-8"))
                service.save_assessments(
                    args.claim_id, triage=TriagePrediction.model_validate(payload["triage"]),
                    anomaly=RegisteredAnomalyAssessment.model_validate(payload["anomaly"]),
                    investigation=InvestigationAssessment.model_validate(payload["investigation"]),
                )
                print("Appended triage, anomaly and investigation assessments.")
            history = service.get_history(args.claim_id)
        except ClaimNotFoundError as error:
            print(str(error))
            return 1
        print("Claim (including customer, policy, documents, damage and images):")
        print(history.claim.model_dump_json(indent=2))
        for kind in ("triage", "anomaly", "investigation"):
            records = getattr(history, kind)
            print(f"\n{kind} history ({len(records)} records, oldest first):")
            for record in records:
                print(f"  {record.id} at {record.created_at.isoformat()}")
                print(record.result.model_dump_json(indent=2))
        return 0
    finally:
        engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
