"""HTTP contract smoke with real PostgreSQL, Qdrant, embeddings and Ollama."""

import argparse
import json
from pathlib import Path

from fastapi.testclient import TestClient

from api.main import create_app
from knowledge.config import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claim-id", default="CLAIM-2026-00001")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/rag/api_smoke.json")
    args = parser.parse_args()
    responses = []
    with TestClient(create_app()) as client:
        for path, body in (
            ("/v1/knowledge/search", {"query": "collision deductible", "source_type": "POLICY", "product": "AUTO_PREMIUM"}),
            ("/v1/knowledge/answer", {"question": "What is the AUTO_PREMIUM collision deductible?", "source_type": "POLICY", "product": "AUTO_PREMIUM"}),
            (f"/v1/claims/{args.claim_id}/knowledge/search", {"query": "collision deductible", "source_type": "POLICY"}),
            (f"/v1/claims/{args.claim_id}/knowledge/answer", {"question": "What is the collision deductible?", "source_type": "POLICY"}),
            (f"/v1/claims/{args.claim_id}/knowledge/search", {"query": "collision deductible", "source_type": "POLICY", "product": "AUTO_BASIC"}),
            (f"/v1/claims/{args.claim_id}/knowledge/search", {"query": "missing documents", "source_type": "PROCEDURE"}),
        ):
            response = client.post(path, json=body)
            data = response.json()
            responses.append({"path": path, "request": body, "status": response.status_code, "response": data})
            expected_status = 422 if "product" in body and path.startswith("/v1/claims/") else 200
            assert response.status_code == expected_status, data
            if "results" in data and body["source_type"] == "POLICY":
                assert data["results"] and all(r["product"] == "AUTO_PREMIUM" for r in data["results"])
            if "answer" in data:
                assert data["citations"] and not data["insufficient_evidence"], data
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(responses, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(responses, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
