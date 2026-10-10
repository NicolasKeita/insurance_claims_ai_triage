"""Run a real read-only advisory review using PostgreSQL, Qdrant and Ollama."""

import argparse
import json
from pathlib import Path

from agent.factory import create_claims_agent_service
from persistence.agent_runs import AgentRunStore
from agent.service import AgentRunError
from persistence.database import DatabaseConfig, create_database_engine, create_session_factory

ROOT = Path(__file__).resolve().parents[1]
SMOKE_OBJECTIVE = "Review this claim and recommend the next human review step."


def display_run(result, *, debug=False):
    recommendation = result.recommendation
    summary = {
        "run_id": result.run_id,
        "claim_id": result.claim_id,
        "objective": result.objective,
        "status": result.status,
        "iterations": result.iteration_count,
        "tool_call_count": result.tool_call_count,
        "tools_called": result.tools_used,
        "elapsed_seconds": result.elapsed_seconds,
        "evidence_ids": [item.evidence_id for item in result.evidence],
        "recommendation": recommendation.model_dump(mode="json") if recommendation else None,
        "uncertainties": recommendation.uncertainties if recommendation else [],
        "error": result.error,
    }
    if debug:
        # Explicit action justifications only; no hidden model reasoning is requested.
        summary["trace"] = [step.model_dump(mode="json") for step in result.trace]
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claim-id", default="CLAIM-2026-00001")
    parser.add_argument("--objective", default=SMOKE_OBJECTIVE)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/agent/claims_agent_smoke.json")
    parser.add_argument("--debug", action="store_true", help="Print structured planner/tool audit steps")
    args = parser.parse_args()
    engine = create_database_engine(DatabaseConfig.from_env())
    service = None
    try:
        factory = create_session_factory(engine)
        service = create_claims_agent_service(factory)
        failure = None
        try:
            result = service.review_claim(args.claim_id, objective=args.objective)
        except AgentRunError as error:
            failure = error
            result = AgentRunStore(factory).get_by_run_id(error.run_id)
            if result is None:
                raise SystemExit("Agent review failed and its audit run could not be found") from error
        display_run(result, debug=args.debug)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
        print(f"Saved run: {args.output.resolve()}")
        if failure is not None:
            raise SystemExit(f"Agent review failed ({failure.failure_kind}); inspect persisted run {result.run_id}")
    finally:
        if service is not None:
            service.tools.close()
        engine.dispose()


if __name__ == "__main__":
    main()
