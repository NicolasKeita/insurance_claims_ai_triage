"""Real agent review stage and its lazy service loaders."""

import os
from contextlib import contextmanager

from evaluation.models import EvaluationStatus

ASSESSMENT_TOOLS = {"GET_TRIAGE", "GET_ANOMALY", "GET_INVESTIGATION", "GET_VISUAL_ASSESSMENT"}


@contextmanager
def llm_environment(model):
    previous = os.environ.get("CLAIMS_LLM_MODEL")
    os.environ["CLAIMS_LLM_MODEL"] = model
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("CLAIMS_LLM_MODEL", None)
        else:
            os.environ["CLAIMS_LLM_MODEL"] = previous


class AgentSteps:
    """Run the real agent service against the existing application tools."""

    def assess_agent(self, run):
        """
        The normal application factory supplies lazy real services. Closing here only closes those
        owned by this service, not preflight clients.
        """
        observed, scenario = run.observed, run.scenario
        if not self.config.include_agent:
            self._skip(observed, "agent", "Disabled by configuration")
            return
        if not self._available(observed, "agent", ["postgresql", "ollama"]):
            return
        unavailable = set(scenario.inputs.get("unavailable_tools", []))
        if unavailable and (not self.config.isolated_demo_stack or unavailable - ASSESSMENT_TOOLS):
            self._skip(
                observed,
                "agent",
                "Requested deterministic tool outage cannot be safely imposed on the real application service; "
                "no fake tools used",
            )
            return
        vision = observed.components.get("vision")
        if scenario.vision_required and vision and vision.status == "SKIP":
            self._skip(
                observed,
                "agent",
                "This golden case requires unavailable real visual ground truth; its document subset remains observed",
            )
            return
        history = observed.components.get("history")
        if history and history.status != "PASS":
            self._skip(
                observed, "agent", "Controlled claim has no matching stored source; no fake agent tool backend used"
            )
            return
        from agent.factory import create_claims_agent_service

        with llm_environment(self.resources["llm_model"]):
            service = create_claims_agent_service(
                self.resources["session_factory"],
                similar_service_loader=self.similar_service,
                knowledge_retriever_loader=self.knowledge_retriever,
            )
            result = self._record(
                observed,
                "agent",
                lambda: self.review(service, run),
                inputs={"claim_id": run.case.claim.claim_id, "source": "existing PostgreSQL claim"},
            )
            service.tools.close()
        if result is not None:
            observed.agent = result.model_dump(mode="json")
            observed.facts["agent"] = observed.agent
            observed.model_versions["agent_llm"] = self.resources["llm_model"]
            if result.status != "COMPLETED":
                observed.components["agent"].status = EvaluationStatus.FAIL
                observed.components["agent"].reason = result.error or "Real agent did not complete"
                observed.errors.append("agent: " + observed.components["agent"].reason)

    def review(self, service, run):
        """Review the stored claim, tolerating a completed run recorded as an error."""
        from agent.service import AgentRunError
        from persistence.agent_runs import AgentRunStore

        try:
            return service.review_claim(
                run.case.claim.claim_id,
                objective="Review the controlled demonstration claim and recommend the next human review step.",
            )
        except AgentRunError as error:
            result = AgentRunStore(self.resources["session_factory"]).get_by_run_id(error.run_id)
            if result is None:
                raise
            return result

    def knowledge_retriever(self):
        """Load the real knowledge retriever, or fail the agent tool explicitly."""
        from retrieval.models import RetrievalUnavailableError

        available = ("qdrant", "knowledge_collections", "embeddings")
        if not self.config.include_rag or not all(self.preflight.available(name) for name in available):
            raise RetrievalUnavailableError("Knowledge retrieval unavailable or disabled by evaluation configuration")
        from qdrant_client import QdrantClient

        from knowledge.qdrant_store import QdrantKnowledgeStore
        from knowledge.retrieval import KnowledgeRetriever

        config = self.resources["knowledge_config"]
        client = QdrantClient(
            url=config.qdrant_url, api_key=os.environ.get("QDRANT_API_KEY"), timeout=self.config.timeout_seconds
        )
        return KnowledgeRetriever(QdrantKnowledgeStore(client, self.resources["embeddings"], config))
